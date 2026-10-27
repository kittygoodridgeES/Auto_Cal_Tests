"""
Validation cut: cuts a band of grooves at each target width, then photographs the
stationary disc several times and checks the measured widths against the targets.

run_validation() returns True (pass), False (fail) or None (stopped before finishing).
"""

import logging
import random
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from camera.camera_grab import Stream
from camera.gw_analysis_stationary import analyse_frame
from cutter_functions import move_piezo

from constants import cutter_positioning as cutter
from constants import timeouts as timeout


# PackML numbers used by the cutter - check these match your PLC
PACKML_STATES_BY_NAME = {name: code for code, name in cutter.PACKML_STATES.items()}
PACKML_MODES_BY_NAME = {name: code for code, name in cutter.PACKML_MODES.items()}

STATE_STOPPED = PACKML_STATES_BY_NAME["PML_StateStopped"]
STATE_IDLE = PACKML_STATES_BY_NAME["PML_StateIdle"]
STATE_EXECUTE  = PACKML_STATES_BY_NAME["PML_StateExecute"]


MODE_MAINTENANCE = PACKML_MODES_BY_NAME["ModeMaintenance"]
MODE_SETUP = PACKML_MODES_BY_NAME["ModeSetUp"]


GROOVES_PER_BAND = round(cutter.VALIDATION_S / cutter.REV_S)        # 10
SAMPLES_PER_BAND = round(GROOVES_PER_BAND / 2)        # 5 measured grooves per band


if cutter.PIEZO_GROOVES + SAMPLES_PER_BAND > GROOVES_PER_BAND:
    raise ValueError("Not enough grooves in each band - increase cutter.VALIDATION_S or reduce cutter.PIEZO_GROOVES")


class LatestFrame:
    """Holds only the newest camera frame; written by the camera thread."""

    def __init__(self):
        self._lock = threading.Lock()
        self._new = threading.Event()
        self._img = None
        self.received = 0

    def put(self, img):
        # convert now: gives our own copy, so the camera can safely reuse its buffer
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.gui.COLOR_BGR2GRAY)
        with self._lock:
            self._img = gray
            self.received += 1
        self._new.set()

    def get(self, timeout):
        """Newest frame not yet returned, or None if none arrives within timeout."""
        if not self._new.wait(timeout):
            return None
        with self._lock:
            self._new.clear()
            return self._img


class ValidationSide:
    def __init__(self, logger, log_folder, stop_event, api, mqtt, mmum, y_int, frame_sink=None):
        self.logger = logger
        self.log_folder = log_folder
        self.stop_event = stop_event
        self.api = api
        self.mqtt = mqtt
        self.mmum = float(mmum)
        self.y_int = float(y_int)
        self.frame_sink = frame_sink      # called with frames for the GUI camera panel
        self._cutter_running = False      # True while the cutter needs stopping on exit
        self._last_shown = 0.0

    # --- helpers --- #
    def _state(self):
        return int(self.api.get("packml_state"))

    def _mode(self):
        return int(self.api.get("packml_mode"))

    def wait_for(self, condition, description, timeout_=timeout.STATE_TIMEOUT, poll=0.1):
        """Poll condition() until it returns True, or raise TimeoutError."""
        deadline = time.monotonic() + timeout_
        while not condition():
            if self.stop_event.is_set():
                raise RuntimeError(f"Stopped while waiting for: {description}")
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Timed out after {timeout_}s waiting for: {description} "
                    f"(packml_state={self.api.get('packml_state')}, "
                    f"packml_mode={self.api.get('packml_mode')})")
            time.sleep(poll)

    def _sleep_until(self, deadline):
        """Wait until deadline. Returns False if stopped first."""
        while time.monotonic() < deadline:
            if self.stop_event.is_set():
                return False
            time.sleep(0.05)
        return True

    def _publish(self, img):
        """Send a scaled-down copy of a camera frame to the GUI, about 10 times a second."""
        if self.frame_sink is None:
            return
        now = time.monotonic()
        if now - self._last_shown < 0.1:
            return
        self._last_shown = now
        s = min(1.0, 1280 / img.shape[1])
        self.frame_sink(cv2.resize(img, None, fx=s, fy=s) if s < 1 else img.copy())

    # --- main entry point --- #
    def run_validation(self, cutter_no):
        """Returns True (pass), False (fail) or None (stopped before finishing)."""
        stop = self.stop_event
        latest = LatestFrame()
        stream = Stream(self.logger, cutter_no)
        camera_error = []
        self._cutter_running = False

        self.logger.info("Starting validation")
        config = self.api.get("config")
        original_head_drop_pos = float(config["PROCESS_POSITIONS_HEADDROPPOSITION"])
        piezo_positions = [self.mmum * w + self.y_int for w in cutter.VALIDATION_WIDTHS]
        for w, p in zip(cutter.VALIDATION_WIDTHS, piezo_positions):
            self.logger.info(f"Target {w} um -> piezo {p:.3f}")

        # start the camera
        def on_frame(img):
            latest.put(img)
            self._publish(img)

        def run_camera():
            try:
                stream.run(stop, frame_callback=on_frame)
            except BaseException as e:
                camera_error.append(e)
            finally:
                stop.set()

        cam_thread = threading.Thread(target=run_camera, name="camera", daemon=True)
        cam_thread.start()

        # make sure the camera is running before changing anything on the cutter
        if latest.get(timeout=15) is None or camera_error:
            stop.set()
            cam_thread.join(timeout=5)
            raise RuntimeError(f"Camera did not start "
                               f"({camera_error[0] if camera_error else 'no frames'!r}) "
                               "- cutter left unchanged")
        self.logger.info("Camera running")

        saved_samples = []
        finished = False
        try:
            finished = self._cut_and_sample(piezo_positions, latest, saved_samples)
        finally:
            if self._cutter_running:
                try:
                    self.api.command("stop")
                    self.logger.info("Cutter stopped during clean-up")
                except Exception:
                    self.logger.exception("Could not stop the cutter during clean-up")
            stop.set()                                    # ends the camera thread
            cam_thread.join(timeout=5)
            if cam_thread.is_alive():
                self.logger.error("Camera thread still running after 5 s - camera may not be released")
            try:
                self.api.set("head_drop_pos", original_head_drop_pos)
                self.logger.info(f"Head drop position restored to {original_head_drop_pos}")
            except Exception:
                self.logger.exception("Could not restore head drop position")

        if camera_error:
            raise RuntimeError(f"Camera stopped during validation: {camera_error[0]!r}")
        if not finished:
            self.logger.info("Validation stopped before it finished")
            return None
        return self._analyse(saved_samples)

    # --- cutting and photographing --- #
    def _cut_and_sample(self, piezo_positions, latest, saved_samples):
        """Cut each validation band, then photograph the stopped disc.
        Returns True when done, False if stopped. Raises on errors."""
        api, mqtt, stop = self.api, self.mqtt, self.stop_event

        api.set("head_drop_pos", cutter.DROPIN_PITCH)
        self.logger.info(f"Drop in pitch set to {cutter.DROPIN_PITCH}")

        # cutter stopped -> setup mode -> reset to IDLE
        if self._state() != STATE_STOPPED:
            api.command("stop")
            self.wait_for(lambda: self._state() == STATE_STOPPED, "cutter to stop")
        if self._mode() != MODE_SETUP:
            api.command("mode_setup")
            self.wait_for(lambda: self._mode() == MODE_SETUP, "setup mode")
        else:
            self.logger.info("Cutter already in setup mode")
        api.command("reset")
        self.wait_for(lambda: self._state() == STATE_IDLE, "reset to finish (IDLE)")
        self.logger.info("Cutter reset")

        # first position is set while the cutter is idle
        move_piezo(api, mqtt, piezo_positions[0])
        self.logger.info("Piezo moved to starting position")

        # start the cut
        api.command("start")
        self._cutter_running = True
        api.set("setup_cut", True)
        self.logger.info("Setup cut started")

        while float(mqtt.monitor("head_pos")) > cutter.DROPIN_PITCH:
            if stop.is_set():
                return False
            time.sleep(0.1)
        self.logger.info("Head has reached cutting position")

        # cut each band for VALIDATION_S
        next_move = time.monotonic() + cutter.VALIDATION_S
        self.logger.info(f"Cutting at {cutter.VALIDATION_WIDTHS[0]} um")
        for width, pos in zip(cutter.VALIDATION_WIDTHS[1:], piezo_positions[1:]):
            if not self._sleep_until(next_move):
                return False
            next_move = time.monotonic() + cutter.VALIDATION_S    # band timer starts as the move is asked for
            move_piezo(api, mqtt, pos)
            self.logger.info(f"Cutting at {width} um")
            if time.monotonic() > next_move:
                self.logger.warning(f"Piezo move to {width} um took longer than {cutter.VALIDATION_S} s")
        if not self._sleep_until(next_move):               # last band gets its full window
            return False

        # stop the cut
        api.command("stop")
        self.wait_for(lambda: self._state() == STATE_STOPPED, "cutter to stop after the cut")
        self._cutter_running = False
        self.logger.info("Validation cut complete")

        api.command("reset")
        self.wait_for(lambda: self._state() == STATE_IDLE, "reset to finish (IDLE)")

# logger = get_logger()

# val = ValidationSide(logger, r"logs\3.2.4_20261026_164214", cal_stop, api, mqtt,
#                          ctx.values["mmum"], ctx.values["y_int"], frame_sink=ctx.frames.set)
#     try:
#         pass_fail = val.run_validation(ctx.selection.machine_id)