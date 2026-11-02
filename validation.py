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
import matplotlib.pyplot as plt
import warnings
import cv2
import numpy as np

from camera.camera_grab import Stream
from camera.gw_analysis_stationary import analyse_frame
from cutter_functions import move_piezo

from constants import cutter_positioning as cutter
from constants import timeouts as timeout
from constants import general as gen

from api_requests import APIRequests
from mqtt_retrieval import MQTTClient


# PackML numbers used by the cutter - check these match your PLC
PACKML_STATES_BY_NAME = {name: code for code, name in cutter.PACKML_STATES.items()}
PACKML_MODES_BY_NAME = {name: code for code, name in cutter.PACKML_MODES.items()}

STATE_STOPPED = PACKML_STATES_BY_NAME["PML_StateStopped"]
STATE_IDLE = PACKML_STATES_BY_NAME["PML_StateIdle"]
STATE_EXECUTE  = PACKML_STATES_BY_NAME["PML_StateExecute"]


MODE_MAINTENANCE = PACKML_MODES_BY_NAME["ModeMaintenance"]
MODE_SETUP = PACKML_MODES_BY_NAME["ModeSetUp"]


GROOVES_PER_BAND = int(round(cutter.VALIDATION_S / cutter.REV_S))        # 10
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
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
        # stream = Stream(self.logger, cutter_no)
        camera_error = []
        self._cutter_running = False
        pass_fail = False

        self.logger.info("Starting validation")
        config = self.api.get("config")
        original_head_drop_pos = float(config["PROCESS_POSITIONS_HEADDROPPOSITION"])
        self.logger.info(f"For gws: {cutter.VALIDATION_WIDTHS}")
        piezo_positions = [((self.mmum * w) + self.y_int) for w in cutter.VALIDATION_WIDTHS]
        self.logger.info(f"Piezo positions: {piezo_positions}")
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
            # make the validation cut and sample with photos
            self._cut_and_sample(piezo_positions, latest, saved_samples)

            #iterate through the photo list and perform stationary analysis
            width_array = []
            for i in range(1,(gen.SAMPLE_NO +1),1):
                image_pathway = f"{self.log_folder}/val_sample_{i}.png"
                frame = cv2.imread(image_pathway)
                widths, land = analyse_frame(frame)
                width_array.append(np.asarray(widths).tolist())
            #self.plot_rows(width_array)

            self.logger.info(width_array)

            #ensure groove widths across all array are within tolerance
            val_widths = cutter.VALIDATION_WIDTHS

            pass_fail, data = self.check_all_sides(width_array, val_widths[::-1])
            self.logger.info(f"Validation result: {pass_fail}")
            fig = self.plot_results(width_array, data, tolerance=cutter.ABS_ERROR_MARGIN, save_path=f"{self.log_folder}/groove_width_validation.png")
                    
            finished = True

        finally:
            self.logger.info("Validation has ended")
            if self._cutter_running:
                try:
                    self.api.command("stop")
                    self.wait_for(lambda: self._state() == STATE_STOPPED, "cutter to stop after the cut")
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
        return pass_fail

    # --- cutting and photographing --- #
    def _cut_and_sample(self, piezo_positions, latest, saved_samples):
        # """Cut each validation band, then photograph the stopped disc.
        # Returns True when done, False if stopped. Raises on errors."""
        api, mqtt, stop = self.api, self.mqtt, self.stop_event

        api.set("head_drop_pos", cutter.VAL_DROP_IN)
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

        #attempt to move to initial piezo position
        move_piezo(api, mqtt, piezo_positions[0])
        self.logger.info(f"Piezo goal: {piezo_positions[0]} moved to {mqtt.moitor("piezo_actual")}")

        # start the cut
        api.command("start")
        self._cutter_running = True
        api.set("setup_cut", True)
        self.logger.info("Setup cut started")

        # move again to the next position as a double check in case not achieved
        move_piezo(api, mqtt, piezo_positions[0])
        self.logger.info("Piezo moved to starting position")

        while float(mqtt.monitor("head_pos")) > cutter.DROPIN_PITCH:
            if stop.is_set():
                return
            time.sleep(0.1)
        self.logger.info("Head has reached cutting position")

        # cut each band for VALIDATION_S
        next_move = time.monotonic() + cutter.VALIDATION_S
        self.logger.info(f"Cutting at {cutter.VALIDATION_WIDTHS[0]} um")
        for width, pos in zip(cutter.VALIDATION_WIDTHS[1:], piezo_positions[1:]):
            if not self._sleep_until(next_move):
                return
            next_move = time.monotonic() + cutter.VALIDATION_S    # band timer starts as the move is asked for
            move_piezo(api, mqtt, pos)
            self.logger.info(f"Cutting at {width} um")
            if time.monotonic() > next_move:
                self.logger.warning(f"Piezo move to {width} um took longer than {cutter.VALIDATION_S} s")
        if not self._sleep_until(next_move):               # last band gets its full window
            return

        # stop the cut
        api.command("stop")
        self.wait_for(lambda: self._state() == STATE_STOPPED, "cutter to stop after the cut")
        self._cutter_running = False
        self.logger.info("Validation cut complete")

        api.command("reset")
        self.wait_for(lambda: self._state() == STATE_IDLE, "reset to finish (IDLE)")

        #sample 4 places on the disc
        #put cutter into maintenance mode
        if self._mode() != MODE_MAINTENANCE:
            api.command("mode_maintenance")
            self.wait_for(lambda: self._mode() == MODE_MAINTENANCE, "setup mode")
                
        # start cutter for a random time period between 0 and 1.8 seconds
        photos = []
        for i in range(0,gen.SAMPLE_NO, 1):
            # turn platter for a random amount of time
            api.command("reset")
            self.wait_for(lambda: self._state() == STATE_IDLE, "reset to IDLE before sample")

            api.command("start")
            self._cutter_running = True
            r = random.uniform(0, 1.8)
            time.sleep(cutter.REV_S + r)
            api.command("stop")
            self.wait_for(lambda: self._state() == STATE_STOPPED, "cutter to stop after the cut")
            self._cutter_running = False

            #save frame
            frame = latest.get(timeout=5)
            if frame is None:
                raise RuntimeError("No camera frame received")
            path = Path(self.log_folder) / f"val_sample_{i+1}.png"
            photos.append(frame)
            cv2.imwrite(str(path), frame)
        return




    
    def find_plateaus(self, values, jump_tol=6.0, edge_tol=3.0, min_size=5):
        """
        Split `values` into blocks of roughly constant value.
    
        jump_tol : a value further than this from the running median of the
                current block starts a new block.
        edge_tol : after splitting, values at the start/end of a block further
                than this from the block's median are trimmed off (transition
                values ramping in/out of a plateau).
        min_size : blocks shorter than this after trimming are discarded
                (stray transition points, spikes).
    
        Returns a list of (start, end) index pairs, end exclusive.
        """
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return []
    
        # 1. Split wherever a value jumps away from the current block's median
        raw = []
        start = 0
        for i in range(1, len(values)):
            if abs(values[i] - np.median(values[start:i])) > jump_tol:
                raw.append((start, i))
                start = i
        raw.append((start, len(values)))
    
        # 2. Trim off-value edges and drop blocks that are too short
        blocks = []
        for s, e in raw:
            m = np.median(values[s:e])
            while s < e and abs(values[s] - m) > edge_tol:
                s += 1
            while e > s and abs(values[e - 1] - m) > edge_tol:
                e -= 1
            if e - s >= min_size:
                blocks.append((s, e))
        return blocks
    
    
    def check_against_targets(self, values, targets, tolerance=cutter.ABS_ERROR_MARGIN, drop_last=True, **plateau_kwargs):
        """
        Group `values` into plateaus, ignore the final plateau, and check every
        value in plateau i is within `tolerance` of targets[i].
    
        Returns (all_passed, results) where results is a list of dicts per block.
        """
        values = np.asarray(values, dtype=float)
        blocks = self.find_plateaus(values, **plateau_kwargs)
        if drop_last and blocks:
            blocks = blocks[:-1]
    
        if len(blocks) != len(targets):
            self.logger.warning(
                f"Found {len(blocks)} blocks but {len(targets)} targets; "
                "comparing the first {min(len(blocks), len(targets))}."
            )
    
        results = []
        for i, ((s, e), target) in enumerate(zip(blocks, targets)):
            block = values[s:e]
            bad = np.where(np.abs(block - target) > tolerance)[0] + s
            results.append({
                "block": i,
                "start": s,
                "end": e,                      # exclusive
                "median": float(np.median(block)),
                "target": target,
                "max_deviation": float(np.max(np.abs(block - target))),
                "failing_indices": bad.tolist(),
                "passed": bad.size == 0,
            })
    
        all_passed = len(blocks) == len(targets) and all(r["passed"] for r in results)
        return all_passed, results
    
    
    def check_all_sides(self, width_array, targets, tolerance=5.0, **plateau_kwargs):
        """
        width_array : list of per-side lists (e.g. sides A-D), lengths may differ.
        targets     : one list used for every side, or a list of per-side lists.
    
        Returns (all_passed, {side_index: (passed, results)}).
        """
        per_side = isinstance(targets[0], (list, tuple, np.ndarray))
        out = {}
        for i, side in enumerate(width_array):
            t = targets[i] if per_side else targets
            out[i] = self.check_against_targets(side, t, tolerance=tolerance, **plateau_kwargs)
        return all(p for p, _ in out.values()), out

    def plot_rows(self, data, labels=None, xlabel="Index", ylabel="GW", title="Validation gw measurements"):
        """Plot each row of a 4 x N array (rows may differ in length) as its own line."""
        save_path=f"{self.log_folder}/validation_graph.png"
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, row in enumerate(data):
            row = np.asarray(row, dtype=float)          # None -> NaN, shown as a gap
            label = labels[i] if labels else f"Row {i + 1}"
            ax.plot(np.arange(len(row)), row, marker=".", label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
        if save_path:
            fig.savefig(save_path, dpi=150)
        return fig, ax


    def plot_results(self, width_array, per_side, tolerance=cutter.ABS_ERROR_MARGIN, side_names="1234", save_path=None):
        """
        Plot every side's readings against its block targets.
        Each block gets its target line and a shaded ±tolerance band over its span.
        Points in a block are green (pass) or red (fail); trimmed / ignored
        readings are grey.
        """
        import matplotlib.pyplot as plt
    
        n = len(width_array)
        cols = 2 if n > 1 else 1
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.5 * rows), squeeze=False)
    
        for i, values in enumerate(width_array):
            ax = axes[i // cols][i % cols]
            values = np.asarray(values, dtype=float)
            x = np.arange(len(values))
            passed, blocks = per_side[i]
            name = side_names[i] if i < len(side_names) else str(i)
    
            in_block = np.zeros(len(values), dtype=bool)
            for b in blocks:
                s, e, t = b["start"], b["end"], b["target"]
                in_block[s:e] = True
                ax.fill_between([s - 0.5, e - 0.5], t - tolerance, t + tolerance,
                                color="tab:blue", alpha=0.15, linewidth=0)
                ax.hlines(t, s - 0.5, e - 0.5, colors="tab:blue", linewidth=2)
                ax.text((s + e - 1) / 2, t - tolerance - 1, f"target {t:g}", ha="center",
                        va="top", fontsize=8, color="tab:blue")
    
                idx = np.arange(s, e)
                fail = np.isin(idx, b["failing_indices"])
                ax.scatter(idx[~fail], values[s:e][~fail], c="tab:green", s=18, zorder=3)
                ax.scatter(idx[fail], values[s:e][fail], c="tab:red", s=18, zorder=3)
    
            ax.scatter(x[~in_block], values[~in_block], c="lightgrey", s=18, zorder=2)
            ax.plot(x, values, color="grey", linewidth=0.6, alpha=0.5, zorder=1)
    
            ax.set_title(f"Sample {name} — {'PASS' if passed else 'FAIL'}",
                        color="tab:green" if passed else "tab:red", fontweight="bold")
            ax.set_xlabel("Reading index")
            ax.set_ylabel("Groove width")
            ax.grid(alpha=0.3)
    
        for j in range(n, rows * cols):
            axes[j // cols][j % cols].axis("off")
    
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        handles = [
            Line2D([], [], color="tab:blue", linewidth=2, label="Target"),
            Patch(color="tab:blue", alpha=0.15, label=f"±{tolerance:g} tolerance"),
            Line2D([], [], marker="o", ls="", color="tab:green", label="Within tolerance"),
            Line2D([], [], marker="o", ls="", color="tab:red", label="Out of tolerance"),
            Line2D([], [], marker="o", ls="", color="lightgrey", label="Ignored (transition / final block)"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
    
        if save_path:
            fig.savefig(save_path, dpi=150)
        return fig

api = APIRequests("1", "3.1.1")
mqtt = MQTTClient("1", "3.1.1")
stop_event = threading.Event()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
val = ValidationSide(logger, r"C:\Users\KittyGoodridge\OneDrive - elasticStage\Desktop\Repos\Auto_Cal_Tests\logs\3.1.1_20261102_112724", stop_event, api, mqtt, 0.0196, 0.51, frame_sink=None)
pass_or_fail = val.run_validation("3.1.1")
print(pass_or_fail)