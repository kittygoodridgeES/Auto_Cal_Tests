"""
Live groove width using the Allied Vision (VmbPy) camera code, unchanged.

    python groove_live_av.py 2.5        # cutter number, as used in camera-settings.txt

The manufacturer's Stream.run() is blocking, so it runs in a background
thread. Its frame_callback only keeps the newest frame (converted to
grayscale, which also makes our own copy of the pixels). The main thread
analyses the newest frame each time, so if analysis is slower than the
camera, old frames are skipped instead of building up a delay.

Keys:  q = quit   r = reset (re-scan for the innermost groove)
"""

import logging
import sys
import threading
import time
import json
import os
import csv
import dotenv
import cv2
import numpy as np
from scipy import stats
from datetime import datetime
from pathlib import Path


from camera.camera_grab import Stream
from camera.gw_analysis_live import GrooveMonitor
from api_requests import APIRequests
from mqtt_retrieval import MQTTClient
from cutter_functions import move_piezo
from airtable_retrieval import AirtableReader
from constants import cutter_positioning as cutter
from constants import timeouts as timeouts
from constants import camera_constants as cam_const

class LatestFrame:
    """Holds only the newest camera frame; written by the camera thread."""

    def __init__(self):
        self._lock = threading.Lock()
        self._new = threading.Event()
        self._img = None
        self.received = 0

    def put(self, img):                  # used as Stream.run(frame_callback=...)
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

class CalibrationSide:
    def __init__(self, logger, log_folder, stop_event, api, mqtt, change_in_diamond_length, frame_sink=None, new_drop_in = None):
        self.logger = logger
        self.log_folder = log_folder
        self.stop_event = stop_event
        self.api = api
        self.mqtt = mqtt
        self.change_in_diamond_length = change_in_diamond_length
        self.new_drop_in = new_drop_in
        self.frame_sink = frame_sink      # called with each annotated BGR frame for display


    def save_samples(self, samples):
        path = Path(self.log_folder) / f"groove_piezo_{datetime.now():%Y%m%d_%H%M%S}.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "step", "piezo_pos", "groove", "average_5_um", "width_um"])
            w.writerows(samples)
        self.logger.info("saved %d samples to %s", len(samples), path)
        return path


    def show(self, frame, mon, text):
        disp = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if mon.r_lock is not None:           # red line along the measured groove
            pts = np.stack([mon.centre[0] + mon.r_lock * mon.cosT,
                            mon.centre[1] + mon.r_lock * mon.sinT], 1)
            pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < disp.shape[1]) &
                    (pts[:, 1] >= 0) & (pts[:, 1] < disp.shape[0])]
            if len(pts) > 1:
                cv2.polylines(disp, [pts.astype(np.int32)], False, (0, 0, 255), 3)
        s = min(1.0, 1280 / disp.shape[1])
        if s < 1:
            disp = cv2.resize(disp, None, fx=s, fy=s)
        cv2.putText(disp, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if self.frame_sink is not None:
            self.frame_sink(disp)

    def calculate_drop_in_pos(self, default_piezo_pos, grams_per_micron, default_groove_width, change_in_diamond_length):
        """
        Calculates a safe drop in position based on the change in diamond length from the previous calibration
        """
        #determine the linear relationship
        y_int = float(default_piezo_pos) - (float(grams_per_micron) * float(default_groove_width))

        #shift it based on the diamond length
        y_int_new = y_int + change_in_diamond_length

        #find piezo pos at 90 micron
        piezo_position = (grams_per_micron*cutter.DROP_IN_AIM) + y_int_new

        return piezo_position

    def wait_for(self, condition, description, api,  timeout=timeouts.STATE_TIMEOUT, poll=0.1):
        """
        Poll `condition()` until it returns True, or raise TimeoutError
        if it hasn't happened within `timeout` seconds.
        """
        deadline = time.monotonic() + timeout
        while not condition():
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Timed out after {timeout}s waiting for: {description} "
                    f"(packml_state={api.get('packml_state')}, "
                    f"packml_mode={api.get('packml_mode')})"
                )
            time.sleep(poll)

    def run_calibration(self, cutter_no):

        dotenv.load_dotenv()

        stop = self.stop_event
        latest = LatestFrame()
        stream = Stream(self.logger, cutter_no)
        camera_error = []
        self.csv_path = None
        
        
        #1. collect previous config information
        config = self.api.get("config")
        grams_per_micron = float(config["HEAD_DROP_AND_CALIBRATION_CFG_GRAMS_PER_MICRON"])
        default_headweight = float(config["HEAD_DROP_AND_CALIBRATION_CFG_DEFAULT_WEIGHT"])
        default_groove_width = float(config["HEAD_DROP_AND_CALIBRATION_CFG_DEFAULT_WIDTH"])
        original_head_drop_pos = float(config["PROCESS_POSITIONS_HEADDROPPOSITION"])
        default_piezo_pos = self.mqtt.monitor("piezo_default_pos")
        self.logger.info(f"Original Config Retrieved: \n mm/um:{grams_per_micron} \n Default piezo pos: {default_piezo_pos} \n Default GW: {default_groove_width}")

        #2. find new drop in piezo position based on change in diamond length and previous calibration
            #validate diamond length
        if abs(self.change_in_diamond_length) > 3:
            raise RuntimeError(f"Change in diamond length is {self.change_in_diamond_length} mm "
                               "(more than 3 mm) - investigate before calibrating")
        self.logger.info(f"Change in diamond length from previous calibration: {self.change_in_diamond_length}")

        if self.new_drop_in != None:
            drop_in_piezo_pos = self.new_drop_in
            self.logger.info(f"Drop in piezo position: {drop_in_piezo_pos}"
                             f"Decided based on manual adjustment that was made")

        else:
            drop_in_piezo_pos = self.calculate_drop_in_pos(default_piezo_pos, grams_per_micron,default_groove_width, self.change_in_diamond_length)
            self.logger.info(f"Drop in piezo pos decided: {drop_in_piezo_pos}"
                             f"Drop in position estimated from change in diamond length")

        #3. start camera
        def run_camera():
            try:
                stream.run(stop, frame_callback=latest.put)
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
                               f"({camera_error[0] if camera_error else 'no frames'!r}) - cutter left unchanged")
        self.logger.info("Camera running")

        samples = []                      # rows: time, step, piezo_pos, groove, average, width

        try:
            #4. start cutter into setup mode 
                #set new head drop position
            self.api.set("head_drop_pos", cutter.DROPIN_PITCH)
            self.logger.info(f"Drop in pitch set from {original_head_drop_pos} to {cutter.DROPIN_PITCH}")

                #check if cutter is stopped
            if self.api.get("packml_state") != 2:
                self.api.command("stop")
                self.wait_for(lambda: self.api.get("packml_state") == 2, "cutter to stop", self.api)

                #put cutter into setup mode
            if self.api.get("packml_mode") != 6:
                self.api.command("mode_setup")
                self.wait_for(lambda: self.api.get("packml_mode") == 6, "setup mode", self.api)
            else:
                self.logger.info("Cutter already in setup mode")

                #reset the cutter
            self.api.command("reset")
            self.logger.info(f"Cutter resetting")
            self.wait_for(lambda: self.api.get("packml_state") == 4, "reset to finish (IDLE)", self.api)
            self.logger.info("Cutter reset")

                # move piezo to starting position post cut
            self.logger.info(f"Moving piezo to starting position \n ===")
            move_piezo(self.api, self.mqtt, drop_in_piezo_pos)
            self.logger.info(f"Piezo moved to starting pos")

                #start the cutter
            self.api.command("start")
            self.logger.info("starting cutter")


                #starting cut
            self.api.set("setup_cut", True)
            self.logger.info("starting setup cut")
            self.logger.info("Setup cut has started")


                #wait until headdrop reaches correct position
            while float(self.mqtt.monitor("head_pos")) > cutter.DROPIN_PITCH:
                if stop.is_set():
                    return
                time.sleep(0.1)
            self.logger.info("Head has reached cutting position")
            t_head = time.perf_counter()      # start of the NO_GROOVE_S window
            
            #4. start analysis
                #wait for cut to reach the camera
            time.sleep(cutter.DELAY_CUT_TO_FRAME)
                #start analysis of frames
            mon = GrooveMonitor()
            analysed = 0

            t_start = time.perf_counter()

            def move_and_wait(api, mqtt, pos):
                """Move the piezo, wait DELAY_CUT_TO_FRAME, then start fresh at the new position."""
                self.logger.info("moving piezo to %s", pos)
                move_piezo(api, mqtt, pos)
                time.sleep(cutter.DELAY_CUT_TO_FRAME)
                latest.get(timeout=0)         # drop the frame captured before/while moving
                mon.recent.clear()            # 5-frame average only uses frames after the move
                return time.perf_counter()    # time we arrived at this position + MEASURE_S
                
            pos = round(float(self.mqtt.monitor("piezo_actual")), 2)
            step = 0
            endflag = False
            moving_heavier = True
            moving_lighter = False
            window_end = None                 # set when the first full 5-frame average arrives
            pos_start = time.perf_counter()
            last_width = None                 # last single-frame width that wasn't a heavy-limit hit
            last_width_t = None
            initial_phase = True              # until the first position finishes measuring (or a heavy-limit hit)
            groove_seen = False               # any groove width seen since the head reached cutting position
            no_groove_bumped = False          # the NO_GROOVE_STEP increase is only done once
        
            while not stop.is_set():
                frame = latest.get(timeout=0.5)
                if frame is None:
                    continue
                t0 = time.perf_counter()
                res = mon.update(frame)
                ms = (time.perf_counter() - t0) * 1000
                analysed += 1
                now_t = time.perf_counter()

                width = res["width"]
                if width is not None:
                    groove_seen = True

                # ---- no groove within NO_GROOVE_S of the head reaching cutting position:
                #      increase the piezo by NO_GROOVE_STEP (capped at PIEZO_UPPER_LIMIT), once
                if not groove_seen and not no_groove_bumped and now_t - t_head > timeouts.NO_GROOVE_S:
                    no_groove_bumped = True
                    current_pos = float(self.mqtt.monitor("piezo_actual"))
                    new_pos = min(current_pos + NO_GROOVE_STEP, cutter.PIEZO_UPPER_LIMIT)
                    if new_pos > current_pos:
                        self.logger.warning("no groove seen %.0f s after head reached cutting position - "
                                            "increasing piezo %.2f -> %.2f", timeouts.NO_GROOVE_S, current_pos, new_pos)
                        step += 1
                        pos_start = move_and_wait(self.api, self.mqtt, new_pos)
                        window_end = None
                        last_width = None
                        pos = round(new_pos, 2)
                        self.logger.info("drop-in piezo position updated %.2f -> %.2f",
                                         drop_in_piezo_pos, new_pos)
                        drop_in_piezo_pos = new_pos
                    else:
                        self.logger.warning("no groove seen %.0f s after head reached cutting position, "
                                            "piezo already at %.2f - not increasing", timeouts.NO_GROOVE_S, current_pos)
                    continue

                # ---- heavy limit: merged grooves (width nearly doubles quickly) or width over MAX_SAFE_GW.
                #      Treated the same as reaching PIEZO_UPPER_LIMIT: go back to the drop-in position
                #      minus HEAVY_STEP and carry on in the lighter direction.
                heavy_reason = None
                if width is not None:
                    if (last_width is not None
                            and now_t - last_width_t <= cutter.MERGE_WINDOW_S
                            and width >= last_width * cutter.MERGE_RATIO):
                        heavy_reason = (f"merged groove detected ({last_width:.1f} -> {width:.1f} um "
                                        f"in {now_t - last_width_t:.2f} s)")
                    elif width > cutter.MAX_SAFE_GW:
                        heavy_reason = f"unsafe heavy groove width: {width:.1f} um"
                    else:
                        last_width, last_width_t = width, now_t

                if heavy_reason is not None:
                    self.logger.warning("%s at piezo %s - treating as heavy limit", heavy_reason, pos)
                    initial_phase = False
                    moving_heavier = False
                    moving_lighter = True
                    current_pos = float(self.mqtt.monitor("piezo_actual"))
                    # back to the drop-in position minus one step; if already at or below that, one step lighter
                    new_pos = min(drop_in_piezo_pos, current_pos) - cutter.HEAVY_STEP
                    step += 1                 # samples after this belong to the new position
                    if new_pos <= cutter.PIEZO_LOWER_LIMIT:
                        self.logger.warning("cannot back off below %.2f (target %.2f) - stopping",
                                            cutter.PIEZO_LOWER_LIMIT, new_pos)
                        endflag = True
                        break
                    pos_start = move_and_wait(self.api, self.mqtt, new_pos)
                    window_end = None
                    last_width = None         # don't compare against widths from the old position
                    pos = round(new_pos, 2)
                    continue                  # don't record the merged / over-limit frame

                # ---- too light while still on the first position: go heavier by HEAVY_STEP instead of stopping
                if initial_phase and width is not None and width < cutter.MIN_SAFE_GW:
                    current_pos = float(self.mqtt.monitor("piezo_actual"))
                    new_pos = min(current_pos + cutter.HEAVY_STEP, cutter.PIEZO_UPPER_LIMIT)
                    if new_pos <= current_pos:
                        self.logger.warning("light groove width %.1f um at piezo %.2f, already at upper limit "
                                            "- stopping cutter", width, current_pos)
                        endflag = True
                        self.api.command("stop")
                        self.wait_for(lambda: self.api.get("packml_state") == 2, "cutter to stop", self.api)
                        break
                    self.logger.warning("light groove width %.1f um on first position - increasing piezo "
                                        "%.2f -> %.2f", width, current_pos, new_pos)
                    step += 1
                    pos_start = move_and_wait(self.api, self.mqtt, new_pos)
                    window_end = None
                    last_width = None
                    pos = round(new_pos, 2)
                    self.logger.info("drop-in piezo position updated %.2f -> %.2f",
                                     drop_in_piezo_pos, new_pos)
                    drop_in_piezo_pos = new_pos
                    continue                  # don't record the too-light frame

                # ---- record; the measuring clock only starts once a groove is being measured
                if res["n"] == cam_const.N_AVG and res["width"] is not None:
                    if window_end is None:
                        window_end = now_t + cutter.MEASURE_S
                        self.logger.info("groove found at piezo %s, measuring for %.1f s", pos, cutter.MEASURE_S)
                    samples.append((round(now_t - t_start, 3), step, pos, res["groove"],
                                    res["average"], res["width"]))
                n_here = sum(1 for s in samples if s[1] == step)

                # (only reached after the first position - during it, light widths go heavier above)
                if res["width"] is not None:
                    if res["width"] < cutter.MIN_SAFE_GW:
                        self.logger.warning(f"Cutting is at an unsafe light groove width: {res['width']}")
                        moving_lighter = False
                        endflag = True
                        #immediately stop cutter
                        self.api.command("stop")
                        self.wait_for(lambda: self.api.get("packml_state") == 2, "cutter to stop", self.api)


                if window_end is None:
                    if now_t - pos_start > timeouts.GROOVE_TIMEOUT_S:
                        raise RuntimeError(f"No groove measured at piezo {pos} within "
                                        f"{timeouts.GROOVE_TIMEOUT_S} s - stopping without moving further")
                    status = f"pos {pos}: waiting for groove ({now_t - pos_start:.0f} s)"
                else:
                    status = f"pos {pos}: {max(window_end - now_t, 0):.1f} s left, {n_here} samples"

                # ---- move on only after the full window AND enough samples
                if window_end is not None and now_t >= window_end and n_here >= cutter.MIN_SAMPLES:
                    vals = np.array([s[4] for s in samples if s[1] == step])
                    if len(vals):
                        self.logger.info("piezo %s: %d samples, mean %.2f um, sd %.2f um",
                                    pos, len(vals), vals.mean(), vals.std(ddof=1) if len(vals) > 1 else 0.0)
                    else:
                        self.logger.warning("piezo %s: no complete 5-frame averages recorded", pos)
                    step += 1
                    initial_phase = False
                    if endflag:
                        self.logger.info("piezo sequence complete")
                        break

                    if moving_heavier:
                        new_pos = float(self.mqtt.monitor("piezo_actual")) + 0.2
                        if new_pos >= cutter.PIEZO_UPPER_LIMIT:
                            self.logger.info("Reached heaviest possible piezo position")
                            moving_lighter = True
                            moving_heavier = False
                        else:
                            pos_start = move_and_wait(self.api, self.mqtt, new_pos)
                            window_end = None
                            last_width = None
                    
                    if moving_lighter:
                        current_pos = float(self.mqtt.monitor("piezo_actual"))
                        if current_pos > drop_in_piezo_pos:
                            #initial movement from moving heavier to moving lighter
                            #first move to default pos -0.2
                            new_pos = drop_in_piezo_pos - 0.2
                        else:
                            new_pos = current_pos - 0.2
                        
                        if new_pos <= cutter.PIEZO_LOWER_LIMIT:
                            self.logger.info("reached lightest default piezo position")
                            moving_lighter = False
                            endflag = True
                            break
                            
                        else:
                            pos_start = move_and_wait(self.api, self.mqtt, new_pos)
                            window_end = None
                            last_width = None
                    #for logging purposes
                    pos = round(new_pos, 2)
                    continue

                # ---- display
                if res["n"] < N_AVG:
                    reading = "waiting for 5 frames"
                else:
                    now = "--" if res["width"] is None else f"{res['width']:.1f}"
                    reading = (f"groove {res['groove']}  now {now} um  "
                            f"avg({res['n']}/{N_AVG}) {res['average']:.1f} um  [{ms:.0f} ms]")
                text = f"{status} | {reading}"
                self.logger.info("%s  (frames: %d received, %d analysed)", text, latest.received, analysed)

                self.show(frame, mon, text)

        except KeyboardInterrupt:
            pass
        finally:
            stop.set()
            self.api.command("stop")
            self.wait_for(lambda: self.api.get("packml_state") == 2, "cutter stopped", self.api)

            
            #reset the cutter
            self.api.command("reset")
            self.logger.info(f"Cutter resetting")
            self.wait_for(lambda: self.api.get("packml_state") == 4, "reset to finish (IDLE)", self.api)
            self.logger.info("Cutter reset")

            cam_thread.join(timeout=5)
            if cam_thread.is_alive():
                self.logger.error("Camera thread still running after 5 s - camera may not be released")
            if samples:
                self.csv_path = self.save_samples(samples)
                
            #resets the headdrop position to original collected from config before closing
            self.api.set("head_drop_pos", original_head_drop_pos)

        if camera_error:
            raise SystemExit(f"camera stopped: {camera_error[0]!r}")
        return self.csv_path


# if __name__ == "__main__":
#     CUTTER_NO = "3.2.2"     # <- edit here
#     FACTORY_ID = "1"        # <- edit here
#     main(CUTTER_NO, FACTORY_ID)