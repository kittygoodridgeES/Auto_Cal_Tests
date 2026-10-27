"""
Live groove width from a camera - follows the newest (innermost) groove.

    python groove_live.py            # camera 0
    python groove_live.py video.mp4  # or a recorded file

The grooves are cut as an inward spiral, so the groove being measured is the
innermost complete groove (closest to the disc centre), and it moves inward
as cutting progresses.

Speed-up: only a thin band (± BAND_UM) around the tracked groove's radius is
unwrapped and analysed. Every frame the band re-centres on the innermost
groove it finds, so it follows the spiral inward. On the first frame (or if
the groove is lost) the frame is scanned from the inside outwards to find it.

A newly cut groove that has only partly entered the frame is ignored until it
covers MIN_COVERAGE of the frame, then measuring switches to it and the
5-frame average restarts, so two grooves are never mixed.

Edge method unchanged from groove_widths.py: edges at the steepest
land-to-groove drop, no calibration, only the pixel size is used.

Keys:  q = quit   r = reset (re-scan for the innermost groove)
"""

import sys
import time
from collections import deque

import cv2
import numpy as np

from camera.gw_analysis_stationary import UM_PER_PX, SECTION_PX, CENTRE, _grooves_in_section, _track, _robust_mean
from constants import camera_constants as cam_const


class GrooveMonitor:
    def __init__(self, centre=CENTRE, um_per_px=UM_PER_PX, n_avg=cam_const.N_AVG, crop=cam_const.CROP,
                 band_um=cam_const.BAND_UM, min_coverage=cam_const.MIN_COVERAGE, new_groove_px=None):
        """
        centre        : disc centre (x, y) in FULL-frame pixels (the crop does not change it)
        crop          : pixels ignored at each edge of the frame
        band_um       : half-height of the band analysed around the tracked groove
        min_coverage  : how much of the visible arc a groove must span before it is used
        new_groove_px : radial jump (px) that counts as a new groove;
                        default = the measured groove width in px
        """
        self.centre = np.asarray(centre, float)
        self.um_per_px = um_per_px
        self.crop = crop
        self.half = int(round(band_um / um_per_px))
        self.min_coverage = min_coverage
        self.new_groove_px = new_groove_px
        self.recent = deque(maxlen=n_avg)
        self.groove_count = 0
        self.r_lock = None          # radius (px) of the groove being tracked
        self._last_r = None
        self._shape = None

    def reset(self):
        self.recent.clear()
        self.r_lock = None
        self._last_r = None

    # -- geometry (built once per frame size) ---------------------------
    def _geometry(self, shape):
        h, w = shape
        c, m = self.centre, self.crop
        x0, x1, y0, y1 = m, w - 1 - m, m, h - 1 - m
        cx = np.array([x0, x1, x0, x1], float)
        cy = np.array([y0, y0, y1, y1], float)
        self.rmin = np.hypot(max(0.0, x0 - c[0], c[0] - x1), max(0.0, y0 - c[1], c[1] - y1))
        self.rmax = np.hypot(cx - c[0], cy - c[1]).max()
        ang = np.unwrap(np.sort(np.arctan2(cy - c[1], cx - c[0])))
        T = np.arange(ang.min(), ang.max(), 2 / (self.rmin + self.rmax))   # ~1 px along the groove
        self.cosT, self.sinT = np.cos(T), np.sin(T)
        self.box = (x0, x1, y0, y1)
        self._shape = shape

    def _band(self, g, r_lo, r_hi):
        """Flattened polar strip for radii r_lo..r_hi (rows = 1 px of radius)."""
        R = np.arange(max(r_lo, self.rmin), min(r_hi, self.rmax))
        if len(R) < 50:
            return None, 0.0
        X = (self.centre[0] + R[:, None] * self.cosT).astype(np.float32)
        Y = (self.centre[1] + R[:, None] * self.sinT).astype(np.float32)
        x0, x1, y0, y1 = self.box
        out = (X < x0) | (X > x1) | (Y < y0) | (Y > y1)
        cols = ~out.all(0)                                    # drop arc outside the frame
        X, Y, out = X[:, cols], Y[:, cols], out[:, cols]
        P = cv2.remap(g, X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        P = cv2.GaussianBlur(P, (0, 0), 1)
        P[out] = np.nan
        # lighting: land level of each column, smoothed along the groove
        land = np.percentile(np.nan_to_num(P, nan=0.0), 90, axis=0).astype(np.float32)
        land = cv2.blur(land.reshape(1, -1), (61, 1)).ravel()
        return P / np.maximum(land, 1e-6), R[0]

    def _grooves(self, ratio, r0):
        """Complete grooves in a strip -> list of (radius px, width µm), inner -> outer."""
        nsec = ratio.shape[1] // SECTION_PX
        try:
            E = _track([_grooves_in_section(ratio, j) for j in range(nsec)])
        except RuntimeError:
            return []
        out = []
        for e in E:
            found = ~np.isnan(e[:, 0])
            if found.sum() < 3:
                continue
            row = int(round(np.nanmedian(e.mean(1))))
            # sections where this groove's radius is inside the (cropped) frame
            vis = ~np.isnan(ratio[row, :nsec * SECTION_PX]).reshape(nsec, SECTION_PX)
            valid = vis.mean(1) > 0.5
            if found.sum() < self.min_coverage * max(valid.sum(), 1):
                continue                                      # newly cut, not far enough in yet
            out.append((r0 + np.nanmedian(e.mean(1)),
                        _robust_mean((e[:, 1] - e[:, 0]) * self.um_per_px)))
        return out

    # -- measurement ---------------------------------------------------
    def measure(self, frame):
        """(width µm, radius px) of the innermost complete groove, or None."""
        g = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.gui.COLOR_BGR2GRAY)
        if g.shape != self._shape:
            self._geometry(g.shape)
            self.r_lock = None

        if self.r_lock is not None:                           # track: band around last groove
            ratio, r0 = self._band(g, self.r_lock - self.half, self.r_lock + self.half + 1)
            found = [] if ratio is None else self._grooves(ratio, r0)
        else:                                                 # search: inside -> outwards
            found, r = [], self.rmin
            while not found and r < self.rmax:
                ratio, r0 = self._band(g, r, r + 2 * self.half + 1)
                if ratio is not None:
                    found = self._grooves(ratio, r0)
                r += self.half                                # bands overlap by half
        if not found:
            return None
        radius, width = found[0]                              # innermost
        self.r_lock = radius
        return width, radius

    # -- per-frame update ----------------------------------------------
    def update(self, frame):
        """
        Returns dict:
          width   : this frame's width (µm) or None
          average : mean of the last <= 5 frames for the current groove (µm)
          n       : frames in the average (5 = full)
          groove  : running groove number
          radius  : radius of the measured groove (px from disc centre)
        """
        m = self.measure(frame)
        if m is not None:
            width, r = m
            jump = self.new_groove_px or max(width / self.um_per_px, 5.0)
            if self._last_r is None or abs(r - self._last_r) > jump:
                self.recent.clear()     # a new groove: start a fresh average
                self.groove_count += 1
            self._last_r = r
            self.recent.append(width)
        avg = round(float(np.mean(self.recent)), 1) if self.recent else None
        return dict(width=None if m is None else round(m[0], 1), average=avg,
                    n=len(self.recent), groove=self.groove_count,
                    radius=None if self.r_lock is None else round(self.r_lock, 1))


# -------------------------------------------------------------------- live loop
def main(source=0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera/source {source!r}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     # always process the newest frame

    mon = GrooveMonitor()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        res = mon.update(frame)
        ms = (time.perf_counter() - t0) * 1000

        if res["average"] is None:
            text = "searching for groove"
        else:
            now = "--" if res["width"] is None else f"{res['width']:.1f}"
            text = (f"groove {res['groove']}  now {now} um  "
                    f"avg({res['n']}/{cam_const.N_AVG}) {res['average']:.1f} um  [{ms:.0f} ms]")
        print(text)

        disp = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        disp = disp.copy()
        if mon.r_lock is not None:                            # draw the tracked groove
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
        cv2.imshow("groove width", disp)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("r"):
            mon.reset()

    cap.release()
    cv2.destroyAllWindows()


# if __name__ == "__main__":
#     src = sys.argv[1] if len(sys.argv) > 1 else 0
#     main(int(src) if str(src).isdigit() else src)