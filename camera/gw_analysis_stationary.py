"""
Groove widths and land widths from a single frame.

    import cv2
    from groove_widths import analyse_frame

    frame = cv2.imread("C_sample_1.jpg")
    widths, land = analyse_frame(frame, centre=(13010, -1478))

    widths  -> (71.1, 69.6, 70.1, ...)   average width of each groove (µm), inner -> outer
    land    -> 2D array [gap, cross-section] (µm): land between groove k and groove k+1
               at each cross-section along the grooves (NaN where not measured)

Each groove edge is placed at the steepest point of the land-to-groove drop,
so no calibration between groove sizes is needed. The only number used is
the pixel size.
"""

import warnings

import cv2
import numpy as np

from constants import camera_constants as cam_const



def analyse_frame(frame, blank=None, um_per_px=cam_const.UM_PER_PX):
    """
    frame  : image array (grayscale or BGR, any dtype)
    centre : (x, y) disc centre in frame pixels; estimated if None
    blank  : optional groove-free frame with the same lighting
    Returns (widths, land).
    """
    crop = _crop_bmp(frame)
    img = _gray(crop)
    ratio = _flatten(img, None if blank is None else _gray(blank))
    c = np.asarray(cam_const.CENTRE, float)
    P = _unwrap(ratio, c)
    E = _track([_grooves_in_section(P, j) for j in range(P.shape[1] // cam_const.SECTION_PX)])

    # keep grooves measured in at least 3 cross-sections, inner -> outer
    E = E[np.sum(~np.isnan(E[:, :, 0]), 1) >= 3]
    W = (E[:, :, 1] - E[:, :, 0]) * um_per_px                 # [groove, section]
    widths = tuple(round(_robust_mean(w), 1) for w in W)
    land = (E[1:, :, 0] - E[:-1, :, 1]) * um_per_px           # [gap, section]
    return widths, land


# ---------------------------------------------------------------- helpers
def _crop_bmp(img, margin=cam_const.CROP):
    h, w = img.shape[:2]
    if w <= 2 * margin or h <= 2 * margin:
        raise ValueError(f"Image {w}x{h} is too small to crop {margin}px from each side")
    return img[margin:h - margin, margin:w - margin]


def _gray(a):
    a = np.asarray(a)
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.gui.COLOR_BGRA2GRAY if a.shape[2] == 4 else cv2.gui.COLOR_BGR2GRAY)
    return a.astype(np.float32)


def _flatten(img, blank=None, k=61):
    """Divide out the lighting so the land is ~1.0 everywhere."""
    if blank is not None:
        r = cv2.GaussianBlur(img, (0, 0), 1) / np.maximum(cv2.GaussianBlur(blank, (0, 0), 1), 1e-6)
        return r / np.percentile(r, 95)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    land = cv2.morphologyEx(cv2.GaussianBlur(img, (0, 0), 1), cv2.MORPH_CLOSE, ker)
    return img / np.maximum(cv2.GaussianBlur(land, (0, 0), k / 2), 1e-6)


def _unwrap(ratio, c):
    """Polar image: rows = radius (1 px), columns = along the grooves (~1 px). Outside = NaN."""
    h, w = ratio.shape
    cx = np.array([0, w - 1, 0, w - 1], float)
    cy = np.array([0, 0, h - 1, h - 1], float)
    rmin = np.hypot(max(0.0, -c[0], c[0] - (w - 1)), max(0.0, -c[1], c[1] - (h - 1)))
    rmax = np.hypot(cx - c[0], cy - c[1]).max()
    ang = np.unwrap(np.sort(np.arctan2(cy - c[1], cx - c[0])))
    R, T = np.meshgrid(np.arange(rmin, rmax), np.arange(ang.min(), ang.max(), 2 / (rmin + rmax)),
                       indexing="ij")
    X = (c[0] + R * np.cos(T)).astype(np.float32)
    Y = (c[1] + R * np.sin(T)).astype(np.float32)
    P = cv2.remap(ratio, X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    P[(X < 0) | (X > w - 1) | (Y < 0) | (Y > h - 1)] = np.nan
    return P


def _grooves_in_section(P, j):
    """(inner edge, outer edge) of each complete groove in cross-section j, as rows of P."""
    blk = P[:, j * SECTION_PX:(j + 1) * SECTION_PX]
    ok = np.isnan(blk).mean(1) < 0.2
    if ok.sum() < 50:
        return np.zeros((0, 2))
    idx = np.flatnonzero(ok)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        prof = np.nanmean(blk[idx[0]:idx[-1] + 1], 1)
    q = cv2.GaussianBlur(np.nan_to_num(prof, nan=1.0).reshape(-1, 1).astype(np.float32), (1, 0), 0.7).ravel()
    d = np.gradient(q)
    below = (q < 0.85).astype(int)                    # finds grooves; edges placed below
    starts = np.flatnonzero(np.diff(below) == 1)
    ends = np.flatnonzero(np.diff(below) == -1)

    def steepest(i, sign):                            # sub-pixel steepest point near i
        lo, hi = max(i - 3, 1), min(i + 4, len(d) - 2)
        k = lo + int(np.argmax(sign * d[lo:hi + 1]))
        y0, y1, y2 = d[k - 1], d[k], d[k + 1]
        den = y0 - 2 * y1 + y2
        return k + (0.5 * (y0 - y2) / den if den != 0 else 0.0)

    out = []
    for a in starts:
        e = ends[ends > a]
        if len(e) == 0:
            break
        l, r = steepest(a, -1), steepest(e[0], +1)
        if l > 2 and r < len(q) - 3 and (r - l) > 2:   # skip grooves cut off by the frame
            out.append((l + idx[0], r + idx[0]))
    return np.array(out, float).reshape(-1, 2)


def _track(sections):
    """Match grooves between cross-sections by radius -> E[groove, section, (inner, outer)]."""
    nb = len(sections)
    j0 = int(np.argmax([len(s) for s in sections]))
    if len(sections[j0]) == 0:
        raise RuntimeError("no grooves found")
    c0 = sections[j0].mean(1)
    tol = (np.median(np.diff(c0)) if len(c0) > 1 else 30.0) / 3
    ids = [[] for _ in range(nb)]
    ids[j0] = list(range(len(c0)))
    n_ids = len(c0)
    for step in (1, -1):
        last = dict(enumerate(c0))                    # groove id -> latest radius
        j = j0 + step
        while 0 <= j < nb:
            for c in sections[j].mean(1):
                gid = min(last, key=lambda i: abs(last[i] - c))
                if abs(last[gid] - c) >= tol:
                    gid, n_ids = n_ids, n_ids + 1
                ids[j].append(gid)
                last[gid] = c
            j += step
    E = np.full((n_ids, nb, 2), np.nan)
    for j in range(nb):
        for gid, edges in zip(ids[j], sections[j]):
            E[gid, j] = edges
    order = np.argsort([np.nanmedian(E[i].mean(1)) for i in range(n_ids)])
    return E[order]


def _robust_mean(w):
    """Mean along the groove, ignoring values more than 3 robust SDs from the median."""
    w = w[~np.isnan(w)]
    med = np.median(w)
    sd = 1.4826 * np.median(np.abs(w - med))
    return float(np.mean(w[np.abs(w - med) <= 3 * sd])) if sd > 0 else float(med)



# import cv2

# frame = cv2.imread(r"archive\3.2.2_19.10.26.bmp")
# widths, land = analyse_frame(frame)
# print(widths)
