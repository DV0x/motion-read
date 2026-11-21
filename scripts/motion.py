#!/usr/bin/env python3
"""motion.py - read the motion in a video as numbers and curves (the motion-read skill).

Dense optical flow between every pair of consecutive frames, at the clip's own frame rate.
A person mask splits the frame into background and subject, so the camera is measured on the
background and the person on herself, on a full-resolution crop.
Mask backends: Apple Vision on macOS (built in), MediaPipe DeepLab v3 elsewhere, or none
(camera, cuts and holds still work, the body measures are skipped).

Per frame pair:
  camera   pan, tilt (% of frame width per second), push (% scale per second), roll (deg/s),
           fitted on background flow only, one vote per patch of the frame, read at the frame centre.
           Camera motion is the opposite of how the background moves.
  shake    camera velocity minus its own 0.5 s average, RMS over 0.5 s, shot by shot. Locked or smooth reads near 0.
  subject  on a full-resolution crop around the person, in % of their own height per second.
           internal: flow left after one affine fit to the whole figure, the body moving against itself.
           spread: how differently the head, middle and bottom bands move.
  cut      frame t+1 warped back onto frame t, the median leftover over a 4x4 tile grid, as a robust z.
           One or two flagged frames is a cut, three or more in a row a blur burst (a whip smeared out).
           In a clip's first half second a flagged frame whose edges stay where they were is not a cut
           (a file's opening frames can pop in brightness or noise with nothing changing in the picture).
  light    background brightness (L, 0 to 100). A dissolve to another room or light swings it.
  holds    0.25 s or more with nothing moving, and the frozen tail at the end.

Usage:
  motion.py [--mask auto|vision|mediapipe|none] [--width 480] [--crop-h 448] [--out DIR] video.mp4 [...]
Writes per clip <name>.csv and <name>.png, and summary.md, summary.json, compare.png (two or more clips).
"""
import argparse
import csv
import json
import math
import os
import sys

import cv2
import numpy as np

CUT_Z = 8.0          # robust z above which a frame pair counts as a cut
CUT_START_S = 0.5    # in a clip's first half second ...
CUT_START_SAME = 0.8 # ... a flagged pair whose edges still point the same way (edge_same, 1 = identical) is not a cut
TEX = 8.0            # texture: smaller eigenvalue of the 9x9 gradient structure tensor (both directions present). Flat sky reads about 0, a soft wall 10 or more
CUT_MIN_RAW = 6.0    # ... and the leftover itself must be this big (grey levels, 0 to 255). In a clean, steady
                     # clip a keyframe's small pop stands far out of the clip's own noise without being a cut.
PATCH = 16           # the camera vote: the frame is cut into PATCH columns of square-ish patches, one vote each
PATCH_MIN_PTS = 4    # textured background points a patch needs to vote
CAM_TOL = 0.15       # patches agree when within this share of the typical patch motion (never under 1 px)
CAM_MIN_PTS = 8      # camera confidence: voting patches needed
CAM_MIN_SPREAD = 0.15  # ... the box they span, as a share of the frame area
CAM_MIN_INLIERS = 0.5  # ... the share of them that agree on one camera move
STILL_BLIP = 0.2     # one frame under this (px) between still frames stays part of the hold (a keyframe's pop)
STILL_PX = 0.04      # mean flow (analysis px per frame) under which a frame counts as frozen

# ---------------- person mask ----------------
MASK = "none"          # set in main: vision | mediapipe | none
_vision = None
_mp_seg = None


def _load_vision():
    global _vision
    if _vision is None:
        import Quartz  # noqa: F401  (CoreVideo functions live here)
        import Vision
        from Foundation import NSData
        _vision = (Vision, Quartz, NSData)
    return _vision


def person_mask(bgr):
    """uint8 mask 0/255 at the frame's size, everyone in frame counted as subject."""
    if MASK == "vision":
        return _mask_vision(bgr)
    if MASK == "mediapipe":
        return _mask_mediapipe(bgr)
    return np.zeros(bgr.shape[:2], np.uint8)


def _mediapipe_model():
    home = os.environ.get("MOTION_READ_HOME", os.path.expanduser("~/.cache/motion-read"))
    return os.environ.get("MOTION_READ_MODEL", os.path.join(home, "deeplab_v3.tflite"))


def _mask_mediapipe(bgr):
    """MediaPipe DeepLab v3 (Pascal VOC classes), person is class 15."""
    global _mp_seg
    os.environ.setdefault("GLOG_minloglevel", "2")
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    if _mp_seg is None:
        opts = vision.ImageSegmenterOptions(base_options=BaseOptions(model_asset_path=_mediapipe_model()),
                                            output_category_mask=True, output_confidence_masks=False)
        _mp_seg = vision.ImageSegmenter.create_from_options(opts)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    cat = _mp_seg.segment(img).category_mask.numpy_view()
    m = np.where(cat == 15, 255, 0).astype(np.uint8)
    if m.shape != bgr.shape[:2]:
        m = cv2.resize(m, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    return m


def pick_mask(want):
    if want in ("vision", "auto") and sys.platform == "darwin":
        try:
            _load_vision()
            return "vision"
        except Exception as e:
            if want == "vision":
                raise SystemExit(f"Apple Vision not available: {e}")
    if want in ("mediapipe", "auto"):
        try:
            import mediapipe  # noqa: F401
            if os.path.exists(_mediapipe_model()):
                return "mediapipe"
            if want == "mediapipe":
                raise SystemExit(f"MediaPipe model missing at {_mediapipe_model()}, run setup.sh mediapipe")
        except ImportError as e:
            if want == "mediapipe":
                raise SystemExit(f"MediaPipe not installed: {e}")
    if want not in ("auto", "none"):
        raise SystemExit(f"mask backend {want} not available")
    return "none"


def _mask_vision(bgr):
    Vision, Quartz, NSData = _load_vision()
    ok, png = cv2.imencode(".png", bgr)
    data = NSData.dataWithBytes_length_(png.tobytes(), len(png))
    req = Vision.VNGeneratePersonSegmentationRequest.alloc().initWithCompletionHandler_(None)
    req.setQualityLevel_(Vision.VNGeneratePersonSegmentationRequestQualityLevelAccurate)
    req.setOutputPixelFormat_(1278226488)  # kCVPixelFormatType_OneComponent8 'L008'
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
    ok, err = handler.performRequests_error_([req], None)
    h, w = bgr.shape[:2]
    if not ok or not req.results():
        return np.zeros((h, w), np.uint8)
    pb = req.results()[0].pixelBuffer()
    Quartz.CVPixelBufferLockBaseAddress(pb, 1)
    try:
        mw = Quartz.CVPixelBufferGetWidth(pb)
        mh = Quartz.CVPixelBufferGetHeight(pb)
        bpr = Quartz.CVPixelBufferGetBytesPerRow(pb)
        base = Quartz.CVPixelBufferGetBaseAddress(pb)
        arr = np.frombuffer(base.as_buffer(bpr * mh), np.uint8).reshape(mh, bpr)[:, :mw].copy()
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(pb, 1)
    arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.where(arr > 127, 255, 0).astype(np.uint8)


# ---------------- helpers ----------------
def affine_apply(M, pts):
    return pts @ M[:, :2].T + M[:, 2]


def bbox(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) < 30:
        return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


class Frame:
    __slots__ = ("full", "small", "gray", "mask", "bgL")


def load_frame(bgr, width):
    f = Frame()
    f.full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = bgr.shape[:2]
    small = bgr if w <= width else cv2.resize(bgr, (width, round(h * width / w / 2) * 2), interpolation=cv2.INTER_AREA)
    f.small = small
    f.gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    f.mask = person_mask(small)
    bgm = cv2.dilate(f.mask, np.ones((9, 9), np.uint8)) == 0
    L = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32) * (100 / 255)
    f.bgL = float(L[bgm].mean()) if bgm.sum() > 100 else float(L.mean())
    return f


# ---------------- per pair measures ----------------
def camera_and_cut(a, b, flow, fps):
    H, W = a.gray.shape
    step = 8
    ys, xs = np.mgrid[step // 2:H:step, step // 2:W:step]
    ys, xs = ys.ravel(), xs.ravel()
    src = np.stack([xs, ys], 1).astype(np.float32)
    dst = src + flow[ys, xs]
    bg = cv2.erode(255 - a.mask, np.ones((7, 7), np.uint8))[ys, xs] > 0
    # flow is only meaningful where the picture has texture in two directions. Plain sky or a flat wall
    # gives zero flow whatever the camera does, so the camera is fitted on textured background only.
    g = cv2.GaussianBlur(a.gray.astype(np.float32), (3, 3), 0)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    k = (9, 9)
    sxx, syy, sxy = cv2.boxFilter(gx * gx, -1, k), cv2.boxFilter(gy * gy, -1, k), cv2.boxFilter(gx * gy, -1, k)
    lam_min = (sxx + syy) / 2 - np.sqrt(((sxx - syy) / 2) ** 2 + sxy ** 2)
    use = bg & (lam_min[ys, xs] > TEX)
    r = {"flow_mean": float(np.linalg.norm(flow, axis=2).mean())}
    # one vote per patch of the frame, not per point. A detailed object (a car, a sign) has many points
    # but covers few patches, so it cannot outvote the background. Each patch votes with its median flow.
    pw = W / PATCH
    rows_n = max(2, round(H / pw))
    pid = (np.minimum(ys * rows_n // H, rows_n - 1) * PATCH + np.minimum(xs * PATCH // W, PATCH - 1))
    vs, vd = [], []
    for q in np.unique(pid[use]):
        sel = use & (pid == q)
        if sel.sum() >= PATCH_MIN_PTS:
            c = src[sel].mean(0)
            vs.append(c)
            vd.append(c + np.median(flow[ys[sel], xs[sel]], axis=0))
    n = len(vs)
    vs, vd = np.array(vs, np.float32).reshape(-1, 2), np.array(vd, np.float32).reshape(-1, 2)
    r["cam_pts"] = n
    if n >= 2:
        r["cam_spread"] = float((vs[:, 0].max() - vs[:, 0].min() + pw) / W * (vs[:, 1].max() - vs[:, 1].min() + pw) / H)
    else:
        r["cam_spread"] = 0.0
    cam, inl = None, None
    if n >= 4:
        # how closely patches must agree grows with the motion: blur and depth spread the flow in a fast move
        tol = max(1.0, CAM_TOL * float(np.median(np.linalg.norm(vd - vs, axis=1))))
        cam, inl = cv2.estimateAffinePartial2D(vs, vd, method=cv2.RANSAC,
                                               ransacReprojThreshold=tol, maxIters=2000)
    r["cam_inliers"] = float(inl.mean()) if inl is not None and len(inl) else 0.0
    r["cam_conf"] = int(cam is not None and n >= CAM_MIN_PTS and r["cam_spread"] >= CAM_MIN_SPREAD
                        and r["cam_inliers"] >= CAM_MIN_INLIERS)
    if cam is not None and not r["cam_conf"]:
        cam = None
    if cam is not None:
        sc = math.hypot(cam[0, 0], cam[1, 0])
        # the shift is read at the frame centre. Read at the corner, every zoom and roll leaks into pan and tilt.
        cx, cy = (W - 1) / 2, (H - 1) / 2
        dx = cam[0, 0] * cx + cam[0, 1] * cy + cam[0, 2] - cx
        dy = cam[1, 0] * cx + cam[1, 1] * cy + cam[1, 2] - cy
        r["pan"] = -dx / W * fps * 100
        r["tilt"] = -dy / W * fps * 100
        r["push"] = (sc - 1) * fps * 100
        r["roll"] = -math.degrees(math.atan2(cam[1, 0], cam[0, 0])) * fps
        r["cam_ok"] = 1
    else:
        r.update(pan=np.nan, tilt=np.nan, push=np.nan, roll=np.nan, cam_ok=0)
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    warped = cv2.remap(b.gray, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
    err = cv2.absdiff(warped, a.gray).astype(np.float32)[8:-8, 8:-8]
    # median over a 4x4 grid of tiles: a cut breaks the whole frame, something passing the lens only part of it
    th, tw = err.shape[0] // 4, err.shape[1] // 4
    tiles = [err[i * th:(i + 1) * th, j * tw:(j + 1) * tw].mean() for i in range(4) for j in range(4)]
    r["cut_raw"] = float(np.median(tiles))
    # do the edges of the two frames, unwarped, still point the same way where both have an edge.
    # Near 1 for the same picture under a brightness or noise pop, 0.1 to 0.6 across a real cut.
    ga, gb = cv2.GaussianBlur(a.gray, (0, 0), 2).astype(np.float32), cv2.GaussianBlur(b.gray, (0, 0), 2).astype(np.float32)
    ax, ay = cv2.Sobel(ga, cv2.CV_32F, 1, 0), cv2.Sobel(ga, cv2.CV_32F, 0, 1)
    bx, by = cv2.Sobel(gb, cv2.CV_32F, 1, 0), cv2.Sobel(gb, cv2.CV_32F, 0, 1)
    ma, mb = np.hypot(ax, ay), np.hypot(bx, by)
    wt = np.minimum(ma, mb)
    r["edge_same"] = float(((ax * bx + ay * by) / (ma * mb + 1e-6) * wt).sum() / (wt.sum() + 1e-6))
    r["subj_area"] = float(a.mask.mean() / 255 * 100)
    return r


def subject(a, b, fps, dis, crop_h):
    """internal and spread on a full-resolution crop around her, % of her height per second."""
    bb = bbox(cv2.bitwise_or(a.mask, b.mask))
    if bb is None:
        return dict(subj_internal=np.nan, subj_spread=np.nan, subj_h_px=np.nan)
    sh, sw = a.gray.shape
    fh, fw = a.full.shape
    s = fw / sw
    x0, y0, x1, y1 = bb
    pad = 0.15 * (y1 - y0)
    x0, y0 = max(0, int((x0 - pad) * s)), max(0, int((y0 - pad) * s))
    x1, y1 = min(fw, int((x1 + pad) * s)), min(fh, int((y1 + pad) * s))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return dict(subj_internal=np.nan, subj_spread=np.nan, subj_h_px=np.nan)
    k = crop_h / (y1 - y0)
    size = (max(16, int((x1 - x0) * k)), crop_h)
    ca = cv2.resize(a.full[y0:y1, x0:x1], size, interpolation=cv2.INTER_AREA)
    cb = cv2.resize(b.full[y0:y1, x0:x1], size, interpolation=cv2.INTER_AREA)
    m_full = cv2.resize(a.mask, (fw, fh), interpolation=cv2.INTER_NEAREST)[y0:y1, x0:x1]
    m = cv2.resize(m_full, size, interpolation=cv2.INTER_NEAREST)
    m = cv2.erode(m, np.ones((5, 5), np.uint8))
    flow = dis.calc(ca, cb, None)
    step = 4
    ys, xs = np.mgrid[step // 2:size[1]:step, step // 2:size[0]:step]
    ys, xs = ys.ravel(), xs.ravel()
    inside = m[ys, xs] > 0
    if inside.sum() < 40:
        return dict(subj_internal=np.nan, subj_spread=np.nan, subj_h_px=np.nan)
    sp = np.stack([xs[inside], ys[inside]], 1).astype(np.float32)
    sd = sp + flow[ys[inside], xs[inside]]
    hp = float(sp[:, 1].max() - sp[:, 1].min()) or 1.0
    norm = fps * 100 / hp
    A, _ = cv2.estimateAffine2D(sp, sd, method=cv2.LMEDS)
    internal = float(np.mean(np.linalg.norm(sd - affine_apply(A, sp), axis=1))) * norm if A is not None else np.nan
    f = sd - sp
    yt = sp[:, 1].min()
    bands = []
    for lo, hi in ((0, .3), (.3, .7), (.7, 1.0)):
        sel = (sp[:, 1] >= yt + lo * hp) & (sp[:, 1] <= yt + hi * hp)
        if sel.sum() >= 8:
            bands.append(np.median(f[sel], 0))
    if len(bands) >= 2:
        d = [np.linalg.norm(bands[i] - bands[j]) for i in range(len(bands)) for j in range(i + 1, len(bands))]
        spread = float(np.mean(d)) * norm
    else:
        spread = np.nan
    return dict(subj_internal=internal, subj_spread=spread, subj_h_px=hp / k)


def analyse(path, width, crop_h, strip_n=8):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    strip_idx = set(np.linspace(0, max(total - 1, 0), strip_n).astype(int).tolist()) if total else set()
    dis_small = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis_crop = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis_crop.setFinestScale(0)          # flow at the crop's full resolution, for the body detail
    rows, bgL, strip = [], [], []
    prev, i = None, 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        cur = load_frame(bgr, width)
        bgL.append(cur.bgL)
        if i in strip_idx:
            strip.append((i, cur.small.copy(), cur.mask.copy()))
        if prev is not None:
            flow = dis_small.calc(prev.gray, cur.gray, None)
            r = camera_and_cut(prev, cur, flow, fps)
            r.update(subject(prev, cur, fps, dis_crop, crop_h))
            r["t"] = i / fps
            r["frame"] = i
            r["light"] = cur.bgL
            r["light_rate"] = (cur.bgL - prev.bgL) * fps
            rows.append(r)
        prev, i = cur, i + 1
    cap.release()
    if len(rows) < 3:
        raise SystemExit(f"{path}: fewer than 3 frames")
    cr = np.array([r["cut_raw"] for r in rows])
    med = np.median(cr)
    mad = np.median(np.abs(cr - med)) * 1.4826 + 1e-6
    for r, v in zip(rows, cr):
        r["cut_z"] = float((v - med) / mad)
    # cut frames and their neighbours, left out of the subject stats
    cut = np.array([r["cut_z"] >= CUT_Z and r["cut_raw"] >= CUT_MIN_RAW
                    and not (r["t"] <= CUT_START_S and r["edge_same"] >= CUT_START_SAME) for r in rows])
    near = cut.copy()
    near[1:] |= cut[:-1]
    near[:-1] |= cut[1:]
    # one or two flagged frames in a row is a cut, three or more is a blur burst (a whip smeared out)
    kind = [""] * len(rows)
    j = 0
    while j < len(rows):
        if cut[j]:
            k = j
            while k + 1 < len(rows) and cut[k + 1]:
                k += 1
            if k - j + 1 >= 3:
                for q in range(j, k + 1):
                    kind[q] = "blur"
            else:
                kind[j] = "cut"
            j = k + 1
        else:
            j += 1
    # frozen tail
    fmv = np.array([r["flow_mean"] for r in rows])
    still = fmv < STILL_PX
    for j in range(1, len(rows) - 1):
        if not still[j] and fmv[j] < STILL_BLIP and fmv[j - 1] < STILL_PX and fmv[j + 1] < STILL_PX:
            still[j] = True
    tail = 0
    for s in still[::-1]:
        if not s:
            break
        tail += 1
    for j, r in enumerate(rows):
        r["is_cut"] = int(near[j])
        r["cut_kind"] = kind[j]
        r["is_frozen_tail"] = int(j >= len(rows) - tail)
        r["is_still"] = int(still[j])
        if near[j]:
            r["subj_internal"] = r["subj_spread"] = np.nan
            r["pan"] = r["tilt"] = r["push"] = r["roll"] = np.nan
    # shake: camera velocity minus its own 0.5 s average, RMS over 0.5 s. A smooth move or a locked
    # camera reads near zero, handheld jitter does not.
    w = max(3, int(round(0.5 * fps)) | 1)
    def nanavg(x, w):
        v = np.nan_to_num(x); c = (~np.isnan(x)).astype(float); k = np.ones(w)
        num, den = np.convolve(v, k, "same"), np.convolve(c, k, "same")
        return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
    pan = np.array([r["pan"] for r in rows], float)
    tilt = np.array([r["tilt"] for r in rows], float)
    roll = np.array([r["roll"] for r in rows], float)
    # shot by shot, so the average never runs across a cut into another shot's camera
    shake, rj = np.full(len(rows), np.nan), np.full(len(rows), np.nan)
    edges = [0] + [j for j, k in enumerate(kind) if k == "cut"] + [len(rows)]
    for a, e in zip(edges[:-1], edges[1:]):
        ws = min(w, (e - a) if (e - a) % 2 else (e - a) - 1)
        if ws < 3:
            continue
        P, T, R = pan[a:e], tilt[a:e], roll[a:e]
        hf = (P - nanavg(P, ws)) ** 2 + (T - nanavg(T, ws)) ** 2
        shake[a:e] = np.sqrt(nanavg(hf, ws))
        rj[a:e] = np.sqrt(nanavg((R - nanavg(R, ws)) ** 2, ws))
    for j, r in enumerate(rows):
        r["shake"] = float(shake[j]) if not near[j] else np.nan
        r["roll_jitter"] = float(rj[j]) if not near[j] else np.nan
    return dict(path=path, fps=fps, n=i, rows=rows, strip=strip, frozen_tail=tail / fps, bgL=bgL)


# ---------------- summaries and charts ----------------
def smooth(x, k=5):
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    for j in range(len(x)):
        w = x[max(0, j - k // 2): j + k // 2 + 1]
        w = w[~np.isnan(w)]
        if len(w):
            out[j] = np.median(w)
    return out


WHIP = 60.0     # camera speed, % of frame width per second, held 2+ frames
ZOOM = 40.0     # push or pull, % scale per second


def shots(res):
    """shots between cuts, each with its camera and subject numbers"""
    rows, fps = res["rows"], res["fps"]
    bounds, start = [], 0.0
    for j, r in enumerate(rows):
        if r["cut_kind"] == "cut":
            bounds.append((start, r["t"] - 1 / fps))
            start = r["t"]
    bounds.append((start, rows[-1]["t"]))
    out = []
    for a, b in bounds:
        sel = [r for r in rows if a <= r["t"] <= b and not r["is_frozen_tail"]]
        if not sel:
            continue
        g = lambda k: np.array([r[k] for r in sel], float)
        m = lambda x: round(float(np.nanmean(x)), 1) if np.isfinite(x).any() else None
        sp, sh, pu = np.hypot(g("pan"), g("tilt")), g("shake"), g("push")
        msp, msh, mpu = (np.nanmean(x) if np.isfinite(x).any() else 0 for x in (sp, sh, np.abs(pu)))
        conf = float(np.mean(g("cam_conf")))
        if conf < 0.5:
            kind = "unsure"
        elif msh >= 4:
            kind = "handheld"
        elif np.isfinite(sp).any() and np.nanmax(sp) >= WHIP:
            kind = "whip"
        elif mpu >= 8 and mpu >= msp:
            kind = "push/pull"
        elif msp >= 3:
            kind = "moving"
        else:
            kind = "locked"
        out.append({"start": round(a, 2), "end": round(b, 2), "dur": round(b - a + 1 / fps, 2), "camera": kind,
                    "cam_conf_pct": round(100 * conf),
                    "speed": m(sp), "shake": m(sh), "push": m(pu), "body": m(g("subj_internal"))})
    return out


def summarise(res):
    rows = res["rows"]
    fps = res["fps"]
    live = np.array([not r["is_frozen_tail"] for r in rows])
    col = lambda k: np.array([r[k] for r in rows], float)
    nm = lambda a: round(float(np.nanmean(a)), 2) if np.isfinite(a).any() else None
    p90 = lambda a: round(float(np.nanpercentile(a, 90)), 2) if np.isfinite(a).any() else None
    cam_speed = np.hypot(col("pan"), col("tilt"))[live]
    internal, spread = col("subj_internal")[live], col("subj_spread")[live]
    cz = col("cut_z")
    cuts = [round(r["t"], 2) for r in rows if r["cut_kind"] == "cut"]
    blurs = [round(rows[j]["t"], 2) for j in range(len(rows))
             if rows[j]["cut_kind"] == "blur" and (j == 0 or rows[j - 1]["cut_kind"] != "blur")]
    # light swings: biggest change in background brightness within any 0.5 s window
    L = np.array(res["bgL"])
    win = max(2, int(round(0.5 * fps)))
    swing, swing_t = 0.0, None
    for j in range(len(L) - win):
        d = L[j + win] - L[j]
        if abs(d) > abs(swing):
            swing, swing_t = float(d), (j + win / 2) / fps
    speed_all = np.hypot(col("pan"), col("tilt"))
    whips, j = [], 0
    while j < len(rows):
        if speed_all[j] >= WHIP and not math.isnan(speed_all[j]):
            k = j
            while k + 1 < len(rows) and speed_all[k + 1] >= WHIP:
                k += 1
            if k - j + 1 >= 2:
                whips.append(round(rows[j]["t"], 2))
            j = k + 1
        else:
            j += 1
    zooms = [round(rows[j]["t"], 2) for j in range(1, len(rows))
             if abs(col("push")[j]) >= ZOOM and not abs(col("push")[j - 1]) >= ZOOM]
    shake = col("shake")[live]
    # holds: 0.25 s or more where nothing in the frame moves, outside the frozen tail (freeze frames, title holds)
    holds, st, j, need = [], col("is_still") > 0, 0, max(3, int(round(0.25 * fps)))
    while j < len(rows):
        if st[j] and live[j]:
            k = j
            while k + 1 < len(rows) and st[k + 1] and live[k + 1]:
                k += 1
            if k - j + 1 >= need:
                holds.append(f"{rows[j]['t']:.2f}+{(k - j + 1) / fps:.2f}s")
            j = k + 1
        else:
            j += 1
    return {
        "clip": os.path.splitext(os.path.basename(res["path"]))[0],
        "mask": MASK,
        "camera_confident_pct": round(100 * float(np.mean(col("cam_conf")[live]))),
        "shake_mean": nm(shake),
        "shake_p90": p90(shake),
        "roll_jitter_mean": nm(col("roll_jitter")[live]),
        "whips_at_s": whips,
        "zooms_at_s": zooms,
        "shots": shots(res),
        "seconds": round(res["n"] / fps, 2),
        "live_seconds": round(float(live.sum()) / fps, 2),
        "frozen_tail_s": round(res["frozen_tail"], 2),
        "camera_speed_mean": nm(cam_speed),
        "camera_speed_p90": p90(cam_speed),
        "subject_internal_mean": nm(internal),
        "subject_internal_p90": p90(internal),
        "subject_spread_mean": nm(spread),
        "subject_spread_p90": p90(spread),
        "subject_height_px": nm(col("subj_h_px")),
        "cuts_at_s": cuts,
        "blur_bursts_at_s": blurs,
        "holds": holds,
        "light_swing": round(swing, 1),
        "light_swing_at_s": round(swing_t, 2) if swing_t is not None else None,
        "light_start_end": [round(L[0], 1), round(L[-1], 1)],
    }


def write_csv(res, path):
    keys = ["frame", "t", "flow_mean", "pan", "tilt", "push", "roll", "shake", "roll_jitter", "cam_ok", "cam_conf", "cam_pts", "cam_spread", "cam_inliers", "subj_area", "subj_h_px",
            "subj_internal", "subj_spread", "cut_raw", "cut_z", "edge_same", "is_cut", "cut_kind", "light", "light_rate", "is_frozen_tail"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        for r in res["rows"]:
            w.writerow([("%.4f" % r[k]) if isinstance(r[k], float) else r[k] for k in keys])


def strip_image(res, h=90):
    tiles, ts = [], []
    for i, small, mask in res["strip"]:
        f = small.copy()
        f[cv2.Canny(mask, 50, 150) > 0] = (0, 255, 255)
        s = h / f.shape[0]
        tiles.append(cv2.resize(f, (int(f.shape[1] * s), h)))
        ts.append(i / res["fps"])
    return cv2.cvtColor(np.hstack(tiles), cv2.COLOR_BGR2RGB), ts


def shade(ax, res, camera=False):
    rows = res["rows"]
    for r in rows:
        if camera and not r["cam_conf"] and not r["is_cut"] and not r["is_frozen_tail"]:
            ax.axvspan(r["t"] - 1 / res["fps"], r["t"], color="#e2e2e2", lw=0)
        if r["is_frozen_tail"]:
            ax.axvspan(r["t"] - 1 / res["fps"], r["t"], color="#cfd8ff", lw=0)
        elif r["is_cut"]:
            ax.axvspan(r["t"] - 1 / res["fps"], r["t"], color="#ffc9c9", lw=0)


def plot_clip(res, summ, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = res["rows"]
    t = np.array([r["t"] for r in rows])
    col = lambda k: np.array([r[k] for r in rows], float)
    fig, ax = plt.subplots(5, 1, figsize=(11, 11), gridspec_kw={"height_ratios": [1.1, 1, 1, .7, .7]})
    strip, ts = strip_image(res)
    ax[0].imshow(strip)
    ax[0].set_title(f"{summ['clip']}   frames at " + ", ".join(f"{x:.1f}s" for x in ts)
                    + "   (yellow = person mask, blue = frozen tail, red = cut, grey = camera unsure)", fontsize=9)
    ax[0].axis("off")
    ax[1].plot(t, smooth(col("pan")), label="pan (+right)")
    ax[1].plot(t, smooth(col("tilt")), label="tilt (+down)")
    ax[1].plot(t, smooth(col("push")), label="push (+in)")
    ax[1].plot(t, col("shake"), label="shake", color="k", lw=1.2)
    ax[1].axhline(0, color="#999", lw=.5)
    ax[1].set_ylabel("camera\n% frame/s")
    ax[1].legend(fontsize=8, loc="upper right")
    ax[2].plot(t, smooth(col("subj_internal")), label="internal (body against itself)", color="C3")
    ax[2].plot(t, smooth(col("subj_spread")), label="spread (head vs middle vs bottom)", color="C4")
    ax[2].set_ylabel("subject\n% her height/s")
    ax[2].legend(fontsize=8, loc="upper right")
    ax[3].plot(t, col("cut_z"), color="k", lw=1)
    ax[3].axhline(CUT_Z, color="r", lw=.6, ls="--")
    ax[3].set_ylabel("cut\n(robust z)")
    ax[4].plot(t, col("light"), color="C1")
    ax[4].set_ylabel("background\nbrightness L")
    ax[4].set_xlabel("seconds")
    for j, a in enumerate(ax[1:]):
        shade(a, res, camera=(j == 0))
        a.set_xlim(0, t[-1])
        a.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_compare(results, summaries, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    for j, (res, s) in enumerate(zip(results, summaries)):
        rows = res["rows"]
        t = np.array([r["t"] for r in rows])
        col = lambda k: np.array([r[k] for r in rows], float)
        ls = "-" if j % 2 == 0 else "--"
        c = f"C{j // 2}"
        ax[0].plot(t, smooth(np.hypot(col("pan"), col("tilt"))), ls, color=c, label=s["clip"])
        ax[0].plot(t, col("shake"), ":", color=c, lw=1)
        ax[1].plot(t, smooth(col("subj_internal"), 7), ls, color=c)
        ax[2].plot(t, smooth(col("subj_spread"), 7), ls, color=c)
        ax[3].plot(t, col("light"), ls, color=c)
    ax[0].set_ylabel("camera speed\n(dotted = shake)")
    ax[1].set_ylabel("subject internal\n% her height/s")
    ax[2].set_ylabel("subject spread\n% her height/s")
    ax[3].set_ylabel("background\nbrightness L")
    ax[3].set_xlabel("seconds")
    ax[0].legend(fontsize=7, ncol=2, loc="upper left")
    for a in ax:
        a.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_summary_md(summaries, path):
    cols = ["clip", "live_seconds", "frozen_tail_s", "camera_confident_pct", "camera_speed_mean", "camera_speed_p90",
            "subject_internal_mean", "subject_internal_p90", "subject_spread_mean", "subject_spread_p90",
            "shake_mean", "shake_p90", "roll_jitter_mean", "whips_at_s", "zooms_at_s",
            "subject_height_px", "cuts_at_s", "blur_bursts_at_s", "holds", "light_swing", "light_swing_at_s"]
    with open(path, "w") as fh:
        fh.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for s in summaries:
            fh.write("| " + " | ".join(
                (" ".join(str(x) for x in s[c]) or "none") if isinstance(s[c], list) else str(s[c]) for c in cols) + " |\n")
        for s in summaries:
            fh.write(f"\n### {s['clip']}, shots\n\n| # | start | end | dur | camera | sure % | speed | shake | push | body |\n|---|---|---|---|---|---|---|---|---|---|\n")
            for i, sh in enumerate(s["shots"], 1):
                fh.write(f"| {i} | {sh['start']} | {sh['end']} | {sh['dur']} | {sh['camera']} | {sh['cam_conf_pct']} | {sh['speed']} | {sh['shake']} | {sh['push']} | {sh['body']} |\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--width", type=int, default=480, help="analysis width for camera and cuts (default 480)")
    ap.add_argument("--crop-h", type=int, default=448, help="height of the full-res subject crop (default 448)")
    ap.add_argument("--out", default=None, help="output folder (default: motion/ next to the first video)")
    ap.add_argument("--mask", default="auto", choices=["auto", "vision", "mediapipe", "none"],
                    help="person mask backend (default auto: Vision on macOS, MediaPipe elsewhere)")
    a = ap.parse_args()
    global MASK
    MASK = pick_mask(a.mask)
    print(f"person mask: {MASK}" + ("  (body measures skipped, camera and cuts still work)" if MASK == "none" else ""),
          file=sys.stderr)
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.videos[0])), "motion")
    os.makedirs(out, exist_ok=True)
    results, summaries = [], []
    for v in a.videos:
        print(f"reading {v} ...", file=sys.stderr)
        res = analyse(v, a.width, a.crop_h)
        s = summarise(res)
        write_csv(res, os.path.join(out, s["clip"] + ".csv"))
        plot_clip(res, s, os.path.join(out, s["clip"] + ".png"))
        res["strip"] = None
        results.append(res)
        summaries.append(s)
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summaries, fh, indent=1)
    write_summary_md(summaries, os.path.join(out, "summary.md"))
    if len(results) > 1:
        plot_compare(results, summaries, os.path.join(out, "compare.png"))
    print(out)


if __name__ == "__main__":
    main()
