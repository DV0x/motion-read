#!/usr/bin/env python3
"""selftest.py - check motion.py against clips whose motion is known exactly.

Draws a textured picture, moves it by a known amount each frame (pan, tilt, zoom, roll, shake, a cut, a
hold, an object moving on its own), writes short clips, runs motion.py on them and compares what it reads
with what was put in. No person mask, no downloads. Run it after any change to motion.py.

  selftest.py [--keep DIR]

Exits 0 when every check passes, 1 otherwise.
"""
import argparse
import math
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import motion  # noqa: E402

W, H, FPS, N = 640, 360, 24.0, 48


def texture(seed, size=(1600, 2400), detail=1.0):
    """a picture with texture at every scale: layered noise plus hard-edged shapes"""
    rng = np.random.default_rng(seed)
    h, w = size
    img = np.zeros((h, w), np.float32)
    for cell, amp in ((160, 60), (48, 40), (16, 30 * detail), (5, 22 * detail)):
        n = rng.random((h // cell + 2, w // cell + 2)).astype(np.float32)
        img += amp * cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
    img = np.clip(img, 0, 255).astype(np.uint8)
    for _ in range(int(260 * detail)):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        r, c = int(rng.integers(8, 46)), int(rng.integers(20, 235))
        if rng.random() < 0.5:
            cv2.circle(img, (x, y), r, c, -1)
        else:
            cv2.rectangle(img, (x, y), (x + r * 2, y + r), c, -1)
    return cv2.cvtColor(cv2.GaussianBlur(img, (3, 3), 0), cv2.COLOR_GRAY2BGR)


def step(dx=0.0, dy=0.0, scale=1.0, deg=0.0):
    """how the picture moves on screen in one frame, about the frame centre"""
    c, s = math.cos(math.radians(deg)) * scale, math.sin(math.radians(deg)) * scale
    return np.array([[c, -s, dx], [s, c, dy], [0, 0, 1]], np.float64)


def render(canvas, G, size=(W, H)):
    """screen = centre + G (canvas point - canvas centre)"""
    ch, cw = canvas.shape[:2]
    to_c = np.array([[1, 0, -cw / 2], [0, 1, -ch / 2], [0, 0, 1]], np.float64)
    to_s = np.array([[1, 0, (size[0] - 1) / 2], [0, 1, (size[1] - 1) / 2], [0, 0, 1]], np.float64)
    return cv2.warpAffine(canvas, (to_s @ G @ to_c)[:2], size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def clip(path, frames):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in frames:
        vw.write(f)
    vw.release()


def camera_clip(canvas, steps):
    G, out = np.eye(3), []
    for S in steps:
        out.append(render(canvas, G))
        G = S @ G
    return out


def object_clip(bg, obj, obj_steps, box=(0.42, 0.62)):
    """a still background with a patch of another picture moving on its own. box: the patch's share of
    the frame's width and height at the start."""
    back = render(bg, np.eye(3))
    G, out = np.eye(3), []
    bw, bh = box[0] * W / 2, box[1] * H / 2
    corners = np.array([[-bw, -bh, 1], [bw, -bh, 1], [bw, bh, 1], [-bw, bh, 1]], np.float64).T
    for S in obj_steps:
        f = back.copy()
        o = render(obj, G)
        pts = (G @ corners)[:2].T + [(W - 1) / 2, (H - 1) / 2]
        m = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(m, pts.astype(np.int32), 255)
        f[m > 0] = o[m > 0]
        out.append(f)
        G = S @ G
    return out


def true_shake(jit):
    """the tool's own shake rule (speed minus its 0.5 s average, RMS over 0.5 s) run on the shifts put in"""
    v = -jit[:N - 1] / W * FPS * 100
    k = np.ones(13)
    avg = lambda x: np.convolve(x, k, "same") / np.convolve(np.ones(len(x)), k, "same")
    hf = (v[:, 0] - avg(v[:, 0])) ** 2 + (v[:, 1] - avg(v[:, 1])) ** 2
    return float(np.sqrt(avg(hf))[2:-2].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", help="keep the test clips in this folder")
    a = ap.parse_args()
    out = a.keep or tempfile.mkdtemp(prefix="motion-selftest-")
    os.makedirs(out, exist_ok=True)
    motion.MASK = "none"
    A, B, plain = texture(1), texture(2), texture(3, detail=0.45)
    rng = np.random.default_rng(7)
    pct = lambda v: v / 100 * W / FPS            # % of frame width a second, as px a frame
    sc = lambda v: 1 + v / 100 / FPS             # % scale a second, as scale a frame
    sig = 1.2                                    # shake: px a frame, each axis
    jit = rng.normal(0, sig, (N, 2))
    # the picture moves opposite to the camera, so a pan right of 20 shifts the picture left by 20
    cases = [
        ("locked", camera_clip(A, [step()] * N), dict(pan=0, tilt=0, push=0, roll=0, shake=0)),
        ("pan-right-20", camera_clip(A, [step(dx=-pct(20))] * N), dict(pan=20, tilt=0, push=0, roll=0, shake=0)),
        ("tilt-down-12", camera_clip(A, [step(dy=-pct(12))] * N), dict(pan=0, tilt=12, push=0, roll=0, shake=0)),
        ("zoom-in-30", camera_clip(A, [step(scale=sc(30))] * N), dict(pan=0, tilt=0, push=30, roll=0, shake=0)),
        ("zoom-out-25", [f for f in camera_clip(A, [step(scale=sc(30))] * N)][::-1],
         dict(pan=0, tilt=0, push=None, roll=0, shake=0)),
        ("roll-10", camera_clip(A, [step(deg=-10 / FPS)] * N), dict(pan=0, tilt=0, push=0, roll=10, shake=0)),
        ("pan-and-zoom", camera_clip(A, [step(dx=-pct(15), scale=sc(20))] * N),
         dict(pan=15, tilt=0, push=20, roll=0, shake=0)),
        ("shake", camera_clip(A, [step(dx=j[0], dy=j[1]) for j in jit]),
         dict(pan=None, tilt=None, push=0, roll=0, shake=true_shake(jit))),
        ("object-crossing", object_clip(plain, B, [step(dx=pct(25))] * N), dict(pan=0, tilt=0, push=0, roll=0, shake=0)),
        ("object-approaching", object_clip(plain, B, [step(scale=sc(45))] * N, box=(0.36, 0.5)),
         dict(pan=0, tilt=0, push=0, roll=0, shake=0)),
    ]
    cut_frames = camera_clip(A, [step(dx=-pct(10))] * 24) + camera_clip(B, [step(dx=pct(10))] * 24)
    hold_frames = camera_clip(A, [step(dx=-pct(10))] * 16) + [render(A, step(dx=-pct(10) * 16))] * 16 + \
        camera_clip(B, [step(dx=-pct(10))] * 16)
    # the shot label each clip should get: its main move with direction, else handheld or locked
    labels = {"locked": "frozen", "pan-right-20": "pan right", "tilt-down-12": "tilt down", "zoom-in-30": "push in",
              "zoom-out-25": "pull out", "roll-10": "roll clockwise", "pan-and-zoom": "push in",
              "shake": "handheld" if true_shake(jit) >= motion.SHAKY else "locked",
              "object-crossing": "locked", "object-approaching": "locked"}
    fails, lines = 0, []

    def check(name, what, got, want, tol):
        nonlocal fails
        ok = got is not None and abs(got - want) <= tol
        fails += not ok
        lines.append(f"{'ok  ' if ok else 'FAIL'} {name:20s} {what:6s} read {got if got is None else round(got, 2)!s:>8}  put in {round(want, 2):>7}  (within {tol})")

    for name, frames, want in cases:
        p = os.path.join(out, name + ".mp4")
        clip(p, frames)
        res = motion.analyse(p, 480, 448)
        rows = res["rows"][2:-2]
        col = lambda k: np.array([r[k] for r in rows], float)
        med = lambda k: float(np.nanmedian(col(k))) if np.isfinite(col(k)).any() else None
        if name == "zoom-out-25":
            want = dict(want, push=(1 / sc(30) - 1) * FPS * 100)
        for k, tol in (("pan", 1.0), ("tilt", 1.0), ("push", 1.5), ("roll", 0.7)):
            if want[k] is not None:
                check(name, k, med(k), want[k], tol)
        sh = float(np.nanmean(col("shake"))) if np.isfinite(col("shake")).any() else None
        check(name, "shake", sh, want["shake"], max(0.5, 0.15 * want["shake"]))
        check(name, "sure %", 100 * float(np.mean(col("cam_conf"))), 100, 10)
        sh = motion.shots(res)
        label = sh[0]["camera"] if sh else "frozen"   # a still picture is one frozen tail, no shot
        exp = labels[name]
        ok = label == exp
        fails += not ok
        lines.append(f"{'ok  ' if ok else 'FAIL'} {name:20s} label  read {label!r}  put in {exp!r}")

    p = os.path.join(out, "cut.mp4")
    clip(p, cut_frames)
    s = motion.summarise(motion.analyse(p, 480, 448))
    check("cut", "count", len(s["cuts_at_s"]), 1, 0)
    check("cut", "time", s["cuts_at_s"][0] if s["cuts_at_s"] else None, 24 / FPS, 0.03)
    # a file whose first frame pops in brightness with nothing changing in the picture is not a cut
    pop = camera_clip(A, [step(dx=-pct(10))] * 48)
    pop[0] = np.clip(pop[0].astype(np.float32) * 0.6, 0, 255).astype(np.uint8)
    p = os.path.join(out, "start-pop.mp4")
    clip(p, pop)
    s = motion.summarise(motion.analyse(p, 480, 448))
    check("start-pop", "cuts", len(s["cuts_at_s"]), 0, 0)
    # a real cut inside the first half second still counts
    early = camera_clip(A, [step(dx=-pct(10))] * 6) + camera_clip(B, [step(dx=pct(10))] * 42)
    p = os.path.join(out, "early-cut.mp4")
    clip(p, early)
    s = motion.summarise(motion.analyse(p, 480, 448))
    check("early-cut", "count", len(s["cuts_at_s"]), 1, 0)
    check("early-cut", "time", s["cuts_at_s"][0] if s["cuts_at_s"] else None, 6 / FPS, 0.03)
    # a fast pan reads as a whip, with its direction
    p = os.path.join(out, "whip.mp4")
    clip(p, camera_clip(A, [step(dx=-pct(90))] * N))
    label = motion.shots(motion.analyse(p, 480, 448))[0]["camera"]
    ok = label == "whip right"
    fails += not ok
    lines.append(f"{'ok  ' if ok else 'FAIL'} {'whip-right-90':20s} label  read {label!r}  put in 'whip right'")
    # a fast zoom reads as a crash zoom
    p = os.path.join(out, "crash-zoom.mp4")
    clip(p, camera_clip(A, [step(scale=sc(150))] * 24))
    label = motion.shots(motion.analyse(p, 480, 448))[0]["camera"]
    ok = label == "crash zoom in"
    fails += not ok
    lines.append(f"{'ok  ' if ok else 'FAIL'} {'crash-zoom-in-150':20s} label  read {label!r}  put in 'crash zoom in'")
    p = os.path.join(out, "hold.mp4")
    clip(p, hold_frames)
    s = motion.summarise(motion.analyse(p, 480, 448))
    hold = s["holds"][0].replace("s", "").split("+") if s["holds"] else [None, None]
    check("hold", "start", float(hold[0]) if hold[0] else None, 17 / FPS, 0.09)
    check("hold", "length", float(hold[1]) if hold[1] else None, 15 / FPS, 0.09)

    print("\n".join(lines))
    print(f"\n{len(lines) - fails} of {len(lines)} checks pass. clips in {out}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
