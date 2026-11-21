# motion-read

A Claude Code skill that measures the motion in a video.

Claude reads still frames, never video. Frames show what is in a shot. They do not show how fast the camera moves, whether it shakes, or where a take freezes. Video models that do watch clips sample a frame or so a second and name what happens without measuring it. So I built a tool that measures, and gives Claude numbers and charts it can read.

## What it reads

| Measure | What you get |
|---|---|
| Camera | pan, tilt, push and roll per frame, and how sure the reading is |
| Shot label | each shot's main move with direction, like whip right, crash zoom in, pan left or tilt up, and whether it is handheld |
| Shake | handheld jitter, per shot |
| Cuts | a cut list, plus whips smeared into blur |
| Holds | freeze frames and frozen ends, common in AI video |
| Body | how much a person moves against themselves |
| Light | brightness swings from dissolves and light changes |

Per clip it writes a CSV with every frame pair, a chart PNG, and a `summary.md` with a shot table.

## Install

```
git clone https://github.com/DV0x/motion-read ~/.claude/skills/motion-read
```

Then ask Claude Code to read the motion in a file or a link. The first run sets up its own Python environment in `~/.cache/motion-read`. It needs Python 3.10 to 3.13 and ffmpeg. macOS uses Apple Vision for the person mask, Linux and Windows use MediaPipe.

It also runs without Claude.

```
bash scripts/run.sh --section 0:30-1:00 --out out/ <file-or-link>
```

## How it works

Dense optical flow on every pair of frames, at the clip's own frame rate. The frame is cut into patches and each patch gets one vote on the camera move, so no single object decides it. A person mask keeps the body out of the camera fit and gives the body its own measure. Details and thresholds are in `reference.md`.

## How I checked it

- **Known motion.** `scripts/selftest.py` draws clips with an exact pan, tilt, zoom, roll, shake, whip, crash zoom, cut and hold, and checks the tool reads back what was put in. 77 of 77 checks pass, shot labels included.
- **By eye.** On 30 seconds of the Beastie Boys' Sabotage it found 23 cuts where ffmpeg scene detection found 17. The 6 extra were checked frame by frame and are real. A locked one-take video (OK Go, Here It Goes Again) read shake 0.12, the Sabotage handheld footage read 17.
- **CameraBench.** On 200 clips from the public test set, mean average precision 0.46 against 0.14 for guessing. In, out and right score about 0.7. Up and down score about 0.3. 49 of the 200 clips came out unsure, measured before the agreement gate was loosened on 2026-11-21 (Sabotage went from 10 unsure shots of 24 to 3). The clips were the GIF previews, which drop frames, so the shake score there is not a fair test.

## What it gets wrong

- It reads the flat picture. It cannot tell a zoom from the camera moving forward, or a pan from the camera sliding sideways.
- When a car or a face fills the frame there is no background to fit the camera on. Those shots come out marked unsure.
- It called 32 cuts across 200 single-shot CameraBench clips. A light going off, a cartoon pop and a TV cutting inside the shot all read as cuts.
- A whip and handheld jitter both read as shake.
- It cannot tell a stiff turn from a living one. Whether a move feels right is still for the person watching.

## Licence

MIT
