---
name: motion-read
description: >-
  Measure the motion in a video, because Claude reads still frames only. Optical flow on every frame pair gives
  camera pan, tilt, push and roll, handheld shake, whips, crash zooms, cuts (more of them than ffmpeg scene
  detection finds), freeze frames and held ends, light changes from dissolves or room swaps, and how much a
  person's body moves, per shot and over time. Takes video files or links (YouTube and anything yt-dlp reads, with
  a section). Use it to analyse a reference ad, film or music video, map a video's cuts and camera, compare takes or
  AI renders (frozen ends, morphs, stiff subjects), or before saying anything about how a video moves. Trigger on
  "read the motion", "analyse this video", "what does the camera do", "is it handheld", "cut list", "compare these
  takes". Not for editing, trimming or converting video.
---

# motion-read

Claude takes images, never video. Frames from ffmpeg show what is in a shot, never its speed, easing or shake. Gemini samples video at 1 frame a second by default and 24 at most, with timestamps to the second, so it can name what happens but cannot measure it. This skill measures. It hands back numbers and charts Claude can read, checked against clips with known motion and against a human eye on two music videos (see `reference.md`). Feel and taste stay with the person watching.

## 1. Run it

```
bash "<this skill's folder>/scripts/run.sh" [--section 0:30-1:00] [--out DIR] <file-or-link> [more ...]
```

- The first run sets itself up in `~/.cache/motion-read` (wheels only, a minute or two). macOS uses Apple Vision for the person mask. Linux and Windows use MediaPipe DeepLab v3. `--mask none` skips the body measures and keeps everything else.
- Links download at up to 720p into `DIR/source`. `--section` keeps only that stretch, which saves time on long videos.
- Several clips in one call also give `compare.png`, with clips plotted in pairs (same colour, solid then dashed), so pass takes in the order you want compared.
- It costs about 10 seconds of compute per second of 24 fps video on an M-series Mac. Tell the person before a long run.

## 2. Read what it wrote

Start with `summary.md` in the output folder. It has one row per clip and a shot table per clip. Then read `<clip>.png` as an image (frames on top, then camera with shake, body, cut score and background light over time, with red bands at cuts and blue for a frozen end). `<clip>.csv` has every frame pair if a question needs one moment.

| Measure | Unit | Reads near zero when |
|---|---|---|
| camera speed (pan, tilt), push | % of frame width a second, % scale a second | the camera is locked |
| camera sure | % of frames where most patches of textured background agree on one camera move | the background is plain, or a face, a car or anything else moving fills much of the frame |
| shake | % of frame a second, jitter left after the 0.5 s average | the camera is locked or moves smoothly |
| body internal, spread | % of the person's height a second | the person holds still |
| cuts, blur bursts, whips, zooms | seconds into the clip | none happen |
| holds, frozen tail | start and length in seconds | something always moves |
| light swing | background brightness change within 0.5 s | the light and the room stay put |

Anchors from the validation runs. A locked one-take camera read shake 0.12. Handheld 70s cop-show footage read 17. A smooth AI camera arc read 1.4. Shots labelled unsure, and grey stretches on the camera chart, are where the patches of the frame do not agree on one camera move, so say so instead of reporting their camera numbers. Fast footage full of cars and close faces can still leave some shots unsure (3 of 24 on Sabotage), and that is the honest reading. Full definitions, thresholds and shot types are in `reference.md`.

## 3. Check the flags before reporting them

Every flagged moment is a claim. Before reporting cuts, whips, zooms or holds, look at the frames around them.

```
bash "<this skill's folder>/scripts/peek.sh" <video> 3.38 15.10 18.69
```

It prints a sheet, one row per time, two frames before and the frame at it. Read it as an image. On the validation run this check showed that six cuts ffmpeg missed were real, and that three "cuts" in a row were one whip smeared into blur, which the tool now labels as a blur burst.

## 4. Report

- Lead with the numbers and the shot table. Do not describe motion from stills.
- Say which mask backend made the body numbers. Vision and MediaPipe body numbers differ by up to about 1.8x on the same clip, so compare body numbers only within one backend. Camera, shake, cuts and holds agree across backends.
- Name the limits below when they touch the question.
- Whether a move looks good, feels alive or suits the film is the viewer's call. Offer to stack takes side by side for them to watch.

## After changing the tool

Run `scripts/selftest.py` with the skill's Python (the venv path `setup.sh` prints). It draws clips with a known pan, tilt, zoom, roll, shake, whip, crash zoom, cut, hold and an object moving on its own, and checks the tool reads back what was put in, shot labels included. It takes under a minute.

## Limits

- A whip and handheld jitter both read as shake, so a whip shot also reads handheld in its label.
- Shot labels name the main move with direction (whip right, crash zoom in, pan left, tilt up), with ", handheld" when the shot shakes. Near the thresholds a label can go either way (a fast tilt can read as a whip), so check a label that matters on the frames.
- Body numbers break up in heavy motion blur. With several people in frame the body numbers describe the group, not one person.
- The camera is whatever most of the frame agrees on. Anything moving on its own that fills about half the frame or more (a car skidding past, a face against a flat wall, a view over a car's bonnet) leaves no majority, and those shots come out unsure.
- The tool reads the flat picture. It cannot tell a zoom from the camera moving forward, or a pan from the camera sliding sideways.
- Fades and dissolves are not counted as cuts. They show up as a light swing.
- To settle a camera dispute, look at every frame of the shot, not a few. A low angle looked locked on four stills and the viewer saw it move.
- The body measure tells movement from stillness but not a stiff turn from a living one. A rigid figure turning in place still changes its outline.
- Moving graphics read as camera. A title that zooms in reads as a push.
- Times mark the second frame of each pair, so a cut at 3.38 s is between the frames at about 3.34 and 3.38 s.
