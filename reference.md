# motion-read reference

## How each number is made

Every frame is scaled to 480 px wide (`--width`). DIS optical flow runs between each consecutive pair at the clip's own frame rate, so nothing between samples is lost. The person mask splits each frame into background and person.

| Measure | How |
|---|---|
| pan, tilt, push, roll | The frame is cut into a grid of patches, 16 across. Each patch with textured background in it casts one vote, the median flow of its textured points. A similarity transform (shift, scale, rotation) is fitted to the votes with RANSAC. One vote per patch means a detailed object such as a car cannot outvote the background, whatever the object is. Textured means both gradient directions are present in a 9x9 window (smaller structure-tensor eigenvalue above 8, flat sky reads about 0). Plain sky shows zero flow whatever the camera does, so it gets no vote. Patches count as agreeing within 15% of the typical patch motion, never under 1 px, since blur and depth spread the flow in a fast move. The shift is read at the frame centre. Camera motion is the opposite of how the background moves on screen. Pan and tilt in % of frame width a second, push in % scale a second, roll in degrees a second |
| camera confidence | A frame's camera reading counts only with 8 or more voting patches, spanning 15% or more of the frame area, with half or more of them agreeing on one move. Otherwise the camera numbers for that frame are left blank. The share of patches that disagree (1 minus `cam_inliers` in the CSV) is how much of the frame moves on its own |
| shake | Camera velocity minus its own 0.5 s running average, RMS over 0.5 s, worked out shot by shot so the average never runs across a cut. Roll jitter is the same on roll |
| body internal | On a crop around the person at full resolution (448 px tall, `--crop-h`), the flow left after one affine fit to the whole figure, mean, in % of the person's height a second |
| body spread | The same crop split into head (top 30%), middle and bottom (last 30%) bands, median flow of each, mean distance between bands, same unit |
| cut score | Frame t+1 warped back onto frame t with the flow. The mean leftover difference per tile on a 4x4 grid, then the median tile, as a robust z against the clip's own median. A cut breaks the whole frame, something passing the lens breaks only a few tiles |
| light | Mean L (0 to 100) of the background. The swing is the largest change within any 0.5 s window |
| still | Mean flow magnitude under 0.04 px a frame at 480 px wide. One frame under 0.2 px between still frames stays still, since a keyframe pops slightly in a frozen stretch |

## Thresholds

| Event | Rule |
|---|---|
| cut | cut score z of 8 or more, and a raw leftover of 6 grey levels or more, for one or two frames in a row. The raw floor stops a keyframe's pop in a clean steady clip reading as a cut. In a clip's first half second a flagged frame whose edges stay where they were (edge_same of 0.8 or more) is not a cut, since a file's opening frames can pop in brightness with nothing changing in the picture |
| blur burst | the same for three or more frames in a row, a whip or a fast pass smeared into blur |
| whip | camera speed of 60% of frame width a second or more for two or more frames |
| zoom | push or pull of 40% scale a second or more, listed where it starts |
| hold | 0.25 s or more of still frames anywhere before the end |
| frozen tail | the still frames at the very end, left out of all the averages |

Frames at a cut and next to it are left out of the camera and body numbers, since the flow across a cut means nothing.

## Shot types in the shot table

Shots run between cuts. Each gets one label, checked in this order after the unsure rule below.

| Label | Rule |
|---|---|
| handheld | mean shake 4 or more |
| whip | peak camera speed at the whip threshold |
| push/pull | mean absolute push 8 or more and at least the pan and tilt speed |
| moving | mean camera speed 3 or more |
| locked | anything slower |

A shot where fewer than half the frames have a confident camera reading is labelled unsure before any of these, with its sure % in the shot table. On the Sabotage clip ten of 24 shots came out unsure. The ones checked on frames were a face against a flat wall, a car skidding across most of the frame, a view over a car's bonnet with the street going by, two people filling the frame, and a car driving up on plain tarmac.

## Known-motion test

`scripts/selftest.py` draws a textured picture, moves it by an exact amount each frame and checks the tool reads the same amount back. Pan 20 read 19.97, tilt 12 read 12.05, zoom 30 read 29.84, roll 10 read 9.92, a pan with a zoom read both right, random shake read within 15% of what was put in, one cut found at the right frame, a hold found at the right start and length. A brightness pop on the first frame reads no cut, and a real cut a quarter second in is still found. Two clips have a detailed object on a plainer still background, one crossing the frame and one growing as if driving at the camera. Both read a still camera. All 65 checks pass. Run it after any change.

## Validation, 2026-11-20, numbers from the 2026-11-21 review

Run on an M-series Mac with the Vision mask unless noted. The human check was the user watching the flagged moments.

| Clip | What it is | What the tool read | Human check |
|---|---|---|---|
| OK Go, "Here It Goes Again", 0:45 to 1:15 | one take, locked camera, four people dancing | camera 0.16, shake 0.12, camera sure 100%, no cuts, body internal 23, a 0.67 s hold at 22.4 s not yet checked by eye | locked and uncut, as known |
| Beastie Boys, "Sabotage", 0:30 to 1:00 | handheld 70s cop show, crash zooms, whips, fast cuts | camera 30, shake 17, camera sure 53%, 23 cuts (ffmpeg scene detection found 17, the extra 6 real on the frames), title crash zoom and 1.25 s hold at 9.2 s, the crash zoom on the man on the roof from 8.2 s (shot push 130), whips at 14.2 to 16.6 s, a blur burst at 18.7 s, ten unsure shots | cuts, title zoom and hold, roof zoom, whips at 14 to 15.5 s and the blur burst confirmed by eye on the first run. The unsure shots checked on frames only |
| MiniMax H3 FL2VA render, 5.17 s, camera arc around a standing woman | AI image-to-video between two stills | camera 10.5, shake 1.4, camera sure 99%, still for 0.5 s at the start, frozen for 0.75 s at the end | |

A review on 2026-11-21 found the tool read the camera's shift at the frame's corner, so every zoom leaked into pan and tilt (half the push into pan). On Sabotage that had inflated shake from 22 to 32 and turned four zooms into whips. The same review replaced one vote per textured point with one vote per patch, after a car driving at the camera read as a push of 28. It now reads 3.7, and the shot is marked unsure. Sabotage went from 87% sure to 53%, because shots where a car fills the frame used to get a confident reading they had not earned. Earlier numbers for these clips (shake 32, 0.19 and 2.0) are from before the fix.

Backends on the H3 render. Camera speed 10.47 (Vision), 10.57 (MediaPipe), 10.41 (none). Shake 2.21, 2.31, 2.00. Body internal 3.37 against 4.88, spread 4.46 against 8.04 (DeepLab's 257 px mask takes in some background at the edges, so its body numbers run higher).

## What Gemini can and cannot do here (checked 2026-11-20)

Gemini's API samples video at 1 fps by default, takes a custom `fps` in `(0, 24]`, adds a timestamp every second and refers to moments as MM:SS (aistudio.google.com/docs/video-understanding). Published tests put it behind dedicated tools on motion. On real footage, shot transition detection scored segment F1 0.549 for Gemini 3 Pro against 0.752 for TransNetV2, with boundaries off by 2.18 s on average against 1.87 s (TransVLM, arXiv 2604.27975, Table 3). The same paper found that feeding optical flow to its own model lifted F1 from 0.694 to 0.783. On camera motion, Gemini 3.1 Pro finds the intervals where the camera does something (SegLoc F1 76.1 at IoU 0.5) but names the exact set of moves in them only 23.7, and scores 43.3 frame-level micro F1 against 67.5 for a 4B model trained with 3D geometry (CamChoreo, arXiv 2608.10932, Table 2). No published test measures shake, roll or speed with it.

Our own test the same day, Gemini 4 Carbon on the Sabotage section, same prompt for both runs (cuts, per-shot camera and shake, holds, whips, zooms, as JSON). At 24 fps (57k input tokens, $0.13, 28 s) it found all 23 cuts with no extras, within 0.04 s on average, as good as this tool and better than ffmpeg. At the default 1 fps ($0.02) it found 13 of 23 within 0.25 s. At 24 fps it also got the title crash zoom and hold and the whip at 18.6 s, and missed the whips at 14 to 15.5 s that the viewer confirmed. On five shots where the two disagreed, this tool was right on four (a tracking shot Gemini called locked, a car driving at the camera Gemini called a push in, a low angle that tilts up after a jumper that Gemini called locked, a balcony shot with handheld drift) and Gemini on one, a crash zoom against plain sky that this tool read as nearly still at the time, since sky gives it no texture to fit the camera on (fixed since, see the validation notes). The low angle first looked locked on four frames per shot. The viewer saw it move, and every frame of the shot confirmed the tilt, so settle a camera dispute on every frame of the shot, not a few. Gemini gives labels, not numbers. Use them together, this tool for the numbers, Gemini at 24 fps for labels and a second cut list, and look at any shot where they disagree.

## Output files

| File | What |
|---|---|
| `summary.md` | one row per clip, then a shot table per clip |
| `summary.json` | the same, for code |
| `<clip>.png` | frame strip with the person mask outlined, then camera with shake, body, cut score, background light |
| `<clip>.csv` | every frame pair, columns frame, t, flow_mean, pan, tilt, push, roll, shake, roll_jitter, cam_ok, cam_conf, cam_pts (voting patches), cam_spread, cam_inliers (share of patches agreeing), subj_area, subj_h_px, subj_internal, subj_spread, cut_raw, cut_z, is_cut, cut_kind, light, light_rate, is_frozen_tail |
| `compare.png` | two or more clips over time, in pairs |
| `source/` | downloaded links |
