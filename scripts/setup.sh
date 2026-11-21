#!/usr/bin/env bash
# setup.sh - one-time environment for motion-read. Wheels only, nothing compiles.
#
#   setup.sh [auto|vision|mediapipe]
#
# MOTION_READ_HOME (default ~/.cache/motion-read) holds one venv per mask backend and the MediaPipe model.
# auto picks Apple Vision on macOS (built in, no model download) and MediaPipe DeepLab v3 everywhere else.
# Prints the venv path on the last line.
set -euo pipefail

HOME_DIR="${MOTION_READ_HOME:-$HOME/.cache/motion-read}"
BACKEND="${1:-auto}"
if [ "$BACKEND" = auto ]; then
  if [ "$(uname)" = Darwin ]; then BACKEND=vision; else BACKEND=mediapipe; fi
fi
case "$BACKEND" in vision|mediapipe) ;; *) echo "setup.sh: backend must be auto, vision or mediapipe" >&2; exit 1;; esac

# Python 3.10 to 3.13, newest first among the ones the wheels cover well
PY=""
for c in python3.12 python3.11 python3.13 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1; then
    v="$("$c" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    case "$v" in 3.10|3.11|3.12|3.13) PY="$c"; break;; esac
  fi
done
[ -n "$PY" ] || { echo "setup.sh: needs Python 3.10 to 3.13 (found none)" >&2; exit 1; }
command -v ffmpeg >/dev/null 2>&1 || echo "setup.sh: note, ffmpeg is not installed. peek.sh and URL sections need it." >&2

mkdir -p "$HOME_DIR"
VENV="$HOME_DIR/venv-$BACKEND"
[ -x "$VENV/bin/python" ] || "$PY" -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
if [ "$BACKEND" = vision ]; then
  "$VENV/bin/pip" install -q --only-binary=:all: opencv-python-headless numpy matplotlib yt-dlp \
    pyobjc-framework-Vision pyobjc-framework-Quartz
else
  # mediapipe brings its own opencv-contrib-python, so no second OpenCV here
  "$VENV/bin/pip" install -q --only-binary=:all: mediapipe numpy matplotlib yt-dlp
  M="$HOME_DIR/deeplab_v3.tflite"
  [ -s "$M" ] || curl -fsSL -o "$M" \
    https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite
fi
"$VENV/bin/python" -c "import cv2, numpy, matplotlib; print('motion-read ready, mask $BACKEND, opencv', cv2.__version__)" >&2
echo "$VENV"
