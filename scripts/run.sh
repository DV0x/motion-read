#!/usr/bin/env bash
# run.sh - read the motion in video files or links.
#
#   run.sh [--section M:SS-M:SS] [--out DIR] [--mask auto|vision|mediapipe|none] [--width N] <file-or-url> [...]
#
# Links (YouTube and anything yt-dlp reads) download at up to 720p into DIR/source. --section takes only that
# stretch of every link. Sets itself up on first use (setup.sh). Prints the output folder on the last line.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOME_DIR="${MOTION_READ_HOME:-$HOME/.cache/motion-read}"
section="" out="" mask=auto
pass=() inputs=()
while [ $# -gt 0 ]; do
  case "$1" in
    --section) section="$2"; shift 2;;
    --out) out="$2"; shift 2;;
    --mask) mask="$2"; shift 2;;
    --width|--crop-h) pass+=("$1" "$2"); shift 2;;
    -h|--help) sed -n 2,8p "$0"; exit 0;;
    *) inputs+=("$1"); shift;;
  esac
done
[ ${#inputs[@]} -gt 0 ] || { echo "run.sh: give a video file or a link" >&2; exit 1; }

backend="$mask"
if [ "$backend" = auto ] || [ "$backend" = none ]; then
  if [ "$(uname)" = Darwin ]; then backend=vision; else backend=mediapipe; fi
fi
VENV="$HOME_DIR/venv-$backend"
[ -x "$VENV/bin/python" ] || bash "$HERE/setup.sh" "$backend" >/dev/null

out="${out:-$PWD/motion-read-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$out"
files=()
for x in "${inputs[@]}"; do
  if [[ "$x" =~ ^https?:// ]]; then
    mkdir -p "$out/source"
    args=(-q --no-warnings --no-update --no-playlist -f "bv*[height<=720][ext=mp4]/b[height<=720]"
          -o "$out/source/%(title).60B-%(id)s.%(ext)s" --print after_move:filepath)
    [ -n "$section" ] && args+=(--download-sections "*$section" --force-keyframes-at-cuts)
    echo "downloading $x ${section:+(section $section)} ..." >&2
    f="$("$VENV/bin/yt-dlp" "${args[@]}" "$x" | tail -1)"
    [ -s "$f" ] || { echo "run.sh: download failed for $x" >&2; exit 1; }
    files+=("$f")
  else
    [ -s "$x" ] || { echo "run.sh: no such video $x" >&2; exit 1; }
    files+=("$x")
  fi
done

MOTION_READ_HOME="$HOME_DIR" "$VENV/bin/python" "$HERE/motion.py" --mask "$mask" --out "$out" \
  ${pass[@]+"${pass[@]}"} "${files[@]}"
