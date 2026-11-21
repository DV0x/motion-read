#!/usr/bin/env bash
# peek.sh - look at the frames around flagged moments before trusting them.
#
#   peek.sh [--out sheet.png] <video> <t1> [t2 ...]
#
# One row per time, three frames each: two before the time and the frame at it. ffmpeg only.
# Prints the sheet path. Read the sheet as an image.
set -euo pipefail

out=""
if [ "${1:-}" = "--out" ]; then out="$2"; shift 2; fi
[ $# -ge 2 ] || { sed -n 2,8p "$0"; exit 1; }
video="$1"; shift
command -v ffmpeg >/dev/null 2>&1 || { echo "peek.sh: needs ffmpeg" >&2; exit 1; }
fps="$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$video" | head -1)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
i=0
for t in "$@"; do
  i=$((i + 1))
  start="$(awk -v t="$t" -v f="$fps" 'BEGIN { split(f, a, "/"); r = (a[2] ? a[1] / a[2] : a[1]); s = t - 2 / r; if (s < 0) s = 0; printf "%.3f", s }')"
  ffmpeg -v error -ss "$start" -i "$video" -vf "scale=320:-2,tile=3x1" \
    -frames:v 1 -y "$tmp/$(printf %03d $i).png"
done
out="${out:-$(dirname "$video")/peek-$(date +%H%M%S).png}"
ffmpeg -v error -pattern_type glob -i "$tmp/*.png" -vf "tile=1x$i" -frames:v 1 -y "$out"
echo "rows, top to bottom: $*" >&2
echo "$out"
