#!/usr/bin/env bash
# Evaluates the drum model on RBMA13 (Vogl et al., ISMIR 2017).
#
# The annotations are downloaded from the dataset page; the audio is the Red
# Bull Music Academy album "Various Assets - Not For Sale" (Bandcamp, a paid
# download), which has to be downloaded by hand:
#   https://rbma.bandcamp.com/album/various-assets-not-for-sale-red-bull-music-academy-new-york-2013
#
# Usage: tool/eval-rbma13.sh <album_dir> [work_dir]
#   album_dir: the unpacked album (any format ffmpeg reads); track NN is the
#              file whose name starts with or contains "NN " / "-NN" (the
#              Bandcamp numbering).
#   work_dir:  where annotations, converted audio and cached activations go
#              (default: build/rbma13).
# Requires curl, unzip, ffmpeg and cargo.
set -euo pipefail

ALBUM="${1:?usage: tool/eval-rbma13.sh <album_dir> [work_dir]}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK="${2:-$SCRIPT_DIR/../build/rbma13}"
mkdir -p "$WORK/audio"

if [ ! -d "$WORK/rbma_13/annotations/drums" ]; then
  curl -fL --retry 3 -o "$WORK/rbma13-drums.zip" "http://ifs.tuwien.ac.at/~vogl/datasets/rbma13-drums.zip"
  unzip -q -o "$WORK/rbma13-drums.zip" -d "$WORK"
fi

for annotation in "$WORK"/rbma_13/annotations/drums/RBMA-13-Track-*.txt; do
  name="$(basename "$annotation" .txt)"
  number="${name##*-}"
  target="$WORK/audio/$name.f32"
  [ -f "$target" ] && continue
  source="$(find "$ALBUM" -type f \( -iname "* ${number} *" -o -iname "${number} *" -o -iname "*-${number}*" -o -iname "*- ${number} -*" \) \
    \( -iname '*.mp3' -o -iname '*.flac' -o -iname '*.wav' -o -iname '*.m4a' -o -iname '*.aiff' -o -iname '*.ogg' \) | head -n 1)"
  if [ -z "$source" ]; then
    echo "no audio for $name (track $number) in $ALBUM" >&2
    continue
  fi
  echo "$name <- $(basename "$source")"
  ffmpeg -v error -y -i "$source" -ac 1 -ar 44100 -f f32le "$target"
done

cd "$SCRIPT_DIR/../rust"
cargo run --release --example eval_rbma -- "$WORK/audio" "$WORK/rbma_13/annotations/drums"
