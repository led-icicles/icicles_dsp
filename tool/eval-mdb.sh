#!/usr/bin/env bash
# Evaluates the drum model on MDB Drums (Southall et al., ISMIR 2017): 23
# MedleyDB tracks (rock, pop, jazz, …) with kick / snare / hi-hat / toms /
# cymbals annotations, CC BY-NC-SA 4.0, from
# https://github.com/CarlSouthall/MDBDrums (about 250 MB of full mixes).
#
# Usage: tool/eval-mdb.sh [work_dir]
#   work_dir: where annotations, converted audio and cached activations go
#             (default: build/mdb).
# Requires curl, ffmpeg and cargo.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK="${1:-$SCRIPT_DIR/../build/mdb}"
BASE="https://raw.githubusercontent.com/CarlSouthall/MDBDrums/master/MDB%20Drums"
TRACKS="80sRock Beatles BebopJazz Britpop CoolJazz Country1 Disco FreeJazz FunkJazz FusionJazz Gospel Grunge
Hendrix LatinJazz ModalJazz Punk Reggae Rock Rockabilly Shadows SpeedMetal SwingJazz Zeppelin"
mkdir -p "$WORK/ann" "$WORK/audio"

for track in $TRACKS; do
  name="MusicDelta_$track"
  [ -f "$WORK/ann/$name.txt" ] || curl -fsSL --retry 3 -o "$WORK/ann/$name.txt" "$BASE/annotations/class/${name}_class.txt"
  target="$WORK/audio/$name.f32"
  [ -f "$target" ] && continue
  echo "$name"
  curl -fsSL --retry 3 -o "$WORK/mix.wav" "$BASE/audio/full_mix/${name}_MIX.wav"
  ffmpeg -v error -y -i "$WORK/mix.wav" -ac 1 -ar 44100 -f f32le "$target"
  rm -f "$WORK/mix.wav"
done

cd "$SCRIPT_DIR/../rust"
cargo run --release --example eval_rbma -- "$WORK/audio" "$WORK/ann"
