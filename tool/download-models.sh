#!/usr/bin/env bash
# Downloads the full-accuracy Beat This! FP32 model (beat_this.onnx, ~83 MB)
# from the beat-this-rs "model-large" GitHub release into the given directory
# (default: this package's models/), verifying its SHA-256. The bundled small
# model keeps working without it; the library prefers beat_this.onnx when the
# model directory contains it.
#
# Usage: tool/download-models.sh [target-dir]
# Requires curl and sha256sum or shasum.
set -euo pipefail

REPO="danigb/beat-this-rs"
ASSET="beat_this.onnx"
RELEASE_TAG="model-large"
BASE="https://github.com/${REPO}/releases/download/${RELEASE_TAG}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET="${1:-$SCRIPT_DIR/../models}"
mkdir -p "$TARGET"

echo "Downloading ${ASSET} from the ${RELEASE_TAG} release of ${REPO} into ${TARGET}..."
curl -fL --retry 3 -o "${TARGET}/${ASSET}" "${BASE}/${ASSET}"
curl -fL --retry 3 -o "${TARGET}/${ASSET}.sha256" "${BASE}/${ASSET}.sha256"

echo "Verifying SHA-256..."
(
  cd "$TARGET"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum -c "${ASSET}.sha256"
  else
    shasum -a 256 -c "${ASSET}.sha256"
  fi
)
rm -f "${TARGET}/${ASSET}.sha256"
echo "Saved ${TARGET}/${ASSET}"
