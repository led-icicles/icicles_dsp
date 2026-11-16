#!/usr/bin/env bash
# Downloads the deep-analysis models (~200 MB) into the given directory
# (default: this package's models/), verifying their SHA-256:
# - htdemucs_core.onnx: the Hybrid Transformer Demucs network (Défossez, MIT);
# - allin1_fold0..7.onnx: the eight All-In-One Harmonix folds (Kim & Nam, MIT).
# ONNX conversions published by LightCut (MIT) on Hugging Face, pinned to a
# commit. They are not committed here (the app bundles them through Git LFS).
#
# Usage: tool/download-deep-models.sh [target-dir]
set -euo pipefail

REPO="azstorm/lightcut-music-analysis-models"
REVISION="971366ce07f76f7236b6bca636e162ecc12e985f"
BASE="https://huggingface.co/${REPO}/resolve/${REVISION}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET="${1:-$SCRIPT_DIR/../models}"
mkdir -p "$TARGET"

# local name | remote name | sha256
MODELS="
htdemucs_core.onnx htdemucs-core.onnx ab89734188dad008d23d45e17c49d919ed71bf8d48531b7e7679bc38b94510d3
allin1_fold0.onnx harmonix-fold0.onnx c8e2612cd340ef57268f612eafba72360af3948dc8f3aaca0a8bf53333f13430
allin1_fold1.onnx harmonix-fold1.onnx 71bbf13583d0bbcf0c608b0dbe9da98000340494c51ccc2397302db28eef6c52
allin1_fold2.onnx harmonix-fold2.onnx 8fee6f4d27dcdc0ed8644d310887527125d5e0d6caf0585f5fd953174d82bfa2
allin1_fold3.onnx harmonix-fold3.onnx e6685c8b2027f5b7e83639725c900f09de0e8efb308cb263c34897aae85b5df4
allin1_fold4.onnx harmonix-fold4.onnx bece14436e70417f0bf65cfc288e1f465f9599c2cafad2226f10e0fc48741aae
allin1_fold5.onnx harmonix-fold5.onnx a37d14dfc2dae21c85fddec1752792d88b70bcdbfc5d974df35c854993a035e3
allin1_fold6.onnx harmonix-fold6.onnx 353bf6ef0c6f6abf5c3fc31585a56885c1e3fa4e70fe95c5841d560056d65c7a
allin1_fold7.onnx harmonix-fold7.onnx fffbfa68890489baaf83038a53b1d73b23485e1bc9a79877c202fa45d6c471ea
"

sha() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

echo "$MODELS" | while read -r local remote digest; do
  [ -z "$local" ] && continue
  file="${TARGET}/${local}"
  if [ -f "$file" ] && [ "$(sha "$file")" = "$digest" ]; then
    echo "${local}: up to date"
    continue
  fi
  echo "Downloading ${local}..."
  curl -fL --retry 3 -o "${file}.part" "${BASE}/${remote}"
  if [ "$(sha "${file}.part")" != "$digest" ]; then
    echo "${local}: SHA-256 mismatch" >&2
    rm -f "${file}.part"
    exit 1
  fi
  mv "${file}.part" "$file"
done
echo "Deep models in ${TARGET}"
