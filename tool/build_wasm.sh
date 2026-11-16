#!/usr/bin/env bash
# Builds the analysis for the browser: the Rust crate for wasm32-wasip1 with
# WebAssembly SIMD, to build/web/icicles_dsp.wasm (loaded by
# package:icicles_dsp/icicles_dsp_web.dart). wasip1 rather than
# wasm32-unknown-unknown: the standard library's clock (used by beat-this for
# stage timings) needs an import there; IciclesDspWasm provides the few WASI
# calls. Needs rustup (the toolchain of rust/rust-toolchain.toml lists the
# target); wasm-opt is used when installed.
set -euo pipefail
cd "$(dirname "$0")/.."
out="build/web"
mkdir -p "$out"
(cd rust && RUSTFLAGS="-C target-feature=+simd128" cargo build --release --lib --target wasm32-wasip1)
wasm="rust/target/wasm32-wasip1/release/icicles_dsp.wasm"
if command -v wasm-opt >/dev/null 2>&1; then
  wasm-opt -O3 --enable-simd --enable-bulk-memory --enable-nontrapping-float-to-int --enable-sign-ext "$wasm" -o "$out/icicles_dsp.wasm"
else
  cp "$wasm" "$out/icicles_dsp.wasm"
fi
echo "Built $out/icicles_dsp.wasm ($(du -h "$out/icicles_dsp.wasm" | cut -f1))."
