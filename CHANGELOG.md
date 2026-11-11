## 0.1.0

- Rust crate `icicles_dsp` (cdylib + staticlib) built by the Dart build hook
  with `native_toolchain_rust`, toolchain pinned in `rust/rust-toolchain.toml`.
- `icicles_dsp_analyze`: Beat This! beats and downbeats through `beat-this`
  (pure-Rust `rten` inference) plus the `icicles_generators` `AudioFeatures`
  matrix (22 columns every 10 ms), bit-identical to the Dart extractor on the
  fixture signals; `icicles_dsp_version`, `icicles_dsp_last_error`,
  `icicles_dsp_free`; panics never cross the FFI boundary.
- Dart facade `IciclesDsp.analyze` → `DspAnalysis { beats, downbeats, features }`
  on a background isolate, `DspException` with `DspErrorKind`, `isAvailable`,
  `bundledModelDir`, a small `WavAudio` codec and an example CLI.
- Bundled models: `mel_spectrogram.onnx` and `beat_this_small.onnx` from
  beat-this-rs (MIT); `tool/download-models.sh` fetches the full model.
