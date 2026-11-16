## 0.5.0

- Depends on `icicles_generators` 0.10.0.

- Deep analysis: drum hits transcribed (ADTOF) on the separated drum stem — `DeepAnalysis::drums` / `drum_error`
  (`deep.rs`), `drum_times` … `drum_error` in `IciclesDspDeepResult` (as in `IciclesDspResult`). The drum model is
  optional: read from the model directory when present; `DeepModels::Bytes` and `icicles_dsp_analyze_deep_models`
  take it as bytes (new `drum`/`drum_len` parameters before `control`). A missing model or a failed detection only
  leaves the drums empty.
- `drums::detect_with` (a loaded model on 44.1 kHz PCM), `drums::load_model` / `load_model_file` /
  `load_model_bytes`; `DrumError` is `Clone`.
- Dart: `DspDeepAnalysis.drums` (`DrumHits?`) and `drumError`, in `fromRaw` (optional `drumTimes`/`drumClasses`/
  `drumStrengths`/`drumError`) and the wire form; `IciclesDspWasm.analyzeDeep(drum:)`.

## 0.4.0

- Deep analysis: `rust/src/deep.rs` runs Hybrid Transformer Demucs (`rust/src/demucs.rs`, stems drums/bass/other/
  vocals, matching PyTorch `demucs` 4.1) and the All-In-One song structure model (`rust/src/structure.rs`, 8 Harmonix
  folds, madmom spectrogram port in `rust/src/spectrum.rs`) → labelled sections, per-stem loudness every 10 ms and the
  chroma of the harmonic stems. Models via `tool/download-deep-models.sh` (LightCut ONNX conversions, MIT).
- C ABI: `icicles_dsp_analyze_deep` / `icicles_dsp_analyze_deep_models` with an `IciclesDspControl` block (progress,
  cancel), `icicles_dsp_free_deep`, `icicles_dsp_control_size`; status `ICICLES_DSP_CANCELLED`.
- Dart: `IciclesDsp.analyzeDeep` (progress, cancel), `DspDeepAnalysis` / `DspSection` / `DspSectionLabel` /
  `DspStem` (with a Web Worker wire form), `DspErrorKind.cancelled`; `IciclesDspWasm.analyzeDeep` in the browser.
- `rust/examples/deep_analyze.rs` (sections and timings of a WAV).

## 0.3.0

- Browser build: `tool/build_wasm.sh` compiles the crate to `wasm32-wasip1` with SIMD;
  `package:icicles_dsp/icicles_dsp_web.dart` (`IciclesDspWasm`) runs it with the models as bytes and a minimal WASI
  shim, one fresh instance per analysis. Results equal the FFI build (`test/web/`, Chrome with dart2js and dart2wasm).
- C ABI: `icicles_dsp_analyze_models` (models as bytes), `icicles_dsp_alloc` / `icicles_dsp_dealloc`.
- Rust: `model::OnnxModel` (rten, from a file or bytes) and `model::ModelSource` (`Dir` / `Bytes`) for
  `beats::track` and `drums::detect`.
- Dart: `package:icicles_dsp/icicles_dsp_types.dart` (platform-neutral types) and `DspResultData` (the raw result
  arrays, shared by the FFI and WebAssembly paths).

## 0.2.1

- Depends on `icicles_generators` v0.9.0.

## 0.2.0

- Drum hits: `rust/src/drums.rs` runs the ADTOF Frame_RNN drum transcription model (`models/adtof_frame_rnn.onnx`,
  ~1.8 MB, CC BY-NC-SA 4.0, see `models/NOTICE.md`) on `rten` — madmom-style log filtered spectrogram front end
  (44.1 kHz via `rubato`, 100 fps, 84 bands), 30 s windows with 5 s of context, madmom note peak picking with
  ADTOF's per-class thresholds and a strength per hit. Matches the PyTorch reference on the fixture
  (`rust/tests/fixtures/adtof/`, written by `tool/export_adtof/export_adtof.py`).
- C ABI: `icicles_dsp_analyze` takes `flags` (`ICICLES_DSP_TRACK_BEATS`, `ICICLES_DSP_DETECT_DRUMS`); the result
  carries `drum_times` / `drum_classes` / `drum_strengths` / `drum_count`, and `drum_status` / `drum_error` (a failed
  drum detection never fails the call).
- Dart: `IciclesDsp.analyze(…, detectDrums: true)`, `IciclesDsp.drumModelFile`, `DspAnalysis.drums` (`DrumHits` of
  `icicles_generators` 0.8.0) and `drumError`. The example CLI prints the drum counts.
- Tools: `tool/export_adtof/export_adtof.py` (ONNX export + fixture), `rust/examples/eval_rbma.rs`
  (F-measure and threshold sweep) with `tool/eval-mdb.sh` (MDB Drums) and `tool/eval-rbma13.sh` (RBMA13),
  `rust/examples/detect_drums.rs`. MDB Drums: kick 0.836, snare 0.711, hi-hat 0.846 (0.975 / 0.901 / 0.916
  without the jazz tracks); leave-one-track-out cross-validated retuning gains nothing, so ADTOF's thresholds
  stay.
- Depends on icicles_generators 0.8.0.

## 0.1.1

- Depend on icicles_generators 0.7.0.

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
