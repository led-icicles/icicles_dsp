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
