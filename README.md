# icicles_dsp

Native audio analysis for icicles animations over `dart:ffi`: one call turns a
mono PCM signal into

- **beats and downbeats** from [Beat This!](https://github.com/CPJKU/beat_this)
  (ISMIR 2024, the current state of the art in beat tracking) through the Rust
  crate [`beat-this`](https://crates.io/crates/beat-this) — pure-Rust `rten`
  inference, no ONNX Runtime or other system libraries;
- **drum hits** — kick, snare, toms, hi-hat, cymbals with a strength per hit —
  from the [ADTOF](https://github.com/MZehren/ADTOF) Frame_RNN drum
  transcription model (Zehren et al.), exported to ONNX and run on the same
  `rten` runtime;
- the **`AudioFeatures` matrix** of `icicles_generators` (22 columns every
  10 ms: level, bass, mid, treble, onset, peak frequency, 16 mel bands), a Rust
  port of the Dart `AudioFeatureExtractor` with the same arithmetic — the output
  is bit-identical on the fixture signals, so bakes and plugins see the same
  numbers whichever extractor ran.

The crate lives in `rust/` and is built by the Dart build hook
(`hook/build.dart`, [`native_toolchain_rust`](https://pub.dev/packages/native_toolchain_rust)),
so `dart run`, `dart test`, `flutter test`, `flutter run` and `flutter build`
compile and bundle it; nothing is prebuilt.

Three libraries:

- `package:icicles_dsp/icicles_dsp.dart` — `IciclesDsp` over `dart:ffi`
  (macOS, Windows); do not import it on the web.
- `package:icicles_dsp/icicles_dsp_web.dart` — `IciclesDspWasm`, the same crate
  compiled to WebAssembly (`tool/build_wasm.sh` → `build/web/icicles_dsp.wasm`)
  for the browser, the models passed as bytes. Same results as the FFI build
  (pinned by `test/web/`). Single-threaded and synchronous: run it on a Web
  Worker (a 3-minute song with the full beat model takes ~30 s and ~600 MB of
  module memory, released after each call).
- `package:icicles_dsp/icicles_dsp_types.dart` — the platform-neutral types
  (`DspAnalysis`, `DspException`, `DspResultData`).

### Browser build

`tool/build_wasm.sh` builds `wasm32-wasip1` with SIMD (`-C target-feature=+simd128`;
`wasm-opt -O3` when installed). wasip1 rather than `wasm32-unknown-unknown`:
`beat-this` times its stages with `std::time::Instant`, which panics without a
clock import there. The module imports only WASI preview 1 calls;
`IciclesDspWasm` provides the clock, `crypto.getRandomValues` and stderr
(collected into the panic message) and answers the file system calls with
"no such file". A Rust panic aborts in WebAssembly (no unwinding), so every
analysis runs on a fresh instance of the compiled module.

## Usage

```dart
import 'package:icicles_dsp/icicles_dsp.dart';

final wav = WavAudio.read('song.wav'); // or PCM from any decoder
final analysis = await const IciclesDsp().analyze(
  wav.mono, // Float32List, -1..1
  wav.sampleRate,
  modelDir: '/path/to/models', // null: the package's models/ (dev only)
);
analysis.beats;     // List<Duration>, sorted
analysis.downbeats; // List<Duration>, a subset of beats
analysis.features;  // AudioFeatures (icicles_generators)
analysis.drums;     // DrumHits (icicles_generators), DrumHits.empty when not detected
analysis.drumError; // why the drum detection failed (it never fails the analysis), else null
analysis.bpm;       // mean tempo, null below two beats
```

`analyze` runs on a background isolate (`Isolate.run`) and never blocks the
caller; `IciclesDsp.analyzeSync` is the blocking call behind it.
`trackBeats: false` / `detectDrums: false` skip the beat / drum models; with
both off only the features are computed (no models needed).
Failures are `DspException { kind: DspErrorKind, message }`
(`libraryUnavailable`, `invalidArgument`, `modelNotFound`, `modelLoadFailed`,
`analysisFailed`, `nativePanic`). `IciclesDsp.isAvailable` tells whether the
native library loads, so a caller can fall back to the Dart
`AudioFeatureExtractor`.

The example CLI prints the tempo and the drum hit counts of a WAV file:

```sh
cd example && dart run bin/analyze.dart song.wav [model-dir]
```

## Models

`models/` holds, copied from
[beat-this-rs](https://github.com/danigb/beat-this-rs) (MIT, see
`models/NOTICE.md`):

| File | Size | Role |
| --- | --- | --- |
| `mel_spectrogram.onnx` | ~270 KB | log-mel front end, always required |
| `beat_this_small.onnx` | ~10 MB | small beat model, bundled |
| `beat_this.onnx` | ~83 MB | full-accuracy model, **not bundled** |
| `adtof_frame_rnn.onnx` | ~1.8 MB | ADTOF drum model (CC BY-NC-SA 4.0), bundled |

The library reads them from the `modelDir` passed to `analyze`: it needs
`mel_spectrogram.onnx` and uses `beat_this.onnx` when present, else
`beat_this_small.onnx`. `tool/download-models.sh [dir]` fetches the full
model from the upstream GitHub release and verifies its SHA-256; an app can run
the same download itself into its support directory. The default
`modelDir` (`IciclesDsp.bundledModelDir()`) resolves the package's own
`models/` through the package config and works for `dart run`/`dart test`/
`flutter test`; a compiled Flutter app has no package directory, so it ships
the two files as assets, copies them to a writable directory once and passes
that path. The drum model `adtof_frame_rnn.onnx` is read from the same
directory.

### Drum detection

`rust/src/drums.rs` reproduces the model's training front end (madmom's log
filtered spectrogram: 44.1 kHz, Hann 2048, hop 441 = 100 fps, 12 bands per
octave 20 Hz–20 kHz on unique FFT bins = 84 bands, `log10(1 + x)`), runs the
network on 30 s windows with 5 s of context on each side (bounded memory; the
GRUs are bidirectional), picks peaks per class like madmom's
`NotePeakPickingProcessor` with ADTOF's thresholds and gives every hit a
strength from the loudness of the class's band (relative to the class's
90th-percentile hit, 24 dB range) weighted by the model's confidence.

The ONNX file is produced by `uv run tool/export_adtof/export_adtof.py` from
the PyTorch port [xavriley/ADTOF-pytorch](https://github.com/xavriley/ADTOF-pytorch)
(weights pinned by commit and SHA-256): the GRUs become ONNX `GRU` ops with
`linear_before_reset = 1` (the Keras `reset_after` maths, which rten
supports); the script checks onnxruntime against PyTorch and writes the parity
fixture `rust/tests/fixtures/adtof/` (a synthetic drum loop, its spectrogram
and activations), which `rust/tests/drums.rs` holds rten to (< 1e-3).

Evaluation: `rust/examples/eval_rbma.rs` computes the F-measure per class
(50 ms tolerance, one-to-one matching, hits summed over tracks) with a
threshold sweep. `tool/eval-mdb.sh` downloads and runs MDB Drums (23 MedleyDB
tracks, CC BY-NC-SA 4.0); `tool/eval-rbma13.sh <album_dir>` runs RBMA13
(electronic music; the album is a paid Bandcamp download).

Results on MDB Drums with ADTOF's thresholds. They are kept: thresholds
belong to a validation set, not the test set, and leave-one-track-out
cross-validation (each track scored with thresholds tuned on the others)
gains nothing — mean F 0.801 vs 0.797 over all tracks, 0.931 vs 0.931
without jazz; kick and hi-hat get slightly worse, and the snare thresholds
chosen per fold scatter over 0.05–0.15 (brushed jazz snares):

| Tracks | Kick F (P / R) | Snare F (P / R) | Hi-hat F (P / R) |
| --- | --- | --- | --- |
| all 23 | 0.836 (0.79 / 0.89) | 0.711 (0.85 / 0.61) | 0.846 (0.89 / 0.81) |
| 15 without jazz | 0.975 (0.97 / 0.98) | 0.901 (0.93 / 0.87) | 0.916 (0.89 / 0.94) | `rust/examples/detect_drums.rs` prints the hits of one file per time
window.

The models are loaded on every `analyze` call (well under 0.2 s for the small
model); the beat model runs on 30 s chunks, so short signals are fine.

## Build requirements

- [rustup](https://rustup.rs) with the toolchain pinned in
  `rust/rust-toolchain.toml` (`channel = "1.99.1"`; rustup installs it and the
  listed targets on first use — `native_toolchain_rust` refuses a bare
  `stable`). Builds are release mode (`lto`, `panic = "unwind"` so panics can be
  caught at the FFI boundary).
- Targets listed there: `aarch64-apple-darwin`, `x86_64-apple-darwin`,
  `aarch64-pc-windows-msvc`, `x86_64-pc-windows-msvc`. Windows needs the MSVC
  build tools (Visual Studio C++ workload), as every `*-msvc` Rust target does.
- Dart ≥ 3.13 / Flutter ≥ 3.47 (build hooks). No flags were needed on macOS:
  `dart test`, `flutter test` and `dart run` all ran the hook as is.
- Regenerating the bindings (after changing `rust/include/icicles_dsp.h`, kept
  in sync with `rust/src/ffi.rs` by hand): `dart run ffigen --config ffigen.yaml`
  (needs libclang, i.e. Xcode or LLVM).

## C ABI

`rust/include/icicles_dsp.h`:

- `const char *icicles_dsp_version(void)` — static string.
- `int32_t icicles_dsp_analyze(const float *samples, size_t len, uint32_t sample_rate, const char *model_dir, uint32_t flags, IciclesDspResult *out)`
  — `flags`: `ICICLES_DSP_TRACK_BEATS` (1), `ICICLES_DSP_DETECT_DRUMS` (2);
  `model_dir` NULL = features only. Returns `IciclesDspStatus`
  (0 ok, 1 invalid argument, 2 model not found, 3 model load failed,
  4 analysis failed, 5 panic caught). `out` holds `beats`/`beat_count`,
  `downbeats`/`downbeat_count` (seconds, f32), `features`/`feature_rows`/
  `feature_columns` (row-major, the `AudioFeatures` layout),
  `feature_hop_micros` (10 000) and the drum hits `drum_times` (seconds) /
  `drum_classes` (0 kick … 4 cymbals) / `drum_strengths` / `drum_count`; a
  failed drum detection does not fail the call — it sets `drum_status` and
  `drum_error`.
- `const char *icicles_dsp_last_error(void)` — message of the last failure on
  the calling thread.
- `int32_t icicles_dsp_analyze_models(samples, len, sample_rate, mel, mel_len, beat, beat_len, drum, drum_len, flags, out)`
  — the same with the models as bytes (the browser build); NULL `mel`/`beat` =
  features only, NULL `drum` fails only the drum detection.
- `uint8_t *icicles_dsp_alloc(size_t len)` / `void icicles_dsp_dealloc(uint8_t *, size_t len)`
  — memory inside the library for a host that writes the PCM and models there.
- `void icicles_dsp_free(IciclesDspResult *out)` — releases the arrays.

Every entry point is wrapped in `catch_unwind`; a panic becomes status 5 and
the library stays usable.

## Tests

- `cd rust && cargo test` — unit tests, the feature matrix against the Dart
  reference fixtures (`rust/tests/fixtures/*.json`, regenerated with
  `dart run tool/dump_features.dart`; the synthetic signals are defined in
  both `tool/dump_features.dart` and `rust/tests/common/mod.rs`), a 120 BPM
  click track within ±30 ms, silence → no beats, the drum front end and model
  against the PyTorch fixture, the drum loop's hits.
  `cargo test --release --test beats -- --ignored --nocapture timing` prints
  the timing of a 3-minute signal.
- `dart test` or `flutter test` — the Dart facade on a generated WAV (the hook
  builds the crate first).
- `tool/build_wasm.sh && dart test -p chrome test/web/` (also `-c dart2wasm`) —
  the WebAssembly build reproduces the FFI analysis of a click track
  (`test/fixtures/click_120_reference.json`, written by
  `dart run tool/dump_reference_analysis.dart`).

## Performance

Apple M-series, release build, 3 minutes at 44.1 kHz: features in 0.21 s,
beats (small model) in 1.45 s; through the Dart facade about 2.1 s in all.
Drums: 390 s of music in about 6.4 s (spectrogram 0.3 s, model 6 s).
