# icicles_dsp

Native audio analysis for icicles animations over `dart:ffi`: one call turns a
mono PCM signal into

- **beats and downbeats** from [Beat This!](https://github.com/CPJKU/beat_this)
  (ISMIR 2024, the current state of the art in beat tracking) through the Rust
  crate [`beat-this`](https://crates.io/crates/beat-this) — pure-Rust `rten`
  inference, no ONNX Runtime or other system libraries;
- the **`AudioFeatures` matrix** of `icicles_generators` (22 columns every
  10 ms: level, bass, mid, treble, onset, peak frequency, 16 mel bands), a Rust
  port of the Dart `AudioFeatureExtractor` with the same arithmetic — the output
  is bit-identical on the fixture signals, so bakes and plugins see the same
  numbers whichever extractor ran.

The crate lives in `rust/` and is built by the Dart build hook
(`hook/build.dart`, [`native_toolchain_rust`](https://pub.dev/packages/native_toolchain_rust)),
so `dart run`, `dart test`, `flutter test`, `flutter run` and `flutter build`
compile and bundle it; nothing is prebuilt. Native only (macOS, Windows): do
not import the package on the web.

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
analysis.bpm;       // mean tempo, null below two beats
```

`analyze` runs on a background isolate (`Isolate.run`) and never blocks the
caller; `IciclesDsp.analyzeSync` is the blocking call behind it.
`trackBeats: false` skips the models and computes only the features.
Failures are `DspException { kind: DspErrorKind, message }`
(`libraryUnavailable`, `invalidArgument`, `modelNotFound`, `modelLoadFailed`,
`analysisFailed`, `nativePanic`). `IciclesDsp.isAvailable` tells whether the
native library loads, so a caller can fall back to the Dart
`AudioFeatureExtractor`.

The example CLI prints the tempo of a WAV file:

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

The library reads them from the `modelDir` passed to `analyze`: it needs
`mel_spectrogram.onnx` and uses `beat_this.onnx` when present, else
`beat_this_small.onnx`. `tool/download-models.sh [dir]` fetches the full
model from the upstream GitHub release and verifies its SHA-256; an app can run
the same download itself into its support directory. The default
`modelDir` (`IciclesDsp.bundledModelDir()`) resolves the package's own
`models/` through the package config and works for `dart run`/`dart test`/
`flutter test`; a compiled Flutter app has no package directory, so it ships
the two files as assets, copies them to a writable directory once and passes
that path.

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
- `int32_t icicles_dsp_analyze(const float *samples, size_t len, uint32_t sample_rate, const char *model_dir, IciclesDspResult *out)`
  — `model_dir` NULL = features only. Returns `IciclesDspStatus`
  (0 ok, 1 invalid argument, 2 model not found, 3 model load failed,
  4 analysis failed, 5 panic caught). `out` holds `beats`/`beat_count`,
  `downbeats`/`downbeat_count` (seconds, f32), `features`/`feature_rows`/
  `feature_columns` (row-major, the `AudioFeatures` layout) and
  `feature_hop_micros` (10 000).
- `const char *icicles_dsp_last_error(void)` — message of the last failure on
  the calling thread.
- `void icicles_dsp_free(IciclesDspResult *out)` — releases the arrays.

Every entry point is wrapped in `catch_unwind`; a panic becomes status 5 and
the library stays usable.

## Tests

- `cd rust && cargo test` — unit tests, the feature matrix against the Dart
  reference fixtures (`rust/tests/fixtures/*.json`, regenerated with
  `dart run tool/dump_features.dart`; the synthetic signals are defined in
  both `tool/dump_features.dart` and `rust/tests/common/mod.rs`), a 120 BPM
  click track within ±30 ms, silence → no beats.
  `cargo test --release --test beats -- --ignored --nocapture timing` prints
  the timing of a 3-minute signal.
- `dart test` or `flutter test` — the Dart facade on a generated WAV (the hook
  builds the crate first).

## Performance

Apple M-series, release build, 3 minutes at 44.1 kHz: features in 0.21 s,
beats (small model) in 1.45 s; through the Dart facade about 2.1 s in all.
