/// Native audio analysis for icicles animations over `dart:ffi`.
///
/// [IciclesDsp.analyze] runs the Rust crate in `rust/` (built by the Dart build
/// hook) on a background isolate and returns a [DspAnalysis]: Beat This! beats
/// and downbeats plus the `icicles_generators` `AudioFeatures` matrix, computed
/// with the same arithmetic as the Dart `AudioFeatureExtractor`.
///
/// Native only (macOS, Windows); do not import this package on the web.
library;

export 'package:icicles_generators/icicles_generators.dart' show AudioFeatureKey, AudioFeatures;

export 'src/dsp_analysis.dart';
export 'src/dsp_exception.dart';
export 'src/icicles_dsp_impl.dart';
export 'src/wav.dart';
