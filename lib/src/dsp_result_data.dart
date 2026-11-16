import 'dart:typed_data';

import 'package:icicles_generators/icicles_generators.dart';

import 'dsp_analysis.dart';
import 'dsp_exception.dart';

/// The raw arrays of one analysis, as the Rust library returns them (the
/// `IciclesDspResult` struct): times in seconds, the row-major feature
/// matrix, the drum hits as parallel arrays. Read from native memory (FFI) or
/// from the WebAssembly module's memory (web), and plain enough to cross a
/// Web Worker boundary; [toAnalysis] builds the [DspAnalysis].
final class DspResultData {
  final Float32List beats;
  final Float32List downbeats;
  final Float32List features;
  final int featureRows;
  final int featureColumns;
  final int featureHopMicros;
  final Float32List drumTimes;

  /// 0 kick, 1 snare, 2 toms, 3 hi-hat, 4 cymbals ([DrumClass] order).
  final Uint8List drumClasses;
  final Float32List drumStrengths;
  final String? drumError;

  DspResultData({
    required this.beats,
    required this.downbeats,
    required this.features,
    required this.featureRows,
    required this.featureColumns,
    required this.featureHopMicros,
    required this.drumTimes,
    required this.drumClasses,
    required this.drumStrengths,
    this.drumError,
  });

  /// The analysis; throws a [DspException] ([DspErrorKind.unknown]) when the
  /// library's feature layout is not this package's.
  DspAnalysis toAnalysis() {
    if (featureColumns != AudioFeatureKey.columnCount) {
      throw DspException(
        DspErrorKind.unknown,
        'Native feature layout has $featureColumns columns, expected ${AudioFeatureKey.columnCount}.',
      );
    }
    if (features.length != featureRows * featureColumns) {
      throw DspException(
        DspErrorKind.unknown,
        'Feature matrix of ${features.length} values, expected '
        '$featureRows × $featureColumns.',
      );
    }
    return DspAnalysis(
      beats: _durations(beats),
      downbeats: _durations(downbeats),
      features: AudioFeatures(features, hopDuration: Duration(microseconds: featureHopMicros)),
      drums: _drums(),
      drumError: drumError,
    );
  }

  DrumHits? _drums() {
    final count = drumTimes.length;
    if (count == 0) return null;
    final hits = <DrumClass, List<DrumHit>>{};
    for (var i = 0; i < count; i++) {
      if (drumClasses[i] >= DrumClass.values.length) continue;
      (hits[DrumClass.values[drumClasses[i]]] ??= []).add(DrumHit(_duration(drumTimes[i]), drumStrengths[i]));
    }
    return DrumHits(hits, source: DrumHits.sourceAdtof);
  }

  static List<Duration> _durations(Float32List seconds) => [for (final s in seconds) _duration(s)];

  static Duration _duration(double seconds) =>
      Duration(microseconds: (seconds * Duration.microsecondsPerSecond).round());
}

/// The [DspErrorKind] of a library status code (`IciclesDspStatus`).
DspErrorKind dspErrorKindOf(int status) => switch (status) {
  1 => DspErrorKind.invalidArgument,
  2 => DspErrorKind.modelNotFound,
  3 => DspErrorKind.modelLoadFailed,
  4 => DspErrorKind.analysisFailed,
  5 => DspErrorKind.nativePanic,
  _ => DspErrorKind.unknown,
};

/// The library's analysis flags (`ICICLES_DSP_TRACK_BEATS`,
/// `ICICLES_DSP_DETECT_DRUMS`).
int dspFlags({required bool trackBeats, required bool detectDrums}) => (trackBeats ? 1 : 0) | (detectDrums ? 2 : 0);
