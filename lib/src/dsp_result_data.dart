import 'dart:typed_data';

import 'package:icicles_generators/icicles_generators.dart';

import 'dsp_analysis.dart';
import 'dsp_chords.dart';
import 'dsp_exception.dart';
import 'drum_hits_raw.dart';

/// The raw arrays of one analysis, as the Rust library returns them (the
/// `IciclesDspResult` struct): times in seconds, the row-major feature
/// matrix, the drum hits and the chords as parallel arrays. Read from native memory (FFI) or
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

  /// The chords: start and end seconds, root pitch classes (C = 0) and
  /// quality indices ([DspChord.qualityNames]); empty when there are none.
  final Float32List chordStarts;
  final Float32List chordEnds;
  final Uint8List chordRoots;
  final Uint8List chordQualities;
  final String? chordError;

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
    Float32List? chordStarts,
    Float32List? chordEnds,
    Uint8List? chordRoots,
    Uint8List? chordQualities,
    this.chordError,
  }) : chordStarts = chordStarts ?? Float32List(0),
       chordEnds = chordEnds ?? Float32List(0),
       chordRoots = chordRoots ?? Uint8List(0),
       chordQualities = chordQualities ?? Uint8List(0);

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
      drums: drumHitsFromRaw(drumTimes, drumClasses, drumStrengths),
      drumError: drumError,
      chords: chordsFromRaw(chordStarts, chordEnds, chordRoots, chordQualities),
      chordError: chordError,
    );
  }

  static List<Duration> _durations(Float32List seconds) => [for (final s in seconds) durationOfSeconds(s)];
}

/// The [DspErrorKind] of a library status code (`IciclesDspStatus`).
DspErrorKind dspErrorKindOf(int status) => switch (status) {
  1 => DspErrorKind.invalidArgument,
  2 => DspErrorKind.modelNotFound,
  3 => DspErrorKind.modelLoadFailed,
  4 => DspErrorKind.analysisFailed,
  5 => DspErrorKind.nativePanic,
  6 => DspErrorKind.cancelled,
  _ => DspErrorKind.unknown,
};

/// The library's analysis flags (`ICICLES_DSP_TRACK_BEATS`,
/// `ICICLES_DSP_DETECT_DRUMS`, `ICICLES_DSP_DETECT_CHORDS`).
int dspFlags({required bool trackBeats, required bool detectDrums, required bool detectChords}) =>
    (trackBeats ? 1 : 0) | (detectDrums ? 2 : 0) | (detectChords ? 4 : 0);
