import 'dart:typed_data';

import 'package:icicles_generators/icicles_generators.dart';

import 'drum_hits_raw.dart';

/// The label of a [DspSection] (All-In-One's Harmonix classes, in the
/// library's index order).
enum DspSectionLabel { start, end, intro, outro, breakdown, bridge, inst, solo, verse, chorus }

/// One labelled section of the song structure.
final class DspSection {
  final Duration start;
  final Duration end;
  final DspSectionLabel label;

  const DspSection(this.start, this.end, this.label);

  @override
  String toString() => 'DspSection(${label.name} $start–$end)';
}

/// The stems of [DspDeepAnalysis.stemLevels], in column order.
enum DspStem { drums, bass, other, vocals }

/// The result of the deep analysis ([IciclesDsp.analyzeDeep] / the browser
/// build): the song structure, per-stem loudness, harmonic chroma and the
/// drum hits transcribed on the separated drum stem.
final class DspDeepAnalysis {
  /// Sections covering the song, in order.
  final List<DspSection> sections;

  /// Stem loudness `0..1`, row-major `stemRows × 4` ([DspStem] order), one
  /// row every 10 ms (the rows of the feature matrix).
  final Float32List stemLevels;
  final int stemRows;

  /// Chroma of the harmonic stems, row-major `chromaRows × 12` (C … B),
  /// [chromaRowsPerSecond] rows a second; rows sum to 1 (all zero in silence).
  final Float32List chroma;
  final int chromaRows;

  static const chromaRowsPerSecond = 10;

  /// The drum hits (ADTOF) of the separated drum stem — cleaner than the
  /// first pass's on the full mix, where loud bass and other instruments
  /// trigger extra kicks. `null` when none were detected or the detection
  /// failed ([drumError]).
  final DrumHits? drums;

  /// Why the drum detection on the drum stem failed (e.g. no drum model; it
  /// never fails the deep analysis), else `null`.
  final String? drumError;

  DspDeepAnalysis({
    required List<DspSection> sections,
    required this.stemLevels,
    required this.stemRows,
    required this.chroma,
    required this.chromaRows,
    this.drums,
    this.drumError,
  }) : sections = List.unmodifiable(sections);

  /// The loudness of [stem] at row [row] (10 ms each).
  double stemLevel(int row, DspStem stem) => stemLevels[row * DspStem.values.length + stem.index];

  /// Builds the result from the library's raw arrays (seconds, label
  /// indices; the drum hits as parallel arrays like [DspResultData]'s, none
  /// when omitted); unknown label and drum class indices are dropped.
  factory DspDeepAnalysis.fromRaw({
    required Float32List sectionStarts,
    required Float32List sectionEnds,
    required Uint8List sectionLabels,
    required Float32List stemLevels,
    required int stemRows,
    required Float32List chroma,
    required int chromaRows,
    Float32List? drumTimes,
    Uint8List? drumClasses,
    Float32List? drumStrengths,
    String? drumError,
  }) => DspDeepAnalysis(
    sections: [
      for (var i = 0; i < sectionStarts.length; i++)
        if (sectionLabels[i] < DspSectionLabel.values.length)
          DspSection(
            durationOfSeconds(sectionStarts[i]),
            durationOfSeconds(sectionEnds[i]),
            DspSectionLabel.values[sectionLabels[i]],
          ),
    ],
    stemLevels: stemLevels,
    stemRows: stemRows,
    chroma: chroma,
    chromaRows: chromaRows,
    drums: drumTimes == null || drumClasses == null || drumStrengths == null
        ? null
        : drumHitsFromRaw(drumTimes, drumClasses, drumStrengths),
    drumError: drumError,
  );

  /// A structured-clone-safe form (to cross a Web Worker): lists, numbers
  /// and typed data only.
  Map<String, Object?> toWire() {
    final drums = drumHitsToRaw(this.drums);
    return {
      'starts': Float32List.fromList([for (final s in sections) s.start.inMicroseconds / 1e6]),
      'ends': Float32List.fromList([for (final s in sections) s.end.inMicroseconds / 1e6]),
      'labels': Uint8List.fromList([for (final s in sections) s.label.index]),
      'stemLevels': stemLevels,
      'stemRows': stemRows,
      'chroma': chroma,
      'chromaRows': chromaRows,
      'drumTimes': drums.times,
      'drumClasses': drums.classes,
      'drumStrengths': drums.strengths,
      'drumError': drumError,
    };
  }

  /// The inverse of [toWire] (a wire without drum entries has no drums).
  factory DspDeepAnalysis.fromWire(Map<Object?, Object?> wire) => DspDeepAnalysis.fromRaw(
    sectionStarts: wire['starts']! as Float32List,
    sectionEnds: wire['ends']! as Float32List,
    sectionLabels: wire['labels']! as Uint8List,
    stemLevels: wire['stemLevels']! as Float32List,
    stemRows: (wire['stemRows']! as num).toInt(),
    chroma: wire['chroma']! as Float32List,
    chromaRows: (wire['chromaRows']! as num).toInt(),
    drumTimes: wire['drumTimes'] as Float32List?,
    drumClasses: wire['drumClasses'] as Uint8List?,
    drumStrengths: wire['drumStrengths'] as Float32List?,
    drumError: wire['drumError'] as String?,
  );
}
