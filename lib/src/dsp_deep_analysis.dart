import 'dart:typed_data';

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
/// build): the song structure, per-stem loudness and harmonic chroma.
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

  DspDeepAnalysis({
    required List<DspSection> sections,
    required this.stemLevels,
    required this.stemRows,
    required this.chroma,
    required this.chromaRows,
  }) : sections = List.unmodifiable(sections);

  /// The loudness of [stem] at row [row] (10 ms each).
  double stemLevel(int row, DspStem stem) => stemLevels[row * DspStem.values.length + stem.index];

  /// Builds the result from the library's raw arrays (seconds, label
  /// indices); unknown label indices are dropped.
  factory DspDeepAnalysis.fromRaw({
    required Float32List sectionStarts,
    required Float32List sectionEnds,
    required Uint8List sectionLabels,
    required Float32List stemLevels,
    required int stemRows,
    required Float32List chroma,
    required int chromaRows,
  }) {
    Duration at(double seconds) => Duration(microseconds: (seconds * Duration.microsecondsPerSecond).round());
    return DspDeepAnalysis(
      sections: [
        for (var i = 0; i < sectionStarts.length; i++)
          if (sectionLabels[i] < DspSectionLabel.values.length)
            DspSection(at(sectionStarts[i]), at(sectionEnds[i]), DspSectionLabel.values[sectionLabels[i]]),
      ],
      stemLevels: stemLevels,
      stemRows: stemRows,
      chroma: chroma,
      chromaRows: chromaRows,
    );
  }

  /// A structured-clone-safe form (to cross a Web Worker): lists, numbers
  /// and typed data only.
  Map<String, Object?> toWire() => {
    'starts': Float32List.fromList([for (final s in sections) s.start.inMicroseconds / 1e6]),
    'ends': Float32List.fromList([for (final s in sections) s.end.inMicroseconds / 1e6]),
    'labels': Uint8List.fromList([for (final s in sections) s.label.index]),
    'stemLevels': stemLevels,
    'stemRows': stemRows,
    'chroma': chroma,
    'chromaRows': chromaRows,
  };

  /// The inverse of [toWire].
  factory DspDeepAnalysis.fromWire(Map<Object?, Object?> wire) => DspDeepAnalysis.fromRaw(
    sectionStarts: wire['starts']! as Float32List,
    sectionEnds: wire['ends']! as Float32List,
    sectionLabels: wire['labels']! as Uint8List,
    stemLevels: wire['stemLevels']! as Float32List,
    stemRows: (wire['stemRows']! as num).toInt(),
    chroma: wire['chroma']! as Float32List,
    chromaRows: (wire['chromaRows']! as num).toInt(),
  );
}
