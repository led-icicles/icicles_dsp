import 'dart:typed_data';

import 'drum_hits_raw.dart';

/// One chord of a song, as the ChordMini BTC model recognized it.
final class DspChord {
  /// The qualities in the model's order ([quality] indexes them), in the
  /// Harte notation of chord annotations.
  static const qualityNames = [
    'min',
    'maj',
    'dim',
    'aug',
    'min6',
    'maj6',
    'min7',
    'minmaj7',
    'maj7',
    '7',
    'dim7',
    'hdim7',
    'sus2',
    'sus4',
  ];

  /// From the start of the signal.
  final Duration start;
  final Duration end;

  /// Pitch class of the root, C = 0 … B = 11.
  final int root;

  /// Index into [qualityNames].
  final int quality;

  const DspChord({required this.start, required this.end, required this.root, required this.quality});

  @override
  bool operator ==(Object other) =>
      other is DspChord && other.start == start && other.end == end && other.root == root && other.quality == quality;

  @override
  int get hashCode => Object.hash(start, end, root, quality);

  @override
  String toString() => 'DspChord($start–$end, root $root, ${qualityNames[quality]})';
}

/// The chords of the library's parallel arrays (start and end seconds, root
/// pitch classes, quality indices). Entries with an unknown root or quality
/// are dropped.
List<DspChord> chordsFromRaw(Float32List starts, Float32List ends, Uint8List roots, Uint8List qualities) => [
  for (var i = 0; i < starts.length; i++)
    if (roots[i] < 12 && qualities[i] < DspChord.qualityNames.length)
      DspChord(
        start: durationOfSeconds(starts[i]),
        end: durationOfSeconds(ends[i]),
        root: roots[i],
        quality: qualities[i],
      ),
];

/// The parallel arrays of [chords] (the inverse of [chordsFromRaw]).
({Float32List starts, Float32List ends, Uint8List roots, Uint8List qualities}) chordsToRaw(List<DspChord> chords) => (
  starts: Float32List.fromList([for (final c in chords) c.start.inMicroseconds / Duration.microsecondsPerSecond]),
  ends: Float32List.fromList([for (final c in chords) c.end.inMicroseconds / Duration.microsecondsPerSecond]),
  roots: Uint8List.fromList([for (final c in chords) c.root]),
  qualities: Uint8List.fromList([for (final c in chords) c.quality]),
);
