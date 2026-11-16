import 'dart:typed_data';

import 'package:icicles_generators/icicles_generators.dart';

/// The drum hits of the library's parallel arrays (onset seconds, class
/// indices in [DrumClass] order, strengths `0..1`), or `null` without any.
/// Unknown class indices are dropped. Shared by the first pass
/// (`DspResultData`) and the deep pass (`DspDeepAnalysis`).
DrumHits? drumHitsFromRaw(Float32List times, Uint8List classes, Float32List strengths) {
  final count = times.length;
  if (count == 0) return null;
  final hits = <DrumClass, List<DrumHit>>{};
  for (var i = 0; i < count; i++) {
    if (classes[i] >= DrumClass.values.length) continue;
    (hits[DrumClass.values[classes[i]]] ??= []).add(DrumHit(durationOfSeconds(times[i]), strengths[i]));
  }
  return DrumHits(hits, source: DrumHits.sourceAdtof);
}

/// The parallel arrays of [hits] (the inverse of [drumHitsFromRaw]), sorted
/// by class, then time.
({Float32List times, Uint8List classes, Float32List strengths}) drumHitsToRaw(DrumHits? hits) {
  final times = <double>[], strengths = <double>[];
  final classes = <int>[];
  for (final cls in DrumClass.values) {
    for (final hit in hits?.of(cls) ?? const <DrumHit>[]) {
      times.add(hit.time.inMicroseconds / Duration.microsecondsPerSecond);
      classes.add(cls.index);
      strengths.add(hit.strength);
    }
  }
  return (
    times: Float32List.fromList(times),
    classes: Uint8List.fromList(classes),
    strengths: Float32List.fromList(strengths),
  );
}

/// [seconds] as a [Duration] (rounded to microseconds).
Duration durationOfSeconds(double seconds) =>
    Duration(microseconds: (seconds * Duration.microsecondsPerSecond).round());
