import 'dart:math' as math;
import 'dart:typed_data';

import 'package:icicles_generators/icicles_generators.dart';

/// `seconds` of 1 kHz clicks (10 ms, exponential decay) at `bpm` over faint
/// noise; the first click at `t = 0`.
Float32List clickTrack({required double bpm, required double seconds, required int sampleRate}) {
  final period = 60 / bpm;
  final random = math.Random(7);
  final out = Float32List((seconds * sampleRate).round());
  for (var i = 0; i < out.length; i++) {
    final t = i / sampleRate;
    final dt = t - (t / period).floor() * period;
    final click = dt < 0.01 ? 0.8 * math.exp(-dt * 300) * math.sin(2 * math.pi * 1000 * dt) : 0.0;
    out[i] = click + 0.002 * (random.nextDouble() * 2 - 1);
  }
  return out;
}

/// The sum of each feature column, a compact fingerprint of a matrix.
List<double> featureSums(AudioFeatures features) => [
  for (final key in AudioFeatureKey.values.where((k) => k.isColumn))
    features.column(key).fold<double>(0, (sum, v) => sum + v),
];
