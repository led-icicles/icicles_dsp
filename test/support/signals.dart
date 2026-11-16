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

/// [mono] as a 16-bit PCM WAV file (platform-neutral, for the transcoding
/// tests on both builds).
Uint8List wav16(Float32List mono, int sampleRate) {
  final data = mono.length * 2;
  final out = ByteData(44 + data);
  void tag(int at, String s) {
    for (var i = 0; i < 4; i++) {
      out.setUint8(at + i, s.codeUnitAt(i));
    }
  }

  tag(0, 'RIFF');
  out.setUint32(4, 36 + data, Endian.little);
  tag(8, 'WAVE');
  tag(12, 'fmt ');
  out
    ..setUint32(16, 16, Endian.little)
    ..setUint16(20, 1, Endian.little)
    ..setUint16(22, 1, Endian.little)
    ..setUint32(24, sampleRate, Endian.little)
    ..setUint32(28, sampleRate * 2, Endian.little)
    ..setUint16(32, 2, Endian.little)
    ..setUint16(34, 16, Endian.little);
  tag(36, 'data');
  out.setUint32(40, data, Endian.little);
  for (var i = 0; i < mono.length; i++) {
    out.setInt16(44 + 2 * i, (mono[i].clamp(-1.0, 1.0) * 32767).round(), Endian.little);
  }
  return out.buffer.asUint8List();
}
