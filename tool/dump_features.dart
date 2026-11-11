// Dumps the `AudioFeatureExtractor` output of `icicles_generators` for the
// synthetic test signals into `rust/tests/fixtures/*.json`, the reference the
// Rust port is compared against (`cargo test`, tolerance 1e-3).
//
// Run from the package root: `fvm dart run tool/dump_features.dart`.
// The signals are defined identically in `rust/tests/common/mod.rs`; change
// both together and regenerate.
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:icicles_generators/icicles_generators.dart';

final class SyntheticSignal {
  final String name;
  final int sampleRate;
  final int length;
  final double Function(double t, int i, Lcg noise) sample;

  const SyntheticSignal(this.name, this.sampleRate, this.length, this.sample);

  Float32List generate() {
    final noise = Lcg();
    final out = Float32List(length);
    for (var i = 0; i < length; i++) {
      out[i] = sample(i / sampleRate, i, noise);
    }
    return out;
  }
}

/// A 32-bit linear congruential generator; `next()` is uniform in `-1..1`.
final class Lcg {
  int _state = 0x12345678;

  double next() {
    _state = (_state * 1664525 + 1013904223) & 0xFFFFFFFF;
    return (_state >> 8) / 0x1000000 * 2 - 1;
  }
}

double _sin(double x) => math.sin(2 * math.pi * x);

double _click(double t, double period, double frequency) {
  final beat = (t / period).floor() * period;
  final dt = t - beat;
  if (dt >= 0.01) return 0;
  return 0.8 * math.exp(-dt * 300) * _sin(frequency * dt);
}

final signals = <SyntheticSignal>[
  SyntheticSignal('tones_44100', 44100, 88641, (t, i, noise) {
    final env = 0.5 + 0.5 * _sin(0.5 * t);
    final tones = 0.4 * _sin(110 * t) + 0.25 * _sin(440 * t) + 0.1 * _sin(3000 * t);
    return env * tones + _click(t, 0.5, 1000) + 0.01 * noise.next();
  }),
  SyntheticSignal('chirp_48000', 48000, 72000, (t, i, noise) {
    const f0 = 100.0, f1 = 8000.0, duration = 1.5;
    final phase = f0 * t + (f1 - f0) * t * t / (2 * duration);
    return 0.6 * _sin(phase) + 0.005 * noise.next();
  }),
  SyntheticSignal('bursts_22050', 22050, 26460, (t, i, noise) {
    final amplitude = (t * 4).floor().isEven ? 0.7 : 0.02;
    return amplitude * noise.next();
  }),
  SyntheticSignal('quiet_44100', 44100, 22050, (t, i, noise) => 0.0005 * noise.next()),
  SyntheticSignal('silence_44100', 44100, 13230, (t, i, noise) => 0),
];

void main() {
  final directory = Directory('rust/tests/fixtures');
  directory.createSync(recursive: true);
  for (final signal in signals) {
    final samples = signal.generate();
    final features = AudioFeatureExtractor.extract(samples, signal.sampleRate);
    final data = <double>[];
    for (var row = 0; row < features.frameCount; row++) {
      for (final key in AudioFeatureKey.values.take(AudioFeatureKey.columnCount)) {
        data.add(features.value(row, key));
      }
    }
    final json = {
      'name': signal.name,
      'sampleRate': signal.sampleRate,
      'length': signal.length,
      'rows': features.frameCount,
      'columns': AudioFeatureKey.columnCount,
      'data': data,
    };
    final file = File('${directory.path}/${signal.name}.json');
    file.writeAsStringSync(jsonEncode(json));
    stdout.writeln('${file.path}: ${features.frameCount} rows');
  }
}
