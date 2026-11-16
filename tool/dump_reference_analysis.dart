// Writes test/fixtures/click_120_reference.json: the native (FFI) analysis
// of the 12 s click track at 120 BPM with the package models, which
// test/web/icicles_dsp_wasm_test.dart expects the WebAssembly build to
// reproduce. Run `dart run tool/dump_reference_analysis.dart` after changing
// the signal, the models or the analysis.
import 'dart:convert';
import 'dart:io';

import 'package:icicles_dsp/icicles_dsp.dart';

import '../test/support/signals.dart';

Future<void> main() async {
  final modelDir = (await IciclesDsp.bundledModelDir())!;
  final analysis = await const IciclesDsp().analyze(
    clickTrack(bpm: 120, seconds: 12, sampleRate: 44100),
    44100,
    modelDir: modelDir,
  );
  final json = {
    'beats': [for (final b in analysis.beats) b.inMicroseconds],
    'downbeats': [for (final b in analysis.downbeats) b.inMicroseconds],
    'drums': {
      for (final cls in DrumClass.values) cls.name: [for (final h in analysis.drums.of(cls)) h.time.inMicroseconds],
    },
    'featureRows': analysis.features.frameCount,
    'featureSums': featureSums(analysis.features),
  };
  File('test/fixtures/click_120_reference.json').writeAsStringSync(const JsonEncoder.withIndent('  ').convert(json));
  stdout.writeln('Wrote test/fixtures/click_120_reference.json (${analysis.beats.length} beats).');
}
