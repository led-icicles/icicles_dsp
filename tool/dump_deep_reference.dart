// Writes test/fixtures/deep_reference.json: the native (FFI) deep analysis
// (2 folds) of the 12 s click track at 120 BPM, which
// test/web/icicles_dsp_wasm_test.dart expects the WebAssembly build to
// reproduce. Needs the deep models (tool/download-deep-models.sh). Run
// `dart run tool/dump_deep_reference.dart` after changing the analysis.
import 'dart:convert';
import 'dart:io';

import 'package:icicles_dsp/icicles_dsp.dart';

import '../test/support/signals.dart';

Future<void> main() async {
  final modelDir = (await IciclesDsp.bundledModelDir())!;
  final result = await const IciclesDsp().analyzeDeep(
    clickTrack(bpm: 120, seconds: 12, sampleRate: 44100),
    44100,
    modelDir: modelDir,
    folds: 2,
  );
  final json = {
    'sections': [
      for (final s in result.sections) [s.start.inMilliseconds, s.end.inMilliseconds, s.label.name],
    ],
    'stemRows': result.stemRows,
    'stemSums': [
      for (final stem in DspStem.values)
        [for (var r = 0; r < result.stemRows; r++) result.stemLevel(r, stem)].fold<double>(0, (a, b) => a + b),
    ],
    'chromaRows': result.chromaRows,
    'chromaSum': result.chroma.fold<double>(0, (a, b) => a + b),
  };
  File('test/fixtures/deep_reference.json').writeAsStringSync(const JsonEncoder.withIndent('  ').convert(json));
  stdout.writeln('Wrote test/fixtures/deep_reference.json (${result.sections.length} sections).');
}
