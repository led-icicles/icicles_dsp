import 'dart:async';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:icicles_dsp/icicles_dsp.dart';
import 'package:test/test.dart';

/// 24 s at 44.1 kHz: 12 s of a bass + chord loop, then 12 s with a lead
/// melody and a kick on top, so stems and structure have something to find.
Float32List song() {
  const rate = 44100;
  final out = Float32List(24 * rate);
  for (var i = 0; i < out.length; i++) {
    final t = i / rate;
    final beat = t % 0.5;
    var x =
        0.25 * math.sin(2 * math.pi * 55 * t) +
        0.1 * math.sin(2 * math.pi * 220 * t) +
        0.1 * math.sin(2 * math.pi * 277.18 * t) +
        0.1 * math.sin(2 * math.pi * 329.63 * t);
    if (t >= 12) {
      x += 0.2 * math.sin(2 * math.pi * (440 + 110 * (t ~/ 1 % 4)) * t);
      x += 0.6 * math.exp(-beat * 30) * math.sin(2 * math.pi * 60 * beat);
    }
    out[i] = (x * 0.6).toDouble();
  }
  return out;
}

void main() {
  test('the wire form carries the drum hits and the drum error', () {
    final analysis = DspDeepAnalysis.fromRaw(
      sectionStarts: Float32List.fromList([0]),
      sectionEnds: Float32List.fromList([2]),
      sectionLabels: Uint8List.fromList([DspSectionLabel.intro.index]),
      stemLevels: Float32List(8),
      stemRows: 2,
      chroma: Float32List(12),
      chromaRows: 1,
      drumTimes: Float32List.fromList([0.5, 0.25, 1, 0.75]),
      drumClasses: Uint8List.fromList([0, 3, 0, 9]),
      drumStrengths: Float32List.fromList([1, 0.5, 0.25, 1]),
    );
    void expectDrums(DspDeepAnalysis a) {
      expect(
        [for (final h in a.drums!.of(DrumClass.kick)) h.time],
        [const Duration(milliseconds: 500), const Duration(seconds: 1)],
      );
      expect(a.drums!.of(DrumClass.kick).last.strength, 0.25);
      expect([for (final h in a.drums!.of(DrumClass.hihat)) h.time], [const Duration(milliseconds: 250)]);
      expect(a.drums!.count, 3, reason: 'class 9 is unknown and dropped');
      expect(a.drums!.source, DrumHits.sourceAdtof);
      expect(a.drumError, isNull);
    }

    expectDrums(analysis);
    expectDrums(DspDeepAnalysis.fromWire(analysis.toWire()));

    final failed = DspDeepAnalysis.fromRaw(
      sectionStarts: Float32List(0),
      sectionEnds: Float32List(0),
      sectionLabels: Uint8List(0),
      stemLevels: Float32List(0),
      stemRows: 0,
      chroma: Float32List(0),
      chromaRows: 0,
      drumTimes: Float32List(0),
      drumClasses: Uint8List(0),
      drumStrengths: Float32List(0),
      drumError: 'drum model not found: adtof_frame_rnn.onnx',
    );
    final back = DspDeepAnalysis.fromWire(failed.toWire());
    expect(back.drums, isNull);
    expect(back.drumError, failed.drumError);
    // An older wire without drum entries.
    final wire = analysis.toWire()..removeWhere((key, _) => key.startsWith('drum'));
    expect(DspDeepAnalysis.fromWire(wire).drums, isNull);
  });

  late String modelDir;
  var available = false;

  setUpAll(() async {
    modelDir = (await IciclesDsp.bundledModelDir())!;
    available = IciclesDsp.deepModelFiles.every((f) => File('$modelDir/$f').existsSync());
  });

  test('a missing deep model is reported', () async {
    final dir = Directory.systemTemp.createTempSync('deep-');
    addTearDown(() => dir.deleteSync(recursive: true));
    await expectLater(
      const IciclesDsp().analyzeDeep(song(), 44100, modelDir: dir.path),
      throwsA(isA<DspException>().having((e) => e.kind, 'kind', DspErrorKind.modelNotFound)),
    );
  });

  test('analyses stems, chroma and structure with progress', () async {
    if (!available) {
      markTestSkipped('deep models missing: run tool/download-deep-models.sh');
      return;
    }
    final progress = <double>[];
    final mono = song();
    final result = await const IciclesDsp().analyzeDeep(
      mono,
      44100,
      modelDir: modelDir,
      folds: 2,
      onProgress: progress.add,
    );
    expect(result.stemRows, (mono.length * 100 / 44100).ceil());
    expect(result.stemLevels.length, result.stemRows * 4);
    expect(result.chromaRows, (mono.length / 4410).ceil());
    expect(result.sections, isNotEmpty);
    expect(result.sections.first.start, Duration.zero);
    expect(result.sections.last.end.inMilliseconds, closeTo(24000, 20));
    expect(progress.last, 1.0);
    // The kick only plays in the second half.
    double mean(int from, int to, DspStem stem) {
      var sum = 0.0;
      for (var r = from; r < to; r++) {
        sum += result.stemLevel(r, stem);
      }
      return sum / (to - from);
    }

    expect(mean(1300, 2300, DspStem.drums), greaterThan(mean(100, 1100, DspStem.drums) + 0.1));
    // Chroma rows are distributions.
    var sum = 0.0;
    for (var c = 0; c < 12; c++) {
      sum += result.chroma[50 * 12 + c];
    }
    expect(sum, closeTo(1, 1e-3));
    // Drums of the drum stem. The 33 ms 60 Hz blip of song() is a weak kick
    // (the first pass finds none on the mix); hits must still only come
    // where it plays. One kick per beat is pinned in rust/tests/deep.rs.
    expect(result.drumError, isNull);
    final kicks = result.drums!.of(DrumClass.kick);
    expect(kicks, isNotEmpty);
    expect(kicks.every((h) => h.time >= const Duration(milliseconds: 11900)), isTrue, reason: '$kicks');
  }, timeout: const Timeout(Duration(minutes: 5)));

  test('without the drum model the deep analysis has no drums, only the reason', () async {
    if (!available) {
      markTestSkipped('deep models missing: run tool/download-deep-models.sh');
      return;
    }
    final dir = Directory.systemTemp.createTempSync('deep-no-drums-');
    addTearDown(() => dir.deleteSync(recursive: true));
    for (final file in [IciclesDsp.demucsModelFile, IciclesDsp.foldModelFile(0)]) {
      Link('${dir.path}/$file').createSync('$modelDir/$file');
    }
    final result = await const IciclesDsp().analyzeDeep(song(), 44100, modelDir: dir.path, folds: 1);
    expect(result.sections, isNotEmpty);
    expect(result.drums, isNull);
    expect(result.drumError, contains(IciclesDsp.drumModelFile));
  }, timeout: const Timeout(Duration(minutes: 5)));

  test('cancelling stops the analysis', () async {
    if (!available) {
      markTestSkipped('deep models missing');
      return;
    }
    final cancel = Completer<void>();
    final future = const IciclesDsp().analyzeDeep(song(), 44100, modelDir: modelDir, cancel: cancel.future);
    cancel.complete();
    await expectLater(future, throwsA(isA<DspException>().having((e) => e.kind, 'kind', DspErrorKind.cancelled)));
  }, timeout: const Timeout(Duration(minutes: 2)));
}
