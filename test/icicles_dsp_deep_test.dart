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
