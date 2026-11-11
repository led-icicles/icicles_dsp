import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:icicles_dsp/icicles_dsp.dart';
import 'package:icicles_generators/icicles_generators.dart';
import 'package:test/test.dart';

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

void main() {
  late Directory temp;
  late String modelDir;

  setUpAll(() async {
    temp = Directory.systemTemp.createTempSync('icicles_dsp_test');
    modelDir = (await IciclesDsp.bundledModelDir())!;
  });

  tearDownAll(() => temp.deleteSync(recursive: true));

  test('the native library loads', () {
    expect(IciclesDsp.isAvailable, isTrue);
    expect(IciclesDsp.version, '0.1.0');
  });

  test('bundled models resolve to the package models directory', () {
    expect(File('$modelDir/${IciclesDsp.melModelFile}').existsSync(), isTrue);
    expect(File('$modelDir/${IciclesDsp.smallModelFile}').existsSync(), isTrue);
  });

  test('analyzes a generated 120 BPM click-track WAV', () async {
    final path = '${temp.path}/click.wav';
    WavAudio.write(path, clickTrack(bpm: 120, seconds: 20, sampleRate: 44100), 44100);
    final wav = WavAudio.read(path);
    expect(wav.sampleRate, 44100);
    expect(wav.mono.length, 44100 * 20);

    final stopwatch = Stopwatch()..start();
    final analysis = await const IciclesDsp().analyze(wav.mono, wav.sampleRate, modelDir: modelDir);
    stopwatch.stop();
    printOnFailure('20 s analysed in ${stopwatch.elapsedMilliseconds} ms');

    expect(analysis.beats.length, inInclusiveRange(30, 42));
    const period = Duration(milliseconds: 500);
    for (final beat in analysis.beats.where((b) => b < const Duration(seconds: 19, milliseconds: 500))) {
      final offset = beat.inMicroseconds % period.inMicroseconds;
      final distance = math.min(offset, period.inMicroseconds - offset);
      expect(distance, lessThanOrEqualTo(30000), reason: 'beat at $beat is off the grid');
    }
    expect(analysis.downbeats, everyElement(isIn(analysis.beats)));
    expect(analysis.bpm, closeTo(120, 2));

    final features = analysis.features;
    expect(features.hopDuration, const Duration(milliseconds: 10));
    expect(features.frameCount, 2000);
    expect(features.value(0, AudioFeatureKey.level), greaterThan(0.5));
    expect(features.value(25, AudioFeatureKey.level), lessThan(0.1));
  });

  test('features are identical to the Dart extractor', () async {
    final random = math.Random(3);
    final signal = Float32List.fromList([
      for (var i = 0; i < 48000 * 2; i++)
        0.5 * math.sin(2 * math.pi * 220 * i / 48000) * (0.5 + 0.5 * math.sin(2 * math.pi * 0.7 * i / 48000)) +
            0.05 * (random.nextDouble() * 2 - 1),
    ]);
    final analysis = await const IciclesDsp().analyze(signal, 48000, trackBeats: false);
    final reference = AudioFeatureExtractor.extract(signal, 48000);
    expect(analysis.beats, isEmpty);
    expect(analysis.features.frameCount, reference.frameCount);
    expect(analysis.features.toBytes(), reference.toBytes());
    expect(analysis.features.contentHash, reference.contentHash);
  });

  test('silence has no beats', () async {
    final analysis = await const IciclesDsp().analyze(Float32List(44100 * 10), 44100, modelDir: modelDir);
    expect(analysis.beats, isEmpty);
    expect(analysis.downbeats, isEmpty);
    expect(analysis.bpm, isNull);
  });

  test('rejects bad input', () async {
    final dsp = const IciclesDsp();
    await expectLater(
      dsp.analyze(Float32List(0), 44100),
      throwsA(isA<DspException>().having((e) => e.kind, 'kind', DspErrorKind.invalidArgument)),
    );
    await expectLater(
      dsp.analyze(Float32List(100), 500),
      throwsA(isA<DspException>().having((e) => e.kind, 'kind', DspErrorKind.invalidArgument)),
    );
    await expectLater(
      dsp.analyze(Float32List(4410), 44100, modelDir: '${temp.path}/missing'),
      throwsA(
        isA<DspException>()
            .having((e) => e.kind, 'kind', DspErrorKind.modelNotFound)
            .having((e) => e.message, 'message', contains(IciclesDsp.melModelFile)),
      ),
    );
  });

  test('a broken model file is a load failure', () async {
    final dir = Directory('${temp.path}/broken')..createSync();
    File('${dir.path}/${IciclesDsp.melModelFile}').writeAsBytesSync([1, 2, 3]);
    File('${dir.path}/${IciclesDsp.smallModelFile}').writeAsBytesSync([4, 5, 6]);
    await expectLater(
      const IciclesDsp().analyze(Float32List(4410), 44100, modelDir: dir.path),
      throwsA(isA<DspException>().having((e) => e.kind, 'kind', DspErrorKind.modelLoadFailed)),
    );
  });

  test('WAV codec round-trips 16-bit PCM', () {
    final mono = Float32List.fromList([0, 0.5, -0.5, 1, -1, 0.25]);
    final decoded = WavAudio.decode(WavAudio.encode16(mono, 22050));
    expect(decoded.sampleRate, 22050);
    for (var i = 0; i < mono.length; i++) {
      expect(decoded.mono[i], closeTo(mono[i], 1 / 32767));
    }
  });
}
