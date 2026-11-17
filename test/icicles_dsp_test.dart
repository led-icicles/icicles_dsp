import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:icicles_dsp/icicles_dsp.dart';
import 'package:icicles_generators/icicles_generators.dart';
import 'package:test/test.dart';

import 'support/signals.dart';

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
    expect(IciclesDsp.version, '0.8.0');
  });

  test('bundled models resolve to the package models directory', () {
    expect(File('$modelDir/${IciclesDsp.melModelFile}').existsSync(), isTrue);
    expect(File('$modelDir/${IciclesDsp.smallModelFile}').existsSync(), isTrue);
    expect(File('$modelDir/${IciclesDsp.drumModelFile}').existsSync(), isTrue);
    expect(File('$modelDir/${IciclesDsp.chordModelFile}').existsSync(), isTrue);
  });

  test('recognizes the chords of the synthetic progression', () async {
    // The fixture of tool/export_chords/export_chords.py: twelve chords of
    // two seconds each.
    final fixture =
        jsonDecode(File('rust/tests/fixtures/chords/progression.json').readAsStringSync()) as Map<String, Object?>;
    final progression = (fixture['progression']! as List<Object?>).cast<List<Object?>>();
    final bytes = File('rust/tests/fixtures/chords/progression_44100.pcm').readAsBytesSync();
    final pcm = Int16List.view(Uint8List.fromList(bytes).buffer);
    final mono = Float32List.fromList([for (final v in pcm) v / 32768]);
    final analysis = await const IciclesDsp().analyze(
      mono,
      44100,
      modelDir: modelDir,
      trackBeats: false,
      detectDrums: false,
    );
    expect(analysis.chordError, isNull);
    expect(analysis.drums.isEmpty, isTrue);
    expect(analysis.chords, hasLength(progression.length));
    for (final (i, chord) in analysis.chords.indexed) {
      expect(chord.root, progression[i][0], reason: '$chord');
      expect((chord.start.inMilliseconds - 2000 * i).abs(), lessThanOrEqualTo(150), reason: '$chord');
      expect(chord.end, greaterThan(chord.start));
      if (i > 0) expect(chord.start, greaterThanOrEqualTo(analysis.chords[i - 1].end));
    }
    expect(DspChord.qualityNames[analysis.chords.first.quality], 'maj');
    expect(DspChord.qualityNames[analysis.chords[1].quality], 'min');
  });

  test('detects the drums of the synthetic loop', () async {
    // The fixture of tool/export_adtof/export_adtof.py: 120 BPM from 0.1 s,
    // kick on every beat, snare on 2 and 4, hi-hats on eighths.
    final bytes = File('rust/tests/fixtures/adtof/drum_loop_44100.pcm').readAsBytesSync();
    final pcm = Int16List.view(Uint8List.fromList(bytes).buffer);
    final mono = Float32List.fromList([for (final v in pcm) v / 32768]);
    final analysis = await const IciclesDsp().analyze(mono, 44100, modelDir: modelDir, trackBeats: false);
    expect(analysis.drumError, isNull);
    expect(analysis.drums.source, DrumHits.sourceAdtof);
    List<int> ms(DrumClass c) => [for (final h in analysis.drums.of(c)) h.time.inMilliseconds];
    final kicks = ms(DrumClass.kick);
    expect(kicks, hasLength(4));
    for (final (i, t) in kicks.indexed) {
      expect((t - (100 + 500 * i)).abs(), lessThanOrEqualTo(30), reason: '$kicks');
    }
    expect(ms(DrumClass.snare), hasLength(2));
    expect(ms(DrumClass.hihat).length, greaterThanOrEqualTo(6));
    for (final cls in DrumClass.values) {
      for (final hit in analysis.drums.of(cls)) {
        expect(hit.strength, inInclusiveRange(0.05, 1.0));
      }
    }
  });

  test('a missing drum model leaves the beats and features intact', () async {
    final dir = Directory('${temp.path}/beats-only')..createSync();
    for (final name in [IciclesDsp.melModelFile, IciclesDsp.smallModelFile]) {
      File('$modelDir/$name').copySync('${dir.path}/$name');
    }
    final analysis = await const IciclesDsp().analyze(
      clickTrack(bpm: 120, seconds: 8, sampleRate: 44100),
      44100,
      modelDir: dir.path,
    );
    expect(analysis.beats, isNotEmpty);
    expect(analysis.features.frameCount, 800);
    expect(analysis.drums.isEmpty, isTrue);
    expect(analysis.drumError, contains(IciclesDsp.drumModelFile));
    expect(analysis.chords, isEmpty);
    expect(analysis.chordError, contains(IciclesDsp.chordModelFile));
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
    final analysis = await const IciclesDsp().analyze(
      signal,
      48000,
      trackBeats: false,
      detectDrums: false,
      detectChords: false,
    );
    final reference = AudioFeatureExtractor.extract(signal, 48000);
    expect(analysis.beats, isEmpty);
    expect(analysis.drums.isEmpty, isTrue);
    expect(analysis.chords, isEmpty);
    expect(analysis.chordError, isNull);
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
