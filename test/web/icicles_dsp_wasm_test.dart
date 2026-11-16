@TestOn('browser')
library;

import 'dart:convert';
import 'dart:js_interop';
import 'dart:typed_data';

import 'package:icicles_dsp/icicles_dsp_web.dart';
import 'package:test/test.dart';

import '../support/signals.dart';

@JS('fetch')
external JSPromise<_Response> _fetch(JSString url);

extension type _Response._(JSObject _) implements JSObject {
  external bool get ok;

  external JSPromise<JSArrayBuffer> arrayBuffer();
}

/// A package file served by the test runner (the package root, two levels
/// above this test's page).
Future<Uint8List> packageFile(String path) async {
  final response = await _fetch('../../$path'.toJS).toDart;
  if (!response.ok) throw StateError('Cannot fetch $path (run tool/build_wasm.sh for the module).');
  return (await response.arrayBuffer().toDart).toDart.asUint8List();
}

void main() {
  late IciclesDspWasm dsp;
  late Uint8List mel, beat, drum;

  setUpAll(() async {
    dsp = await IciclesDspWasm.compile(await packageFile('build/web/icicles_dsp.wasm'));
    mel = await packageFile('models/mel_spectrogram.onnx');
    beat = await packageFile('models/beat_this_small.onnx');
    drum = await packageFile('models/adtof_frame_rnn.onnx');
  });

  test('reproduces the native analysis of the click track', () async {
    final reference =
        jsonDecode(utf8.decode(await packageFile('test/fixtures/click_120_reference.json'))) as Map<String, Object?>;
    final analysis = dsp.analyze(
      clickTrack(bpm: 120, seconds: 12, sampleRate: 44100),
      44100,
      mel: mel,
      beat: beat,
      drum: drum,
    );
    expect([for (final b in analysis.beats) b.inMicroseconds], reference['beats']);
    expect([for (final b in analysis.downbeats) b.inMicroseconds], reference['downbeats']);
    final drums = reference['drums']! as Map<String, Object?>;
    for (final cls in DrumClass.values) {
      expect([for (final h in analysis.drums.of(cls)) h.time.inMicroseconds], drums[cls.name], reason: cls.name);
    }
    expect(analysis.drumError, isNull);
    expect(analysis.features.frameCount, reference['featureRows']);
    final sums = featureSums(analysis.features);
    final expected = (reference['featureSums']! as List<Object?>).cast<num>();
    for (var i = 0; i < sums.length; i++) {
      expect(sums[i], closeTo(expected[i], 1e-3 * (1 + expected[i].abs())), reason: 'column $i');
    }
  });

  test('every call runs on a fresh instance; a missing drum model fails only the drums', () {
    final signal = clickTrack(bpm: 100, seconds: 6, sampleRate: 48000);
    final first = dsp.analyze(signal, 48000, mel: mel, beat: beat, drum: drum);
    final again = dsp.analyze(signal, 48000, mel: mel, beat: beat, drum: drum);
    expect(again.beats, first.beats);
    final noDrums = dsp.analyze(signal, 48000, mel: mel, beat: beat);
    expect(noDrums.beats, first.beats);
    expect(noDrums.drums.isEmpty, isTrue);
    expect(noDrums.drumError, isNotNull);
  });

  test('failures are DspExceptions, never a broken module', () {
    expect(
      () => dsp.analyze(Float32List(0), 44100, mel: mel, beat: beat),
      throwsA(isA<DspException>().having((e) => e.kind, 'kind', DspErrorKind.invalidArgument)),
    );
    expect(
      () => dsp.analyze(Float32List(4410), 44100, mel: Uint8List.fromList([1, 2, 3]), beat: beat),
      throwsA(isA<DspException>().having((e) => e.kind, 'kind', DspErrorKind.modelLoadFailed)),
    );
    expect(dsp.analyze(Float32List(44100), 44100, mel: mel, beat: beat, drum: drum).features.frameCount, 100);
  });
}
