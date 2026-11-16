import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:icicles_dsp/icicles_dsp.dart';
import 'package:test/test.dart';

import 'support/signals.dart';

/// The stereo signal both the native and the WebAssembly encoding are pinned
/// with: 2 s of the 120 bpm click track, the right channel at half level.
Float32List stereoClick() {
  final mono = clickTrack(bpm: 120, seconds: 2, sampleRate: 44100);
  final out = Float32List(mono.length * 2);
  for (var i = 0; i < mono.length; i++) {
    out[2 * i] = mono[i];
    out[2 * i + 1] = mono[i] * 0.5;
  }
  return out;
}

/// FNV-1a over [bytes], as hex.
String fnv(Uint8List bytes) {
  var h = 0x811c9dc5;
  for (final b in bytes) {
    final x = h ^ b;
      // 32-bit safe under JavaScript numbers.
      h = ((x & 0xffff) * 0x01000193 + (((x >> 16) * 0x0193 & 0xffff) << 16)) & 0xffffffff;
  }
  return h.toRadixString(16).padLeft(8, '0');
}

void main() {
  test('encodes an Ogg Vorbis file far smaller than the PCM', () async {
    final pcm = stereoClick();
    final ogg = await const IciclesDsp().encodeVorbis(pcm, channels: 2, sampleRate: 44100);
    expect(ascii.decode(ogg.sublist(0, 4)), 'OggS');
    expect(ogg.length, lessThan(pcm.length * 2 ~/ 4), reason: 'at least 4× smaller than 16-bit PCM');
    final reference =
        jsonDecode(File('test/fixtures/vorbis_reference.json').readAsStringSync()) as Map<String, Object?>;
    expect({'length': ogg.length, 'fnv': fnv(ogg)}, reference);
  });

  test('mono is accepted, bad input refused', () async {
    final mono = clickTrack(bpm: 120, seconds: 1, sampleRate: 48000);
    final ogg = IciclesDsp.encodeVorbisSync(mono, channels: 1, sampleRate: 48000, quality: 3);
    expect(ascii.decode(ogg.sublist(0, 4)), 'OggS');
    expect(() => IciclesDsp.encodeVorbisSync(mono, channels: 3, sampleRate: 48000),
        throwsA(isA<DspException>().having((e) => e.kind, 'kind', DspErrorKind.invalidArgument)));
    expect(() => IciclesDsp.encodeVorbisSync(Float32List(0), channels: 1, sampleRate: 48000),
        throwsA(isA<DspException>()));
  });
}
