import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

/// A decoded WAV file as mono PCM, the input shape of [IciclesDsp.analyze].
///
/// A small codec for the example CLI and tests: PCM 8/16/24/32-bit integer
/// and 32/64-bit float, any channel count (averaged to mono). It is not a
/// general audio decoder (the app decodes through SoLoud).
final class WavAudio {
  final int sampleRate;
  final Float32List mono;

  const WavAudio(this.sampleRate, this.mono);

  Duration get duration => Duration(microseconds: mono.length * Duration.microsecondsPerSecond ~/ sampleRate);

  /// Decodes a RIFF/WAVE file.
  ///
  /// Throws [FormatException] for anything that is not PCM or float WAV.
  static WavAudio read(String path) => decode(File(path).readAsBytesSync());

  /// Decodes RIFF/WAVE [bytes].
  static WavAudio decode(Uint8List bytes) {
    final view = ByteData.sublistView(bytes);
    if (bytes.length < 12 || _tag(bytes, 0) != 'RIFF' || _tag(bytes, 8) != 'WAVE') {
      throw const FormatException('Not a WAV file.');
    }
    int? format, channels, sampleRate, bitsPerSample;
    var offset = 12;
    while (offset + 8 <= bytes.length) {
      final id = _tag(bytes, offset);
      final size = view.getUint32(offset + 4, Endian.little);
      final body = offset + 8;
      if (id == 'fmt ') {
        if (size < 16) throw const FormatException('Malformed WAV fmt chunk.');
        format = view.getUint16(body, Endian.little);
        channels = view.getUint16(body + 2, Endian.little);
        sampleRate = view.getUint32(body + 4, Endian.little);
        bitsPerSample = view.getUint16(body + 14, Endian.little);
        if (format == 0xFFFE && size >= 26) {
          // WAVE_FORMAT_EXTENSIBLE: the real format is the sub-format's first word.
          format = view.getUint16(body + 24, Endian.little);
        }
      } else if (id == 'data') {
        if (format == null || channels == null || sampleRate == null || bitsPerSample == null) {
          throw const FormatException('WAV data before fmt chunk.');
        }
        final end = math.min(bytes.length, body + size);
        return WavAudio(
          sampleRate,
          _decodeSamples(view, body, end, format: format, channels: channels, bits: bitsPerSample),
        );
      }
      offset = body + size + (size & 1);
    }
    throw const FormatException('WAV file without a data chunk.');
  }

  static Float32List _decodeSamples(
    ByteData view,
    int start,
    int end, {
    required int format,
    required int channels,
    required int bits,
  }) {
    if (channels < 1) throw const FormatException('WAV without channels.');
    final bytesPerSample = bits ~/ 8;
    final frameBytes = bytesPerSample * channels;
    final frames = (end - start) ~/ frameBytes;
    final double Function(int offset) sample = switch ((format, bits)) {
      (1, 8) => (o) => (view.getUint8(o) - 128) / 128,
      (1, 16) => (o) => view.getInt16(o, Endian.little) / 32768,
      (1, 24) => (o) => ((view.getUint8(o) | view.getUint8(o + 1) << 8 | view.getInt8(o + 2) << 16)) / 8388608,
      (1, 32) => (o) => view.getInt32(o, Endian.little) / 2147483648,
      (3, 32) => (o) => view.getFloat32(o, Endian.little),
      (3, 64) => (o) => view.getFloat64(o, Endian.little),
      _ => throw FormatException('Unsupported WAV format $format with $bits bits per sample.'),
    };
    final mono = Float32List(frames);
    for (var i = 0; i < frames; i++) {
      var sum = 0.0;
      final frame = start + i * frameBytes;
      for (var c = 0; c < channels; c++) {
        sum += sample(frame + c * bytesPerSample);
      }
      mono[i] = sum / channels;
    }
    return mono;
  }

  /// Encodes [mono] as a 16-bit PCM mono WAV.
  static Uint8List encode16(Float32List mono, int sampleRate) {
    final dataBytes = mono.length * 2;
    final bytes = Uint8List(44 + dataBytes);
    final view = ByteData.sublistView(bytes);
    void tag(int offset, String value) => bytes.setAll(offset, value.codeUnits);
    tag(0, 'RIFF');
    view.setUint32(4, 36 + dataBytes, Endian.little);
    tag(8, 'WAVE');
    tag(12, 'fmt ');
    view.setUint32(16, 16, Endian.little);
    view.setUint16(20, 1, Endian.little);
    view.setUint16(22, 1, Endian.little);
    view.setUint32(24, sampleRate, Endian.little);
    view.setUint32(28, sampleRate * 2, Endian.little);
    view.setUint16(32, 2, Endian.little);
    view.setUint16(34, 16, Endian.little);
    tag(36, 'data');
    view.setUint32(40, dataBytes, Endian.little);
    for (var i = 0; i < mono.length; i++) {
      view.setInt16(44 + i * 2, (mono[i].clamp(-1.0, 1.0) * 32767).round(), Endian.little);
    }
    return bytes;
  }

  /// Writes [mono] to [path] as a 16-bit PCM mono WAV.
  static void write(String path, Float32List mono, int sampleRate) {
    File(path).writeAsBytesSync(encode16(mono, sampleRate));
  }

  static String _tag(Uint8List bytes, int offset) => String.fromCharCodes(bytes, offset, offset + 4);
}
