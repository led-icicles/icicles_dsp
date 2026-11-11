// Analyzes a WAV file with icicles_dsp and prints the tempo.
//
//   dart run bin/analyze.dart <file.wav> [model-dir]
//
// Without a model directory the package's bundled small model is used.
import 'dart:io';

import 'package:icicles_dsp/icicles_dsp.dart';

Future<void> main(List<String> args) async {
  if (args.isEmpty || args.length > 2) {
    stderr.writeln('Usage: dart run bin/analyze.dart <file.wav> [model-dir]');
    exitCode = 64;
    return;
  }
  if (!IciclesDsp.isAvailable) {
    stderr.writeln('The icicles_dsp native library is not available on this platform.');
    exitCode = 1;
    return;
  }
  final WavAudio wav;
  try {
    wav = WavAudio.read(args[0]);
  } on FormatException catch (e) {
    stderr.writeln('Cannot read ${args[0]}: ${e.message}');
    exitCode = 65;
    return;
  }
  stdout.writeln('${args[0]}: ${_clock(wav.duration)} at ${wav.sampleRate} Hz, icicles_dsp ${IciclesDsp.version}');

  final stopwatch = Stopwatch()..start();
  final DspAnalysis analysis;
  try {
    analysis = await const IciclesDsp().analyze(wav.mono, wav.sampleRate, modelDir: args.length > 1 ? args[1] : null);
  } on DspException catch (e) {
    stderr.writeln('Analysis failed (${e.kind.name}): ${e.message}');
    exitCode = 1;
    return;
  }
  stopwatch.stop();

  final bpm = analysis.bpm;
  stdout
    ..writeln('beats: ${analysis.beats.length}, downbeats: ${analysis.downbeats.length}')
    ..writeln('bpm: ${bpm == null ? 'n/a' : bpm.toStringAsFixed(1)}')
    ..writeln(
      'features: ${analysis.features.frameCount} rows × ${AudioFeatureKey.columnCount} '
      'every ${analysis.features.hopDuration.inMilliseconds} ms',
    )
    ..writeln('analysed in ${stopwatch.elapsedMilliseconds} ms');
  if (analysis.beats.isNotEmpty) {
    final first = analysis.beats.take(8).map(_clock).join(', ');
    stdout.writeln('first beats: $first${analysis.beats.length > 8 ? ', …' : ''}');
  }
}

String _clock(Duration d) {
  final minutes = d.inMinutes;
  final seconds = (d.inMilliseconds % 60000) / 1000;
  return '$minutes:${seconds.toStringAsFixed(3).padLeft(6, '0')}';
}
