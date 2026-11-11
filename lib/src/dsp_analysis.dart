import 'package:icicles_generators/icicles_generators.dart';

/// The result of [IciclesDsp.analyze]: the Beat This! beats and downbeats
/// and the `icicles_generators` feature matrix of one signal.
final class DspAnalysis {
  /// Beat instants from the start of the signal, sorted.
  final List<Duration> beats;

  /// Downbeat instants (a subset of [beats]), sorted.
  final List<Duration> downbeats;

  /// The song-wide feature matrix (one row every 10 ms).
  final AudioFeatures features;

  DspAnalysis({required List<Duration> beats, required List<Duration> downbeats, required this.features})
    : beats = List.unmodifiable(beats),
      downbeats = List.unmodifiable(downbeats);

  /// The mean beat period (the span from the first to the last beat over the
  /// number of intervals), or `null` with fewer than two beats.
  ///
  /// The mean rather than the median: the model places beats on a 20 ms
  /// frame grid, so single intervals alternate around the true period.
  Duration? get beatPeriod {
    if (beats.length < 2) return null;
    final span = beats.last.inMicroseconds - beats.first.inMicroseconds;
    return Duration(microseconds: (span / (beats.length - 1)).round());
  }

  /// The tempo from [beatPeriod], or `null` with fewer than two beats.
  double? get bpm {
    final period = beatPeriod;
    if (period == null || period <= Duration.zero) return null;
    return 60e6 / period.inMicroseconds;
  }
}
