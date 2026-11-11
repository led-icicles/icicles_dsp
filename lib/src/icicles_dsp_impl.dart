import 'dart:convert';
import 'dart:ffi' as ffi;
import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:icicles_generators/icicles_generators.dart';

import 'dsp_analysis.dart';
import 'dsp_exception.dart';
import 'icicles_dsp_bindings_generated.dart' as bindings;

/// Native audio analysis: Beat This! beat tracking plus the
/// `icicles_generators` feature matrix, computed by the Rust crate behind
/// `dart:ffi` on a background isolate.
///
/// Native only (macOS, Windows): on the web the package must not be used.
final class IciclesDsp {
  const IciclesDsp();

  /// File name of the mel spectrogram model (required in every model directory).
  static const melModelFile = 'mel_spectrogram.onnx';

  /// File name of the full-accuracy beat model (optional, about 83 MB; see
  /// `tool/download-models.sh`). Preferred over [smallModelFile] when present.
  static const fullModelFile = 'beat_this.onnx';

  /// File name of the bundled small beat model (about 10 MB).
  static const smallModelFile = 'beat_this_small.onnx';

  /// Whether the native library can be loaded on this platform.
  static bool get isAvailable => version != null;

  /// The native library version, or `null` when it cannot be loaded.
  static String? get version {
    try {
      return bindings.icicles_dsp_version().cast<Utf8>().toDartString();
    } on Error {
      return null;
    }
  }

  /// The package's own `models/` directory (the bundled small model), or
  /// `null` when the package root cannot be resolved (compiled Flutter apps:
  /// pass an explicit `modelDir` to [analyze] there).
  static Future<String?> bundledModelDir() async {
    final root = await _packageRoot();
    if (root == null) return null;
    final directory = Directory.fromUri(root.resolve('models/'));
    return directory.existsSync() ? directory.path : null;
  }

  /// The package root: from the isolate's package resolution (`dart run`,
  /// `dart test`), else from the nearest `.dart_tool/package_config.json`
  /// above the working directory (`flutter test`, where package URIs cannot
  /// be resolved); `null` in a compiled app.
  static Future<Uri?> _packageRoot() async {
    try {
      final uri = await Isolate.resolvePackageUri(Uri.parse('package:icicles_dsp/icicles_dsp.dart'));
      if (uri != null && uri.scheme == 'file') return uri.resolve('../');
    } on UnsupportedError {
      // Fall through to the package config lookup.
    }
    for (var directory = Directory.current; ; directory = directory.parent) {
      final config = File(
        '${directory.path}${Platform.pathSeparator}.dart_tool${Platform.pathSeparator}package_config.json',
      );
      if (config.existsSync()) {
        final root = _rootFromPackageConfig(config);
        if (root != null) return root;
      }
      if (directory.parent.path == directory.path) return null;
    }
  }

  static Uri? _rootFromPackageConfig(File config) {
    try {
      final json = jsonDecode(config.readAsStringSync());
      if (json is! Map<String, Object?>) return null;
      final packages = json['packages'];
      if (packages is! List<Object?>) return null;
      for (final entry in packages) {
        if (entry is Map<String, Object?> && entry['name'] == 'icicles_dsp') {
          final rootUri = entry['rootUri'];
          if (rootUri is! String) return null;
          final root = config.uri.resolve(rootUri);
          return root.path.endsWith('/') ? root : root.replace(path: '${root.path}/');
        }
      }
    } on FormatException {
      return null;
    } on IOException {
      return null;
    }
    return null;
  }

  /// Analyses [mono] PCM (`-1..1`) at [sampleRate] Hz on a background isolate.
  ///
  /// With [trackBeats] the Beat This! models are read from [modelDir]
  /// ([melModelFile] plus [fullModelFile] or [smallModelFile]); a `null`
  /// [modelDir] falls back to [bundledModelDir] and fails with
  /// [DspErrorKind.modelNotFound] when that is unavailable. Without
  /// [trackBeats] only the features are computed (no models needed).
  ///
  /// Throws [DspException].
  Future<DspAnalysis> analyze(Float32List mono, int sampleRate, {String? modelDir, bool trackBeats = true}) async {
    if (mono.isEmpty) throw const DspException(DspErrorKind.invalidArgument, 'The signal is empty.');
    if (sampleRate < 1000) {
      throw DspException(DspErrorKind.invalidArgument, 'Sample rate $sampleRate Hz is below the 1000 Hz minimum.');
    }
    String? models;
    if (trackBeats) {
      models = modelDir ?? await bundledModelDir();
      if (models == null) {
        throw const DspException(
          DspErrorKind.modelNotFound,
          'No model directory: pass modelDir (the package models/ directory is not resolvable here).',
        );
      }
    }
    final transfer = TransferableTypedData.fromList([mono]);
    return Isolate.run(() => analyzeSync(transfer.materialize().asFloat32List(), sampleRate, modelDir: models));
  }

  /// The synchronous native call behind [analyze]; blocks the calling isolate
  /// for the whole analysis. A `null` [modelDir] skips beat tracking.
  static DspAnalysis analyzeSync(Float32List mono, int sampleRate, {String? modelDir}) {
    final samples = calloc<ffi.Float>(mono.length);
    final out = calloc<bindings.IciclesDspResult>();
    final path = modelDir?.toNativeUtf8() ?? ffi.nullptr;
    try {
      samples.asTypedList(mono.length).setAll(0, mono);
      final int status;
      try {
        status = bindings.icicles_dsp_analyze(samples, mono.length, sampleRate, path.cast(), out);
      } on Error catch (e) {
        throw DspException(DspErrorKind.libraryUnavailable, 'The icicles_dsp native library is not available: $e');
      }
      if (status != bindings.IciclesDspStatus.ICICLES_DSP_OK.value) {
        final message = bindings.icicles_dsp_last_error().cast<Utf8>().toDartString();
        throw DspException(_kindOf(status), message.isEmpty ? 'Native status $status.' : message);
      }
      final result = out.ref;
      if (result.feature_columns != AudioFeatureKey.columnCount) {
        throw DspException(
          DspErrorKind.unknown,
          'Native feature layout has ${result.feature_columns} columns, expected ${AudioFeatureKey.columnCount}.',
        );
      }
      final matrix = Float32List.fromList(result.features.asTypedList(result.feature_rows * result.feature_columns));
      return DspAnalysis(
        beats: _durations(result.beats, result.beat_count),
        downbeats: _durations(result.downbeats, result.downbeat_count),
        features: AudioFeatures(matrix, hopDuration: Duration(microseconds: result.feature_hop_micros)),
      );
    } finally {
      try {
        bindings.icicles_dsp_free(out);
      } on Error {
        // The library is unavailable: nothing was allocated natively.
      }
      calloc.free(out);
      calloc.free(samples);
      if (path != ffi.nullptr) calloc.free(path);
    }
  }

  static List<Duration> _durations(ffi.Pointer<ffi.Float> seconds, int count) {
    if (count == 0) return const [];
    final values = seconds.asTypedList(count);
    return [for (final s in values) Duration(microseconds: (s * Duration.microsecondsPerSecond).round())];
  }

  static DspErrorKind _kindOf(int status) => switch (status) {
    1 => DspErrorKind.invalidArgument,
    2 => DspErrorKind.modelNotFound,
    3 => DspErrorKind.modelLoadFailed,
    4 => DspErrorKind.analysisFailed,
    5 => DspErrorKind.nativePanic,
    _ => DspErrorKind.unknown,
  };
}
