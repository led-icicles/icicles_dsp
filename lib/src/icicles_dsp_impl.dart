import 'dart:async';
import 'dart:convert';
import 'dart:ffi' as ffi;
import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'dsp_analysis.dart';
import 'dsp_deep_analysis.dart';
import 'dsp_result_data.dart';
import 'dsp_exception.dart';
import 'icicles_dsp_bindings_generated.dart' as bindings;

/// Native audio analysis: Beat This! beat tracking, ADTOF drum hits and the
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

  /// File name of the bundled drum model (ADTOF Frame_RNN, about 2 MB).
  static const drumModelFile = 'adtof_frame_rnn.onnx';

  /// File name of the Demucs network of the deep analysis.
  static const demucsModelFile = 'htdemucs_core.onnx';

  /// Number of All-In-One folds of the deep analysis.
  static const foldCount = 8;

  /// File name of All-In-One fold [k] (`0 ≤ k < foldCount`).
  static String foldModelFile(int k) => 'allin1_fold$k.onnx';

  /// Every model file the deep analysis needs with all folds. The drum model
  /// ([drumModelFile]) is optional for it: read from the same directory when
  /// present (drums of the drum stem), else [DspDeepAnalysis.drums] is `null`
  /// and [DspDeepAnalysis.drumError] says why.
  static List<String> get deepModelFiles => [demucsModelFile, for (var k = 0; k < foldCount; k++) foldModelFile(k)];

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
  /// ([melModelFile] plus [fullModelFile] or [smallModelFile]), with
  /// [detectDrums] the drum model ([drumModelFile]); a `null` [modelDir]
  /// falls back to [bundledModelDir] and fails with
  /// [DspErrorKind.modelNotFound] when that is unavailable. Without either
  /// only the features are computed (no models needed). A failed drum
  /// detection does not fail the analysis: it leaves [DspAnalysis.drums]
  /// empty and explains itself in [DspAnalysis.drumError].
  ///
  /// Throws [DspException].
  Future<DspAnalysis> analyze(
    Float32List mono,
    int sampleRate, {
    String? modelDir,
    bool trackBeats = true,
    bool detectDrums = true,
  }) async {
    if (mono.isEmpty) throw const DspException(DspErrorKind.invalidArgument, 'The signal is empty.');
    if (sampleRate < 1000) {
      throw DspException(DspErrorKind.invalidArgument, 'Sample rate $sampleRate Hz is below the 1000 Hz minimum.');
    }
    String? models;
    if (trackBeats || detectDrums) {
      models = modelDir ?? await bundledModelDir();
      if (models == null) {
        throw const DspException(
          DspErrorKind.modelNotFound,
          'No model directory: pass modelDir (the package models/ directory is not resolvable here).',
        );
      }
    }
    final transfer = TransferableTypedData.fromList([mono]);
    return Isolate.run(
      () => analyzeSync(
        transfer.materialize().asFloat32List(),
        sampleRate,
        modelDir: models,
        trackBeats: trackBeats,
        detectDrums: detectDrums,
      ),
    );
  }

  /// The synchronous native call behind [analyze]; blocks the calling isolate
  /// for the whole analysis. A `null` [modelDir] skips beat tracking and drum
  /// detection.
  static DspAnalysis analyzeSync(
    Float32List mono,
    int sampleRate, {
    String? modelDir,
    bool trackBeats = true,
    bool detectDrums = true,
  }) {
    final flags = dspFlags(trackBeats: trackBeats, detectDrums: detectDrums);
    final samples = calloc<ffi.Float>(mono.length);
    final out = calloc<bindings.IciclesDspResult>();
    final path = modelDir?.toNativeUtf8() ?? ffi.nullptr;
    try {
      samples.asTypedList(mono.length).setAll(0, mono);
      final int status;
      try {
        status = bindings.icicles_dsp_analyze(samples, mono.length, sampleRate, path.cast(), flags, out);
      } on Error catch (e) {
        throw DspException(DspErrorKind.libraryUnavailable, 'The icicles_dsp native library is not available: $e');
      }
      if (status != bindings.IciclesDspStatus.ICICLES_DSP_OK.value) {
        final message = bindings.icicles_dsp_last_error().cast<Utf8>().toDartString();
        throw DspException(dspErrorKindOf(status), message.isEmpty ? 'Native status $status.' : message);
      }
      final result = out.ref;
      Float32List floats(ffi.Pointer<ffi.Float> pointer, int count) =>
          count == 0 ? Float32List(0) : Float32List.fromList(pointer.asTypedList(count));
      return DspResultData(
        beats: floats(result.beats, result.beat_count),
        downbeats: floats(result.downbeats, result.downbeat_count),
        features: floats(result.features, result.feature_rows * result.feature_columns),
        featureRows: result.feature_rows,
        featureColumns: result.feature_columns,
        featureHopMicros: result.feature_hop_micros,
        drumTimes: floats(result.drum_times, result.drum_count),
        drumClasses: result.drum_count == 0
            ? Uint8List(0)
            : Uint8List.fromList(result.drum_classes.asTypedList(result.drum_count)),
        drumStrengths: floats(result.drum_strengths, result.drum_count),
        drumError: result.drum_error == ffi.nullptr ? null : result.drum_error.cast<Utf8>().toDartString(),
      ).toAnalysis();
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

  /// Default Vorbis quality of [encodeVorbis] (`-q 5`, about 160 kbit/s on
  /// music).
  static const defaultVorbisQuality = 5.0;

  /// [interleaved] f32 PCM (`-1..1`, [channels] 1 or 2) at [sampleRate] Hz
  /// as an Ogg Vorbis file (always stereo; mono is duplicated) at Vorbis
  /// quality [quality] (`-1..10`), encoded on a background isolate.
  ///
  /// Throws [DspException].
  Future<Uint8List> encodeVorbis(
    Float32List interleaved, {
    required int channels,
    required int sampleRate,
    double quality = defaultVorbisQuality,
  }) {
    final transfer = TransferableTypedData.fromList([interleaved]);
    return Isolate.run(() => encodeVorbisSync(transfer.materialize().asFloat32List(),
        channels: channels, sampleRate: sampleRate, quality: quality));
  }

  /// The synchronous native call behind [encodeVorbis].
  static Uint8List encodeVorbisSync(
    Float32List interleaved, {
    required int channels,
    required int sampleRate,
    double quality = defaultVorbisQuality,
  }) {
    if (interleaved.isEmpty) throw const DspException(DspErrorKind.invalidArgument, 'The signal is empty.');
    final samples = calloc<ffi.Float>(interleaved.length);
    final out = calloc<ffi.Pointer<ffi.Uint8>>();
    final outLength = calloc<ffi.Size>();
    try {
      samples.asTypedList(interleaved.length).setAll(0, interleaved);
      final int status;
      try {
        status = bindings.icicles_dsp_encode_vorbis(
            samples, interleaved.length, channels, sampleRate, quality, out, outLength);
      } on Error catch (e) {
        throw DspException(DspErrorKind.libraryUnavailable, 'The icicles_dsp native library is not available: $e');
      }
      if (status != bindings.IciclesDspStatus.ICICLES_DSP_OK.value) {
        final message = bindings.icicles_dsp_last_error().cast<Utf8>().toDartString();
        throw DspException(dspErrorKindOf(status), message.isEmpty ? 'Native status $status.' : message);
      }
      final bytes = Uint8List.fromList(out.value.asTypedList(outLength.value));
      bindings.icicles_dsp_free_bytes(out.value, outLength.value);
      return bytes;
    } finally {
      calloc.free(samples);
      calloc.free(out);
      calloc.free(outLength);
    }
  }

  /// The deep analysis of [mono] PCM at [sampleRate] on a background
  /// isolate: Demucs stems → per-stem loudness, harmonic chroma, the drum
  /// hits of the drum stem and the All-In-One song structure, with [folds] of
  /// the [foldCount] folds (fewer is faster and slightly less accurate).
  /// Reads [demucsModelFile] and the fold files from [modelDir] (default
  /// [bundledModelDir]), and [drumModelFile] when it is there (without it
  /// only the drums are missing, see [DspDeepAnalysis.drumError]). About a
  /// minute of CPU for a 3-minute song on Apple silicon.
  ///
  /// [onProgress] receives `0..1` a few times a second; completing [cancel]
  /// stops the analysis with [DspErrorKind.cancelled].
  ///
  /// Throws [DspException].
  Future<DspDeepAnalysis> analyzeDeep(
    Float32List mono,
    int sampleRate, {
    String? modelDir,
    int folds = foldCount,
    void Function(double fraction)? onProgress,
    Future<void>? cancel,
  }) async {
    if (mono.isEmpty) throw const DspException(DspErrorKind.invalidArgument, 'The signal is empty.');
    if (sampleRate < 1000) {
      throw DspException(DspErrorKind.invalidArgument, 'Sample rate $sampleRate Hz is below the 1000 Hz minimum.');
    }
    final models = modelDir ?? await bundledModelDir();
    if (models == null) {
      throw const DspException(DspErrorKind.modelNotFound, 'No model directory: pass modelDir.');
    }
    // The control block lives on the native heap, shared with the isolate
    // that runs the analysis: progress is read from here, cancel written.
    final control = calloc<bindings.IciclesDspControl>();
    var done = false;
    final poll = onProgress == null
        ? null
        : Timer.periodic(const Duration(milliseconds: 250), (_) {
            if (!done) onProgress(_progressOf(control));
          });
    unawaited(
      cancel?.then((_) {
        if (!done) control.ref.cancel = 1;
      }),
    );
    try {
      final wire = await _runDeep(mono, sampleRate, models, folds, control.address);
      onProgress?.call(1);
      return DspDeepAnalysis.fromWire(wire);
    } finally {
      done = true;
      poll?.cancel();
      calloc.free(control);
    }
  }

  /// Spawns the deep analysis isolate from a scope that captures nothing
  /// but sendable values (closures share their enclosing context).
  static Future<Map<String, Object?>> _runDeep(
    Float32List mono,
    int sampleRate,
    String modelDir,
    int folds,
    int controlAddress,
  ) {
    final transfer = TransferableTypedData.fromList([mono]);
    return Isolate.run(
      () => analyzeDeepSync(
        transfer.materialize().asFloat32List(),
        sampleRate,
        modelDir: modelDir,
        folds: folds,
        controlAddress: controlAddress,
      ).toWire(),
    );
  }

  static double _progressOf(ffi.Pointer<bindings.IciclesDspControl> control) {
    final bits = ByteData(4)..setUint32(0, control.ref.progress, Endian.host);
    return bits.getFloat32(0, Endian.host).clamp(0.0, 1.0);
  }

  /// The synchronous native call behind [analyzeDeep]; blocks the calling
  /// isolate. [controlAddress] is the address of an `IciclesDspControl`
  /// (0 = none).
  static DspDeepAnalysis analyzeDeepSync(
    Float32List mono,
    int sampleRate, {
    required String modelDir,
    int folds = foldCount,
    int controlAddress = 0,
  }) {
    final samples = calloc<ffi.Float>(mono.length);
    final out = calloc<bindings.IciclesDspDeepResult>();
    final path = modelDir.toNativeUtf8();
    try {
      samples.asTypedList(mono.length).setAll(0, mono);
      final int status;
      try {
        status = bindings.icicles_dsp_analyze_deep(
          samples,
          mono.length,
          sampleRate,
          path.cast(),
          folds,
          ffi.Pointer.fromAddress(controlAddress),
          out,
        );
      } on Error catch (e) {
        throw DspException(DspErrorKind.libraryUnavailable, 'The icicles_dsp native library is not available: $e');
      }
      if (status != bindings.IciclesDspStatus.ICICLES_DSP_OK.value) {
        final message = bindings.icicles_dsp_last_error().cast<Utf8>().toDartString();
        throw DspException(dspErrorKindOf(status), message.isEmpty ? 'Native status $status.' : message);
      }
      final r = out.ref;
      Float32List floats(ffi.Pointer<ffi.Float> pointer, int count) =>
          count == 0 ? Float32List(0) : Float32List.fromList(pointer.asTypedList(count));
      return DspDeepAnalysis.fromRaw(
        sectionStarts: floats(r.section_starts, r.section_count),
        sectionEnds: floats(r.section_ends, r.section_count),
        sectionLabels: r.section_count == 0
            ? Uint8List(0)
            : Uint8List.fromList(r.section_labels.asTypedList(r.section_count)),
        stemLevels: floats(r.stem_levels, r.stem_rows * DspStem.values.length),
        stemRows: r.stem_rows,
        chroma: floats(r.chroma, r.chroma_rows * 12),
        chromaRows: r.chroma_rows,
        drumTimes: floats(r.drum_times, r.drum_count),
        drumClasses: r.drum_count == 0 ? Uint8List(0) : Uint8List.fromList(r.drum_classes.asTypedList(r.drum_count)),
        drumStrengths: floats(r.drum_strengths, r.drum_count),
        drumError: r.drum_error == ffi.nullptr ? null : r.drum_error.cast<Utf8>().toDartString(),
      );
    } finally {
      try {
        bindings.icicles_dsp_free_deep(out);
      } on Error {
        // The library is unavailable: nothing was allocated natively.
      }
      calloc.free(out);
      calloc.free(samples);
      calloc.free(path);
    }
  }
}
