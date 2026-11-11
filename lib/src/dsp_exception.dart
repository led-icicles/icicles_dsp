/// Why [IciclesDsp.analyze] failed.
enum DspErrorKind {
  /// The native library could not be loaded (not built for this platform, or
  /// the build hook did not run).
  libraryUnavailable,

  /// An empty signal, a sample rate below 1000 Hz, or a bad model path.
  invalidArgument,

  /// A model file is missing from the model directory.
  modelNotFound,

  /// A model file exists but could not be loaded.
  modelLoadFailed,

  /// Feature extraction or beat inference failed.
  analysisFailed,

  /// The native code panicked (caught; the library stays usable).
  nativePanic,

  /// The native library returned a status this version does not know.
  unknown,
}

/// A failure of the native analysis, with the native error message.
final class DspException implements Exception {
  final DspErrorKind kind;
  final String message;

  const DspException(this.kind, this.message);

  @override
  String toString() => 'DspException(${kind.name}): $message';
}
