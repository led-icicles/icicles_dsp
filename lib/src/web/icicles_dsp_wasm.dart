import 'dart:js_interop';
import 'dart:typed_data';

import '../dsp_analysis.dart';
import '../dsp_exception.dart';
import '../dsp_result_data.dart';

/// The Rust analysis compiled to WebAssembly (`wasm32-wasip1`, built by
/// `tool/build_wasm.sh`), for the browser: the same Beat This! beats, ADTOF
/// drum hits and feature matrix as [IciclesDsp] on the desktop.
///
/// [compile] the module once; every [analyze] runs on a fresh instance, so
/// its memory (several hundred MB with the full beat model) is released
/// afterwards and a trap (a Rust panic aborts in WebAssembly) never leaves a
/// broken instance behind. The module only imports a handful of WASI calls:
/// the clock, randomness and text output are provided here, the file system
/// calls report "no such file" (the models arrive as bytes).
///
/// Runs synchronously: call it on a Web Worker. Single-threaded (rayon falls
/// back to the calling thread).
final class IciclesDspWasm {
  final _Module _module;

  IciclesDspWasm._(this._module);

  /// Compiles the module from its bytes. Throws a [DspException]
  /// ([DspErrorKind.libraryUnavailable]) when the browser cannot compile it
  /// (no WebAssembly SIMD, a corrupt file).
  static Future<IciclesDspWasm> compile(Uint8List wasm) async {
    try {
      return IciclesDspWasm._(await _compile(wasm.toJS).toDart);
    } catch (error) {
      throw DspException(DspErrorKind.libraryUnavailable, 'The icicles_dsp WebAssembly module did not compile: $error');
    }
  }

  /// Analyses [mono] PCM (`-1..1`) at [sampleRate] Hz with the models'
  /// bytes: [mel] and [beat] (the Beat This! front end and beat model) for
  /// [trackBeats], [drum] (ADTOF) for [detectDrums]. Without a [drum] model
  /// the drum detection fails on its own ([DspAnalysis.drumError]), never
  /// the analysis. Throws [DspException].
  DspAnalysis analyze(
    Float32List mono,
    int sampleRate, {
    required Uint8List mel,
    required Uint8List beat,
    Uint8List? drum,
    bool trackBeats = true,
    bool detectDrums = true,
  }) => analyzeData(
    mono,
    sampleRate,
    mel: mel,
    beat: beat,
    drum: drum,
    trackBeats: trackBeats,
    detectDrums: detectDrums,
  ).toAnalysis();

  /// [analyze] as the raw result arrays, e.g. to send them from a Web Worker
  /// to the page ([DspResultData.toAnalysis] there).
  DspResultData analyzeData(
    Float32List mono,
    int sampleRate, {
    required Uint8List mel,
    required Uint8List beat,
    Uint8List? drum,
    bool trackBeats = true,
    bool detectDrums = true,
  }) {
    if (mono.isEmpty) throw const DspException(DspErrorKind.invalidArgument, 'The signal is empty.');
    return _Instance(this)
        .analyze(mono, sampleRate, mel, beat, drum, dspFlags(trackBeats: trackBeats, detectDrums: detectDrums));
  }
}

/// One instantiation of the module with its WASI shim.
final class _Instance {
  late final _Exports _exports;
  final _stderr = StringBuffer();

  _Instance(IciclesDspWasm wasm) {
    final imports = JSObject();
    imports['wasi_snapshot_preview1'] = _wasi();
    try {
      _exports = _instantiate(wasm._module, imports).exports;
    } catch (error) {
      throw DspException(DspErrorKind.libraryUnavailable, 'The icicles_dsp WebAssembly module did not start: $error');
    }
  }

  ByteData get _memory => _exports.memory.buffer.toDart.asByteData();

  Uint8List get _bytes => _exports.memory.buffer.toDart.asUint8List();

  int _put(Uint8List bytes) {
    if (bytes.isEmpty) return 0;
    final pointer = _u32(_exports.alloc(bytes.length));
    if (pointer == 0) throw const DspException(DspErrorKind.analysisFailed, 'Out of WebAssembly memory.');
    _bytes.setAll(pointer, bytes);
    return pointer;
  }

  DspResultData analyze(Float32List mono, int sampleRate, Uint8List mel, Uint8List beat, Uint8List? drum, int flags) {
    final int status;
    final int out;
    try {
      final samples = _put(mono.buffer.asUint8List(mono.offsetInBytes, mono.lengthInBytes));
      final melAt = _put(mel), beatAt = _put(beat), drumAt = drum == null ? 0 : _put(drum);
      out = _u32(_exports.alloc(_resultSize));
      status = _exports.analyzeModels(
        samples,
        mono.length,
        sampleRate,
        melAt,
        mel.length,
        beatAt,
        beat.length,
        drumAt,
        drum?.length ?? 0,
        flags,
        out,
      );
    } on DspException {
      rethrow;
    } catch (error) {
      // A trap: a Rust panic aborts the instance (its message went to stderr).
      final message = _stderr.toString().trim();
      throw DspException(DspErrorKind.nativePanic, message.isEmpty ? '$error' : message);
    }
    if (status != 0) {
      throw DspException(dspErrorKindOf(status), _cString(_u32(_exports.lastError())) ?? 'Status $status.');
    }
    return _read(out);
  }

  /// The `IciclesDspResult` at [at] (wasm32 layout: pointers and `usize`
  /// are 4 bytes, see `rust/src/ffi.rs`), copied out of the module memory.
  DspResultData _read(int at) {
    final m = _memory;
    int u(int field) => m.getUint32(at + 4 * field, Endian.little);
    final buffer = _exports.memory.buffer.toDart;
    Float32List floats(int pointer, int count) =>
        count == 0 ? Float32List(0) : Float32List.fromList(buffer.asFloat32List(pointer, count));
    final drumCount = u(11);
    return DspResultData(
      beats: floats(u(0), u(1)),
      downbeats: floats(u(2), u(3)),
      features: floats(u(4), u(5) * u(6)),
      featureRows: u(5),
      featureColumns: u(6),
      featureHopMicros: u(7),
      drumTimes: floats(u(8), drumCount),
      drumClasses: drumCount == 0 ? Uint8List(0) : Uint8List.fromList(buffer.asUint8List(u(9), drumCount)),
      drumStrengths: floats(u(10), drumCount),
      drumError: _cString(u(13)),
    );
  }

  /// `sizeof(IciclesDspResult)` on wasm32: 12 pointer-sized fields, 2 × u32/i32.
  static const _resultSize = 56;

  String? _cString(int pointer) {
    if (pointer == 0) return null;
    final bytes = _bytes;
    var end = pointer;
    while (bytes[end] != 0) {
      end++;
    }
    final text = String.fromCharCodes(bytes, pointer, end);
    return text.isEmpty ? null : text;
  }

  static int _u32(int value) => value & 0xFFFFFFFF;

  static const _ok = 0, _badFile = 8, _noSys = 52;

  /// The WASI preview 1 calls the Rust standard library imports.
  JSObject _wasi() {
    final wasi = JSObject();
    void fn(String name, JSFunction function) => wasi[name] = function;
    fn(
      'clock_time_get',
      ((JSAny? id, JSAny? precision, JSNumber out) {
        final micros = DateTime.now().microsecondsSinceEpoch;
        final m = _memory, at = _u32(out.toDartInt);
        // Nanoseconds as a little-endian u64, written as two halves (no 64-bit
        // ints in dart2js).
        final nanos = micros * 1000;
        m.setUint32(at, nanos % 0x100000000, Endian.little);
        m.setUint32(at + 4, nanos ~/ 0x100000000, Endian.little);
        return _ok;
      }).toJS,
    );
    fn(
      'random_get',
      ((JSNumber at, JSNumber length) {
        _randomValues(JSUint8Array(_exports.memory.buffer, _u32(at.toDartInt), length.toDartInt));
        return _ok;
      }).toJS,
    );
    fn(
      'environ_sizes_get',
      ((JSNumber count, JSNumber size) {
        _memory
          ..setUint32(_u32(count.toDartInt), 0, Endian.little)
          ..setUint32(_u32(size.toDartInt), 0, Endian.little);
        return _ok;
      }).toJS,
    );
    fn('environ_get', ((JSAny? environ, JSAny? buffer) => _ok).toJS);
    fn(
      'fd_write',
      ((JSNumber fd, JSNumber iovs, JSNumber count, JSNumber written) {
        final m = _memory, bytes = _bytes;
        var total = 0;
        for (var i = 0; i < count.toDartInt; i++) {
          final base = _u32(iovs.toDartInt) + 8 * i;
          final at = m.getUint32(base, Endian.little), length = m.getUint32(base + 4, Endian.little);
          _stderr.write(String.fromCharCodes(bytes, at, at + length));
          total += length;
        }
        m.setUint32(_u32(written.toDartInt), total, Endian.little);
        return _ok;
      }).toJS,
    );
    int procExit(JSNumber code) =>
        throw DspException(DspErrorKind.nativePanic, 'proc_exit(${code.toDartInt}): $_stderr');
    fn('proc_exit', procExit.toJS);
    fn('sched_yield', (() => _ok).toJS);
    // No file system: the models arrive as bytes.
    fn('fd_prestat_get', ((JSAny? fd, JSAny? out) => _badFile).toJS);
    fn('fd_prestat_dir_name', ((JSAny? fd, JSAny? path, JSAny? length) => _badFile).toJS);
    fn('fd_close', ((JSAny? fd) => _badFile).toJS);
    fn('fd_fdstat_get', ((JSAny? fd, JSAny? out) => _badFile).toJS);
    fn('fd_filestat_get', ((JSAny? fd, JSAny? out) => _badFile).toJS);
    fn('fd_read', ((JSAny? fd, JSAny? iovs, JSAny? count, JSAny? read) => _badFile).toJS);
    fn('fd_seek', ((JSAny? fd, JSAny? offset, JSAny? whence, JSAny? out) => _badFile).toJS);
    fn('path_filestat_get', ((JSAny? fd, JSAny? flags, JSAny? path, JSAny? length, JSAny? out) => _noSys).toJS);
    fn(
      'path_open',
      ((
        JSAny? fd,
        JSAny? dirflags,
        JSAny? path,
        JSAny? length,
        JSAny? oflags,
        JSAny? base,
        JSAny? inheriting,
        JSAny? fdflags,
        JSAny? out,
      ) => _noSys).toJS,
    );
    return wasi;
  }
}

@JS('crypto.getRandomValues')
external void _randomValues(JSUint8Array bytes);

@JS('WebAssembly.compile')
external JSPromise<_Module> _compile(JSUint8Array bytes);

extension type _Module._(JSObject _) implements JSObject {}

@JS('WebAssembly.Instance')
extension type _WasmInstance._(JSObject _) implements JSObject {
  external factory _WasmInstance(_Module module, JSObject imports);

  external _Exports get exports;
}

_WasmInstance _instantiate(_Module module, JSObject imports) => _WasmInstance(module, imports);

extension type _Memory._(JSObject _) implements JSObject {
  external JSArrayBuffer get buffer;
}

extension type _Exports._(JSObject _) implements JSObject {
  external _Memory get memory;

  @JS('icicles_dsp_alloc')
  external int alloc(int length);

  @JS('icicles_dsp_last_error')
  external int lastError();

  @JS('icicles_dsp_analyze_models')
  external int analyzeModels(
    int samples,
    int length,
    int sampleRate,
    int mel,
    int melLength,
    int beat,
    int beatLength,
    int drum,
    int drumLength,
    int flags,
    int out,
  );
}

extension on JSObject {
  external void operator []=(String key, JSAny? value);
}
