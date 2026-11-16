/// The `icicles_dsp` analysis in the browser: the Rust crate compiled to
/// WebAssembly (`tool/build_wasm.sh` → `build/web/icicles_dsp.wasm`), run by
/// [IciclesDspWasm] with the models as bytes. Web only (`dart:js_interop`);
/// run it on a Web Worker.
library;

export 'icicles_dsp_types.dart';
export 'src/web/icicles_dsp_wasm.dart';
