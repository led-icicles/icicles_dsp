/// The platform-neutral types of `icicles_dsp` — the result, its raw arrays
/// and the failures — importable everywhere (also on the web, unlike
/// `icicles_dsp.dart`, which needs `dart:ffi`).
library;

export 'package:icicles_generators/icicles_generators.dart'
    show AudioFeatureKey, AudioFeatures, DrumClass, DrumHit, DrumHits;

export 'src/dsp_analysis.dart';
export 'src/dsp_deep_analysis.dart';
export 'src/dsp_exception.dart';
export 'src/dsp_result_data.dart' show DspResultData;
