// C ABI of the icicles_dsp crate (rust/src/ffi.rs). Keep both in sync; the
// Dart bindings are generated from this header with ffigen.
#ifndef ICICLES_DSP_H
#define ICICLES_DSP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Status codes of icicles_dsp_analyze.
typedef enum IciclesDspStatus {
  // Success.
  ICICLES_DSP_OK = 0,
  // A null pointer, an empty signal or a sample rate below 1000 Hz.
  ICICLES_DSP_INVALID_ARGUMENT = 1,
  // A model file is missing from model_dir.
  ICICLES_DSP_MODEL_NOT_FOUND = 2,
  // A model file exists but could not be loaded.
  ICICLES_DSP_MODEL_LOAD_FAILED = 3,
  // Feature extraction or beat inference failed.
  ICICLES_DSP_ANALYSIS_FAILED = 4,
  // A panic was caught; the library stays usable.
  ICICLES_DSP_PANIC = 5,
} IciclesDspStatus;

// The analysis output. The arrays are owned by the library: release them with
// icicles_dsp_free. A zeroed struct is valid and free-able.
typedef struct IciclesDspResult {
  // Beat instants in seconds, sorted.
  float *beats;
  size_t beat_count;
  // Downbeat instants in seconds, sorted (a subset of the beats).
  float *downbeats;
  size_t downbeat_count;
  // The feature matrix, row-major feature_rows x feature_columns, in the
  // icicles_generators AudioFeatures layout (values 0..1).
  float *features;
  size_t feature_rows;
  size_t feature_columns;
  // Time between two feature rows in microseconds (10 000).
  uint32_t feature_hop_micros;
  // Drum hits (with ICICLES_DSP_DETECT_DRUMS), sorted by time: onset times in
  // seconds, classes (0 kick, 1 snare, 2 toms, 3 hi-hat, 4 cymbals) and
  // strengths 0..1, drum_count of each.
  float *drum_times;
  uint8_t *drum_classes;
  float *drum_strengths;
  size_t drum_count;
  // Status of the drum detection: ICICLES_DSP_OK when it ran (or was not
  // requested), otherwise the cause with the message in drum_error. A failed
  // drum detection never fails the analysis.
  int32_t drum_status;
  // NUL-terminated message of a failed drum detection, else NULL.
  char *drum_error;
} IciclesDspResult;

// icicles_dsp_analyze flags.
enum {
  // Track beats and downbeats (Beat This!).
  ICICLES_DSP_TRACK_BEATS = 1,
  // Detect drum hits (ADTOF).
  ICICLES_DSP_DETECT_DRUMS = 2,
};

// The library version as a static NUL-terminated string (never freed).
const char *icicles_dsp_version(void);

// The message of the last failure on this thread (NUL-terminated, valid until
// the next icicles_dsp_* call on the same thread; empty after a success).
const char *icicles_dsp_last_error(void);

// Analyses `len` mono float samples (-1..1) at `sample_rate` Hz.
//
// `model_dir` is a NUL-terminated path to the directory with the models: with
// ICICLES_DSP_TRACK_BEATS in `flags` the Beat This! models (mel_spectrogram.onnx
// plus beat_this.onnx or beat_this_small.onnx), with ICICLES_DSP_DETECT_DRUMS
// the drum model (adtof_frame_rnn.onnx). A NULL `model_dir` computes the
// features only, whatever the flags. On success `out` holds the result
// (release it with icicles_dsp_free); on failure it is zeroed and the returned
// IciclesDspStatus names the cause, with details in icicles_dsp_last_error. A
// failed drum detection is not a failure: see drum_status / drum_error.
int32_t icicles_dsp_analyze(const float *samples, size_t len, uint32_t sample_rate, const char *model_dir,
                            uint32_t flags, IciclesDspResult *out);

// Releases the arrays of a result filled by icicles_dsp_analyze and zeroes it.
// NULL and already-freed results are no-ops.
void icicles_dsp_free(IciclesDspResult *out);

#ifdef __cplusplus
}
#endif

#endif  // ICICLES_DSP_H
