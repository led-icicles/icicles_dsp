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
  // The deep analysis was cancelled through its control block.
  ICICLES_DSP_CANCELLED = 6,
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

// Like icicles_dsp_analyze, with the models given as bytes instead of a
// directory (the browser build has no file system): `mel`/`beat` for
// ICICLES_DSP_TRACK_BEATS, `drum` for ICICLES_DSP_DETECT_DRUMS (a NULL `drum`
// fails only the drum detection, reported in the result). A NULL `mel` or
// `beat` computes the features only, whatever the flags.
int32_t icicles_dsp_analyze_models(const float *samples, size_t len, uint32_t sample_rate, const uint8_t *mel,
                                   size_t mel_len, const uint8_t *beat, size_t beat_len, const uint8_t *drum,
                                   size_t drum_len, uint32_t flags, IciclesDspResult *out);

// Allocates `len` bytes (8-aligned) in the library's memory, for a host that
// has to write the PCM and models there (the browser); NULL for len 0.
uint8_t *icicles_dsp_alloc(size_t len);

// Releases a block of icicles_dsp_alloc (with the same `len`).
void icicles_dsp_dealloc(uint8_t *pointer, size_t len);

// Releases the arrays of a result filled by icicles_dsp_analyze and zeroes it.
// NULL and already-freed results are no-ops.
void icicles_dsp_free(IciclesDspResult *out);

// Progress and cancellation of a deep analysis, shared with another thread:
// `progress` holds the bits of an f32 in 0..1 (written by the library),
// a non-zero `cancel` (written by the caller) stops the analysis.
typedef struct IciclesDspControl {
  uint32_t progress;
  uint32_t cancel;
} IciclesDspControl;

// The deep analysis output. The arrays are owned by the library: release them
// with icicles_dsp_free_deep. A zeroed struct is valid and free-able.
typedef struct IciclesDspDeepResult {
  // Sections covering the song: start/end in seconds and label index
  // (0 start, 1 end, 2 intro, 3 outro, 4 break, 5 bridge, 6 inst, 7 solo,
  // 8 verse, 9 chorus), section_count of each.
  float *section_starts;
  float *section_ends;
  uint8_t *section_labels;
  size_t section_count;
  // Stem loudness 0..1, row-major stem_rows x 4 (drums, bass, other, vocals),
  // one row every 10 ms (the rows of the feature matrix).
  float *stem_levels;
  size_t stem_rows;
  // Chroma of the harmonic stems, row-major chroma_rows x 12 (C..B), 10 rows a
  // second, each row summing to 1 (all zero in silence).
  float *chroma;
  size_t chroma_rows;
  // Drum hits transcribed on the drum stem, sorted by time (as drum_times ..
  // drum_count of IciclesDspResult).
  float *drum_times;
  uint8_t *drum_classes;
  float *drum_strengths;
  size_t drum_count;
  // Status of the drum detection: ICICLES_DSP_OK when it ran, otherwise the
  // cause (e.g. no drum model) with the message in drum_error. A failed drum
  // detection never fails the deep analysis.
  int32_t drum_status;
  // NUL-terminated message of a failed drum detection, else NULL.
  char *drum_error;
} IciclesDspDeepResult;

// Runs the deep analysis (Demucs stems -> stem loudness, chroma, drum hits of
// the drum stem and the All-In-One song structure with `folds` of the 8 folds)
// of `len` mono samples at `sample_rate`. `model_dir` holds htdemucs_core.onnx,
// allin1_fold0..7.onnx and optionally adtof_frame_rnn.onnx (without it only
// the drum detection fails, see drum_status). `control` may be NULL. Minutes
// of CPU; returns ICICLES_DSP_CANCELLED when cancelled.
int32_t icicles_dsp_analyze_deep(const float *samples, size_t len, uint32_t sample_rate, const char *model_dir,
                                 uint32_t folds, IciclesDspControl *control, IciclesDspDeepResult *out);

// Like icicles_dsp_analyze_deep with the models as bytes (the browser): the
// Demucs network, `fold_count` fold models (all of which run) and the drum
// model (NULL `drum`: only the drum detection fails, see drum_status).
int32_t icicles_dsp_analyze_deep_models(const float *samples, size_t len, uint32_t sample_rate,
                                        const uint8_t *demucs, size_t demucs_len, const uint8_t *const *folds,
                                        const size_t *fold_lens, size_t fold_count, const uint8_t *drum,
                                        size_t drum_len, IciclesDspControl *control, IciclesDspDeepResult *out);

// Releases the arrays of a deep result and zeroes it.
void icicles_dsp_free_deep(IciclesDspDeepResult *out);

// Size in bytes of IciclesDspControl (8).
size_t icicles_dsp_control_size(void);

#ifdef __cplusplus
}
#endif

#endif  // ICICLES_DSP_H
