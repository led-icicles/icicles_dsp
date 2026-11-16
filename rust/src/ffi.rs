//! The C ABI (`icicles_dsp_*`), consumed by the Dart side over `dart:ffi`.
//!
//! Every entry point is wrapped in `catch_unwind`: a panic never crosses the
//! boundary, it becomes [`IciclesDspStatus::Panic`] with a message in
//! [`icicles_dsp_last_error`]. The header `include/icicles_dsp.h` mirrors
//! these declarations for ffigen.

use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::ptr;

use crate::beats::{self, BeatError};
use crate::drums::{self, DrumError};
use crate::features::{self, FeatureError, COLUMN_COUNT, HOP_MICROS};
use crate::model::ModelSource;

/// Status codes returned by [`icicles_dsp_analyze`].
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IciclesDspStatus {
    /// Success.
    Ok = 0,
    /// A null pointer, an empty signal or a sample rate below 1000 Hz.
    InvalidArgument = 1,
    /// A model file is missing from `model_dir`.
    ModelNotFound = 2,
    /// A model file exists but could not be loaded.
    ModelLoadFailed = 3,
    /// Feature extraction or beat inference failed.
    AnalysisFailed = 4,
    /// A panic was caught; the library stays usable.
    Panic = 5,
}

/// The analysis output. Arrays are owned by the library: release them with
/// [`icicles_dsp_free`]. A zeroed struct (`beat_count == 0`, null pointers) is
/// valid and free-able.
#[repr(C)]
#[derive(Debug)]
pub struct IciclesDspResult {
    /// Beat instants in seconds, sorted.
    pub beats: *mut f32,
    pub beat_count: usize,
    /// Downbeat instants in seconds, sorted (a subset of the beats).
    pub downbeats: *mut f32,
    pub downbeat_count: usize,
    /// The feature matrix, row-major `feature_rows × feature_columns`, in the
    /// `icicles_generators` `AudioFeatures` layout (values `0..1`).
    pub features: *mut f32,
    pub feature_rows: usize,
    pub feature_columns: usize,
    /// Time between two feature rows in microseconds (10 000).
    pub feature_hop_micros: u32,
    /// Drum hits (with [`ICICLES_DSP_DETECT_DRUMS`]), sorted by time: onset
    /// times in seconds, classes (0 kick, 1 snare, 2 toms, 3 hi-hat,
    /// 4 cymbals) and strengths `0..1`, `drum_count` of each.
    pub drum_times: *mut f32,
    pub drum_classes: *mut u8,
    pub drum_strengths: *mut f32,
    pub drum_count: usize,
    /// Status of the drum detection: [`IciclesDspStatus::Ok`] when it ran
    /// (or was not requested); otherwise the cause, with the message in
    /// `drum_error`. A failed drum detection never fails the analysis.
    pub drum_status: i32,
    /// NUL-terminated message of a failed drum detection, else NULL.
    pub drum_error: *mut c_char,
}

impl IciclesDspResult {
    const EMPTY: IciclesDspResult = IciclesDspResult {
        beats: ptr::null_mut(),
        beat_count: 0,
        downbeats: ptr::null_mut(),
        downbeat_count: 0,
        features: ptr::null_mut(),
        feature_rows: 0,
        feature_columns: 0,
        feature_hop_micros: 0,
        drum_times: ptr::null_mut(),
        drum_classes: ptr::null_mut(),
        drum_strengths: ptr::null_mut(),
        drum_count: 0,
        drum_status: 0,
        drum_error: ptr::null_mut(),
    };
}

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_last_error(message: impl Into<Vec<u8>>) {
    let message = CString::new(message)
        .unwrap_or_else(|_| CString::new("error message contained NUL").expect("literal"));
    LAST_ERROR.with(|slot| *slot.borrow_mut() = message);
}

/// Hands a `Vec` to C: pointer + length, capacity shrunk to the length so
/// [`take_back`] can rebuild it.
fn give_away<T>(values: Vec<T>) -> (*mut T, usize) {
    let boxed = values.into_boxed_slice();
    let len = boxed.len();
    if len == 0 {
        return (ptr::null_mut(), 0);
    }
    (Box::into_raw(boxed) as *mut T, len)
}

/// Reclaims an array made by [`give_away`].
///
/// # Safety
/// `pointer`/`len` must come from `give_away` and not have been freed.
unsafe fn take_back<T>(pointer: *mut T, len: usize) {
    if pointer.is_null() || len == 0 {
        return;
    }
    drop(Vec::from_raw_parts(pointer, len, len));
}

static VERSION_C: &CStr =
    match CStr::from_bytes_with_nul(concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes()) {
        Ok(version) => version,
        Err(_) => panic!("version contains NUL"),
    };

/// The library version as a static NUL-terminated string (never freed).
#[no_mangle]
pub extern "C" fn icicles_dsp_version() -> *const c_char {
    VERSION_C.as_ptr()
}

/// The message of the last failure on this thread (NUL-terminated, valid
/// until the next `icicles_dsp_*` call on the same thread; empty when the
/// last call succeeded).
#[no_mangle]
pub extern "C" fn icicles_dsp_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| slot.borrow().as_ptr())
}

/// [`icicles_dsp_analyze`] flag: track beats and downbeats (Beat This!).
pub const ICICLES_DSP_TRACK_BEATS: u32 = 1;
/// [`icicles_dsp_analyze`] flag: detect drum hits (ADTOF).
pub const ICICLES_DSP_DETECT_DRUMS: u32 = 2;

/// Analyses `len` mono f32 samples (`-1..1`) at `sample_rate` Hz.
///
/// `model_dir` is a NUL-terminated path to the directory with the models:
/// with [`ICICLES_DSP_TRACK_BEATS`] in `flags` the Beat This! models
/// (`mel_spectrogram.onnx` plus `beat_this.onnx` or `beat_this_small.onnx`),
/// with [`ICICLES_DSP_DETECT_DRUMS`] the drum model (`adtof_frame_rnn.onnx`).
/// A NULL `model_dir` computes the features only, whatever the flags. On
/// success `out` holds the result (release it with [`icicles_dsp_free`]); on
/// failure it is zeroed and the status names the cause, with details in
/// [`icicles_dsp_last_error`]. A failed drum detection is not a failure: it
/// is reported in `drum_status` / `drum_error` of the result.
///
/// # Safety
/// `samples` must point to `len` readable `f32`s and `out` to a writable
/// `IciclesDspResult`; `model_dir` must be NULL or a valid C string.
#[no_mangle]
pub unsafe extern "C" fn icicles_dsp_analyze(
    samples: *const f32,
    len: usize,
    sample_rate: u32,
    model_dir: *const c_char,
    flags: u32,
    out: *mut IciclesDspResult,
) -> i32 {
    if out.is_null() {
        set_last_error("out is null");
        return IciclesDspStatus::InvalidArgument as i32;
    }
    ptr::write(out, IciclesDspResult::EMPTY);
    finish(
        catch_unwind(AssertUnwindSafe(|| {
            analyze_checked(samples, len, sample_rate, model_dir, flags)
        })),
        out,
    )
}

/// Like [`icicles_dsp_analyze`], with the models given as bytes instead of a
/// directory (the browser build has no file system): `mel`/`beat` are needed
/// with [`ICICLES_DSP_TRACK_BEATS`], `drum` with [`ICICLES_DSP_DETECT_DRUMS`]
/// (a NULL `drum` fails only the drum detection, reported in the result).
/// A NULL `mel` or `beat` computes the features only, whatever the flags.
///
/// # Safety
/// As [`icicles_dsp_analyze`]; each non-NULL model pointer must point to its
/// length of readable bytes.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn icicles_dsp_analyze_models(
    samples: *const f32,
    len: usize,
    sample_rate: u32,
    mel: *const u8,
    mel_len: usize,
    beat: *const u8,
    beat_len: usize,
    drum: *const u8,
    drum_len: usize,
    flags: u32,
    out: *mut IciclesDspResult,
) -> i32 {
    if out.is_null() {
        set_last_error("out is null");
        return IciclesDspStatus::InvalidArgument as i32;
    }
    ptr::write(out, IciclesDspResult::EMPTY);
    finish(
        catch_unwind(AssertUnwindSafe(|| {
            let bytes = |pointer: *const u8, len: usize| {
                (!pointer.is_null() && len > 0).then(|| std::slice::from_raw_parts(pointer, len))
            };
            let models = match (bytes(mel, mel_len), bytes(beat, beat_len)) {
                (Some(mel), Some(beat)) => Some(ModelSource::Bytes {
                    mel,
                    beat,
                    drum: bytes(drum, drum_len),
                }),
                _ => None,
            };
            let signal = checked_signal(samples, len, sample_rate)?;
            analyze(signal, sample_rate, models, flags)
        })),
        out,
    )
}

/// Allocates `len` bytes (8-aligned) for a caller that has to write into the
/// library's memory (the browser host: PCM and models); release with
/// [`icicles_dsp_dealloc`] and the same `len`. NULL for `len == 0`.
#[no_mangle]
pub extern "C" fn icicles_dsp_alloc(len: usize) -> *mut u8 {
    match std::alloc::Layout::from_size_align(len, 8) {
        Ok(layout) if len > 0 => unsafe { std::alloc::alloc(layout) },
        _ => ptr::null_mut(),
    }
}

/// Releases a block of [`icicles_dsp_alloc`].
///
/// # Safety
/// `pointer` must be NULL or come from `icicles_dsp_alloc(len)`, not freed.
#[no_mangle]
pub unsafe extern "C" fn icicles_dsp_dealloc(pointer: *mut u8, len: usize) {
    if pointer.is_null() || len == 0 {
        return;
    }
    if let Ok(layout) = std::alloc::Layout::from_size_align(len, 8) {
        std::alloc::dealloc(pointer, layout);
    }
}

type Outcome = std::thread::Result<Result<IciclesDspResult, Failure>>;

/// Writes a caught analysis outcome to `out` and returns its status.
unsafe fn finish(outcome: Outcome, out: *mut IciclesDspResult) -> i32 {
    match outcome {
        Ok(Ok(result)) => {
            set_last_error("");
            ptr::write(out, result);
            IciclesDspStatus::Ok as i32
        }
        Ok(Err((status, message))) => {
            set_last_error(message);
            status as i32
        }
        Err(panic) => {
            let message = panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_string()))
                .unwrap_or_else(|| "unknown panic".to_string());
            set_last_error(format!("panic: {message}"));
            IciclesDspStatus::Panic as i32
        }
    }
}

type Failure = (IciclesDspStatus, String);

unsafe fn analyze_checked(
    samples: *const f32,
    len: usize,
    sample_rate: u32,
    model_dir: *const c_char,
    flags: u32,
) -> Result<IciclesDspResult, Failure> {
    let signal = checked_signal(samples, len, sample_rate)?;
    let model_dir = if model_dir.is_null() {
        None
    } else {
        let raw = CStr::from_ptr(model_dir).to_str().map_err(|_| {
            (
                IciclesDspStatus::InvalidArgument,
                "model_dir is not UTF-8".to_string(),
            )
        })?;
        Some(Path::new(raw).to_path_buf())
    };
    analyze(
        signal,
        sample_rate,
        model_dir.as_deref().map(ModelSource::Dir),
        flags,
    )
}

/// The validated signal of an entry point.
unsafe fn checked_signal<'a>(
    samples: *const f32,
    len: usize,
    sample_rate: u32,
) -> Result<&'a [f32], Failure> {
    if samples.is_null() {
        return Err((IciclesDspStatus::InvalidArgument, "samples is null".into()));
    }
    if len == 0 {
        return Err((
            IciclesDspStatus::InvalidArgument,
            "the signal is empty".into(),
        ));
    }
    if sample_rate < 1000 {
        return Err((
            IciclesDspStatus::InvalidArgument,
            FeatureError::SampleRateTooLow(sample_rate).to_string(),
        ));
    }
    Ok(std::slice::from_raw_parts(samples, len))
}

fn analyze(
    signal: &[f32],
    sample_rate: u32,
    models: Option<ModelSource<'_>>,
    flags: u32,
) -> Result<IciclesDspResult, Failure> {
    let features = features::extract(signal, sample_rate)
        .map_err(|e| (IciclesDspStatus::InvalidArgument, e.to_string()))?;
    let beats = match models.filter(|_| flags & ICICLES_DSP_TRACK_BEATS != 0) {
        Some(models) => beats::track(signal, sample_rate, models).map_err(|e| {
            let status = match e {
                BeatError::ModelNotFound(_) => IciclesDspStatus::ModelNotFound,
                BeatError::ModelLoad(_) => IciclesDspStatus::ModelLoadFailed,
                BeatError::Analysis(_) => IciclesDspStatus::AnalysisFailed,
            };
            (status, e.to_string())
        })?,
        None => beats::Beats::default(),
    };
    let (drum_hits, drum_status, drum_error) = match models
        .filter(|_| flags & ICICLES_DSP_DETECT_DRUMS != 0)
    {
        None => (Vec::new(), IciclesDspStatus::Ok, ptr::null_mut()),
        Some(models) => match drums::detect(signal, sample_rate, models) {
            Ok(hits) => (hits, IciclesDspStatus::Ok, ptr::null_mut()),
            Err(e) => {
                let status = match e {
                    DrumError::ModelNotFound(_) => IciclesDspStatus::ModelNotFound,
                    DrumError::ModelLoad(_) => IciclesDspStatus::ModelLoadFailed,
                    DrumError::Analysis(_) => IciclesDspStatus::AnalysisFailed,
                };
                let message = CString::new(e.to_string())
                    .unwrap_or_else(|_| CString::new("drum detection failed").expect("literal"));
                (Vec::new(), status, message.into_raw())
            }
        },
    };
    let drum_count = drum_hits.len();
    let (drum_times, _) = give_away(drum_hits.iter().map(|h| h.time).collect());
    let (drum_classes, _) = give_away(drum_hits.iter().map(|h| h.class as u8).collect());
    let (drum_strengths, _) = give_away(drum_hits.iter().map(|h| h.strength).collect());
    let (beat_ptr, beat_count) = give_away(beats.beats);
    let (downbeat_ptr, downbeat_count) = give_away(beats.downbeats);
    let (feature_ptr, _) = give_away(features.data);
    Ok(IciclesDspResult {
        beats: beat_ptr,
        beat_count,
        downbeats: downbeat_ptr,
        downbeat_count,
        features: feature_ptr,
        feature_rows: features.rows,
        feature_columns: COLUMN_COUNT,
        feature_hop_micros: HOP_MICROS,
        drum_times,
        drum_classes,
        drum_strengths,
        drum_count,
        drum_status: drum_status as i32,
        drum_error,
    })
}

/// Releases the arrays of a result filled by [`icicles_dsp_analyze`] and
/// zeroes it. NULL and already-freed (zeroed) results are no-ops.
///
/// # Safety
/// `out` must be NULL or point to a result written by `icicles_dsp_analyze`.
#[no_mangle]
pub unsafe extern "C" fn icicles_dsp_free(out: *mut IciclesDspResult) {
    if out.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let result = ptr::read(out);
        take_back(result.beats, result.beat_count);
        take_back(result.downbeats, result.downbeat_count);
        take_back(
            result.features,
            result.feature_rows * result.feature_columns,
        );
        take_back(result.drum_times, result.drum_count);
        take_back(result.drum_classes, result.drum_count);
        take_back(result.drum_strengths, result.drum_count);
        if !result.drum_error.is_null() {
            drop(CString::from_raw(result.drum_error));
        }
        ptr::write(out, IciclesDspResult::EMPTY);
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn last_error() -> String {
        unsafe { CStr::from_ptr(icicles_dsp_last_error()) }
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn version_is_the_crate_version() {
        let version = unsafe { CStr::from_ptr(icicles_dsp_version()) }
            .to_str()
            .unwrap();
        assert_eq!(version, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn features_only_without_a_model_dir() {
        let signal = vec![0.1f32; 44100];
        let mut out = IciclesDspResult::EMPTY;
        let status = unsafe {
            icicles_dsp_analyze(
                signal.as_ptr(),
                signal.len(),
                44100,
                ptr::null(),
                0,
                &mut out,
            )
        };
        assert_eq!(status, IciclesDspStatus::Ok as i32);
        assert_eq!(last_error(), "");
        assert_eq!(out.feature_rows, 100);
        assert_eq!(out.feature_columns, COLUMN_COUNT);
        assert_eq!(out.feature_hop_micros, 10_000);
        assert_eq!(out.beat_count, 0);
        assert!(out.beats.is_null());
        let features = unsafe {
            std::slice::from_raw_parts(out.features, out.feature_rows * out.feature_columns)
        };
        assert!(features.iter().all(|v| (0.0..=1.0).contains(v)));
        unsafe { icicles_dsp_free(&mut out) };
        assert!(out.features.is_null());
        assert_eq!(out.feature_rows, 0);
        unsafe { icicles_dsp_free(&mut out) };
        unsafe { icicles_dsp_free(ptr::null_mut()) };
    }

    #[test]
    fn rejects_bad_arguments() {
        let mut out = IciclesDspResult::EMPTY;
        let signal = [0.0f32; 10];
        assert_eq!(
            unsafe { icicles_dsp_analyze(ptr::null(), 10, 44100, ptr::null(), 0, &mut out) },
            IciclesDspStatus::InvalidArgument as i32
        );
        assert_eq!(last_error(), "samples is null");
        assert_eq!(
            unsafe { icicles_dsp_analyze(signal.as_ptr(), 0, 44100, ptr::null(), 0, &mut out) },
            IciclesDspStatus::InvalidArgument as i32
        );
        assert_eq!(
            unsafe {
                icicles_dsp_analyze(signal.as_ptr(), signal.len(), 999, ptr::null(), 0, &mut out)
            },
            IciclesDspStatus::InvalidArgument as i32
        );
        assert!(last_error().contains("999"));
        assert_eq!(
            unsafe {
                icicles_dsp_analyze(
                    signal.as_ptr(),
                    signal.len(),
                    44100,
                    ptr::null(),
                    0,
                    ptr::null_mut(),
                )
            },
            IciclesDspStatus::InvalidArgument as i32
        );
    }

    #[test]
    fn missing_models_are_reported() {
        let signal = [0.0f32; 4410];
        let dir = CString::new("/nonexistent/icicles_dsp_models").unwrap();
        let mut out = IciclesDspResult::EMPTY;
        let status = unsafe {
            icicles_dsp_analyze(
                signal.as_ptr(),
                signal.len(),
                44100,
                dir.as_ptr(),
                ICICLES_DSP_TRACK_BEATS,
                &mut out,
            )
        };
        assert_eq!(status, IciclesDspStatus::ModelNotFound as i32);
        assert!(
            last_error().contains("mel_spectrogram.onnx"),
            "{}",
            last_error()
        );
        assert!(out.features.is_null());
    }

    #[test]
    fn detects_drums_with_the_bundled_model() {
        // Kicks every 0.5 s from 0.25 s: decaying 60 Hz bursts.
        let sr = 44_100usize;
        let signal: Vec<f32> = (0..sr * 3)
            .map(|i| {
                let t = i as f64 / sr as f64 - 0.25;
                if t < 0.0 {
                    return 0.0;
                }
                let dt = t % 0.5;
                (0.9 * (-dt / 0.12).exp()
                    * (2.0 * std::f64::consts::PI * (55.0 * dt + 3.0 * (1.0 - (-30.0 * dt).exp())))
                        .sin()) as f32
            })
            .collect();
        let dir = CString::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../models")).unwrap();
        let mut out = IciclesDspResult::EMPTY;
        let status = unsafe {
            icicles_dsp_analyze(
                signal.as_ptr(),
                signal.len(),
                44100,
                dir.as_ptr(),
                ICICLES_DSP_DETECT_DRUMS,
                &mut out,
            )
        };
        assert_eq!(status, IciclesDspStatus::Ok as i32, "{}", last_error());
        assert_eq!(out.drum_status, IciclesDspStatus::Ok as i32);
        assert!(out.drum_error.is_null());
        assert_eq!(out.beat_count, 0, "beats were not requested");
        let times = unsafe { std::slice::from_raw_parts(out.drum_times, out.drum_count) };
        let classes = unsafe { std::slice::from_raw_parts(out.drum_classes, out.drum_count) };
        let strengths = unsafe { std::slice::from_raw_parts(out.drum_strengths, out.drum_count) };
        let kicks: Vec<f32> = times
            .iter()
            .zip(classes)
            .filter(|(_, c)| **c == 0)
            .map(|(t, _)| *t)
            .collect();
        assert!(kicks.len() >= 5, "kicks {kicks:?}, classes {classes:?}");
        assert!(strengths.iter().all(|s| (0.0..=1.0).contains(s)));
        assert!(times.windows(2).all(|w| w[0] <= w[1]));
        unsafe { icicles_dsp_free(&mut out) };
        assert!(out.drum_times.is_null());
    }

    #[test]
    fn missing_drum_model_does_not_fail_the_analysis() {
        let signal = vec![0.1f32; 44100];
        let dir = CString::new("/nonexistent/icicles_dsp_models").unwrap();
        let mut out = IciclesDspResult::EMPTY;
        let status = unsafe {
            icicles_dsp_analyze(
                signal.as_ptr(),
                signal.len(),
                44100,
                dir.as_ptr(),
                ICICLES_DSP_DETECT_DRUMS,
                &mut out,
            )
        };
        assert_eq!(status, IciclesDspStatus::Ok as i32);
        assert_eq!(out.feature_rows, 100);
        assert_eq!(out.drum_count, 0);
        assert_eq!(out.drum_status, IciclesDspStatus::ModelNotFound as i32);
        let message = unsafe { CStr::from_ptr(out.drum_error) }
            .to_string_lossy()
            .into_owned();
        assert!(message.contains("adtof_frame_rnn.onnx"), "{message}");
        unsafe { icicles_dsp_free(&mut out) };
        assert!(out.drum_error.is_null());
    }
}
