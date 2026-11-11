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
use crate::features::{self, FeatureError, COLUMN_COUNT, HOP_MICROS};

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
fn give_away(values: Vec<f32>) -> (*mut f32, usize) {
    let boxed = values.into_boxed_slice();
    let len = boxed.len();
    if len == 0 {
        return (ptr::null_mut(), 0);
    }
    (Box::into_raw(boxed) as *mut f32, len)
}

/// Reclaims an array made by [`give_away`].
///
/// # Safety
/// `pointer`/`len` must come from `give_away` and not have been freed.
unsafe fn take_back(pointer: *mut f32, len: usize) {
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

/// Analyses `len` mono f32 samples (`-1..1`) at `sample_rate` Hz.
///
/// `model_dir` is a NUL-terminated path to the directory with the Beat This!
/// models (`mel_spectrogram.onnx` plus `beat_this.onnx` or
/// `beat_this_small.onnx`); pass NULL to skip beat tracking and compute the
/// features only. On success `out` holds the result (release it with
/// [`icicles_dsp_free`]); on failure it is zeroed and the status names the
/// cause, with details in [`icicles_dsp_last_error`].
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
    out: *mut IciclesDspResult,
) -> i32 {
    if out.is_null() {
        set_last_error("out is null");
        return IciclesDspStatus::InvalidArgument as i32;
    }
    ptr::write(out, IciclesDspResult::EMPTY);
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        analyze_checked(samples, len, sample_rate, model_dir)
    }));
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
) -> Result<IciclesDspResult, Failure> {
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
    let signal = std::slice::from_raw_parts(samples, len);
    analyze(signal, sample_rate, model_dir.as_deref())
}

fn analyze(
    signal: &[f32],
    sample_rate: u32,
    model_dir: Option<&Path>,
) -> Result<IciclesDspResult, Failure> {
    let features = features::extract(signal, sample_rate)
        .map_err(|e| (IciclesDspStatus::InvalidArgument, e.to_string()))?;
    let beats = match model_dir {
        Some(dir) => beats::track(signal, sample_rate, dir).map_err(|e| {
            let status = match e {
                BeatError::ModelNotFound(_) => IciclesDspStatus::ModelNotFound,
                BeatError::ModelLoad(_) => IciclesDspStatus::ModelLoadFailed,
                BeatError::Analysis(_) => IciclesDspStatus::AnalysisFailed,
            };
            (status, e.to_string())
        })?,
        None => beats::Beats::default(),
    };
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
            icicles_dsp_analyze(signal.as_ptr(), signal.len(), 44100, ptr::null(), &mut out)
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
            unsafe { icicles_dsp_analyze(ptr::null(), 10, 44100, ptr::null(), &mut out) },
            IciclesDspStatus::InvalidArgument as i32
        );
        assert_eq!(last_error(), "samples is null");
        assert_eq!(
            unsafe { icicles_dsp_analyze(signal.as_ptr(), 0, 44100, ptr::null(), &mut out) },
            IciclesDspStatus::InvalidArgument as i32
        );
        assert_eq!(
            unsafe {
                icicles_dsp_analyze(signal.as_ptr(), signal.len(), 999, ptr::null(), &mut out)
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
            icicles_dsp_analyze(signal.as_ptr(), signal.len(), 44100, dir.as_ptr(), &mut out)
        };
        assert_eq!(status, IciclesDspStatus::ModelNotFound as i32);
        assert!(
            last_error().contains("mel_spectrogram.onnx"),
            "{}",
            last_error()
        );
        assert!(out.features.is_null());
    }
}
