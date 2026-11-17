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
use crate::chords::{self, Chord, ChordError};
use crate::deep::{self, Control, DeepError, DeepModels};
use crate::drums::{self, DrumError, DrumHit};
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
    /// The deep analysis was cancelled through its control block.
    Cancelled = 6,
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
    /// Chords (with [`ICICLES_DSP_DETECT_CHORDS`]), sorted by time and never
    /// overlapping: start and end in seconds, the root's pitch class (C = 0)
    /// and the quality (0 min, 1 maj, 2 dim, 3 aug, 4 min6, 5 maj6, 6 min7,
    /// 7 minmaj7, 8 maj7, 9 7, 10 dim7, 11 hdim7, 12 sus2, 13 sus4),
    /// `chord_count` of each. Stretches without a chord are gaps.
    pub chord_starts: *mut f32,
    pub chord_ends: *mut f32,
    pub chord_roots: *mut u8,
    pub chord_qualities: *mut u8,
    pub chord_count: usize,
    /// Status of the chord recognition, as `drum_status`: a failure never
    /// fails the analysis.
    pub chord_status: i32,
    /// NUL-terminated message of a failed chord recognition, else NULL.
    pub chord_error: *mut c_char,
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
        chord_starts: ptr::null_mut(),
        chord_ends: ptr::null_mut(),
        chord_roots: ptr::null_mut(),
        chord_qualities: ptr::null_mut(),
        chord_count: 0,
        chord_status: 0,
        chord_error: ptr::null_mut(),
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
/// [`icicles_dsp_analyze`] flag: recognize the chords (ChordMini BTC).
pub const ICICLES_DSP_DETECT_CHORDS: u32 = 4;

/// Analyses `len` mono f32 samples (`-1..1`) at `sample_rate` Hz.
///
/// `model_dir` is a NUL-terminated path to the directory with the models:
/// with [`ICICLES_DSP_TRACK_BEATS`] in `flags` the Beat This! models
/// (`mel_spectrogram.onnx` plus `beat_this.onnx` or `beat_this_small.onnx`),
/// with [`ICICLES_DSP_DETECT_DRUMS`] the drum model (`adtof_frame_rnn.onnx`),
/// with [`ICICLES_DSP_DETECT_CHORDS`] the chord model (`btc_chords.onnx`).
/// A NULL `model_dir` computes the features only, whatever the flags. On
/// success `out` holds the result (release it with [`icicles_dsp_free`]); on
/// failure it is zeroed and the status names the cause, with details in
/// [`icicles_dsp_last_error`]. A failed drum detection or chord recognition
/// is not a failure: it is reported in `drum_status` / `drum_error` and
/// `chord_status` / `chord_error` of the result.
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
/// with [`ICICLES_DSP_TRACK_BEATS`], `drum` with [`ICICLES_DSP_DETECT_DRUMS`],
/// `chord` with [`ICICLES_DSP_DETECT_CHORDS`] (a NULL `drum` or `chord` fails
/// only that detection, reported in the result).
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
    chord: *const u8,
    chord_len: usize,
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
                    chord: bytes(chord, chord_len),
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
    let drums = match models.filter(|_| flags & ICICLES_DSP_DETECT_DRUMS != 0) {
        None => DrumArrays::new(Vec::new(), None),
        Some(models) => match drums::detect(signal, sample_rate, models) {
            Ok(hits) => DrumArrays::new(hits, None),
            Err(e) => DrumArrays::new(Vec::new(), Some(&e)),
        },
    };
    let chords = match models.filter(|_| flags & ICICLES_DSP_DETECT_CHORDS != 0) {
        None => ChordArrays::new(Vec::new(), None),
        Some(models) => match chords::detect(signal, sample_rate, models) {
            Ok(chords) => ChordArrays::new(chords, None),
            Err(e) => ChordArrays::new(Vec::new(), Some(&e)),
        },
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
        drum_times: drums.times,
        drum_classes: drums.classes,
        drum_strengths: drums.strengths,
        drum_count: drums.count,
        drum_status: drums.status,
        drum_error: drums.error,
        chord_starts: chords.starts,
        chord_ends: chords.ends,
        chord_roots: chords.roots,
        chord_qualities: chords.qualities,
        chord_count: chords.count,
        chord_status: chords.status,
        chord_error: chords.error,
    })
}

/// Chords as the C arrays of a result (see `IciclesDspResult::chord_starts`).
struct ChordArrays {
    starts: *mut f32,
    ends: *mut f32,
    roots: *mut u8,
    qualities: *mut u8,
    count: usize,
    status: i32,
    error: *mut c_char,
}

impl ChordArrays {
    /// The arrays of `chords`; with `error` the status and message of a
    /// failed recognition.
    fn new(chords: Vec<Chord>, error: Option<&ChordError>) -> Self {
        let (status, error) = match error {
            None => (IciclesDspStatus::Ok, ptr::null_mut()),
            Some(e) => {
                let status = match e {
                    ChordError::ModelNotFound(_) => IciclesDspStatus::ModelNotFound,
                    ChordError::ModelLoad(_) => IciclesDspStatus::ModelLoadFailed,
                    ChordError::Analysis(_) => IciclesDspStatus::AnalysisFailed,
                };
                let message = CString::new(e.to_string())
                    .unwrap_or_else(|_| CString::new("chord recognition failed").expect("literal"));
                (status, message.into_raw())
            }
        };
        let count = chords.len();
        let (starts, _) = give_away(chords.iter().map(|c| c.start).collect());
        let (ends, _) = give_away(chords.iter().map(|c| c.end).collect());
        let (roots, _) = give_away(chords.iter().map(|c| c.root).collect());
        let (qualities, _) = give_away(chords.iter().map(|c| c.quality).collect());
        ChordArrays {
            starts,
            ends,
            roots,
            qualities,
            count,
            status: status as i32,
            error,
        }
    }

    /// Releases the arrays of a result's chords.
    ///
    /// # Safety
    /// The pointers must come from `ChordArrays::new` (via `result`) and not
    /// have been freed.
    unsafe fn free(result: &IciclesDspResult) {
        take_back(result.chord_starts, result.chord_count);
        take_back(result.chord_ends, result.chord_count);
        take_back(result.chord_roots, result.chord_count);
        take_back(result.chord_qualities, result.chord_count);
        if !result.chord_error.is_null() {
            drop(CString::from_raw(result.chord_error));
        }
    }
}

/// Drum hits as the C arrays of a result (see `IciclesDspResult::drum_times`).
struct DrumArrays {
    times: *mut f32,
    classes: *mut u8,
    strengths: *mut f32,
    count: usize,
    status: i32,
    error: *mut c_char,
}

impl DrumArrays {
    /// The arrays of `hits`; with `error` the status and message of a failed
    /// detection.
    fn new(hits: Vec<DrumHit>, error: Option<&DrumError>) -> Self {
        let (status, error) = match error {
            None => (IciclesDspStatus::Ok, ptr::null_mut()),
            Some(e) => {
                let status = match e {
                    DrumError::ModelNotFound(_) => IciclesDspStatus::ModelNotFound,
                    DrumError::ModelLoad(_) => IciclesDspStatus::ModelLoadFailed,
                    DrumError::Analysis(_) => IciclesDspStatus::AnalysisFailed,
                };
                let message = CString::new(e.to_string())
                    .unwrap_or_else(|_| CString::new("drum detection failed").expect("literal"));
                (status, message.into_raw())
            }
        };
        let count = hits.len();
        let (times, _) = give_away(hits.iter().map(|h| h.time).collect());
        let (classes, _) = give_away(hits.iter().map(|h| h.class as u8).collect());
        let (strengths, _) = give_away(hits.iter().map(|h| h.strength).collect());
        DrumArrays {
            times,
            classes,
            strengths,
            count,
            status: status as i32,
            error,
        }
    }

    /// Releases arrays made by [`DrumArrays::new`].
    ///
    /// # Safety
    /// The pointers must come from `DrumArrays::new` (via a result) and not
    /// have been freed.
    unsafe fn free(
        times: *mut f32,
        classes: *mut u8,
        strengths: *mut f32,
        count: usize,
        error: *mut c_char,
    ) {
        take_back(times, count);
        take_back(classes, count);
        take_back(strengths, count);
        if !error.is_null() {
            drop(CString::from_raw(error));
        }
    }
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
        DrumArrays::free(
            result.drum_times,
            result.drum_classes,
            result.drum_strengths,
            result.drum_count,
            result.drum_error,
        );
        ChordArrays::free(&result);
        ptr::write(out, IciclesDspResult::EMPTY);
    }));
}

/// The deep analysis output (see `crate::deep::DeepAnalysis`). Arrays are
/// owned by the library: release them with [`icicles_dsp_free_deep`].
#[repr(C)]
#[derive(Debug)]
pub struct IciclesDspDeepResult {
    /// Sections covering the song: start and end in seconds, label index
    /// (0 start, 1 end, 2 intro, 3 outro, 4 break, 5 bridge, 6 inst, 7 solo,
    /// 8 verse, 9 chorus), `section_count` of each.
    pub section_starts: *mut f32,
    pub section_ends: *mut f32,
    pub section_labels: *mut u8,
    pub section_count: usize,
    /// Stem loudness `0..1`, row-major `stem_rows × 4` (drums, bass, other,
    /// vocals), one row every 10 ms (the feature matrix's rows).
    pub stem_levels: *mut f32,
    pub stem_rows: usize,
    /// Chroma of the harmonic stems, row-major `chroma_rows × 12` (C … B),
    /// 10 rows a second; each row sums to 1 (all zero in silence).
    pub chroma: *mut f32,
    pub chroma_rows: usize,
    /// Drum hits transcribed on the drum stem, sorted by time (as
    /// [`IciclesDspResult::drum_times`] … `drum_count`).
    pub drum_times: *mut f32,
    pub drum_classes: *mut u8,
    pub drum_strengths: *mut f32,
    pub drum_count: usize,
    /// Status of the drum detection: [`IciclesDspStatus::Ok`] when it ran;
    /// otherwise the cause (e.g. no drum model), with the message in
    /// `drum_error`. A failed drum detection never fails the deep analysis.
    pub drum_status: i32,
    /// NUL-terminated message of a failed drum detection, else NULL.
    pub drum_error: *mut c_char,
}

impl IciclesDspDeepResult {
    const EMPTY: IciclesDspDeepResult = IciclesDspDeepResult {
        section_starts: ptr::null_mut(),
        section_ends: ptr::null_mut(),
        section_labels: ptr::null_mut(),
        section_count: 0,
        stem_levels: ptr::null_mut(),
        stem_rows: 0,
        chroma: ptr::null_mut(),
        chroma_rows: 0,
        drum_times: ptr::null_mut(),
        drum_classes: ptr::null_mut(),
        drum_strengths: ptr::null_mut(),
        drum_count: 0,
        drum_status: 0,
        drum_error: ptr::null_mut(),
    };
}

/// Runs the deep analysis (Demucs stems, stem loudness, chroma, All-In-One
/// structure with `folds` of the 8 folds) of `len` mono samples at
/// `sample_rate`, with the drum hits of the drum stem. `model_dir` holds
/// `htdemucs_core.onnx` and `allin1_fold{0..folds}.onnx`, and optionally the
/// drum model `adtof_frame_rnn.onnx` (without it `drum_status` reports
/// [`IciclesDspStatus::ModelNotFound`]; the analysis still succeeds). `control` (may be NULL) receives the
/// progress as f32 bits `0..1` and is polled for a non-zero cancel flag;
/// cancelling returns [`IciclesDspStatus::Cancelled`].
///
/// # Safety
/// `samples` must point to `len` readable `f32`s, `model_dir` be a valid C
/// string, `control` NULL or valid for the whole call (another thread may
/// read/write it), `out` writable.
#[no_mangle]
pub unsafe extern "C" fn icicles_dsp_analyze_deep(
    samples: *const f32,
    len: usize,
    sample_rate: u32,
    model_dir: *const c_char,
    folds: u32,
    control: *const Control,
    out: *mut IciclesDspDeepResult,
) -> i32 {
    if out.is_null() {
        set_last_error("out is null");
        return IciclesDspStatus::InvalidArgument as i32;
    }
    ptr::write(out, IciclesDspDeepResult::EMPTY);
    finish_deep(
        catch_unwind(AssertUnwindSafe(|| {
            let signal = checked_signal(samples, len, sample_rate)?;
            if model_dir.is_null() {
                return Err((
                    IciclesDspStatus::InvalidArgument,
                    "model_dir is null".into(),
                ));
            }
            let dir = CStr::from_ptr(model_dir).to_str().map_err(|_| {
                (
                    IciclesDspStatus::InvalidArgument,
                    "model_dir is not UTF-8".to_string(),
                )
            })?;
            run_deep(
                signal,
                sample_rate,
                &DeepModels::Dir(Path::new(dir)),
                folds,
                control,
            )
        })),
        out,
    )
}

/// Like [`icicles_dsp_analyze_deep`] with the models as bytes (the browser):
/// the Demucs network, `fold_count` fold models (`folds[i]` of
/// `fold_lens[i]` bytes; all of them run) and the drum model (`drum`, NULL
/// when not provided: only the drum detection fails, in the result).
///
/// # Safety
/// As [`icicles_dsp_analyze_deep`]; `demucs` must point to `demucs_len`
/// bytes, `folds`/`fold_lens` to `fold_count` entries each pointing to its
/// length of bytes, `drum` be NULL or point to `drum_len` bytes.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn icicles_dsp_analyze_deep_models(
    samples: *const f32,
    len: usize,
    sample_rate: u32,
    demucs: *const u8,
    demucs_len: usize,
    folds: *const *const u8,
    fold_lens: *const usize,
    fold_count: usize,
    drum: *const u8,
    drum_len: usize,
    control: *const Control,
    out: *mut IciclesDspDeepResult,
) -> i32 {
    if out.is_null() {
        set_last_error("out is null");
        return IciclesDspStatus::InvalidArgument as i32;
    }
    ptr::write(out, IciclesDspDeepResult::EMPTY);
    finish_deep(
        catch_unwind(AssertUnwindSafe(|| {
            let signal = checked_signal(samples, len, sample_rate)?;
            if demucs.is_null()
                || demucs_len == 0
                || folds.is_null()
                || fold_lens.is_null()
                || fold_count == 0
            {
                return Err((
                    IciclesDspStatus::ModelNotFound,
                    "deep models missing".into(),
                ));
            }
            let fold_bytes: Vec<&[u8]> = (0..fold_count)
                .map(|i| std::slice::from_raw_parts(*folds.add(i), *fold_lens.add(i)))
                .collect();
            let models = DeepModels::Bytes {
                demucs: std::slice::from_raw_parts(demucs, demucs_len),
                folds: fold_bytes,
                drum: (!drum.is_null() && drum_len > 0)
                    .then(|| std::slice::from_raw_parts(drum, drum_len)),
            };
            run_deep(signal, sample_rate, &models, fold_count as u32, control)
        })),
        out,
    )
}

unsafe fn run_deep(
    signal: &[f32],
    sample_rate: u32,
    models: &DeepModels<'_>,
    folds: u32,
    control: *const Control,
) -> Result<IciclesDspDeepResult, Failure> {
    let fallback = Control::default();
    let control = if control.is_null() {
        &fallback
    } else {
        &*control
    };
    let analysis =
        deep::analyze(signal, sample_rate, models, folds as usize, control).map_err(|e| {
            let status = match e {
                DeepError::ModelNotFound(_) => IciclesDspStatus::ModelNotFound,
                DeepError::ModelLoad(_) => IciclesDspStatus::ModelLoadFailed,
                DeepError::Analysis(_) => IciclesDspStatus::AnalysisFailed,
                DeepError::Cancelled => IciclesDspStatus::Cancelled,
            };
            (status, e.to_string())
        })?;
    let section_count = analysis.segments.len();
    let (section_starts, _) = give_away(analysis.segments.iter().map(|s| s.start).collect());
    let (section_ends, _) = give_away(analysis.segments.iter().map(|s| s.end).collect());
    let (section_labels, _) = give_away(analysis.segments.iter().map(|s| s.label).collect());
    let (stem_levels, _) = give_away(analysis.stem_levels);
    let (chroma, _) = give_away(analysis.chroma);
    let drums = DrumArrays::new(analysis.drums, analysis.drum_error.as_ref());
    Ok(IciclesDspDeepResult {
        section_starts,
        section_ends,
        section_labels,
        section_count,
        stem_levels,
        stem_rows: analysis.stem_rows,
        chroma,
        chroma_rows: analysis.chroma_rows,
        drum_times: drums.times,
        drum_classes: drums.classes,
        drum_strengths: drums.strengths,
        drum_count: drums.count,
        drum_status: drums.status,
        drum_error: drums.error,
    })
}

type DeepOutcome = std::thread::Result<Result<IciclesDspDeepResult, Failure>>;

unsafe fn finish_deep(outcome: DeepOutcome, out: *mut IciclesDspDeepResult) -> i32 {
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

/// Releases the arrays of a deep result and zeroes it (NULL / zeroed: no-op).
///
/// # Safety
/// `out` must be NULL or point to a result written by a deep entry point.
#[no_mangle]
pub unsafe extern "C" fn icicles_dsp_free_deep(out: *mut IciclesDspDeepResult) {
    if out.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let result = ptr::read(out);
        take_back(result.section_starts, result.section_count);
        take_back(result.section_ends, result.section_count);
        take_back(result.section_labels, result.section_count);
        take_back(result.stem_levels, result.stem_rows * 4);
        take_back(result.chroma, result.chroma_rows * 12);
        DrumArrays::free(
            result.drum_times,
            result.drum_classes,
            result.drum_strengths,
            result.drum_count,
            result.drum_error,
        );
        ptr::write(out, IciclesDspDeepResult::EMPTY);
    }));
}

/// Size in bytes of the control block (`progress` f32 bits, `cancel` u32).
#[no_mangle]
pub extern "C" fn icicles_dsp_control_size() -> usize {
    std::mem::size_of::<Control>()
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

    /// A sustained C major triad with a C bass, 6 s.
    fn c_major() -> Vec<f32> {
        let sr = 44_100usize;
        (0..sr * 6)
            .map(|i| {
                let t = i as f64 / sr as f64;
                [65.406, 261.626, 329.628, 391.995]
                    .iter()
                    .flat_map(|hz| (1..5).map(move |h| (hz * f64::from(h), 0.2 / f64::from(h))))
                    .map(|(hz, gain)| gain * (2.0 * std::f64::consts::PI * hz * t).sin())
                    .sum::<f64>() as f32
            })
            .collect()
    }

    #[test]
    fn recognizes_chords_with_the_bundled_model() {
        let signal = c_major();
        let dir = CString::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../models")).unwrap();
        let mut out = IciclesDspResult::EMPTY;
        let status = unsafe {
            icicles_dsp_analyze(
                signal.as_ptr(),
                signal.len(),
                44100,
                dir.as_ptr(),
                ICICLES_DSP_DETECT_CHORDS,
                &mut out,
            )
        };
        assert_eq!(status, IciclesDspStatus::Ok as i32, "{}", last_error());
        assert_eq!(out.chord_status, IciclesDspStatus::Ok as i32);
        assert!(out.chord_error.is_null());
        assert_eq!(out.drum_count, 0, "drums were not requested");
        assert_eq!(out.chord_count, 1);
        let starts = unsafe { std::slice::from_raw_parts(out.chord_starts, out.chord_count) };
        let ends = unsafe { std::slice::from_raw_parts(out.chord_ends, out.chord_count) };
        let roots = unsafe { std::slice::from_raw_parts(out.chord_roots, out.chord_count) };
        let qualities = unsafe { std::slice::from_raw_parts(out.chord_qualities, out.chord_count) };
        assert_eq!((roots[0], qualities[0]), (0, 1), "C major");
        assert!(starts[0] < 0.5 && ends[0] > 5.5, "{starts:?} {ends:?}");
        unsafe { icicles_dsp_free(&mut out) };
        assert!(out.chord_starts.is_null());
        assert_eq!(out.chord_count, 0);
    }

    #[test]
    fn missing_chord_model_does_not_fail_the_analysis() {
        let signal = c_major();
        let dir = std::env::temp_dir().join("icicles_dsp_ffi_no_chord_model");
        std::fs::create_dir_all(&dir).unwrap();
        let dir = CString::new(dir.to_str().unwrap()).unwrap();
        let mut out = IciclesDspResult::EMPTY;
        let status = unsafe {
            icicles_dsp_analyze(
                signal.as_ptr(),
                signal.len(),
                44100,
                dir.as_ptr(),
                ICICLES_DSP_DETECT_CHORDS,
                &mut out,
            )
        };
        assert_eq!(status, IciclesDspStatus::Ok as i32, "{}", last_error());
        assert_eq!(out.chord_status, IciclesDspStatus::ModelNotFound as i32);
        assert_eq!(out.chord_count, 0);
        let message = unsafe { CStr::from_ptr(out.chord_error) }.to_string_lossy();
        assert!(message.contains("btc_chords.onnx"), "{message}");
        assert!(out.feature_rows > 0);
        unsafe { icicles_dsp_free(&mut out) };
        assert!(out.chord_error.is_null());

        // The same through the byte entry point, without chord bytes: no
        // beat models either, so the features alone come back.
        let status = unsafe {
            icicles_dsp_analyze_models(
                signal.as_ptr(),
                signal.len(),
                44100,
                ptr::null(),
                0,
                ptr::null(),
                0,
                ptr::null(),
                0,
                ptr::null(),
                0,
                ICICLES_DSP_DETECT_CHORDS,
                &mut out,
            )
        };
        assert_eq!(status, IciclesDspStatus::Ok as i32, "{}", last_error());
        assert_eq!(out.chord_count, 0);
        unsafe { icicles_dsp_free(&mut out) };
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
    #[test]
    fn deep_without_models_is_reported() {
        let samples = vec![0.0f32; 44_100];
        let dir = CString::new("/nonexistent").unwrap();
        let mut out = IciclesDspDeepResult::EMPTY;
        let status = unsafe {
            icicles_dsp_analyze_deep(
                samples.as_ptr(),
                samples.len(),
                44_100,
                dir.as_ptr(),
                8,
                ptr::null(),
                &mut out,
            )
        };
        assert_eq!(status, IciclesDspStatus::ModelNotFound as i32);
        assert!(last_error().contains("htdemucs_core.onnx"));
        unsafe { icicles_dsp_free_deep(&mut out) };
        assert_eq!(icicles_dsp_control_size(), 8);
    }

    #[test]
    fn deep_result_carries_the_drums_of_the_drum_stem() {
        let models = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../models"));
        let needed = [deep::DEMUCS_MODEL_FILE, "allin1_fold0.onnx"];
        if !needed.iter().all(|f| models.join(f).is_file()) {
            eprintln!("skipped: deep models missing (tool/download-deep-models.sh)");
            return;
        }
        // Kicks every 0.5 s from 0.25 s over a quiet tone, fading out.
        let sr = 44_100usize;
        let signal: Vec<f32> = (0..sr * 4)
            .map(|i| {
                let t = i as f64 / sr as f64;
                let fade = ((4.0 - t) / 0.4).clamp(0.0, 1.0);
                let tone = 0.1 * fade * (2.0 * std::f64::consts::PI * 220.0 * t).sin();
                let k = t - 0.25;
                let kick = if k < 0.0 || k >= 3.5 {
                    0.0
                } else {
                    let dt = k % 0.5;
                    0.9 * (-dt / 0.12).exp()
                        * (2.0
                            * std::f64::consts::PI
                            * (55.0 * dt + 3.0 * (1.0 - (-30.0 * dt).exp())))
                        .sin()
                };
                (tone + 0.8 * kick) as f32
            })
            .collect();
        let dir = CString::new(models.to_str().unwrap()).unwrap();
        let mut out = IciclesDspDeepResult::EMPTY;
        let status = unsafe {
            icicles_dsp_analyze_deep(
                signal.as_ptr(),
                signal.len(),
                44_100,
                dir.as_ptr(),
                1,
                ptr::null(),
                &mut out,
            )
        };
        assert_eq!(status, IciclesDspStatus::Ok as i32, "{}", last_error());
        if models.join(drums::DRUM_MODEL_FILE).is_file() {
            assert_eq!(out.drum_status, IciclesDspStatus::Ok as i32);
            assert!(out.drum_error.is_null());
            let classes = unsafe { std::slice::from_raw_parts(out.drum_classes, out.drum_count) };
            assert_eq!(
                classes.iter().filter(|c| **c == 0).count(),
                7,
                "{classes:?}"
            );
        }
        unsafe { icicles_dsp_free_deep(&mut out) };
        assert!(out.drum_times.is_null());

        // No drum model as bytes: the deep analysis succeeds without drums.
        let demucs = std::fs::read(models.join(deep::DEMUCS_MODEL_FILE)).unwrap();
        let fold = std::fs::read(models.join("allin1_fold0.onnx")).unwrap();
        let folds = [fold.as_ptr()];
        let fold_lens = [fold.len()];
        let status = unsafe {
            icicles_dsp_analyze_deep_models(
                signal.as_ptr(),
                signal.len(),
                44_100,
                demucs.as_ptr(),
                demucs.len(),
                folds.as_ptr(),
                fold_lens.as_ptr(),
                1,
                ptr::null(),
                0,
                ptr::null(),
                &mut out,
            )
        };
        assert_eq!(status, IciclesDspStatus::Ok as i32, "{}", last_error());
        assert!(out.stem_rows > 0);
        assert_eq!(out.drum_count, 0);
        assert_eq!(out.drum_status, IciclesDspStatus::ModelNotFound as i32);
        let message = unsafe { CStr::from_ptr(out.drum_error) }
            .to_string_lossy()
            .into_owned();
        assert!(message.contains(drums::DRUM_MODEL_FILE), "{message}");
        unsafe { icicles_dsp_free_deep(&mut out) };
        assert!(out.drum_error.is_null());
    }
}

/// Encodes `len` interleaved f32 samples (`channels` 1 or 2, `-1..1`) at
/// `sample_rate` Hz as an Ogg Vorbis file at Vorbis quality `quality`
/// (`-1..=10`). On success `*out` / `*out_len` hold the file (release it
/// with [`icicles_dsp_free_bytes`]); on failure they are NULL / 0 and the
/// status names the cause, with details in [`icicles_dsp_last_error`].
///
/// # Safety
/// `samples` must point to `len` readable `f32`s; `out` and `out_len` must
/// be writable.
#[no_mangle]
pub unsafe extern "C" fn icicles_dsp_encode_vorbis(
    samples: *const f32,
    len: usize,
    channels: u32,
    sample_rate: u32,
    quality: f32,
    out: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    if out.is_null() || out_len.is_null() {
        set_last_error("out is null");
        return IciclesDspStatus::InvalidArgument as i32;
    }
    ptr::write(out, ptr::null_mut());
    ptr::write(out_len, 0);
    if samples.is_null() || len == 0 {
        set_last_error("no samples");
        return IciclesDspStatus::InvalidArgument as i32;
    }
    let pcm = std::slice::from_raw_parts(samples, len);
    match catch_unwind(AssertUnwindSafe(|| {
        crate::encode::encode_vorbis(pcm, channels as usize, sample_rate, quality)
    })) {
        Ok(Ok(bytes)) => {
            let (pointer, length) = give_away(bytes);
            ptr::write(out, pointer);
            ptr::write(out_len, length);
            set_last_error("");
            IciclesDspStatus::Ok as i32
        }
        Ok(Err(error)) => {
            set_last_error(format!("{error:#}"));
            IciclesDspStatus::InvalidArgument as i32
        }
        Err(_) => {
            set_last_error("panic while encoding");
            IciclesDspStatus::Panic as i32
        }
    }
}

/// Decodes the audio file `bytes` (`len` bytes: WAV, AIFF, FLAC, MP3, Ogg
/// Vorbis, AAC/M4A) from `start` seconds for `length` seconds (`<= 0`: to
/// the end) and encodes it as an Ogg Vorbis file at quality `quality`; the
/// result and failures as in [`icicles_dsp_encode_vorbis`].
///
/// # Safety
/// `bytes` must point to `len` readable bytes; `out` and `out_len` must be
/// writable.
#[no_mangle]
pub unsafe extern "C" fn icicles_dsp_transcode_vorbis(
    bytes: *const u8,
    len: usize,
    start: f64,
    length: f64,
    quality: f32,
    out: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    if out.is_null() || out_len.is_null() {
        set_last_error("out is null");
        return IciclesDspStatus::InvalidArgument as i32;
    }
    ptr::write(out, ptr::null_mut());
    ptr::write(out_len, 0);
    if bytes.is_null() || len == 0 {
        set_last_error("no bytes");
        return IciclesDspStatus::InvalidArgument as i32;
    }
    let file = std::slice::from_raw_parts(bytes, len);
    match catch_unwind(AssertUnwindSafe(|| {
        crate::encode::transcode_vorbis(file, start, length, quality)
    })) {
        Ok(Ok(encoded)) => {
            let (pointer, length) = give_away(encoded);
            ptr::write(out, pointer);
            ptr::write(out_len, length);
            set_last_error("");
            IciclesDspStatus::Ok as i32
        }
        Ok(Err(error)) => {
            set_last_error(format!("{error:#}"));
            IciclesDspStatus::InvalidArgument as i32
        }
        Err(_) => {
            set_last_error("panic while transcoding");
            IciclesDspStatus::Panic as i32
        }
    }
}

/// Releases a byte array returned by [`icicles_dsp_encode_vorbis`] or
/// [`icicles_dsp_transcode_vorbis`].
///
/// # Safety
/// `pointer`/`len` must come from that call and not have been freed.
#[no_mangle]
pub unsafe extern "C" fn icicles_dsp_free_bytes(pointer: *mut u8, len: usize) {
    take_back(pointer, len);
}
