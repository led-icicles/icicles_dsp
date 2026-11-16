//! The deep analysis: Demucs stems → per-stem loudness, harmonic chroma,
//! drum hits transcribed on the drum stem and the All-In-One song structure. Minutes of CPU per song, so it is a
//! separate pass from [`crate::ffi::icicles_dsp_analyze`], with progress and
//! cancellation through [`Control`].

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use crate::demucs::{self, Stem, STEM_COUNT};
use crate::drums::{self, DrumError, DrumHit};
use crate::model::OnnxModel;
use crate::structure::{self, Segment};

/// File name of the Demucs network.
pub const DEMUCS_MODEL_FILE: &str = "htdemucs_core.onnx";
/// Number of All-In-One folds shipped.
pub const FOLD_COUNT: usize = 8;

/// File name of All-In-One fold `k`.
pub fn fold_model_file(k: usize) -> String {
    format!("allin1_fold{k}.onnx")
}

// The drum stem goes to the drum model without resampling.
const _: () = assert!(demucs::SAMPLE_RATE == drums::SAMPLE_RATE);

/// Chroma rows per second.
pub const CHROMA_FPS: usize = 10;

/// Where the deep models come from. The drum model
/// ([`drums::DRUM_MODEL_FILE`]) is optional: without it the drums stay empty
/// and [`DeepAnalysis::drum_error`] says why.
#[derive(Debug, Clone)]
pub enum DeepModels<'a> {
    /// A directory with [`DEMUCS_MODEL_FILE`], the fold files and, when
    /// present, the drum model.
    Dir(&'a Path),
    /// The files' bytes (browser): Demucs, one entry per fold to run and the
    /// drum model (`None` when not provided).
    Bytes {
        demucs: &'a [u8],
        folds: Vec<&'a [u8]>,
        drum: Option<&'a [u8]>,
    },
}

/// Progress (`0..1`, f32 bits) and a cancel flag shared with another thread.
#[derive(Default)]
#[repr(C)]
pub struct Control {
    pub progress: AtomicU32,
    pub cancel: AtomicU32,
}

impl Control {
    fn set(&self, value: f32) {
        self.progress
            .store(value.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
    }

    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed) != 0
    }
}

/// The deep analysis of one song.
#[derive(Debug, Clone, Default)]
pub struct DeepAnalysis {
    /// Labelled sections covering the song.
    pub segments: Vec<Segment>,
    /// `rows × 4` stem loudness `0..1` (drums, bass, other, vocals), one row
    /// every 10 ms aligned with the feature matrix.
    pub stem_levels: Vec<f32>,
    pub stem_rows: usize,
    /// `rows × 12` chroma (C, C#, … B; each row sums to 1, or all zero in
    /// silence) of the harmonic stems, [`CHROMA_FPS`] rows a second.
    pub chroma: Vec<f32>,
    pub chroma_rows: usize,
    /// Drum hits transcribed (ADTOF) on the separated drum stem, sorted by
    /// time — cleaner than on the full mix, where loud bass and other
    /// instruments trigger false kicks. Empty when the drum detection failed.
    pub drums: Vec<DrumHit>,
    /// Why the drum detection failed (it never fails the deep analysis).
    pub drum_error: Option<DrumError>,
}

/// Why the deep analysis failed.
#[derive(Debug)]
pub enum DeepError {
    ModelNotFound(PathBuf),
    ModelLoad(String),
    Analysis(String),
    Cancelled,
}

impl std::fmt::Display for DeepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeepError::ModelNotFound(path) => write!(f, "model file not found: {}", path.display()),
            DeepError::ModelLoad(message) => write!(f, "model load failed: {message}"),
            DeepError::Analysis(message) => write!(f, "deep analysis failed: {message}"),
            DeepError::Cancelled => write!(f, "cancelled"),
        }
    }
}

impl std::error::Error for DeepError {}

/// Share of the progress bar spent separating (the rest: drums on the drum
/// stem, then structure).
const SEPARATION_SHARE: f32 = 0.55;

/// Runs the deep analysis of mono PCM (`-1..1`) at `sample_rate`, with up to
/// `folds` All-In-One folds (1..=8; fewer is faster and slightly worse).
pub fn analyze(
    samples: &[f32],
    sample_rate: u32,
    models: &DeepModels<'_>,
    folds: usize,
    control: &Control,
) -> Result<DeepAnalysis, DeepError> {
    let load = |e: anyhow::Error| DeepError::ModelLoad(format!("{e:#}"));
    let fail = |e: anyhow::Error| {
        if e.to_string() == "cancelled" {
            DeepError::Cancelled
        } else {
            DeepError::Analysis(format!("{e:#}"))
        }
    };
    let folds = folds.clamp(1, FOLD_COUNT);
    let fold_paths: Vec<PathBuf>;
    let mut demucs_model = match models {
        DeepModels::Dir(dir) => {
            let path = dir.join(DEMUCS_MODEL_FILE);
            if !path.is_file() {
                return Err(DeepError::ModelNotFound(path));
            }
            fold_paths = (0..folds).map(|k| dir.join(fold_model_file(k))).collect();
            if let Some(missing) = fold_paths.iter().find(|p| !p.is_file()) {
                return Err(DeepError::ModelNotFound(missing.clone()));
            }
            OnnxModel::from_file(&path).map_err(load)?
        }
        DeepModels::Bytes {
            demucs,
            folds: bytes,
            ..
        } => {
            fold_paths = Vec::new();
            if bytes.is_empty() {
                return Err(DeepError::ModelNotFound(PathBuf::from(fold_model_file(0))));
            }
            OnnxModel::from_bytes(demucs).map_err(load)?
        }
    };
    control.set(0.0);
    let mix = drums::resample(samples, sample_rate).map_err(DeepError::Analysis)?;
    let stems = demucs::separate(&mut demucs_model, &mix, |done, total| {
        control.set(SEPARATION_SHARE * done as f32 / total as f32);
        !control.cancelled()
    })
    .map_err(fail)?;
    drop(demucs_model);

    let rows = (samples.len() * 100).div_ceil(sample_rate as usize);
    let stem_levels = stem_levels(&mix, &stems, rows);
    let (chroma, chroma_rows) = chroma(&stems);
    if control.cancelled() {
        return Err(DeepError::Cancelled);
    }
    // The stems are at 44.1 kHz already: no resampling.
    let (drums, drum_error) = match drum_hits(models, stems.get(Stem::Drums)) {
        Ok(hits) => (hits, None),
        Err(e) => (Vec::new(), Some(e)),
    };

    let (features, frames) = structure::features(
        stems.get(Stem::Bass),
        stems.get(Stem::Drums),
        stems.get(Stem::Other),
        stems.get(Stem::Vocals),
    );
    drop(stems);
    let fold_total = match models {
        DeepModels::Dir(_) => fold_paths.len(),
        DeepModels::Bytes { folds: bytes, .. } => bytes.len().min(folds),
    };
    let mut sum = None;
    for k in 0..fold_total {
        if control.cancelled() {
            return Err(DeepError::Cancelled);
        }
        let mut model = match models {
            DeepModels::Dir(_) => OnnxModel::from_file(&fold_paths[k]).map_err(load)?,
            DeepModels::Bytes { folds: bytes, .. } => {
                OnnxModel::from_bytes(bytes[k]).map_err(load)?
            }
        };
        structure::run_fold(&mut model, &features, frames, &mut sum).map_err(fail)?;
        control
            .set(SEPARATION_SHARE + (1.0 - SEPARATION_SHARE) * (k + 1) as f32 / fold_total as f32);
    }
    let segments = sum
        .as_ref()
        .map(|logits| structure::segments(logits, fold_total))
        .unwrap_or_default();
    control.set(1.0);
    Ok(DeepAnalysis {
        segments,
        stem_levels,
        stem_rows: rows,
        chroma,
        chroma_rows,
        drums,
        drum_error,
    })
}

/// The drum hits of the drum stem (44.1 kHz) with the drum model of `models`.
fn drum_hits(models: &DeepModels<'_>, stem: &[f32]) -> Result<Vec<DrumHit>, DrumError> {
    let mut model = match models {
        DeepModels::Dir(dir) => drums::load_model_file(dir)?,
        DeepModels::Bytes { drum, .. } => drums::load_model_bytes(*drum)?,
    };
    drums::detect_with(&mut model, stem)
}

const LEVEL_WINDOW: usize = 2048;
const QUIET_AMPLITUDE: f64 = 0.003;

/// Hann-windowed RMS of `x` (44.1 kHz) around row `k · 10 ms`, like the
/// `level` column of the feature matrix.
fn windowed_rms(x: &[f32], rows: usize, window: &[f64], window_sum: f64) -> Vec<f64> {
    (0..rows)
        .map(|k| {
            let center = ((k * 44_100 + 50) / 100) as isize;
            let start = center - (LEVEL_WINDOW / 2) as isize;
            let mut sum = 0.0;
            for (i, w) in window.iter().enumerate() {
                let index = start + i as isize;
                if index >= 0 && (index as usize) < x.len() {
                    let v = f64::from(x[index as usize]);
                    sum += w * v * v;
                }
            }
            (sum / window_sum).sqrt()
        })
        .collect()
}

/// The 95th percentile, at least a quarter of the maximum and `floor`
/// (as the feature matrix's normalization).
fn reference(values: &[f64], floor: f64) -> f64 {
    if values.is_empty() {
        return floor;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let p95 = sorted[((sorted.len() - 1) as f64 * 0.95).round() as usize];
    p95.max(sorted[sorted.len() - 1] * 0.25).max(floor)
}

/// Per-stem loudness `0..1`, `rows × 4`. Each stem is normalized by its own
/// reference, but never by less than a quarter of the mix's — leakage in a
/// stem that is silent in the song stays near zero.
fn stem_levels(mix: &[f32], stems: &demucs::Stems, rows: usize) -> Vec<f32> {
    let window: Vec<f64> = (0..LEVEL_WINDOW)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / LEVEL_WINDOW as f64).cos())
        .collect();
    let window_sum: f64 = window.iter().sum();
    let mix_reference = reference(
        &windowed_rms(mix, rows, &window, window_sum),
        QUIET_AMPLITUDE,
    );
    let mut out = vec![0.0f32; rows * STEM_COUNT];
    for stem in Stem::ALL {
        let rms = windowed_rms(stems.get(stem), rows, &window, window_sum);
        let r = reference(&rms, QUIET_AMPLITUDE).max(0.25 * mix_reference);
        for (k, value) in rms.iter().enumerate() {
            out[k * STEM_COUNT + stem as usize] = (value / r).clamp(0.0, 1.0) as f32;
        }
    }
    out
}

const CHROMA_FFT: usize = 8192;
const CHROMA_MIN_HZ: f64 = 55.0;
const CHROMA_MAX_HZ: f64 = 2000.0;

/// Chroma of bass + other + vocals (no drums): magnitude spectra of 8192-sample
/// Hann frames every 100 ms, folded onto pitch classes (bins 55 Hz – 2 kHz,
/// nearest equal-tempered pitch, A440), each row normalized to sum 1.
fn chroma(stems: &demucs::Stems) -> (Vec<f32>, usize) {
    let n = stems.get(Stem::Bass).len();
    let hop = 44_100 / CHROMA_FPS;
    let rows = n.div_ceil(hop);
    let fft = FftPlanner::<f32>::new().plan_fft_forward(CHROMA_FFT);
    let window: Vec<f32> = (0..CHROMA_FFT)
        .map(|i| {
            (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / CHROMA_FFT as f64).cos()) as f32
        })
        .collect();
    let classes: Vec<Option<usize>> = (0..CHROMA_FFT / 2)
        .map(|k| {
            let hz = k as f64 * 44_100.0 / CHROMA_FFT as f64;
            if !(CHROMA_MIN_HZ..=CHROMA_MAX_HZ).contains(&hz) {
                return None;
            }
            let midi = (69.0 + 12.0 * (hz / 440.0).log2()).round() as i64;
            Some(midi.rem_euclid(12) as usize)
        })
        .collect();
    let harmonic = |i: usize| -> f32 {
        stems.get(Stem::Bass)[i] + stems.get(Stem::Other)[i] + stems.get(Stem::Vocals)[i]
    };
    let mut out = vec![0.0f32; rows * 12];
    let mut buffer = vec![Complex::new(0.0f32, 0.0); CHROMA_FFT];
    for r in 0..rows {
        let start = (r * hop) as isize - (CHROMA_FFT / 2) as isize;
        for (i, slot) in buffer.iter_mut().enumerate() {
            let index = start + i as isize;
            let x = if index >= 0 && (index as usize) < n {
                harmonic(index as usize)
            } else {
                0.0
            };
            *slot = Complex::new(x * window[i], 0.0);
        }
        fft.process(&mut buffer);
        let row = &mut out[r * 12..(r + 1) * 12];
        for (k, class) in classes.iter().enumerate() {
            if let Some(c) = class {
                row[*c] += buffer[k].norm();
            }
        }
        let total: f32 = row.iter().sum();
        if total > 1e-3 {
            for v in row.iter_mut() {
                *v /= total;
            }
        } else {
            row.fill(0.0);
        }
    }
    (out, rows)
}
