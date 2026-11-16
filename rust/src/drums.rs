//! Drum hit detection with the ADTOF Frame_RNN model (Zehren et al.,
//! CC BY-NC-SA 4.0; ONNX export by `tool/export_adtof/export_adtof.py`).
//!
//! Pipeline, mirroring the model's training front end (madmom's
//! `LogarithmicFilteredSpectrogram`):
//! 1. mono PCM resampled to 44.1 kHz;
//! 2. STFT: symmetric Hann window of 2048, hop 441 (100 fps), frames centered
//!    on `k · hop` with zero padding, magnitudes of bins `0..1024`;
//! 3. a triangular filterbank of 12 bands per octave between 20 Hz and 20 kHz
//!    on unique FFT bins (84 bands, each normalized to sum 1), `log10(1 + x)`;
//! 4. the network, on overlapping windows (bounded memory), gives per-frame
//!    activations of kick, snare, toms, hi-hat and cymbals;
//! 5. madmom-style peak picking per class (moving-average threshold, local
//!    maximum, combined within 20 ms);
//! 6. a strength per hit from the activation and the loudness of the class's
//!    frequency band at the hit.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use beat_this::{Model, RtenRuntime, Runtime, Tensor};
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{
    Async, FixedAsync, Resampler, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

/// File name of the drum model.
pub const DRUM_MODEL_FILE: &str = "adtof_frame_rnn.onnx";

/// Sample rate of the front end.
pub const SAMPLE_RATE: u32 = 44_100;
/// FFT window length.
pub const FRAME_SIZE: usize = 2048;
/// Hop between frames (100 fps).
pub const HOP: usize = 441;
/// Activation frames per second.
pub const FPS: f64 = 100.0;
/// Number of drum classes.
pub const CLASS_COUNT: usize = 5;

/// The drum classes in the model's output order.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DrumClass {
    Kick = 0,
    Snare = 1,
    Toms = 2,
    HiHat = 3,
    Cymbals = 4,
}

impl DrumClass {
    pub const ALL: [DrumClass; CLASS_COUNT] = [
        DrumClass::Kick,
        DrumClass::Snare,
        DrumClass::Toms,
        DrumClass::HiHat,
        DrumClass::Cymbals,
    ];

    /// Peak-picking threshold of the class over the moving average of its
    /// activation (ADTOF's tuned Frame_RNN values).
    pub fn threshold(self) -> f32 {
        THRESHOLDS[self as usize]
    }

    /// The frequency band (Hz) whose loudness scales the strength of a hit.
    pub fn band(self) -> (f64, f64) {
        match self {
            DrumClass::Kick => (30.0, 150.0),
            DrumClass::Snare => (150.0, 5000.0),
            DrumClass::Toms => (80.0, 500.0),
            DrumClass::HiHat | DrumClass::Cymbals => (6000.0, 16000.0),
        }
    }
}

/// Per-class thresholds, see [`DrumClass::threshold`].
pub const THRESHOLDS: [f32; CLASS_COUNT] = [0.22, 0.24, 0.32, 0.22, 0.30];

/// One detected hit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrumHit {
    /// Onset time in seconds.
    pub time: f32,
    pub class: DrumClass,
    /// Strength `0..1` (1 = among the loudest hits of the class in the song).
    pub strength: f32,
}

/// Why drum detection failed.
#[derive(Debug)]
pub enum DrumError {
    ModelNotFound(PathBuf),
    ModelLoad(String),
    Analysis(String),
}

impl std::fmt::Display for DrumError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DrumError::ModelNotFound(path) => {
                write!(f, "drum model not found: {}", path.display())
            }
            DrumError::ModelLoad(message) => write!(f, "drum model load failed: {message}"),
            DrumError::Analysis(message) => write!(f, "drum analysis failed: {message}"),
        }
    }
}

impl std::error::Error for DrumError {}

/// The drum model in `model_dir`.
pub fn resolve_model(model_dir: &Path) -> Result<PathBuf, DrumError> {
    let path = model_dir.join(DRUM_MODEL_FILE);
    if path.is_file() {
        Ok(path)
    } else {
        Err(DrumError::ModelNotFound(path))
    }
}

/// Detects the drum hits of mono PCM (`-1..1`) at `sample_rate` Hz with the
/// model in `model_dir`. Hits are sorted by time.
pub fn detect(
    samples: &[f32],
    sample_rate: u32,
    model_dir: &Path,
) -> Result<Vec<DrumHit>, DrumError> {
    let path = resolve_model(model_dir)?;
    let mut model = RtenRuntime
        .load_model(&path)
        .map_err(|e| DrumError::ModelLoad(format!("{e:#}")))?;
    let audio = resample(samples, sample_rate).map_err(DrumError::Analysis)?;
    let front = FrontEnd::new();
    let (spec, linear) = front.spectrogram(&audio);
    let activations = activations(&mut model, &spec, front.bands())
        .map_err(|e| DrumError::Analysis(format!("{e:#}")))?;
    Ok(hits(&activations, &linear, &front))
}

/// `samples` at 44.1 kHz (a copy when already there).
pub fn resample(samples: &[f32], sample_rate: u32) -> Result<Vec<f32>, String> {
    if sample_rate == SAMPLE_RATE || samples.is_empty() {
        return Ok(samples.to_vec());
    }
    if sample_rate == 0 {
        return Err("sample rate 0".into());
    }
    let ratio = f64::from(SAMPLE_RATE) / f64::from(sample_rate);
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut resampler = Async::<f32>::new_sinc(ratio, 1.0, &params, 8192, 1, FixedAsync::Input)
        .map_err(|e| format!("resampler: {e}"))?;
    let mut out = vec![0.0f32; resampler.process_all_needed_output_len(samples.len())];
    let input = InterleavedSlice::new(samples, 1, samples.len()).map_err(|e| format!("{e:?}"))?;
    let out_len = out.len();
    let mut output =
        InterleavedSlice::new_mut(&mut out, 1, out_len).map_err(|e| format!("{e:?}"))?;
    let (_, written) = resampler
        .process_all_into_buffer(&input, &mut output, samples.len(), None)
        .map_err(|e| format!("resampler: {e}"))?;
    out.truncate(written);
    Ok(out)
}

/// The spectrogram front end (window, FFT plan, filterbank).
pub struct FrontEnd {
    window: Vec<f32>,
    fft: Arc<dyn Fft<f32>>,
    /// `(first_bin, weights)` of each band.
    filters: Vec<(usize, Vec<f32>)>,
    /// Center frequency (Hz) of each band.
    centers: Vec<f64>,
}

impl Default for FrontEnd {
    fn default() -> Self {
        Self::new()
    }
}

impl FrontEnd {
    pub fn new() -> Self {
        // numpy.hanning: symmetric.
        let window = (0..FRAME_SIZE)
            .map(|i| {
                (0.5 - 0.5
                    * (2.0 * std::f64::consts::PI * i as f64 / (FRAME_SIZE - 1) as f64).cos())
                    as f32
            })
            .collect();
        let fft = FftPlanner::new().plan_fft_forward(FRAME_SIZE);
        let (filters, centers) = filterbank();
        Self {
            window,
            fft,
            filters,
            centers,
        }
    }

    /// Number of filterbank bands (84).
    pub fn bands(&self) -> usize {
        self.filters.len()
    }

    /// Center frequencies of the bands in Hz.
    pub fn centers(&self) -> &[f64] {
        &self.centers
    }

    /// The `[frames × bands]` log spectrogram (the model input) and the same
    /// before the log (linear band magnitudes), for 44.1 kHz audio.
    pub fn spectrogram(&self, audio: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let frames = 1 + audio.len() / HOP;
        let bands = self.bands();
        let mut log = vec![0.0f32; frames * bands];
        let mut linear = vec![0.0f32; frames * bands];
        let mut buffer = vec![Complex::new(0.0f32, 0.0); FRAME_SIZE];
        let mut scratch = vec![Complex::new(0.0f32, 0.0); self.fft.get_inplace_scratch_len()];
        let mut magnitudes = vec![0.0f32; FRAME_SIZE / 2];
        let half = (FRAME_SIZE / 2) as isize;
        for t in 0..frames {
            let start = (t * HOP) as isize - half;
            for (i, slot) in buffer.iter_mut().enumerate() {
                let index = start + i as isize;
                let x = if index >= 0 && (index as usize) < audio.len() {
                    audio[index as usize]
                } else {
                    0.0
                };
                *slot = Complex::new(x * self.window[i], 0.0);
            }
            self.fft.process_with_scratch(&mut buffer, &mut scratch);
            for (k, m) in magnitudes.iter_mut().enumerate() {
                *m = buffer[k].norm();
            }
            for (b, (first, weights)) in self.filters.iter().enumerate() {
                let value: f32 = weights
                    .iter()
                    .zip(&magnitudes[*first..])
                    .map(|(w, m)| w * m)
                    .sum();
                linear[t * bands + b] = value;
                log[t * bands + b] = (1.0 + value).log10();
            }
        }
        (log, linear)
    }
}

/// The madmom-style log filterbank: `(first_bin, weights)` per band and the
/// band centers in Hz.
fn filterbank() -> (Vec<(usize, Vec<f32>)>, Vec<f64>) {
    let sr = f64::from(SAMPLE_RATE);
    let n_bins = FRAME_SIZE / 2;
    let bin_freq = |k: usize| k as f64 * sr / FRAME_SIZE as f64;
    let factor = 2f64.powf(1.0 / 12.0);
    let mut bins: Vec<usize> = Vec::new();
    let mut f = 20.0f64;
    while f <= 20000.0 * (1.0 + 1e-12) {
        // argmin |fft_freq − f|, first index on ties (numpy.argmin).
        let mut best = 0;
        let mut best_distance = f64::INFINITY;
        for k in 0..n_bins {
            let d = (bin_freq(k) - f).abs();
            if d < best_distance {
                best_distance = d;
                best = k;
            }
        }
        if bins.last().is_none_or(|&last| best > last) {
            bins.push(best);
        }
        f *= factor;
    }
    let mut filters = Vec::new();
    let mut centers = Vec::new();
    for i in 0..bins.len() - 2 {
        let (left, center, right) = (bins[i], bins[i + 1], bins[i + 2]);
        let mut weights = vec![0.0f64; n_bins];
        if right - left < 2 {
            weights[left] = 1.0;
        } else {
            for (b, w) in weights.iter_mut().enumerate().take(center).skip(left) {
                *w = (b - left) as f64 / (center - left) as f64;
            }
            weights[center] = 1.0;
            for (b, w) in weights
                .iter_mut()
                .enumerate()
                .take(right.min(n_bins))
                .skip(center + 1)
            {
                *w = (right - b) as f64 / (right - center) as f64;
            }
        }
        let sum: f64 = weights.iter().sum();
        let sum = if sum == 0.0 { 1.0 } else { sum };
        let first = weights.iter().position(|w| *w != 0.0).unwrap_or(0);
        let last = weights.iter().rposition(|w| *w != 0.0).unwrap_or(0);
        filters.push((
            first,
            weights[first..=last]
                .iter()
                .map(|w| (w / sum) as f32)
                .collect(),
        ));
        centers.push(bin_freq(center));
    }
    (filters, centers)
}

/// Frames of model context on each side of a window (bidirectional GRUs).
const CONTEXT_FRAMES: usize = 500;
/// Frames each window contributes.
const CORE_FRAMES: usize = 3000;

/// Runs the network over `spec` (`[frames × bands]`) in overlapping windows
/// (`CORE_FRAMES` kept per window, `CONTEXT_FRAMES` of context on each side)
/// and returns `[frames × CLASS_COUNT]` activations.
pub fn activations<M: Model>(
    model: &mut M,
    spec: &[f32],
    bands: usize,
) -> anyhow::Result<Vec<f32>> {
    let frames = spec.len() / bands;
    let mut out = vec![0.0f32; frames * CLASS_COUNT];
    let mut core_start = 0;
    while core_start < frames {
        let core_end = (core_start + CORE_FRAMES).min(frames);
        let from = core_start.saturating_sub(CONTEXT_FRAMES);
        let to = (core_end + CONTEXT_FRAMES).min(frames);
        let input = Tensor {
            shape: vec![1, to - from, bands],
            data: spec[from * bands..to * bands].to_vec(),
        };
        let result = model.run(&[("spectrogram", &input)])?;
        let act = result
            .get("activations")
            .ok_or_else(|| anyhow::anyhow!("the drum model has no 'activations' output"))?;
        anyhow::ensure!(
            act.shape == [1, to - from, CLASS_COUNT],
            "unexpected activation shape {:?}",
            act.shape
        );
        let skip = core_start - from;
        out[core_start * CLASS_COUNT..core_end * CLASS_COUNT].copy_from_slice(
            &act.data[skip * CLASS_COUNT..(skip + core_end - core_start) * CLASS_COUNT],
        );
        core_start = core_end;
    }
    Ok(out)
}

/// madmom `NotePeakPickingProcessor` on one activation curve at 100 fps:
/// a frame is a candidate when it reaches the moving average over
/// `[−100 ms, +10 ms]` plus `threshold`, and is a peak when it is the maximum
/// of the candidates over `[−20 ms, +10 ms]`; peaks within 20 ms of the
/// previous kept one are dropped. Returns frame indices.
pub fn pick_peaks(activation: &[f32], threshold: f32) -> Vec<usize> {
    const PRE_AVG: usize = 10;
    const POST_AVG: usize = 1;
    const PRE_MAX: usize = 2;
    const POST_MAX: usize = 1;
    const COMBINE: usize = 2;
    let n = activation.len();
    let mut detections = vec![0.0f32; n];
    for i in 0..n {
        let lo = i.saturating_sub(PRE_AVG);
        let hi = (i + POST_AVG + 1).min(n);
        let avg = activation[lo..hi].iter().sum::<f32>() / (hi - lo) as f32;
        if activation[i] >= avg + threshold {
            detections[i] = activation[i];
        }
    }
    let mut peaks: Vec<usize> = Vec::new();
    for i in 0..n {
        let value = detections[i];
        if value <= 0.0 {
            continue;
        }
        let lo = i.saturating_sub(PRE_MAX);
        let hi = (i + POST_MAX + 1).min(n);
        let max = detections[lo..hi].iter().copied().fold(0.0f32, f32::max);
        if value < max {
            continue;
        }
        if peaks.last().is_some_and(|&last| i - last <= COMBINE) {
            continue;
        }
        peaks.push(i);
    }
    peaks
}

/// Loudness range (dB below the class's loud hits) mapped onto strength 1..0.
const STRENGTH_RANGE_DB: f32 = 24.0;

/// Picks the hits of every class and gives each a strength: the loudness of
/// the class's band at the hit relative to the class's 90th-percentile hit
/// (`STRENGTH_RANGE_DB` below it = 0), weighted by the model's confidence.
pub fn hits(activations: &[f32], linear: &[f32], front: &FrontEnd) -> Vec<DrumHit> {
    let frames = activations.len() / CLASS_COUNT;
    let bands = front.bands();
    let mut all = Vec::new();
    for class in DrumClass::ALL {
        let c = class as usize;
        let curve: Vec<f32> = (0..frames)
            .map(|t| activations[t * CLASS_COUNT + c])
            .collect();
        let peaks = pick_peaks(&curve, class.threshold());
        if peaks.is_empty() {
            continue;
        }
        let (lo_hz, hi_hz) = class.band();
        let band_bins: Vec<usize> = front
            .centers()
            .iter()
            .enumerate()
            .filter(|(_, f)| **f >= lo_hz && **f <= hi_hz)
            .map(|(b, _)| b)
            .collect();
        let level_db = |t: usize| -> f32 {
            let energy: f32 = band_bins
                .iter()
                .map(|b| {
                    let v = linear[t * bands + b];
                    v * v
                })
                .sum();
            10.0 * (energy + 1e-9).log10()
        };
        // Loudest band level shortly after the onset (the attack).
        let levels: Vec<f32> = peaks
            .iter()
            .map(|&t| {
                (t..(t + 4).min(frames))
                    .map(level_db)
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .collect();
        let mut sorted = levels.clone();
        sorted.sort_by(f32::total_cmp);
        let reference = sorted[((sorted.len() - 1) as f32 * 0.9).round() as usize];
        for (&t, &level) in peaks.iter().zip(&levels) {
            let loudness = (1.0 + (level - reference) / STRENGTH_RANGE_DB).clamp(0.0, 1.0);
            let confidence = (curve[t] / 0.8).clamp(0.0, 1.0);
            let strength = (loudness * (0.5 + 0.5 * confidence)).clamp(0.05, 1.0);
            all.push(DrumHit {
                time: (t as f64 / FPS) as f32,
                class,
                strength,
            });
        }
    }
    all.sort_by(|a, b| {
        a.time
            .total_cmp(&b.time)
            .then((a.class as u8).cmp(&(b.class as u8)))
    });
    all
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filterbank_has_84_bands() {
        let front = FrontEnd::new();
        assert_eq!(front.bands(), 84);
        assert!(front.centers().windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn peaks_of_isolated_spikes() {
        let mut curve = vec![0.0f32; 100];
        curve[10] = 0.9;
        curve[11] = 0.5;
        curve[50] = 0.3;
        curve[51] = 0.1; // within 20 ms: combined
        curve[52] = 0.25;
        curve[80] = 0.15; // below threshold
        assert_eq!(pick_peaks(&curve, 0.2), vec![10, 50]);
    }

    #[test]
    fn resample_keeps_44100_and_converts_48000() {
        let x: Vec<f32> = (0..4800).map(|i| (i as f32 * 0.01).sin()).collect();
        assert_eq!(resample(&x, 44_100).unwrap().len(), 4800);
        let y = resample(&x, 48_000).unwrap();
        assert!((y.len() as i64 - 4410).abs() <= 2, "{}", y.len());
    }
}
