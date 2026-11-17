//! Chord recognition with the ChordMini BTC model (a bi-directional
//! transformer, Park et al. ISMIR 2019, retrained by ChordMini; MIT; ONNX
//! export by `tool/export_chords/export_chords.py`).
//!
//! Pipeline, mirroring the model's front end (`librosa.cqt` 0.10.1 as
//! ChordMini calls it):
//! 1. mono PCM brought to 22.05 kHz;
//! 2. a constant-Q transform of 144 bins, 24 per octave from C1, one column
//!    every 2048 samples (~93 ms): octave by octave from the top, each octave
//!    a 512-point STFT projected onto 24 wavelets, the signal halved in rate
//!    between octaves ([`Cqt`]);
//! 3. `ln(|x| + 1e-6)`, normalized with the checkpoint's mean and deviation;
//! 4. the network on windows of 108 columns (10 s) every 54, the logits of
//!    the windows covering a column averaged, then smoothed over time;
//! 5. the best class per column, merged into chords; chords shorter than
//!    half a second join a neighbour.
//!
//! The halving filter is a windowed sinc of this crate, not librosa's soxr
//! resampler, so the transform matches the reference closely (a few
//! hundredths in the log domain), not bit for bit.

use std::f64::consts::PI;
use std::path::{Path, PathBuf};

use beat_this::{Model, Tensor};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use crate::drums;
use crate::model::{ModelSource, OnnxModel};

/// File name of the chord model.
pub const CHORD_MODEL_FILE: &str = "btc_chords.onnx";

/// Sample rate of the front end.
pub const SAMPLE_RATE: u32 = 22_050;
/// Samples between feature columns.
pub const HOP: usize = 2048;
/// Constant-Q bins.
pub const BINS: usize = 144;
/// Bins per octave.
pub const BINS_PER_OCTAVE: usize = 24;
/// Feature columns the network takes at once (10 s).
pub const SEQ_LEN: usize = 108;
/// Columns between two windows.
pub const STRIDE: usize = 54;
/// Classes of the network: 12 roots × 14 qualities, unknown, no chord.
pub const CLASS_COUNT: usize = 170;
/// Chord qualities per root.
pub const QUALITY_COUNT: usize = 14;
/// Seconds between feature columns.
pub const FRAME_SECONDS: f64 = HOP as f64 / SAMPLE_RATE as f64;

/// The label of a column without a chord (silence, noise, drums alone, or a
/// chord outside the vocabulary).
pub const NO_CHORD: u8 = 255;

/// Normalization of the features (from the checkpoint).
const MEAN: f32 = -2.369_753_8;
const STD: f32 = 1.962_707_9;

/// Frequency of bin 0 (C1, `librosa.note_to_hz('C1')`).
const FMIN: f64 = 32.703_195_662_574_83;
/// Rate the transform's top octave is computed at.
const TOP_RATE: f64 = 11_025.0;
/// FFT size of every octave.
const N_FFT: usize = 512;
/// Columns of the smoothing kernel.
const KERNEL: usize = 9;
/// Windows per model run.
const BATCH: usize = 8;
/// Chords shorter than this many columns (~0.56 s) join a neighbour.
const MIN_FRAMES: usize = 6;
/// Taps of the halving filter.
const HALF_TAPS: usize = 127;

/// One chord of a song.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Chord {
    /// Seconds from the start of the audio.
    pub start: f32,
    pub end: f32,
    /// Pitch class of the root, C = 0.
    pub root: u8,
    /// Index into min, maj, dim, aug, min6, maj6, min7, minmaj7, maj7, 7,
    /// dim7, hdim7, sus2, sus4.
    pub quality: u8,
}

/// Why chord recognition failed.
#[derive(Debug, Clone)]
pub enum ChordError {
    ModelNotFound(PathBuf),
    ModelLoad(String),
    Analysis(String),
}

impl std::fmt::Display for ChordError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChordError::ModelNotFound(path) => {
                write!(f, "chord model not found: {}", path.display())
            }
            ChordError::ModelLoad(message) => write!(f, "chord model load failed: {message}"),
            ChordError::Analysis(message) => write!(f, "chord analysis failed: {message}"),
        }
    }
}

impl std::error::Error for ChordError {}

/// The chord model in `model_dir`.
pub fn resolve_model(model_dir: &Path) -> Result<PathBuf, ChordError> {
    let path = model_dir.join(CHORD_MODEL_FILE);
    if path.is_file() {
        Ok(path)
    } else {
        Err(ChordError::ModelNotFound(path))
    }
}

/// The chord model of `models`.
pub fn load_model(models: ModelSource<'_>) -> Result<OnnxModel, ChordError> {
    match models {
        ModelSource::Dir(dir) => OnnxModel::from_file(&resolve_model(dir)?)
            .map_err(|e| ChordError::ModelLoad(format!("{e:#}"))),
        ModelSource::Bytes { chord, .. } => load_model_bytes(chord),
    }
}

/// The chord model from its file's bytes; `None` is a missing model.
pub fn load_model_bytes(bytes: Option<&[u8]>) -> Result<OnnxModel, ChordError> {
    match bytes {
        Some(bytes) => {
            OnnxModel::from_bytes(bytes).map_err(|e| ChordError::ModelLoad(format!("{e:#}")))
        }
        None => Err(ChordError::ModelNotFound(PathBuf::from(CHORD_MODEL_FILE))),
    }
}

/// Recognizes the chords of mono PCM (`-1..1`) at `sample_rate` Hz with the
/// chord model of `models` (in the directory, or its bytes). Chords are
/// sorted by time and never overlap; stretches without a chord are gaps.
pub fn detect(
    samples: &[f32],
    sample_rate: u32,
    models: ModelSource<'_>,
) -> Result<Vec<Chord>, ChordError> {
    let mut model = load_model(models)?;
    let audio = resample(samples, sample_rate).map_err(ChordError::Analysis)?;
    detect_with(&mut model, &audio)
}

/// Recognizes the chords of mono PCM already at [`SAMPLE_RATE`] with a
/// loaded chord model.
pub fn detect_with<M: Model>(model: &mut M, audio: &[f32]) -> Result<Vec<Chord>, ChordError> {
    let features = Cqt::new().features(audio);
    let labels =
        frame_labels(model, &features).map_err(|e| ChordError::Analysis(format!("{e:#}")))?;
    Ok(chords(&labels, audio.len() as f64 / f64::from(SAMPLE_RATE)))
}

/// `samples` at 22.05 kHz: through 44.1 kHz (the drum front end's resampler,
/// a copy when already there) and one halving.
pub fn resample(samples: &[f32], sample_rate: u32) -> Result<Vec<f32>, String> {
    if sample_rate == SAMPLE_RATE || samples.is_empty() {
        return Ok(samples.to_vec());
    }
    let audio = drums::resample(samples, sample_rate)?;
    Ok(Halver::new().halve(&audio, 1.0))
}

/// Halves the sample rate: a Kaiser-windowed sinc low-pass at a quarter of
/// the rate (flat up to ~0.45 of the new Nyquist, where the transform reads
/// up to 0.38), every second output kept, without delay.
struct Halver {
    /// Non-zero taps as (offset from the output's input sample, weight).
    taps: Vec<(isize, f32)>,
}

impl Halver {
    fn new() -> Self {
        let center = (HALF_TAPS / 2) as isize;
        let beta = 10.0;
        let mut taps: Vec<(isize, f64)> = (0..HALF_TAPS as isize)
            .map(|i| {
                let n = (i - center) as f64;
                let sinc = if n == 0.0 {
                    0.5
                } else {
                    (PI * n / 2.0).sin() / (PI * n)
                };
                let x = n / center as f64;
                (
                    i - center,
                    sinc * bessel_i0(beta * (1.0 - x * x).max(0.0).sqrt()) / bessel_i0(beta),
                )
            })
            .collect();
        let sum: f64 = taps.iter().map(|(_, w)| w).sum();
        for (_, weight) in &mut taps {
            *weight /= sum;
        }
        Self {
            // A half-band filter: every second tap off the center is zero.
            taps: taps
                .into_iter()
                .filter(|(_, w)| w.abs() > 1e-12)
                .map(|(offset, w)| (offset, w as f32))
                .collect(),
        }
    }

    /// `ceil(len / 2)` samples, each multiplied by `gain`.
    fn halve(&self, input: &[f32], gain: f32) -> Vec<f32> {
        let len = input.len() as isize;
        (0..input.len().div_ceil(2))
            .map(|k| {
                let at = 2 * k as isize;
                let mut sum = 0.0f32;
                for &(offset, weight) in &self.taps {
                    let i = at + offset;
                    if i >= 0 && i < len {
                        sum += weight * input[i as usize];
                    }
                }
                sum * gain
            })
            .collect()
    }
}

/// The modified Bessel function of the first kind, order 0.
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    for k in 1..60 {
        term *= (x / (2.0 * k as f64)).powi(2);
        sum += term;
        if term < 1e-14 * sum {
            break;
        }
    }
    sum
}

/// The constant-Q front end: the wavelet basis of one octave in the
/// frequency domain (the same for every octave, since rate and frequencies
/// halve together) and the per-bin scale.
pub struct Cqt {
    /// Per bin of an octave: the non-zero (FFT bin, coefficient) pairs.
    basis: Vec<Vec<(usize, Complex<f32>)>>,
    /// `1 / sqrt(wavelet length)` per bin of the whole transform.
    scale: Vec<f32>,
    halver: Halver,
}

impl Default for Cqt {
    fn default() -> Self {
        Self::new()
    }
}

impl Cqt {
    pub fn new() -> Self {
        // Equal relative bandwidth of geometrically spaced bins.
        let ratio = 2f64.powf(2.0 / BINS_PER_OCTAVE as f64);
        let q = (ratio + 1.0) / (ratio - 1.0);
        let frequency = |bin: usize| FMIN * 2f64.powf(bin as f64 / BINS_PER_OCTAVE as f64);
        let length = |bin: usize| q * TOP_RATE / frequency(bin);
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(N_FFT);
        let basis = (BINS - BINS_PER_OCTAVE..BINS)
            .map(|bin| {
                let length = length(bin);
                // librosa: arange(-length // 2, length // 2), a periodic Hann
                // window over it, the sum of magnitudes normalized to 1,
                // centered in the FFT frame and weighted by length / n_fft.
                let from = (-length / 2.0).floor() as isize;
                let to = (length / 2.0).floor() as isize;
                let count = (to - from) as usize;
                let wavelet: Vec<Complex<f64>> = (0..count)
                    .map(|i| {
                        let n = (from + i as isize) as f64;
                        let window = 0.5 - 0.5 * (2.0 * PI * i as f64 / count as f64).cos();
                        Complex::from_polar(window, 2.0 * PI * frequency(bin) * n / TOP_RATE)
                    })
                    .collect();
                let norm: f64 = wavelet.iter().map(|c| c.norm()).sum();
                let mut frame = vec![Complex::new(0.0, 0.0); N_FFT];
                let pad = (N_FFT - count) / 2;
                for (i, value) in wavelet.iter().enumerate() {
                    frame[pad + i] = value / norm * (length / N_FFT as f64);
                }
                fft.process(&mut frame);
                frame.truncate(N_FFT / 2 + 1);
                // librosa sparsifies each row: the smallest coefficients
                // holding 1 % of the row's magnitude are dropped.
                let mut sorted: Vec<f64> = frame.iter().map(|c| c.norm()).collect();
                sorted.sort_by(f64::total_cmp);
                let total: f64 = sorted.iter().sum();
                let mut cumulative = 0.0;
                let mut threshold = *sorted.last().unwrap();
                for magnitude in &sorted {
                    cumulative += magnitude / total;
                    if cumulative >= 0.01 {
                        threshold = *magnitude;
                        break;
                    }
                }
                frame
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| c.norm() >= threshold)
                    .map(|(k, c)| (k, Complex::new(c.re as f32, c.im as f32)))
                    .collect()
            })
            .collect();
        Self {
            basis,
            scale: (0..BINS)
                .map(|bin| (1.0 / length(bin).sqrt()) as f32)
                .collect(),
            halver: Halver::new(),
        }
    }

    /// `[columns × BINS]` log-magnitudes of mono audio at [`SAMPLE_RATE`]:
    /// column `t` is centered on sample `t · HOP`, `1 + len / HOP` columns.
    pub fn features(&self, audio: &[f32]) -> Vec<f32> {
        if audio.is_empty() {
            return Vec::new();
        }
        let octaves = BINS / BINS_PER_OCTAVE;
        let columns = 1 + audio.len() / HOP;
        let mut out = vec![0.0f32; columns * BINS];
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);
        let mut scratch = vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];
        let mut frame = vec![Complex::new(0.0, 0.0); N_FFT];
        // librosa halves once before the top octave (its filters end well
        // below a quarter of the rate) and scales every halving by sqrt(2).
        let mut signal = self.halver.halve(audio, std::f32::consts::SQRT_2);
        let mut hop = HOP / 2;
        for octave in 0..octaves {
            // The basis is rescaled to compensate for the lower rate.
            let gain = 2f32.powf(octave as f32 / 2.0);
            let first_bin = BINS - BINS_PER_OCTAVE * (octave + 1);
            for column in 0..columns.min(1 + signal.len() / hop) {
                // A rectangular frame centered on the column, zeros outside.
                let start = (column * hop) as isize - (N_FFT / 2) as isize;
                for (i, value) in frame.iter_mut().enumerate() {
                    let at = start + i as isize;
                    let sample = if at >= 0 && (at as usize) < signal.len() {
                        signal[at as usize]
                    } else {
                        0.0
                    };
                    *value = Complex::new(sample, 0.0);
                }
                fft.process_with_scratch(&mut frame, &mut scratch);
                let row = &mut out[column * BINS..(column + 1) * BINS];
                for (k, filter) in self.basis.iter().enumerate() {
                    let mut sum = Complex::new(0.0f32, 0.0);
                    for &(bin, coefficient) in filter {
                        sum += coefficient * frame[bin];
                    }
                    let bin = first_bin + k;
                    row[bin] = (sum.norm() * gain * self.scale[bin] + 1e-6).ln();
                }
            }
            if octave + 1 < octaves {
                signal = self.halver.halve(&signal, std::f32::consts::SQRT_2);
                hop /= 2;
            }
        }
        out
    }
}

/// Runs the network over `features` (`[columns × BINS]`, not normalized) and
/// returns one label per column: the class `0..168` (root × 14 + quality) or
/// [`NO_CHORD`].
pub fn frame_labels<M: Model>(model: &mut M, features: &[f32]) -> anyhow::Result<Vec<u8>> {
    let logits = smooth(&frame_logits(model, features)?);
    Ok(logits
        .as_chunks::<CLASS_COUNT>()
        .0
        .iter()
        .map(|row| {
            let best = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map_or(CLASS_COUNT - 1, |(i, _)| i);
            if best < 12 * QUALITY_COUNT {
                best as u8
            } else {
                NO_CHORD
            }
        })
        .collect())
}

/// `[columns × CLASS_COUNT]`: the mean logits of the windows over a column.
/// The features are zero-padded to whole windows, as the model was trained.
fn frame_logits<M: Model>(model: &mut M, features: &[f32]) -> anyhow::Result<Vec<f32>> {
    let columns = features.len() / BINS;
    let padded = columns.div_ceil(SEQ_LEN) * SEQ_LEN;
    let mut input = vec![(0.0 - MEAN) / (STD + 1e-8); padded * BINS];
    for (to, from) in input.iter_mut().zip(features) {
        *to = (from - MEAN) / (STD + 1e-8);
    }
    let mut total = vec![0.0f32; columns * CLASS_COUNT];
    let mut count = vec![0.0f32; columns];
    if columns == 0 {
        return Ok(total);
    }
    let starts: Vec<usize> = (0..=(padded - SEQ_LEN) / STRIDE)
        .map(|i| i * STRIDE)
        .filter(|start| *start < columns)
        .collect();
    for batch in starts.chunks(BATCH) {
        let mut data = Vec::with_capacity(batch.len() * SEQ_LEN * BINS);
        for &start in batch {
            data.extend_from_slice(&input[start * BINS..(start + SEQ_LEN) * BINS]);
        }
        let tensor = Tensor {
            shape: vec![batch.len(), SEQ_LEN, BINS],
            data,
        };
        let result = model.run(&[("features", &tensor)])?;
        let logits = result
            .get("logits")
            .ok_or_else(|| anyhow::anyhow!("the chord model has no 'logits' output"))?;
        anyhow::ensure!(
            logits.shape == [batch.len(), SEQ_LEN, CLASS_COUNT],
            "unexpected logits shape {:?}",
            logits.shape
        );
        for (window, &start) in batch.iter().enumerate() {
            let valid = SEQ_LEN.min(columns - start);
            let from = window * SEQ_LEN * CLASS_COUNT;
            for (to, value) in total[start * CLASS_COUNT..(start + valid) * CLASS_COUNT]
                .iter_mut()
                .zip(&logits.data[from..from + valid * CLASS_COUNT])
            {
                *to += value;
            }
            for value in &mut count[start..start + valid] {
                *value += 1.0;
            }
        }
    }
    for (row, count) in total
        .as_chunks_mut::<CLASS_COUNT>()
        .0
        .iter_mut()
        .zip(&count)
    {
        for value in row {
            *value /= count;
        }
    }
    Ok(total)
}

/// A Gaussian of [`KERNEL`] columns (sigma 1.5) over time, edges replicated.
fn smooth(logits: &[f32]) -> Vec<f32> {
    let columns = logits.len() / CLASS_COUNT;
    let half = (KERNEL / 2) as isize;
    let sigma = KERNEL as f32 / 6.0;
    let mut kernel: Vec<f32> = (-half..=half)
        .map(|offset| (-0.5 * (offset as f32 / sigma).powi(2)).exp())
        .collect();
    let sum: f32 = kernel.iter().sum();
    for weight in &mut kernel {
        *weight /= sum;
    }
    let mut out = vec![0.0f32; logits.len()];
    for column in 0..columns {
        let row = &mut out[column * CLASS_COUNT..(column + 1) * CLASS_COUNT];
        for (i, weight) in kernel.iter().enumerate() {
            let source = (column as isize + i as isize - half).clamp(0, columns as isize - 1);
            let from = source as usize * CLASS_COUNT;
            for (to, value) in row.iter_mut().zip(&logits[from..from + CLASS_COUNT]) {
                *to += weight * value;
            }
        }
    }
    out
}

/// Merges column labels into chords over `duration` seconds of audio: a run
/// shorter than [`MIN_FRAMES`] joins the run before it (the first one the
/// run after it), equal neighbours become one chord, runs of [`NO_CHORD`]
/// are left out.
pub fn chords(labels: &[u8], duration: f64) -> Vec<Chord> {
    // (label, first column, columns)
    let mut runs: Vec<(u8, usize, usize)> = Vec::new();
    for (column, &label) in labels.iter().enumerate() {
        match runs.last_mut() {
            Some((last, _, count)) if *last == label => *count += 1,
            _ => runs.push((label, column, 1)),
        }
    }
    let mut merged: Vec<(u8, usize, usize)> = Vec::new();
    let mut pending = 0; // columns of a short first run, given to the next
    for (label, start, count) in runs {
        match merged.last_mut() {
            Some((_, _, last_count)) if count < MIN_FRAMES => *last_count += count,
            Some((last, _, last_count)) if *last == label => *last_count += count,
            None if count < MIN_FRAMES => pending += count,
            _ => {
                merged.push((label, start - pending, count + pending));
                pending = 0;
            }
        }
    }
    if merged.is_empty() && pending > 0 {
        // Nothing but one short stretch: keep its most frequent label.
        let mut counts = [0usize; 256];
        for &label in labels {
            counts[label as usize] += 1;
        }
        let label = (0..256).max_by_key(|&l| counts[l]).unwrap_or(255) as u8;
        merged.push((label, 0, pending));
    }
    merged
        .into_iter()
        .filter(|(label, _, _)| *label != NO_CHORD)
        .map(|(label, start, count)| Chord {
            start: (start as f64 * FRAME_SECONDS) as f32,
            end: (((start + count) as f64 * FRAME_SECONDS).min(duration)) as f32,
            root: label / QUALITY_COUNT as u8,
            quality: label % QUALITY_COUNT as u8,
        })
        .filter(|chord| chord.end > chord.start)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halving_keeps_low_tones_and_removes_high_ones() {
        let tone = |hz: f64| -> Vec<f32> {
            (0..8000)
                .map(|i| (2.0 * PI * hz * i as f64 / 44_100.0).sin() as f32)
                .collect()
        };
        let peak = |signal: &[f32]| {
            signal[500..signal.len() - 500]
                .iter()
                .fold(0.0f32, |a, b| a.max(b.abs()))
        };
        let halver = Halver::new();
        let low = halver.halve(&tone(4000.0), 1.0);
        assert_eq!(low.len(), 4000);
        assert!((peak(&low) - 1.0).abs() < 1e-3, "low tone {}", peak(&low));
        assert!(peak(&halver.halve(&tone(15_000.0), 1.0)) < 1e-3);
    }

    #[test]
    fn a_tone_lights_its_bin() {
        // A4 = bin 24 · log2(440 / C1) = 90.
        let audio: Vec<f32> = (0..SAMPLE_RATE as usize * 2)
            .map(|i| (2.0 * PI * 440.0 * i as f64 / f64::from(SAMPLE_RATE)).sin() as f32 * 0.5)
            .collect();
        let features = Cqt::new().features(&audio);
        assert_eq!(features.len(), (1 + audio.len() / HOP) * BINS);
        let row = &features[10 * BINS..11 * BINS];
        let best = (0..BINS)
            .max_by(|a, b| row[*a].total_cmp(&row[*b]))
            .unwrap();
        assert_eq!(best, 90);
    }

    #[test]
    fn short_runs_join_their_neighbours() {
        let mut labels = vec![1u8; 3]; // a short first run
        labels.extend([15u8; 20]);
        labels.extend([NO_CHORD; 2]); // a blip
        labels.extend([15u8; 10]);
        labels.extend([NO_CHORD; 12]);
        labels.extend([29u8; 8]);
        let result = chords(&labels, 100.0);
        assert_eq!(result.len(), 2);
        assert_eq!((result[0].root, result[0].quality), (1, 1));
        assert_eq!(result[0].start, 0.0);
        assert!((f64::from(result[0].end) - 35.0 * FRAME_SECONDS).abs() < 1e-4);
        assert_eq!((result[1].root, result[1].quality), (2, 1));
        assert!((f64::from(result[1].start) - 47.0 * FRAME_SECONDS).abs() < 1e-4);
    }

    #[test]
    fn nothing_in_nothing_out() {
        assert!(Cqt::new().features(&[]).is_empty());
        assert!(chords(&[], 0.0).is_empty());
        assert!(chords(&[NO_CHORD; 40], 4.0).is_empty());
    }
}
