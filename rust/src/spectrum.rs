//! madmom's `LogarithmicFilteredSpectrogram` as configured by All-In-One
//! (`allin1/spectrogram.py`): 44.1 kHz mono, frames of 2048 samples centered
//! on `k · 441` (100 fps, `ceil(len / 441)` frames, zeros outside), symmetric
//! Hann window (`numpy.hanning`), magnitudes of bins `0..1024`, a triangular
//! filterbank of 12 bands per octave between 30 Hz and 17 kHz aligned to
//! A440 (`log_frequencies`), on unique nearest bins (`frequencies2bins`),
//! each band normalized to sum 1, then `log10(1 + x)` — 81 bands.

use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

pub const SAMPLE_RATE: f64 = 44_100.0;
pub const FRAME_SIZE: usize = 2048;
pub const HOP: usize = 441;

/// A madmom log-filtered spectrogram front end.
pub struct LogSpectrogram {
    window: Vec<f32>,
    fft: Arc<dyn Fft<f32>>,
    /// `(first_bin, weights)` per band.
    filters: Vec<(usize, Vec<f32>)>,
}

impl LogSpectrogram {
    /// The All-In-One configuration (30 Hz – 17 kHz, 81 bands).
    pub fn allin1() -> Self {
        Self::new(12, 30.0, 17_000.0)
    }

    pub fn new(bands_per_octave: u32, fmin: f64, fmax: f64) -> Self {
        let window = (0..FRAME_SIZE)
            .map(|i| {
                (0.5 - 0.5
                    * (2.0 * std::f64::consts::PI * i as f64 / (FRAME_SIZE - 1) as f64).cos())
                    as f32
            })
            .collect();
        Self {
            window,
            fft: FftPlanner::new().plan_fft_forward(FRAME_SIZE),
            filters: filterbank(bands_per_octave, fmin, fmax),
        }
    }

    pub fn bands(&self) -> usize {
        self.filters.len()
    }

    /// `[frames × bands]` row-major.
    pub fn compute(&self, audio: &[f32]) -> Vec<f32> {
        let frames = audio.len().div_ceil(HOP);
        let bands = self.bands();
        let mut out = vec![0.0f32; frames * bands];
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
                out[t * bands + b] = (1.0 + value).log10();
            }
        }
        out
    }
}

/// madmom `log_frequencies(bands_per_octave, fmin, fmax, fref=440)`.
fn log_frequencies(bands_per_octave: u32, fmin: f64, fmax: f64) -> Vec<f64> {
    let bpo = f64::from(bands_per_octave);
    let left = ((fmin / 440.0).log2() * bpo).floor() as i64;
    let right = ((fmax / 440.0).log2() * bpo).ceil() as i64;
    (left..right)
        .map(|k| 440.0 * 2f64.powf(k as f64 / bpo))
        .filter(|&f| f >= fmin && f <= fmax)
        .collect()
}

/// madmom `frequencies2bins(..., unique_bins=True)`: the nearest FFT bin of
/// each frequency (ties to the upper bin), deduplicated and sorted.
fn frequencies_to_bins(frequencies: &[f64], bin_frequencies: &[f64]) -> Vec<usize> {
    let mut bins: Vec<usize> = frequencies
        .iter()
        .map(|&f| {
            let mut index = bin_frequencies.partition_point(|&b| b < f);
            index = index.clamp(1, bin_frequencies.len() - 1);
            let left = bin_frequencies[index - 1];
            let right = bin_frequencies[index];
            if f - left < right - f {
                index -= 1;
            }
            index
        })
        .collect();
    bins.sort_unstable();
    bins.dedup();
    bins
}

/// madmom `TriangularFilter.filters(bins, norm=True, overlap=True)` as
/// `(first_bin, weights)`.
fn filterbank(bands_per_octave: u32, fmin: f64, fmax: f64) -> Vec<(usize, Vec<f32>)> {
    let n_bins = FRAME_SIZE / 2;
    let bin_frequencies: Vec<f64> = (0..n_bins)
        .map(|k| k as f64 * SAMPLE_RATE / FRAME_SIZE as f64)
        .collect();
    let bins = frequencies_to_bins(
        &log_frequencies(bands_per_octave, fmin, fmax),
        &bin_frequencies,
    );
    let mut filters = Vec::new();
    for triple in bins.windows(3) {
        let (start, mut center, mut stop) = (triple[0], triple[1], triple[2]);
        if stop - start < 2 {
            center = start;
            stop = start + 1;
        }
        let rel_center = center - start;
        let rel_stop = stop - start;
        let mut data = vec![0.0f64; rel_stop];
        for (i, value) in data.iter_mut().enumerate().take(rel_center) {
            *value = i as f64 / rel_center as f64;
        }
        let falling = rel_stop - rel_center;
        for j in 0..falling {
            data[rel_center + j] = 1.0 - j as f64 / falling as f64;
        }
        let sum: f64 = data.iter().sum();
        let sum = if sum == 0.0 { 1.0 } else { sum };
        filters.push((start, data.iter().map(|w| (w / sum) as f32).collect()));
    }
    filters
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allin1_has_81_bands() {
        assert_eq!(LogSpectrogram::allin1().bands(), 81);
    }

    #[test]
    fn frames_round_up_to_the_hop() {
        let spec = LogSpectrogram::allin1();
        assert_eq!(spec.compute(&vec![0.0; 882]).len(), 2 * 81);
        assert_eq!(spec.compute(&vec![0.0; 883]).len(), 3 * 81);
    }
}
