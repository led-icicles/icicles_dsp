//! Port of `AudioFeatureExtractor` from the Dart package `icicles_generators`
//! (`lib/src/audio/audio_feature_extractor.dart`).
//!
//! The output is the `AudioFeatures` matrix: one row of [`COLUMN_COUNT`]
//! values in `0..1` every 10 ms, row `k` describing the audio around
//! `k · 10 ms`. Columns, in order: level, bass, mid, treble, onset,
//! peakFrequency, band0…band15.
//!
//! Every arithmetic step mirrors the Dart implementation (f64 math, f32
//! storage, the same evaluation order) so that the two agree to float
//! round-off; only the FFT differs (rustfft instead of the hand-written
//! radix-2), which changes bins at the 1e-12 level.

use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

/// Number of columns of the feature matrix.
pub const COLUMN_COUNT: usize = 22;
/// Index of the first mel band column.
pub const FIRST_BAND: usize = 6;
/// Number of mel bands.
pub const BAND_COUNT: usize = 16;
/// Row hop in microseconds.
pub const HOP_MICROS: u32 = 10_000;

const LEVEL: usize = 0;
const BASS: usize = 1;
const MID: usize = 2;
const TREBLE: usize = 3;
const ONSET: usize = 4;
const PEAK_FREQUENCY: usize = 5;

/// Lowest mel filterbank frequency.
pub const MIN_BAND_FREQUENCY: f64 = 40.0;
/// Highest mel filterbank frequency.
pub const MAX_BAND_FREQUENCY: f64 = 16000.0;

const COMPRESSION: f64 = 1000.0;
const QUIET_AMPLITUDE: f64 = 0.003; // about −50 dBFS
const ONSET_FLOOR: f64 = 0.02;

/// The song-wide feature matrix: `rows × COLUMN_COUNT` values, row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioFeatures {
    pub rows: usize,
    pub data: Vec<f32>,
}

/// Why the extractor refused the input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeatureError {
    /// The sample rate is below 1000 Hz.
    SampleRateTooLow(u32),
}

impl std::fmt::Display for FeatureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FeatureError::SampleRateTooLow(rate) => {
                write!(f, "sample rate {rate} Hz is below the 1000 Hz minimum")
            }
        }
    }
}

impl std::error::Error for FeatureError {}

/// The FFT window for `sample_rate`: the power of two nearest to
/// `2048 · sample_rate / 44100` (2048 at 44.1 and 48 kHz).
pub fn window_size_for(sample_rate: u32) -> usize {
    let ideal = 2048.0 * f64::from(sample_rate) / 44100.0;
    let exponent = (ideal.ln() / std::f64::consts::LN_2)
        .round()
        .clamp(7.0, 15.0) as u32;
    1usize << exponent
}

/// Analyses mono PCM (`-1..1`) at `sample_rate` Hz into the feature matrix.
pub fn extract(samples: &[f32], sample_rate: u32) -> Result<AudioFeatures, FeatureError> {
    let mut extractor = Extractor::new(sample_rate)?;
    Ok(extractor.run(samples))
}

struct MelFilter {
    bins: Vec<usize>,
    weights: Vec<f64>,
}

/// FFT bin ranges `[from, to)` of the fixed frequency bands.
struct BinRanges {
    bass: (usize, usize),
    mid: (usize, usize),
    treble: (usize, usize),
    spectrum: (usize, usize),
}

impl BinRanges {
    fn new(sample_rate: u32, window_size: usize) -> Self {
        let half = window_size / 2;
        let bin = |hz: f64| -> usize {
            let raw = (hz * window_size as f64 / f64::from(sample_rate)).ceil() as isize;
            raw.clamp(1, half as isize) as usize
        };
        Self {
            bass: (bin(20.0), bin(250.0)),
            mid: (bin(250.0), bin(4000.0)),
            treble: (bin(4000.0), bin(16000.0)),
            spectrum: (bin(MIN_BAND_FREQUENCY), bin(MAX_BAND_FREQUENCY)),
        }
    }
}

struct Extractor {
    sample_rate: u32,
    window_size: usize,
    fft: Arc<dyn Fft<f64>>,
    window: Vec<f64>,
    window_sum: f64,
    buffer: Vec<Complex<f64>>,
    amplitudes: Vec<f64>,
    log_amplitudes: Vec<f64>,
    previous_log: Vec<f64>,
    has_previous: bool,
    ranges: BinRanges,
    filters: Vec<MelFilter>,
    compression_norm: f64,
}

impl Extractor {
    fn new(sample_rate: u32) -> Result<Self, FeatureError> {
        if sample_rate < 1000 {
            return Err(FeatureError::SampleRateTooLow(sample_rate));
        }
        let size = window_size_for(sample_rate);
        let mut window = vec![0.0; size];
        let mut sum = 0.0;
        for (i, w) in window.iter_mut().enumerate() {
            *w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / size as f64).cos();
            sum += *w;
        }
        let half = size / 2;
        Ok(Self {
            sample_rate,
            window_size: size,
            fft: FftPlanner::new().plan_fft_forward(size),
            window,
            window_sum: sum,
            buffer: vec![Complex::new(0.0, 0.0); size],
            amplitudes: vec![0.0; half],
            log_amplitudes: vec![0.0; half],
            previous_log: vec![0.0; half],
            has_previous: false,
            ranges: BinRanges::new(sample_rate, size),
            filters: mel_filters(sample_rate, size),
            compression_norm: (1.0 + COMPRESSION).ln(),
        })
    }

    fn run(&mut self, samples: &[f32]) -> AudioFeatures {
        let rate = self.sample_rate as usize;
        let frame_count = (samples.len() * 100 + rate - 1) / rate;
        let mut data = vec![0.0f32; frame_count * COLUMN_COUNT];
        for frame in 0..frame_count {
            self.compute_frame(
                frame,
                samples,
                &mut data[frame * COLUMN_COUNT..(frame + 1) * COLUMN_COUNT],
            );
        }
        normalize(&mut data, frame_count, self.compress(QUIET_AMPLITUDE));
        AudioFeatures {
            rows: frame_count,
            data,
        }
    }

    fn center(&self, frame: usize) -> isize {
        ((frame * self.sample_rate as usize + 50) / 100) as isize
    }

    fn window_start(&self, frame: usize) -> isize {
        self.center(frame) - (self.window_size / 2) as isize
    }

    fn compress(&self, amplitude: f64) -> f64 {
        (1.0 + COMPRESSION * amplitude).ln() / self.compression_norm
    }

    fn compute_frame(&mut self, frame: usize, samples: &[f32], row: &mut [f32]) {
        let start = self.window_start(frame);
        let mut weighted_squares = 0.0;
        for i in 0..self.window_size {
            let absolute = start + i as isize;
            let x = if absolute >= 0 && (absolute as usize) < samples.len() {
                f64::from(samples[absolute as usize])
            } else {
                0.0
            };
            let w = self.window[i];
            weighted_squares += w * x * x;
            self.buffer[i] = Complex::new(x * w, 0.0);
        }
        self.fft.process(&mut self.buffer);
        let scale = 2.0 / self.window_sum;
        let half = self.window_size / 2;
        for k in 0..half {
            let c = self.buffer[k];
            let a = (c.re * c.re + c.im * c.im).sqrt() * scale;
            self.amplitudes[k] = a;
            self.log_amplitudes[k] = self.compress(a);
        }

        row[LEVEL] = (weighted_squares / self.window_sum).sqrt() as f32;
        row[BASS] = self.compress(self.band_amplitude(self.ranges.bass)) as f32;
        row[MID] = self.compress(self.band_amplitude(self.ranges.mid)) as f32;
        row[TREBLE] = self.compress(self.band_amplitude(self.ranges.treble)) as f32;
        for (j, filter) in self.filters.iter().enumerate() {
            let mut power = 0.0;
            for (bin, weight) in filter.bins.iter().zip(&filter.weights) {
                let a = self.amplitudes[*bin];
                power += weight * a * a;
            }
            row[FIRST_BAND + j] = self.compress(power.sqrt()) as f32;
        }

        let mut flux = 0.0;
        let mut peak = 0.0;
        let mut peak_bin: Option<usize> = None;
        let (from, to) = self.ranges.spectrum;
        for k in from..to {
            if self.has_previous {
                let rise = self.log_amplitudes[k] - self.previous_log[k];
                if rise > 0.0 {
                    flux += rise;
                }
            }
            if self.amplitudes[k] > peak {
                peak = self.amplitudes[k];
                peak_bin = Some(k);
            }
        }
        row[ONSET] = if to > from {
            (flux / (to - from) as f64) as f32
        } else {
            0.0
        };
        row[PEAK_FREQUENCY] = match peak_bin {
            Some(bin) if peak >= 1e-4 => {
                log_position(bin as f64 * f64::from(self.sample_rate) / self.window_size as f64)
                    as f32
            }
            _ => 0.0,
        };
        self.previous_log.copy_from_slice(&self.log_amplitudes);
        self.has_previous = true;
    }

    fn band_amplitude(&self, (from, to): (usize, usize)) -> f64 {
        let mut power = 0.0;
        for k in from..to {
            power += self.amplitudes[k] * self.amplitudes[k];
        }
        power.sqrt()
    }
}

fn log_position(frequency: f64) -> f64 {
    let p = (frequency.ln() - MIN_BAND_FREQUENCY.ln())
        / (MAX_BAND_FREQUENCY.ln() - MIN_BAND_FREQUENCY.ln());
    p.clamp(0.0, 1.0)
}

fn normalize(data: &mut [f32], frame_count: usize, quiet: f64) {
    for c in 0..FIRST_BAND {
        if c == PEAK_FREQUENCY {
            continue;
        }
        let floor = match c {
            LEVEL => QUIET_AMPLITUDE,
            ONSET => ONSET_FLOOR,
            _ => quiet,
        };
        let values: Vec<f32> = (0..frame_count)
            .map(|i| data[i * COLUMN_COUNT + c])
            .collect();
        let reference = reference(&values, floor);
        for i in 0..frame_count {
            let index = i * COLUMN_COUNT + c;
            data[index] = (f64::from(data[index]) / reference).clamp(0.0, 1.0) as f32;
        }
    }
    let mut bands = Vec::with_capacity(frame_count * BAND_COUNT);
    for i in 0..frame_count {
        for b in 0..BAND_COUNT {
            bands.push(data[i * COLUMN_COUNT + FIRST_BAND + b]);
        }
    }
    let reference = reference(&bands, quiet);
    for i in 0..frame_count {
        for b in 0..BAND_COUNT {
            let index = i * COLUMN_COUNT + FIRST_BAND + b;
            data[index] = (f64::from(data[index]) / reference).clamp(0.0, 1.0) as f32;
        }
    }
}

/// The column's 95th percentile, at least a quarter of its maximum and at
/// least `floor` (the Dart `_reference`).
fn reference(values: &[f32], floor: f64) -> f64 {
    if values.is_empty() {
        return floor;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f32::total_cmp);
    let p95 = f64::from(sorted[((sorted.len() - 1) as f64 * 0.95).round() as usize]);
    let last = f64::from(*sorted.last().expect("non-empty"));
    p95.max(last * 0.25).max(floor)
}

fn mel_points() -> Vec<f64> {
    let mel = |f: f64| 2595.0 * (1.0 + f / 700.0).ln() / std::f64::consts::LN_10;
    let hz = |m: f64| 700.0 * (10f64.powf(m / 2595.0) - 1.0);
    let low = mel(MIN_BAND_FREQUENCY);
    let high = mel(MAX_BAND_FREQUENCY);
    let count = BAND_COUNT + 2;
    (0..count)
        .map(|i| hz(low + (high - low) * i as f64 / (count - 1) as f64))
        .collect()
}

/// Center frequencies (Hz) of the 16 mel bands.
pub fn band_centers() -> Vec<f64> {
    let points = mel_points();
    points[1..=BAND_COUNT].to_vec()
}

fn mel_filters(sample_rate: u32, window_size: usize) -> Vec<MelFilter> {
    let points = mel_points();
    let bin_hz = f64::from(sample_rate) / window_size as f64;
    let half = window_size / 2;
    let nyquist = f64::from(sample_rate) / 2.0;
    (0..BAND_COUNT)
        .map(|j| {
            mel_filter(
                points[j],
                points[j + 1],
                points[j + 2],
                bin_hz,
                half,
                nyquist,
            )
        })
        .collect()
}

fn mel_filter(
    low: f64,
    center: f64,
    high: f64,
    bin_hz: f64,
    half: usize,
    nyquist: f64,
) -> MelFilter {
    let mut bins = Vec::new();
    let mut weights = Vec::new();
    if center < nyquist {
        for k in 1..half {
            let f = k as f64 * bin_hz;
            if f <= low || f >= high {
                continue;
            }
            let w = if f <= center {
                (f - low) / (center - low)
            } else {
                (high - f) / (high - center)
            };
            if w > 0.0 {
                bins.push(k);
                weights.push(w);
            }
        }
        if bins.is_empty() {
            let nearest = ((center / bin_hz).round() as isize).clamp(1, half as isize - 1) as usize;
            bins.push(nearest);
            weights.push(1.0);
        }
    }
    MelFilter { bins, weights }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_sizes_match_dart() {
        assert_eq!(window_size_for(44100), 2048);
        assert_eq!(window_size_for(48000), 2048);
        assert_eq!(window_size_for(22050), 1024);
        assert_eq!(window_size_for(96000), 4096);
        assert_eq!(window_size_for(8000), 512);
        assert_eq!(window_size_for(1000), 128);
    }

    #[test]
    fn row_count_rounds_up_to_the_hop() {
        let features = extract(&vec![0.0; 44100 + 1], 44100).unwrap();
        assert_eq!(features.rows, 101);
        assert_eq!(features.data.len(), 101 * COLUMN_COUNT);
        assert!(features.data.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn empty_input_gives_no_rows() {
        let features = extract(&[], 48000).unwrap();
        assert_eq!(features.rows, 0);
        assert!(features.data.is_empty());
    }

    #[test]
    fn rejects_low_sample_rates() {
        assert_eq!(
            extract(&[0.0; 10], 999),
            Err(FeatureError::SampleRateTooLow(999))
        );
    }

    #[test]
    fn bands_cover_40_hz_to_16_khz() {
        let centers = band_centers();
        assert_eq!(centers.len(), BAND_COUNT);
        assert!(centers[0] > MIN_BAND_FREQUENCY && centers[0] < 200.0);
        assert!(centers[BAND_COUNT - 1] < MAX_BAND_FREQUENCY && centers[BAND_COUNT - 1] > 10_000.0);
    }

    #[test]
    fn a_sine_lights_the_matching_band() {
        let rate = 44100;
        let samples: Vec<f32> = (0..rate * 2)
            .map(|i| {
                (0.5 * (2.0 * std::f64::consts::PI * 440.0 * i as f64 / rate as f64).sin()) as f32
            })
            .collect();
        let features = extract(&samples, rate as u32).unwrap();
        let row = &features.data[100 * COLUMN_COUNT..101 * COLUMN_COUNT];
        assert!(row[MID] > row[BASS] && row[MID] > row[TREBLE], "{row:?}");
        assert!(row[LEVEL] > 0.9, "{row:?}");
        let expected = log_position(440.0) as f32;
        assert!((row[PEAK_FREQUENCY] - expected).abs() < 0.01, "{row:?}");
    }
}
