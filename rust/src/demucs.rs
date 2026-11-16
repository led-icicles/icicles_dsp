//! Source separation with Hybrid Transformer Demucs (`htdemucs`, Défossez
//! 2022, MIT): mix → drums, bass, other, vocals.
//!
//! The ONNX graph (`htdemucs_core.onnx`) is the network only; this module
//! reproduces what `demucs.apply.apply_model` and `HTDemucs._spec`/`_ispec`
//! do around it, so the stems match the PyTorch model (pinned by a fixture
//! test against `demucs` 4.1, `shifts=0`):
//! 1. the whole track is normalized by the mean/std (unbiased) of its mono mix;
//! 2. it is cut into segments of [`SEGMENT`] samples (7.8 s at 44.1 kHz) every
//!    `0.75 · SEGMENT`, a short last segment centered in a full window with
//!    the neighbouring samples as context and zeros past the end;
//! 3. per segment: the spectrogram (`_spec`: reflect padding to a multiple of
//!    the hop, `torch.stft` with a periodic Hann window, `normalized=True`,
//!    `center=True`, last bin and two frames on each side dropped) goes in
//!    with the waveform; the network returns per-stem spectrograms and
//!    waveforms, summed after an inverse STFT (`_ispec`);
//! 4. segments are cross-faded with triangular weights, then denormalized.
//!
//! The app analyses mono, so the mix enters as two identical channels and the
//! stems come back as the mean of their two channels.

use std::sync::Arc;

use beat_this::{Model, Tensor};
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

/// Sample rate of the model.
pub const SAMPLE_RATE: u32 = 44_100;
/// Samples per segment (`int(7.8 · 44100)`).
pub const SEGMENT: usize = 343_980;
/// STFT size and hop of the frequency branch.
pub const NFFT: usize = 4096;
pub const HOP: usize = 1024;
/// Frequency bins the network sees (the Nyquist bin is dropped).
pub const BINS: usize = NFFT / 2;
/// Spectrogram frames per segment (`ceil(SEGMENT / HOP)`).
pub const FRAMES: usize = SEGMENT.div_ceil(HOP);
/// Number of stems, in the model's order.
pub const STEM_COUNT: usize = 4;

/// The stems, in the model's output order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stem {
    Drums = 0,
    Bass = 1,
    Other = 2,
    Vocals = 3,
}

impl Stem {
    pub const ALL: [Stem; STEM_COUNT] = [Stem::Drums, Stem::Bass, Stem::Other, Stem::Vocals];
}

/// Mono stems at 44.1 kHz: `stems[s][i]`, `s` in [`Stem`] order.
pub struct Stems {
    pub stems: [Vec<f32>; STEM_COUNT],
}

impl Stems {
    pub fn get(&self, stem: Stem) -> &[f32] {
        &self.stems[stem as usize]
    }
}

/// STFT / inverse STFT of the frequency branch (`HTDemucs._spec`/`_ispec`).
pub struct Spectro {
    forward: Arc<dyn Fft<f32>>,
    inverse: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
}

impl Default for Spectro {
    fn default() -> Self {
        Self::new()
    }
}

/// numpy/torch `reflect` padding of `x` by `left`/`right` (no edge repeat;
/// pads shorter than `x`).
fn reflect_pad(x: &[f32], left: usize, right: usize) -> Vec<f32> {
    let n = x.len() as isize;
    let index = |i: isize| -> usize {
        // Reflect into 0..n without repeating the edges.
        let period = 2 * (n - 1).max(1);
        let mut j = i.rem_euclid(period);
        if j >= n {
            j = period - j;
        }
        j as usize
    };
    (-(left as isize)..n + right as isize)
        .map(|i| x[index(i)])
        .collect()
}

impl Spectro {
    pub fn new() -> Self {
        let mut planner = FftPlanner::new();
        // torch.hann_window: periodic.
        let window = (0..NFFT)
            .map(|i| {
                (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / NFFT as f64).cos()) as f32
            })
            .collect();
        Self {
            forward: planner.plan_fft_forward(NFFT),
            inverse: planner.plan_fft_inverse(NFFT),
            window,
        }
    }

    /// `_spec` of one segment channel: `[BINS × FRAMES]` complex, bin-major.
    pub fn spec(&self, x: &[f32]) -> Vec<Complex<f32>> {
        debug_assert_eq!(x.len(), SEGMENT);
        let pad = HOP / 2 * 3;
        let x = reflect_pad(x, pad, pad + FRAMES * HOP - x.len());
        let xp = reflect_pad(&x, NFFT / 2, NFFT / 2);
        let norm = 1.0 / (NFFT as f32).sqrt();
        let mut out = vec![Complex::new(0.0, 0.0); BINS * FRAMES];
        let mut buffer = vec![Complex::new(0.0f32, 0.0); NFFT];
        for t in 0..FRAMES {
            // Output frame t is torch frame t + 2.
            let start = (t + 2) * HOP;
            for (i, slot) in buffer.iter_mut().enumerate() {
                *slot = Complex::new(xp[start + i] * self.window[i], 0.0);
            }
            self.forward.process(&mut buffer);
            for f in 0..BINS {
                out[f * FRAMES + t] = buffer[f] * norm;
            }
        }
        out
    }

    /// `_ispec` of `[BINS × FRAMES]` complex (bin-major) to [`SEGMENT`] samples.
    pub fn ispec(&self, z: &[Complex<f32>]) -> Vec<f32> {
        debug_assert_eq!(z.len(), BINS * FRAMES);
        // Two zero frames on each side, a zero Nyquist bin.
        let frames = FRAMES + 4;
        let total = NFFT + HOP * (frames - 1);
        let mut ola = vec![0.0f32; total];
        let mut envelope = vec![0.0f32; total];
        let scale = (NFFT as f32).sqrt() / NFFT as f32; // normalized=True, 1/N of the inverse
        let mut buffer = vec![Complex::new(0.0f32, 0.0); NFFT];
        for j in 0..frames {
            let start = j * HOP;
            for (i, w) in self.window.iter().enumerate() {
                envelope[start + i] += w * w;
            }
            if !(2..FRAMES + 2).contains(&j) {
                continue;
            }
            let t = j - 2;
            buffer[0] = z[t];
            buffer[BINS] = Complex::new(0.0, 0.0);
            for f in 1..BINS {
                let value = z[f * FRAMES + t];
                buffer[f] = value;
                buffer[NFFT - f] = value.conj();
            }
            buffer[0].im = 0.0;
            self.inverse.process(&mut buffer);
            for (i, w) in self.window.iter().enumerate() {
                ola[start + i] += buffer[i].re * scale * w;
            }
        }
        let offset = NFFT / 2 + HOP / 2 * 3;
        (0..SEGMENT)
            .map(|s| {
                let i = offset + s;
                let e = envelope[i];
                if e > 1e-11 {
                    ola[i] / e
                } else {
                    ola[i]
                }
            })
            .collect()
    }
}

/// Separates mono 44.1 kHz `mix` into stems. `progress(done, total)` is
/// called after each segment; returning `false` from it cancels (the result
/// is then `Err("cancelled")`).
pub fn separate<M: Model>(
    model: &mut M,
    mix: &[f32],
    mut progress: impl FnMut(usize, usize) -> bool,
) -> anyhow::Result<Stems> {
    let n = mix.len();
    anyhow::ensure!(n > 0, "empty signal");
    let mean = mix.iter().map(|&x| f64::from(x)).sum::<f64>() / n as f64;
    let variance = mix
        .iter()
        .map(|&x| (f64::from(x) - mean).powi(2))
        .sum::<f64>()
        / (n.max(2) - 1) as f64;
    let std = variance.sqrt() + 1e-8;
    let normalized: Vec<f32> = mix
        .iter()
        .map(|&x| ((f64::from(x) - mean) / std) as f32)
        .collect();

    let stride = (0.75 * SEGMENT as f64) as usize;
    let weight: Vec<f32> = {
        let half = SEGMENT / 2;
        let raw: Vec<f32> = (1..=half)
            .chain((1..=SEGMENT - half).rev())
            .map(|v| v as f32)
            .collect();
        let max = raw.iter().cloned().fold(0.0f32, f32::max);
        raw.into_iter().map(|v| v / max).collect()
    };
    let offsets: Vec<usize> = (0..n).step_by(stride).collect();
    let spectro = Spectro::new();
    let mut acc: [Vec<f32>; STEM_COUNT] = std::array::from_fn(|_| vec![0.0f32; n]);
    let mut weights = vec![0.0f32; n];
    for (done, &offset) in offsets.iter().enumerate() {
        let length = (n - offset).min(SEGMENT);
        let delta = SEGMENT - length;
        let start = offset as isize - (delta / 2) as isize;
        let chunk: Vec<f32> = (0..SEGMENT as isize)
            .map(|i| {
                let k = start + i;
                if k >= 0 && (k as usize) < n {
                    normalized[k as usize]
                } else {
                    0.0
                }
            })
            .collect();
        let z = spectro.spec(&chunk);
        let mut stft = Vec::with_capacity(2 * BINS * FRAMES * 2);
        for _channel in 0..2 {
            for value in &z {
                stft.push(value.re);
                stft.push(value.im);
            }
        }
        let mut mix_in = Vec::with_capacity(2 * SEGMENT);
        mix_in.extend_from_slice(&chunk);
        mix_in.extend_from_slice(&chunk);
        let result = model.run(&[
            (
                "mix",
                &Tensor {
                    shape: vec![1, 2, SEGMENT],
                    data: mix_in,
                },
            ),
            (
                "stft",
                &Tensor {
                    shape: vec![1, 2, BINS, FRAMES, 2],
                    data: stft,
                },
            ),
        ])?;
        let complex = result
            .get("complex_stems")
            .ok_or_else(|| anyhow::anyhow!("demucs: no 'complex_stems' output"))?;
        let time = result
            .get("time_stems")
            .ok_or_else(|| anyhow::anyhow!("demucs: no 'time_stems' output"))?;
        anyhow::ensure!(
            complex.shape == [1, STEM_COUNT, 2, BINS, FRAMES, 2],
            "demucs: complex_stems shape {:?}",
            complex.shape
        );
        anyhow::ensure!(
            time.shape == [1, STEM_COUNT, 2, SEGMENT],
            "demucs: time_stems shape {:?}",
            time.shape
        );
        let trim = delta / 2;
        let per_channel = BINS * FRAMES * 2;
        #[allow(clippy::needless_range_loop)] // s indexes the model output too
        for s in 0..STEM_COUNT {
            // The inverse STFT is linear: average the channels first.
            let base = s * 2 * per_channel;
            let z: Vec<Complex<f32>> = (0..BINS * FRAMES)
                .map(|k| {
                    let a = base + 2 * k;
                    let b = base + per_channel + 2 * k;
                    Complex::new(
                        0.5 * (complex.data[a] + complex.data[b]),
                        0.5 * (complex.data[a + 1] + complex.data[b + 1]),
                    )
                })
                .collect();
            let wave = spectro.ispec(&z);
            let t0 = s * 2 * SEGMENT;
            let t1 = t0 + SEGMENT;
            let target = &mut acc[s];
            for i in 0..length {
                let k = trim + i;
                let value = wave[k] + 0.5 * (time.data[t0 + k] + time.data[t1 + k]);
                target[offset + i] += weight[i] * value;
            }
        }
        for i in 0..length {
            weights[offset + i] += weight[i];
        }
        if !progress(done + 1, offsets.len()) {
            anyhow::bail!("cancelled");
        }
    }
    let stems = acc.map(|mut stem| {
        for (value, w) in stem.iter_mut().zip(&weights) {
            *value = ((f64::from(*value / w)) * std + mean) as f32;
        }
        stem
    });
    Ok(Stems { stems })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflect_pad_matches_numpy() {
        let x = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(
            reflect_pad(&x, 2, 3),
            vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0]
        );
    }

    #[test]
    fn spec_then_ispec_reconstructs_the_signal() {
        let spectro = Spectro::new();
        let x: Vec<f32> = (0..SEGMENT)
            .map(|i| ((i as f32) * 0.01).sin() * 0.5 + ((i as f32) * 0.173).cos() * 0.1)
            .collect();
        let y = spectro.ispec(&spectro.spec(&x));
        // The dropped Nyquist bin and edge frames cost a little at the edges.
        let err: f32 = x[HOP * 4..SEGMENT - HOP * 4]
            .iter()
            .zip(&y[HOP * 4..SEGMENT - HOP * 4])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(err < 1e-3, "max error {err}");
    }
}
