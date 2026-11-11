//! Synthetic test signals, defined identically in `tool/dump_features.dart`
//! (which dumps the Dart reference features into `tests/fixtures/`).
//! Change both together and regenerate the fixtures.
#![allow(dead_code)] // each test binary uses a subset

use std::f64::consts::PI;
use std::path::PathBuf;

/// A 32-bit linear congruential generator; `next()` is uniform in `-1..1`.
pub struct Lcg {
    state: u32,
}

impl Lcg {
    pub fn new() -> Self {
        Self { state: 0x1234_5678 }
    }

    pub fn next(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        f64::from(self.state >> 8) / f64::from(0x0100_0000) * 2.0 - 1.0
    }
}

impl Default for Lcg {
    fn default() -> Self {
        Self::new()
    }
}

fn sin(x: f64) -> f64 {
    (2.0 * PI * x).sin()
}

fn click(t: f64, period: f64, frequency: f64) -> f64 {
    let beat = (t / period).floor() * period;
    let dt = t - beat;
    if dt >= 0.01 {
        return 0.0;
    }
    0.8 * (-dt * 300.0).exp() * sin(frequency * dt)
}

pub struct SyntheticSignal {
    pub name: &'static str,
    pub sample_rate: u32,
    pub length: usize,
    sample: fn(f64, &mut Lcg) -> f64,
}

impl SyntheticSignal {
    pub fn generate(&self) -> Vec<f32> {
        let mut noise = Lcg::new();
        (0..self.length)
            .map(|i| (self.sample)(i as f64 / f64::from(self.sample_rate), &mut noise) as f32)
            .collect()
    }
}

pub fn signals() -> Vec<SyntheticSignal> {
    vec![
        SyntheticSignal {
            name: "tones_44100",
            sample_rate: 44100,
            length: 88641,
            sample: |t, noise| {
                let env = 0.5 + 0.5 * sin(0.5 * t);
                let tones = 0.4 * sin(110.0 * t) + 0.25 * sin(440.0 * t) + 0.1 * sin(3000.0 * t);
                env * tones + click(t, 0.5, 1000.0) + 0.01 * noise.next()
            },
        },
        SyntheticSignal {
            name: "chirp_48000",
            sample_rate: 48000,
            length: 72000,
            sample: |t, noise| {
                let (f0, f1, duration) = (100.0, 8000.0, 1.5);
                let phase = f0 * t + (f1 - f0) * t * t / (2.0 * duration);
                0.6 * sin(phase) + 0.005 * noise.next()
            },
        },
        SyntheticSignal {
            name: "bursts_22050",
            sample_rate: 22050,
            length: 26460,
            sample: |t, noise| {
                let amplitude = if ((t * 4.0).floor() as i64) % 2 == 0 {
                    0.7
                } else {
                    0.02
                };
                amplitude * noise.next()
            },
        },
        SyntheticSignal {
            name: "quiet_44100",
            sample_rate: 44100,
            length: 22050,
            sample: |_, noise| 0.0005 * noise.next(),
        },
        SyntheticSignal {
            name: "silence_44100",
            sample_rate: 44100,
            length: 13230,
            sample: |_, _| 0.0,
        },
    ]
}

/// A click track: `seconds` of 1 kHz clicks (10 ms, exponential decay) at
/// `bpm`, the first click at `t = 0`, over faint noise.
pub fn click_track(bpm: f64, seconds: f64, sample_rate: u32) -> Vec<f32> {
    let period = 60.0 / bpm;
    let mut noise = Lcg::new();
    let length = (seconds * f64::from(sample_rate)) as usize;
    (0..length)
        .map(|i| {
            let t = i as f64 / f64::from(sample_rate);
            (click(t, period, 1000.0) + 0.002 * noise.next()) as f32
        })
        .collect()
}

/// The package's `models/` directory (the bundled small model).
pub fn model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("models")
}
