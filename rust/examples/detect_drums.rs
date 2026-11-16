//! Detects the drum hits of a raw mono f32 little-endian PCM file and prints
//! hits per class in fixed windows, plus timings.
//!
//! ```sh
//! ffmpeg -i song.wav -ac 1 -ar 44100 -f f32le song.f32
//! cargo run --release --example detect_drums -- song.f32 [sample_rate] [window_s] [activations_out.f32]
//! ```
//!
//! With a fourth argument the raw activations (`frames × 5` f32) are written
//! there, for comparison with the PyTorch reference.

use std::path::PathBuf;
use std::time::Instant;

use beat_this::{RtenRuntime, Runtime};
use icicles_dsp::drums::{
    activations, hits, resample, resolve_model, DrumClass, FrontEnd, CLASS_COUNT,
};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .expect("usage: detect_drums <file.f32> [sample_rate] [window_s] [act_out]");
    let sample_rate: u32 = args.get(2).map_or(Ok(44_100), |s| s.parse())?;
    let window: f32 = args.get(3).map_or(Ok(7.5), |s| s.parse())?;
    let bytes = std::fs::read(path)?;
    let samples: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models");

    let t0 = Instant::now();
    let audio = resample(&samples, sample_rate).map_err(anyhow::Error::msg)?;
    let t1 = Instant::now();
    let front = FrontEnd::new();
    let (spec, linear) = front.spectrogram(&audio);
    let t2 = Instant::now();
    let mut model = RtenRuntime.load_model(&resolve_model(&model_dir)?)?;
    let act = activations(&mut model, &spec, front.bands())?;
    let t3 = Instant::now();
    let found = hits(&act, &linear, &front);
    let t4 = Instant::now();
    eprintln!(
        "resample {:?}, spectrogram {:?}, model {:?}, peaks {:?}; {} frames",
        t1 - t0,
        t2 - t1,
        t3 - t2,
        t4 - t3,
        act.len() / CLASS_COUNT
    );
    if let Some(out) = args.get(4) {
        let raw: Vec<u8> = act.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(out, raw)?;
    }

    let duration = audio.len() as f32 / 44_100.0;
    println!("     t | kick snare  toms hihat  cymb | mean strength kick");
    let mut start = 0.0f32;
    while start < duration {
        let end = start + window;
        let in_window = |c: DrumClass| {
            found
                .iter()
                .filter(|h| h.class == c && h.time >= start && h.time < end)
                .count()
        };
        let kicks: Vec<f32> = found
            .iter()
            .filter(|h| h.class == DrumClass::Kick && h.time >= start && h.time < end)
            .map(|h| h.strength)
            .collect();
        let mean = if kicks.is_empty() {
            0.0
        } else {
            kicks.iter().sum::<f32>() / kicks.len() as f32
        };
        println!(
            "{start:6.1} | {:4} {:5} {:5} {:5} {:5} | {mean:.2}",
            in_window(DrumClass::Kick),
            in_window(DrumClass::Snare),
            in_window(DrumClass::Toms),
            in_window(DrumClass::HiHat),
            in_window(DrumClass::Cymbals),
        );
        start = end;
    }
    let total = |c: DrumClass| found.iter().filter(|h| h.class == c).count();
    println!(
        "totals: kick {}, snare {}, toms {}, hihat {}, cymbals {}",
        total(DrumClass::Kick),
        total(DrumClass::Snare),
        total(DrumClass::Toms),
        total(DrumClass::HiHat),
        total(DrumClass::Cymbals)
    );
    Ok(())
}
