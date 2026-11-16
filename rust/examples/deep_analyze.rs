//! Runs the deep analysis on a 16-bit PCM WAV and prints the sections, the
//! timings and (with a third argument) dumps a JSON for comparison with the
//! Python reference (`tool/deep_reference/`).
//!
//! `cargo run --release --example deep_analyze -- <song.wav> <model_dir> [out.json] [folds]`
use std::time::Instant;

use icicles_dsp::deep::{self, Control, DeepModels};
use icicles_dsp::demucs;
use icicles_dsp::model::OnnxModel;
use icicles_dsp::structure;

fn read_wav(path: &str) -> (Vec<f32>, u32) {
    let bytes = std::fs::read(path).expect("read wav");
    let mut i = 12;
    let (mut channels, mut rate, mut bits) = (0u16, 0u32, 0u16);
    while i + 8 <= bytes.len() {
        let id = &bytes[i..i + 4];
        let size = u32::from_le_bytes(bytes[i + 4..i + 8].try_into().unwrap()) as usize;
        let body = &bytes[i + 8..(i + 8 + size).min(bytes.len())];
        if id == b"fmt " {
            channels = u16::from_le_bytes([body[2], body[3]]);
            rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
            bits = u16::from_le_bytes([body[14], body[15]]);
        } else if id == b"data" {
            assert_eq!(bits, 16, "16-bit PCM only");
            let frames = body.len() / (2 * channels as usize);
            let mono = (0..frames)
                .map(|f| {
                    (0..channels as usize)
                        .map(|c| {
                            let k = (f * channels as usize + c) * 2;
                            f32::from(i16::from_le_bytes([body[k], body[k + 1]])) / 32768.0
                        })
                        .sum::<f32>()
                        / f32::from(channels)
                })
                .collect();
            return (mono, rate);
        }
        i += 8 + size + (size & 1);
    }
    panic!("no data chunk");
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (samples, rate) = read_wav(&args[1]);
    let dir = std::path::Path::new(&args[2]);
    let folds = args
        .get(4)
        .map(|s| s.parse().unwrap())
        .unwrap_or(deep::FOLD_COUNT);
    println!("{} s at {rate} Hz", samples.len() as f32 / rate as f32);

    let t = Instant::now();
    let control = Control::default();
    let result = deep::analyze(&samples, rate, &DeepModels::Dir(dir), folds, &control)?;
    println!("deep analysis {:?} ({folds} folds)", t.elapsed());
    for s in &result.segments {
        println!(
            "{:7.2} {:7.2} {}",
            s.start,
            s.end,
            structure::LABELS[s.label as usize]
        );
    }

    if let Some(out) = args.get(3) {
        // Intermediate values for the comparison with the Python reference.
        let mut model = OnnxModel::from_file(&dir.join(deep::DEMUCS_MODEL_FILE))?;
        let t = Instant::now();
        let stems = demucs::separate(&mut model, &samples, |_, _| true)?;
        println!("separation alone {:?}", t.elapsed());
        let (features, frames) = structure::features(
            stems.get(demucs::Stem::Bass),
            stems.get(demucs::Stem::Drums),
            stems.get(demucs::Stem::Other),
            stems.get(demucs::Stem::Vocals),
        );
        let head: Vec<Vec<Vec<f32>>> = (0..4)
            .map(|s| {
                (0..200.min(frames))
                    .map(|t| features[(s * frames + t) * 81..(s * frames + t + 1) * 81].to_vec())
                    .collect()
            })
            .collect();
        let rms: Vec<Vec<f32>> = demucs::Stem::ALL
            .iter()
            .map(|&s| {
                stems
                    .get(s)
                    .chunks_exact(441)
                    .map(|c| (c.iter().map(|x| x * x).sum::<f32>() / 441.0).sqrt())
                    .collect()
            })
            .collect();
        let json = format!(
            "{{\"segments\":[{}],\"frames\":{frames},\"spec_head\":{:?},\"stem_rms\":{:?},\"chroma_rows\":{}}}",
            result
                .segments
                .iter()
                .map(|s| format!(
                    "{{\"start\":{},\"end\":{},\"label\":\"{}\"}}",
                    s.start,
                    s.end,
                    structure::LABELS[s.label as usize]
                ))
                .collect::<Vec<_>>()
                .join(","),
            head,
            rms,
            result.chroma_rows
        );
        std::fs::write(out, json)?;
    }
    Ok(())
}
