//! Evaluates drum detection on an annotated dataset — RBMA13 (Vogl et al.
//! 2017: electronic music) or MDB Drums (Southall et al. 2017: MedleyDB
//! rock/pop/jazz) — for kick / snare / hi-hat and sweeps the peak-picking
//! threshold per class. See `tool/eval-rbma13.sh` and `tool/eval-mdb.sh` for
//! preparing the data.
//!
//! ```sh
//! cargo run --release --example eval_rbma -- <audio_dir> <annotation_dir>
//! ```
//!
//! `audio_dir` holds `<name>.f32` (mono f32 little-endian, 44.1 kHz) for
//! every `<name>.txt` in `annotation_dir` (`<seconds> <class>` per line; the
//! class is RBMA13's number — 0 kick, 1 snare, 2 hi-hat — or MDB Drums'
//! label `KD`/`SD`/`HH`; other classes are ignored). Activations are cached
//! next to the audio as `.act` (delete them after changing the model or the
//! front end). The F-measure sums hits over all tracks (onset tolerance
//! 50 ms, one-to-one matching), as in the drum transcription literature.
//!
//! Thresholds tuned on the whole set overstate the gain (they are fitted to
//! the test data), so the tool also runs leave-one-track-out cross-validation:
//! each track is scored with the per-class threshold tuned on the other
//! tracks — the honest estimate of what retuning would bring.

use std::path::{Path, PathBuf};

use beat_this::{RtenRuntime, Runtime};
use icicles_dsp::drums::{
    activations, pick_peaks, resolve_model, DrumClass, FrontEnd, CLASS_COUNT, FPS,
};

const TOLERANCE: f64 = 0.05;
/// The evaluated model classes.
const CLASSES: [DrumClass; 3] = [DrumClass::Kick, DrumClass::Snare, DrumClass::HiHat];

/// Annotation label (RBMA13 number or MDB Drums label) -> slot in [CLASSES].
fn slot_of(label: &str) -> Option<usize> {
    match label {
        "0" | "KD" => Some(0),
        "1" | "SD" => Some(1),
        "2" | "HH" => Some(2),
        _ => None,
    }
}

struct Track {
    name: String,
    activations: Vec<f32>,
    truth: Vec<Vec<f64>>, // per CLASSES entry
}

fn load_activations(audio: &Path, model_dir: &Path) -> anyhow::Result<Vec<f32>> {
    let cache = audio.with_extension("act");
    if cache.is_file() {
        let bytes = std::fs::read(&cache)?;
        return Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect());
    }
    let bytes = std::fs::read(audio)?;
    let samples: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    let front = FrontEnd::new();
    let (spec, _) = front.spectrogram(&samples);
    let mut model = RtenRuntime.load_model(&resolve_model(model_dir)?)?;
    let act = activations(&mut model, &spec, front.bands())?;
    std::fs::write(
        &cache,
        act.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
    )?;
    Ok(act)
}

/// (true positives, false positives, false negatives) with greedy one-to-one
/// matching of sorted lists within the tolerance.
fn count(detected: &[f64], truth: &[f64]) -> (usize, usize, usize) {
    let mut used = vec![false; truth.len()];
    let mut tp = 0;
    let mut j0 = 0;
    for &d in detected {
        while j0 < truth.len() && truth[j0] < d - TOLERANCE {
            j0 += 1;
        }
        let mut best: Option<usize> = None;
        let mut j = j0;
        while j < truth.len() && truth[j] <= d + TOLERANCE {
            if !used[j] && best.is_none_or(|b| (truth[j] - d).abs() < (truth[b] - d).abs()) {
                best = Some(j);
            }
            j += 1;
        }
        if let Some(b) = best {
            used[b] = true;
            tp += 1;
        }
    }
    (tp, detected.len() - tp, truth.len() - tp)
}

fn f_measure(tp: usize, fp: usize, fne: usize) -> f64 {
    if tp == 0 {
        return 0.0;
    }
    2.0 * tp as f64 / (2 * tp + fp + fne) as f64
}

fn evaluate(tracks: &[Track], slot: usize, threshold: f32) -> (usize, usize, usize) {
    let class = CLASSES[slot] as usize;
    let mut total = (0, 0, 0);
    for track in tracks {
        let frames = track.activations.len() / CLASS_COUNT;
        let curve: Vec<f32> = (0..frames)
            .map(|t| track.activations[t * CLASS_COUNT + class])
            .collect();
        let detected: Vec<f64> = pick_peaks(&curve, threshold)
            .iter()
            .map(|&t| t as f64 / FPS)
            .collect();
        let (tp, fp, fne) = count(&detected, &track.truth[slot]);
        total = (total.0 + tp, total.1 + fp, total.2 + fne);
    }
    total
}

/// The threshold (0.01..0.60) with the best F over [tracks] for [slot]
/// (ties keep the lower one).
fn best_threshold(tracks: &[&Track], slot: usize) -> f32 {
    let mut best = (0.01, -1.0);
    for step in 1..=60 {
        let threshold = step as f32 * 0.01;
        let (tp, fp, fne) = evaluate_refs(tracks, slot, threshold);
        let f = f_measure(tp, fp, fne);
        if f > best.1 + 1e-9 {
            best = (threshold, f);
        }
    }
    best.0
}

fn evaluate_refs(tracks: &[&Track], slot: usize, threshold: f32) -> (usize, usize, usize) {
    tracks
        .iter()
        .map(|t| evaluate(std::slice::from_ref(*t), slot, threshold))
        .fold((0, 0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2))
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let audio_dir = PathBuf::from(
        args.get(1)
            .expect("usage: eval_rbma <audio_dir> <annotation_dir>"),
    );
    let annotation_dir = PathBuf::from(
        args.get(2)
            .expect("usage: eval_rbma <audio_dir> <annotation_dir>"),
    );
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models");

    let mut tracks = Vec::new();
    let mut names: Vec<PathBuf> = std::fs::read_dir(&annotation_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "txt"))
        .collect();
    names.sort();
    for annotation in names {
        let stem = annotation
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();
        let audio = audio_dir.join(format!("{stem}.f32"));
        if !audio.is_file() {
            eprintln!("skip {stem}: no {}", audio.display());
            continue;
        }
        let mut truth = vec![Vec::new(); CLASSES.len()];
        for line in std::fs::read_to_string(&annotation)?.lines() {
            let mut parts = line.split_whitespace();
            let (Some(time), Some(label)) = (parts.next(), parts.next()) else {
                continue;
            };
            let time: f64 = time.parse()?;
            if let Some(slot) = slot_of(label) {
                truth[slot].push(time);
            }
        }
        for t in &mut truth {
            t.sort_by(f64::total_cmp);
        }
        let activations = load_activations(&audio, &model_dir)?;
        eprintln!("{stem}: {} frames", activations.len() / CLASS_COUNT);
        tracks.push(Track {
            name: stem,
            activations,
            truth,
        });
    }
    anyhow::ensure!(!tracks.is_empty(), "no tracks with audio");

    println!(
        "{} tracks, tolerance {} ms",
        tracks.len(),
        TOLERANCE * 1000.0
    );
    println!("class   | default thr     F  P     R  | best thr     F");
    let mut f_default = Vec::new();
    let mut f_best = Vec::new();
    for (slot, class) in CLASSES.iter().enumerate() {
        let (tp, fp, fne) = evaluate(&tracks, slot, class.threshold());
        let f = f_measure(tp, fp, fne);
        let precision = tp as f64 / (tp + fp).max(1) as f64;
        let recall = tp as f64 / (tp + fne).max(1) as f64;
        let mut best = (class.threshold(), f);
        for step in 1..=60 {
            let threshold = step as f32 * 0.01;
            let (tp, fp, fne) = evaluate(&tracks, slot, threshold);
            let candidate = f_measure(tp, fp, fne);
            if candidate > best.1 + 1e-9 {
                best = (threshold, candidate);
            }
        }
        println!(
            "{:<7} |     {:.2}    {:.3} {:.2} {:.2} |   {:.2}    {:.3}",
            format!("{class:?}"),
            class.threshold(),
            f,
            precision,
            recall,
            best.0,
            best.1
        );
        f_default.push(f);
        f_best.push(best.1);
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    println!(
        "mean F (kick/snare/hihat): default {:.3}, tuned on all (optimistic) {:.3}",
        mean(&f_default),
        mean(&f_best)
    );
    println!();
    println!("leave-one-track-out cross-validation (threshold tuned on the other tracks):");
    let mut f_cv = Vec::new();
    for (slot, class) in CLASSES.iter().enumerate() {
        let mut total = (0, 0, 0);
        let mut chosen = Vec::new();
        for held in 0..tracks.len() {
            let rest: Vec<&Track> = tracks
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != held)
                .map(|(_, t)| t)
                .collect();
            let threshold = best_threshold(&rest, slot);
            chosen.push(threshold);
            let (tp, fp, fne) = evaluate(std::slice::from_ref(&tracks[held]), slot, threshold);
            total = (total.0 + tp, total.1 + fp, total.2 + fne);
        }
        let f = f_measure(total.0, total.1, total.2);
        chosen.sort_by(f32::total_cmp);
        println!(
            "{:<7} | cv F {:.3} (default {:.3}) | thresholds {:.2}..{:.2}",
            format!("{class:?}"),
            f,
            f_default[slot],
            chosen[0],
            chosen[chosen.len() - 1]
        );
        f_cv.push(f);
    }
    println!(
        "mean F: default {:.3}, cross-validated retuning {:.3}",
        mean(&f_default),
        mean(&f_cv)
    );
    println!();
    println!("per track (default thresholds), F kick / snare / hihat:");
    for track in &tracks {
        let one = std::slice::from_ref(track);
        let fs: Vec<String> = CLASSES
            .iter()
            .enumerate()
            .map(|(slot, class)| {
                let (tp, fp, fne) = evaluate(one, slot, class.threshold());
                if track.truth[slot].is_empty() && tp + fp == 0 {
                    "  -  ".to_string()
                } else {
                    format!("{:.3}", f_measure(tp, fp, fne))
                }
            })
            .collect();
        println!("  {}: {}", track.name, fs.join(" / "));
    }
    Ok(())
}
