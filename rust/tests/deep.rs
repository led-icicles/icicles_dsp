//! Drums of the deep analysis: ADTOF on the separated drum stem. Tests that
//! need a model skip (with a note) when it is not in `models/` — the deep
//! models come from `tool/download-deep-models.sh`.

mod common;

use std::f64::consts::PI;
use std::path::Path;

use icicles_dsp::deep::{self, Control, DeepModels, DEMUCS_MODEL_FILE};
use icicles_dsp::drums::{self, DrumClass, DrumError, DrumHit, DRUM_MODEL_FILE, SAMPLE_RATE};

const BPM: f64 = 120.0;
const BEAT: f64 = 60.0 / BPM;

/// A decaying 55 Hz kick with a pitch drop, `dt` seconds after its onset.
fn kick(dt: f64) -> f64 {
    if dt < 0.0 {
        return 0.0;
    }
    0.9 * (-dt / 0.12).exp() * (2.0 * PI * (55.0 * dt + 3.0 * (1.0 - (-30.0 * dt).exp()))).sin()
}

/// Four on the floor at [`BPM`]: a kick on every beat from `first`, `bars`
/// bars long (plus half a second of tail).
fn kicks_only(bars: usize, first: f64) -> Vec<f32> {
    let beats = bars * 4;
    let length = ((first + beats as f64 * BEAT + 0.5) * f64::from(SAMPLE_RATE)) as usize;
    (0..length)
        .map(|i| {
            let t = i as f64 / f64::from(SAMPLE_RATE) - first;
            if t < 0.0 {
                return 0.0;
            }
            // The last kick rings into the tail; no new one starts there.
            let beat = ((t / BEAT).floor() as usize).min(beats - 1);
            kick(t - beat as f64 * BEAT) as f32
        })
        .collect()
}

/// A bass line and a chord with the kick pattern on top (a mix Demucs can
/// pull the drums out of), fading out over its tail (an abrupt end of the
/// tones would itself be a transient).
fn song(bars: usize, first: f64) -> Vec<f32> {
    let kicks = kicks_only(bars, first);
    let end = kicks.len() as f64 / f64::from(SAMPLE_RATE);
    kicks
        .iter()
        .enumerate()
        .map(|(i, k)| {
            let t = i as f64 / f64::from(SAMPLE_RATE);
            let fade = ((end - t) / 0.4).clamp(0.0, 1.0);
            let x = 0.25 * (2.0 * PI * 55.0 * t).sin()
                + 0.1 * (2.0 * PI * 220.0 * t).sin()
                + 0.1 * (2.0 * PI * 277.18 * t).sin()
                + 0.1 * (2.0 * PI * 329.63 * t).sin();
            (0.6 * (fade * x + 0.8 * f64::from(*k))) as f32
        })
        .collect()
}

fn kick_times(hits: &[DrumHit]) -> Vec<f32> {
    hits.iter()
        .filter(|h| h.class == DrumClass::Kick)
        .map(|h| h.time)
        .collect()
}

/// Every expected beat has a kick within 50 ms and no kick is off the beats.
fn assert_one_kick_per_beat(kicks: &[f32], first: f64, beats: usize) {
    for b in 0..beats {
        let at = (first + b as f64 * BEAT) as f32;
        let near = kicks.iter().filter(|k| (**k - at).abs() <= 0.05).count();
        assert_eq!(near, 1, "beat {b} at {at}: kicks {kicks:?}");
    }
    assert_eq!(kicks.len(), beats, "kicks {kicks:?}");
}

fn has(dir: &Path, files: &[&str]) -> bool {
    files.iter().all(|f| dir.join(f).is_file())
}

#[test]
fn a_clean_kick_on_every_beat_gives_one_kick_per_beat() {
    let dir = common::model_dir();
    if !has(&dir, &[DRUM_MODEL_FILE]) {
        eprintln!("skipped: {DRUM_MODEL_FILE} missing");
        return;
    }
    let mut model = drums::load_model_file(&dir).unwrap();
    let (bars, first) = (8, 0.25);
    let hits = drums::detect_with(&mut model, &kicks_only(bars, first)).unwrap();
    assert_one_kick_per_beat(&kick_times(&hits), first, bars * 4);
    assert!(hits.windows(2).all(|w| w[0].time <= w[1].time));
}

#[test]
fn deep_transcribes_the_drum_stem_and_survives_a_missing_drum_model() {
    let dir = common::model_dir();
    let fold = deep::fold_model_file(0);
    if !has(&dir, &[DEMUCS_MODEL_FILE, &fold, DRUM_MODEL_FILE]) {
        eprintln!("skipped: deep models missing (tool/download-deep-models.sh)");
        return;
    }
    let (bars, first) = (4, 0.25);
    let mix = song(bars, first);

    // With the drum model (from the directory): one kick per beat.
    let control = Control::default();
    let result = deep::analyze(&mix, SAMPLE_RATE, &DeepModels::Dir(&dir), 1, &control).unwrap();
    assert!(result.drum_error.is_none(), "{:?}", result.drum_error);
    assert_one_kick_per_beat(&kick_times(&result.drums), first, bars * 4);
    assert!(!result.segments.is_empty());

    // Without it: the rest of the analysis, no drums, the reason.
    let demucs = std::fs::read(dir.join(DEMUCS_MODEL_FILE)).unwrap();
    let fold = std::fs::read(dir.join(fold)).unwrap();
    let models = DeepModels::Bytes {
        demucs: &demucs,
        folds: vec![&fold],
        drum: None,
    };
    let without = deep::analyze(&mix, SAMPLE_RATE, &models, 1, &control).unwrap();
    assert!(without.drums.is_empty());
    assert!(
        matches!(without.drum_error, Some(DrumError::ModelNotFound(_))),
        "{:?}",
        without.drum_error
    );
    assert_eq!(without.stem_rows, result.stem_rows);
    assert_eq!(without.segments.len(), result.segments.len());
}
