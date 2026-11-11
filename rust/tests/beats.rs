//! Beat tracking with the bundled small model (`../models`).

mod common;

use std::time::Instant;

use icicles_dsp::beats::{resolve_models, track, SMALL_MODEL_FILE};
use icicles_dsp::features::extract;

#[test]
fn resolves_the_bundled_models() {
    let (mel, beat) = resolve_models(&common::model_dir()).unwrap();
    assert!(mel.ends_with("mel_spectrogram.onnx"));
    assert!(beat.ends_with(SMALL_MODEL_FILE) || beat.ends_with("beat_this.onnx"));
}

#[test]
fn click_track_beats_land_on_the_grid() {
    let bpm = 120.0;
    let period = 60.0 / bpm;
    let signal = common::click_track(bpm, 20.0, 44100);
    let beats = track(&signal, 44100, &common::model_dir()).unwrap();
    assert!(
        beats.beats.len() >= 30,
        "found only {} beats: {:?}",
        beats.beats.len(),
        beats.beats
    );
    assert!(
        beats.beats.len() <= 42,
        "found {} beats: {:?}",
        beats.beats.len(),
        beats.beats
    );
    // The model works on 20 ms frames, so every beat sits within one frame of
    // the grid; the very last click, cut by the end of the signal, is excused.
    let end = 20.0 - period;
    for beat in beats.beats.iter().filter(|b| f64::from(**b) < end) {
        let offset = f64::from(*beat).rem_euclid(period);
        let distance = offset.min(period - offset);
        assert!(
            distance <= 0.030,
            "beat at {beat} s is {:.1} ms off the 120 BPM grid",
            distance * 1000.0
        );
    }
    for downbeat in &beats.downbeats {
        assert!(
            beats.beats.iter().any(|b| (b - downbeat).abs() < 1e-3),
            "downbeat {downbeat} is not a beat"
        );
    }
    let bpm_estimate = 60.0 * (beats.beats.len() - 1) as f64
        / f64::from(beats.beats.last().unwrap() - beats.beats[0]);
    assert!(
        (bpm_estimate - bpm).abs() < 2.0,
        "estimated {bpm_estimate} BPM"
    );
}

#[test]
fn silence_has_no_beats() {
    let signal = vec![0.0f32; 44100 * 20];
    let beats = track(&signal, 44100, &common::model_dir()).unwrap();
    assert!(beats.beats.is_empty(), "{:?}", beats.beats);
    assert!(beats.downbeats.is_empty());
}

/// Timing of a 3-minute signal; run with
/// `cargo test --release --test beats -- --ignored --nocapture timing`.
#[test]
#[ignore]
fn timing_three_minutes() {
    let signal = common::click_track(128.0, 180.0, 44100);
    let started = Instant::now();
    let features = extract(&signal, 44100).unwrap();
    let features_time = started.elapsed();
    let started = Instant::now();
    let beats = track(&signal, 44100, &common::model_dir()).unwrap();
    let beats_time = started.elapsed();
    eprintln!(
        "3 min @ 44.1 kHz: features {} rows in {features_time:?}, beats {} in {beats_time:?}",
        features.rows,
        beats.beats.len()
    );
}
