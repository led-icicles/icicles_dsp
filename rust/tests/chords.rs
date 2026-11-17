//! Chord recognition against the reference written by
//! `tool/export_chords/export_chords.py` (`tests/fixtures/chords/`): the
//! constant-Q front end must stay close to librosa's, the ONNX model on rten
//! must label the reference features like PyTorch, and the whole pipeline
//! must find the fixture's progression.

mod common;

use std::path::PathBuf;

use beat_this::{RtenRuntime, Runtime};
use icicles_dsp::chords::{
    chords, detect, frame_labels, resample, resolve_model, ChordError, Cqt, BINS, FRAME_SECONDS,
    NO_CHORD,
};
use icicles_dsp::model::ModelSource;
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    #[serde(rename = "sampleRate")]
    sample_rate: u32,
    samples: usize,
    frames: usize,
    bins: usize,
    #[serde(rename = "chordSeconds")]
    chord_seconds: f32,
    /// (root, 1 = major / 0 = minor) per chord of the signal.
    progression: Vec<(u8, u8)>,
    features: Vec<f32>,
    labels: Vec<u8>,
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/chords")
}

fn load() -> (Fixture, Vec<f32>) {
    let text = std::fs::read_to_string(fixture_dir().join("progression.json"))
        .expect("fixture (run `uv run tool/export_chords/export_chords.py`)");
    let fixture: Fixture = serde_json::from_str(&text).unwrap();
    let bytes = std::fs::read(fixture_dir().join("progression_44100.pcm")).unwrap();
    let pcm: Vec<f32> = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| f32::from(i16::from_le_bytes(*b)) / 32768.0)
        .collect();
    assert_eq!(pcm.len(), fixture.samples);
    (fixture, pcm)
}

/// The reference's class indices as this crate's labels.
fn reference_labels(fixture: &Fixture) -> Vec<u8> {
    fixture
        .labels
        .iter()
        .map(|&label| if label < 168 { label } else { NO_CHORD })
        .collect()
}

fn agreement(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).filter(|(x, y)| x == y).count() as f64 / a.len() as f64
}

#[test]
fn the_front_end_stays_close_to_librosa() {
    let (fixture, pcm) = load();
    assert_eq!(fixture.bins, BINS);
    let audio = resample(&pcm, fixture.sample_rate).unwrap();
    let features = Cqt::new().features(&audio);
    assert_eq!(features.len(), fixture.frames * BINS);
    // The resampling filters differ from librosa's (soxr), so quiet bins —
    // deep in the log — may wander; bins that carry sound must agree.
    let mut loud = 0usize;
    let mut sum = 0.0f64;
    let mut worst = 0.0f32;
    for (ours, reference) in features.iter().zip(&fixture.features) {
        if *reference > -4.0 {
            let diff = (ours - reference).abs();
            loud += 1;
            sum += f64::from(diff);
            worst = worst.max(diff);
        }
    }
    let mean = sum / loud as f64;
    assert!(loud > fixture.frames * 10, "only {loud} loud bins");
    assert!(mean < 0.02, "loud bins differ by {mean} on average");
    assert!(worst < 0.5, "a loud bin differs by {worst}");
}

#[test]
fn the_model_labels_the_reference_features_like_pytorch() {
    let (fixture, _) = load();
    let path = resolve_model(&common::model_dir()).unwrap();
    let mut model = RtenRuntime.load_model(&path).unwrap();
    let labels = frame_labels(&mut model, &fixture.features).unwrap();
    let agreement = agreement(&labels, &reference_labels(&fixture));
    assert!(agreement >= 0.995, "labels agree on {agreement}");
}

#[test]
fn the_pipeline_labels_the_audio_like_the_reference() {
    let (fixture, pcm) = load();
    let path = resolve_model(&common::model_dir()).unwrap();
    let mut model = RtenRuntime.load_model(&path).unwrap();
    let audio = resample(&pcm, fixture.sample_rate).unwrap();
    let labels = frame_labels(&mut model, &Cqt::new().features(&audio)).unwrap();
    let agreement = agreement(&labels, &reference_labels(&fixture));
    assert!(agreement >= 0.99, "labels agree on {agreement}");
}

#[test]
fn the_progression_is_found() {
    let (fixture, pcm) = load();
    let found = detect(
        &pcm,
        fixture.sample_rate,
        ModelSource::Dir(&common::model_dir()),
    )
    .unwrap();
    assert_eq!(found.len(), fixture.progression.len(), "{found:?}");
    for (k, (chord, (root, major))) in found.iter().zip(&fixture.progression).enumerate() {
        assert_eq!(chord.root, *root, "chord {k}: {chord:?}");
        // Sevenths may come back as their triad: only major / minor counts.
        let is_major = matches!(chord.quality, 1 | 3 | 5 | 8 | 9);
        assert_eq!(is_major, *major == 1, "chord {k}: {chord:?}");
        let start = k as f32 * fixture.chord_seconds;
        assert!(
            (chord.start - start).abs() < 0.15,
            "chord {k} starts at {}",
            chord.start
        );
    }
    for pair in found.windows(2) {
        assert!(pair[0].end <= pair[1].start + 1e-6);
    }
    let duration = pcm.len() as f32 / fixture.sample_rate as f32;
    assert!(found.last().unwrap().end <= duration + 1e-3);
}

#[test]
fn other_sample_rates_give_the_same_chords() {
    let (fixture, pcm) = load();
    // 48 kHz by linear interpolation: good enough below 2 kHz.
    let ratio = 44_100.0 / 48_000.0;
    let resampled: Vec<f32> = (0..(pcm.len() as f64 / ratio) as usize - 1)
        .map(|i| {
            let at = i as f64 * ratio;
            let k = at as usize;
            let t = (at - k as f64) as f32;
            pcm[k] * (1.0 - t) + pcm[k + 1] * t
        })
        .collect();
    let dir = common::model_dir();
    let found = detect(&resampled, 48_000, ModelSource::Dir(&dir)).unwrap();
    let reference = detect(&pcm, fixture.sample_rate, ModelSource::Dir(&dir)).unwrap();
    assert_eq!(found.len(), reference.len());
    for (a, b) in found.iter().zip(&reference) {
        assert_eq!((a.root, a.quality), (b.root, b.quality));
        assert!((a.start - b.start).abs() < 2.0 * FRAME_SECONDS as f32);
    }
}

#[test]
fn silence_has_no_chords() {
    let silence = vec![0.0f32; 44_100 * 12];
    let found = detect(&silence, 44_100, ModelSource::Dir(&common::model_dir())).unwrap();
    assert!(found.is_empty(), "{found:?}");
    assert!(chords(&[NO_CHORD; 30], 3.0).is_empty());
}

#[test]
fn a_missing_model_is_reported_by_name() {
    let dir = std::env::temp_dir().join("icicles_dsp_no_chord_model");
    std::fs::create_dir_all(&dir).unwrap();
    let error = detect(&[0.0; 44_100], 44_100, ModelSource::Dir(&dir)).unwrap_err();
    assert!(matches!(error, ChordError::ModelNotFound(_)));
    assert!(error.to_string().contains("btc_chords.onnx"));
    let bytes = ModelSource::Bytes {
        mel: &[],
        beat: &[],
        drum: None,
        chord: None,
    };
    let error = detect(&[0.0; 44_100], 44_100, bytes).unwrap_err();
    assert!(error.to_string().contains("btc_chords.onnx"));
}
