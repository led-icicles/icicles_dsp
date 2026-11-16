//! Drum detection against the PyTorch reference written by
//! `tool/export_adtof/export_adtof.py` (`tests/fixtures/adtof/`): the front
//! end and the ONNX model on rten must reproduce the reference spectrogram and
//! activations, and the synthetic loop's hits must land on its pattern.

mod common;

use std::path::PathBuf;
use std::time::Instant;

use beat_this::{RtenRuntime, Runtime};
use icicles_dsp::drums::{activations, detect, resolve_model, DrumClass, FrontEnd, CLASS_COUNT};
use icicles_dsp::model::ModelSource;
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    #[serde(rename = "sampleRate")]
    sample_rate: u32,
    samples: usize,
    frames: usize,
    bins: usize,
    classes: usize,
    spectrogram: Vec<f32>,
    activations: Vec<f32>,
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/adtof")
}

fn load() -> (Fixture, Vec<f32>) {
    let text = std::fs::read_to_string(fixture_dir().join("drum_loop.json"))
        .expect("fixture (run `uv run tool/export_adtof/export_adtof.py`)");
    let fixture: Fixture = serde_json::from_str(&text).unwrap();
    let bytes = std::fs::read(fixture_dir().join("drum_loop_44100.pcm")).unwrap();
    let pcm: Vec<f32> = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| f32::from(i16::from_le_bytes(*b)) / 32768.0)
        .collect();
    assert_eq!(pcm.len(), fixture.samples);
    (fixture, pcm)
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn spectrogram_and_activations_match_the_reference() {
    let (fixture, pcm) = load();
    assert_eq!(fixture.sample_rate, 44_100);
    assert_eq!(fixture.classes, CLASS_COUNT);
    let front = FrontEnd::new();
    assert_eq!(front.bands(), fixture.bins);
    let (spec, _) = front.spectrogram(&pcm);
    assert_eq!(spec.len(), fixture.frames * fixture.bins);
    let spec_diff = max_diff(&spec, &fixture.spectrogram);
    assert!(spec_diff < 1e-4, "spectrogram differs by {spec_diff}");

    let path = resolve_model(&common::model_dir()).unwrap();
    let mut model = RtenRuntime.load_model(&path).unwrap();
    let act = activations(&mut model, &fixture.spectrogram, fixture.bins).unwrap();
    let act_diff = max_diff(&act, &fixture.activations);
    assert!(act_diff < 1e-3, "activations differ by {act_diff}");
}

#[test]
fn loop_hits_land_on_the_pattern() {
    let (_, pcm) = load();
    let hits = detect(&pcm, 44_100, ModelSource::Dir(&common::model_dir())).unwrap();
    let times = |class: DrumClass| -> Vec<f32> {
        hits.iter()
            .filter(|h| h.class == class)
            .map(|h| h.time)
            .collect()
    };
    let near = |found: &[f32], expected: &[f32]| {
        assert_eq!(found.len(), expected.len(), "{found:?} vs {expected:?}");
        for (f, e) in found.iter().zip(expected) {
            assert!((f - e).abs() <= 0.03, "{found:?} vs {expected:?}");
        }
    };
    // Kick on every beat from 0.1 s, snare on beats 2 and 4, hats on eighths.
    near(&times(DrumClass::Kick), &[0.1, 0.6, 1.1, 1.6]);
    near(&times(DrumClass::Snare), &[0.6, 1.6]);
    assert!(
        times(DrumClass::HiHat).len() >= 6,
        "{:?}",
        times(DrumClass::HiHat)
    );
    assert!(hits.iter().all(|h| (0.05..=1.0).contains(&h.strength)));
}

#[test]
fn resampled_input_gives_the_same_hits() {
    let (_, pcm) = load();
    // 44.1 kHz -> 48 kHz by linear interpolation (good enough for onsets).
    let n = pcm.len() * 48_000 / 44_100;
    let resampled: Vec<f32> = (0..n)
        .map(|i| {
            let x = i as f64 * 44_100.0 / 48_000.0;
            let j = x.floor() as usize;
            let f = (x - j as f64) as f32;
            let a = pcm[j];
            let b = *pcm.get(j + 1).unwrap_or(&a);
            a + (b - a) * f
        })
        .collect();
    let started = Instant::now();
    let hits = detect(&resampled, 48_000, ModelSource::Dir(&common::model_dir())).unwrap();
    eprintln!("48 kHz detection: {:?}", started.elapsed());
    let kicks: Vec<f32> = hits
        .iter()
        .filter(|h| h.class == DrumClass::Kick)
        .map(|h| h.time)
        .collect();
    assert_eq!(kicks.len(), 4, "{kicks:?}");
    for (k, e) in kicks.iter().zip([0.1f32, 0.6, 1.1, 1.6]) {
        assert!((k - e).abs() <= 0.03, "{kicks:?}");
    }
}

#[test]
fn missing_model_is_reported() {
    let err = detect(
        &[0.0; 4410],
        44_100,
        ModelSource::Dir(std::path::Path::new("/nonexistent")),
    )
    .unwrap_err();
    assert!(err.to_string().contains("adtof_frame_rnn.onnx"), "{err}");
}
