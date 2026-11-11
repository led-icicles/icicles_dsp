//! The Rust feature extractor against the Dart `AudioFeatureExtractor`
//! reference dumped by `tool/dump_features.dart` (`tests/fixtures/*.json`).

mod common;

use std::path::PathBuf;

use icicles_dsp::features::{extract, COLUMN_COUNT};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    name: String,
    #[serde(rename = "sampleRate")]
    sample_rate: u32,
    length: usize,
    rows: usize,
    columns: usize,
    data: Vec<f32>,
}

const TOLERANCE: f32 = 1e-3;

fn load(name: &str) -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(format!("{name}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "{}: {e} (run `fvm dart run tool/dump_features.dart`)",
            path.display()
        )
    });
    serde_json::from_str(&text).expect("fixture JSON")
}

#[test]
fn features_match_the_dart_reference() {
    for signal in common::signals() {
        let fixture = load(signal.name);
        assert_eq!(fixture.name, signal.name);
        assert_eq!(
            fixture.sample_rate, signal.sample_rate,
            "{}: sample rate",
            signal.name
        );
        assert_eq!(fixture.length, signal.length, "{}: length", signal.name);
        assert_eq!(fixture.columns, COLUMN_COUNT, "{}: columns", signal.name);

        let samples = signal.generate();
        let features = extract(&samples, signal.sample_rate).unwrap();
        assert_eq!(features.rows, fixture.rows, "{}: rows", signal.name);
        assert_eq!(
            features.data.len(),
            fixture.data.len(),
            "{}: matrix size",
            signal.name
        );

        let mut worst = 0.0f32;
        let mut worst_at = (0, 0);
        for (i, (ours, theirs)) in features.data.iter().zip(&fixture.data).enumerate() {
            let diff = (ours - theirs).abs();
            if diff > worst {
                worst = diff;
                worst_at = (i / COLUMN_COUNT, i % COLUMN_COUNT);
            }
        }
        assert!(
            worst <= TOLERANCE,
            "{}: worst deviation {worst} at row {} column {} (limit {TOLERANCE})",
            signal.name,
            worst_at.0,
            worst_at.1
        );
        eprintln!(
            "{}: {} rows, worst deviation {worst:e}",
            signal.name, features.rows
        );
    }
}
