//! Beat and downbeat tracking with the Beat This! model through the
//! `beat-this` crate (pure-Rust `rten` inference).
//!
//! The model files are looked up in a directory: `mel_spectrogram.onnx` is
//! required, and the beat model is `beat_this.onnx` (the full ~83 MB model)
//! when present, else `beat_this_small.onnx` (~10 MB, bundled with this
//! package under `models/`).

use std::path::{Path, PathBuf};

use beat_this::{BeatThis, RtenRuntime};

/// File name of the mel spectrogram front end.
pub const MEL_MODEL_FILE: &str = "mel_spectrogram.onnx";
/// File name of the full-accuracy beat model (not bundled; see `download-models.sh`).
pub const FULL_MODEL_FILE: &str = "beat_this.onnx";
/// File name of the bundled small beat model.
pub const SMALL_MODEL_FILE: &str = "beat_this_small.onnx";

/// Beat and downbeat instants in seconds, sorted.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Beats {
    pub beats: Vec<f32>,
    pub downbeats: Vec<f32>,
}

/// Why beat tracking failed.
#[derive(Debug)]
pub enum BeatError {
    /// A model file is missing from the model directory.
    ModelNotFound(PathBuf),
    /// The runtime could not load a model file.
    ModelLoad(String),
    /// Inference or post-processing failed.
    Analysis(String),
}

impl std::fmt::Display for BeatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BeatError::ModelNotFound(path) => write!(f, "model file not found: {}", path.display()),
            BeatError::ModelLoad(message) => write!(f, "model load failed: {message}"),
            BeatError::Analysis(message) => write!(f, "beat analysis failed: {message}"),
        }
    }
}

impl std::error::Error for BeatError {}

/// The model files `model_dir` resolves to: `(mel, beat)`.
pub fn resolve_models(model_dir: &Path) -> Result<(PathBuf, PathBuf), BeatError> {
    let mel = model_dir.join(MEL_MODEL_FILE);
    if !mel.is_file() {
        return Err(BeatError::ModelNotFound(mel));
    }
    let full = model_dir.join(FULL_MODEL_FILE);
    if full.is_file() {
        return Ok((mel, full));
    }
    let small = model_dir.join(SMALL_MODEL_FILE);
    if small.is_file() {
        return Ok((mel, small));
    }
    Err(BeatError::ModelNotFound(small))
}

/// Tracks the beats of mono PCM (`-1..1`) at `sample_rate` Hz with the
/// models in `model_dir`. Any sample rate works (the crate resamples to
/// 22 050 Hz).
pub fn track(samples: &[f32], sample_rate: u32, model_dir: &Path) -> Result<Beats, BeatError> {
    let (mel, beat) = resolve_models(model_dir)?;
    let mut tracker = BeatThis::new(&RtenRuntime, &mel, &beat)
        .map_err(|e| BeatError::ModelLoad(format!("{e:#}")))?;
    let analysis = tracker
        .analyze_audio(samples, sample_rate)
        .map_err(|e| BeatError::Analysis(format!("{e:#}")))?;
    Ok(Beats {
        beats: analysis.beats,
        downbeats: analysis.downbeats,
    })
}
