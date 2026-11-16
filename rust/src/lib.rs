//! Native audio analysis for icicles animations.
//!
//! Three analyses over one mono PCM signal:
//! - [`features`]: the `icicles_generators` `AudioFeatures` matrix (one row of
//!   22 values every 10 ms), a port of the Dart `AudioFeatureExtractor`;
//! - [`beats`]: Beat This! beat and downbeat tracking through the `beat-this`
//!   crate (pure-Rust `rten` inference, no system libraries);
//! - [`drums`]: drum hits (kick, snare, toms, hi-hat, cymbals) with the ADTOF
//!   model on the same runtime.
//!
//! [`ffi`] exposes them behind a C ABI — for `dart:ffi` on the desktop and
//! for the browser through the `wasm32-unknown-unknown` build, where the
//! models arrive as bytes ([`model::ModelSource`]); the Rust API is usable
//! directly as well.

pub mod beats;
pub mod deep;
pub mod demucs;
pub mod drums;
pub mod encode;
pub mod features;
pub mod ffi;
pub mod model;
pub mod spectrum;
pub mod structure;

/// The crate version, also returned by `icicles_dsp_version()`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
