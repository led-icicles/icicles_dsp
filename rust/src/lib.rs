//! Native audio analysis for icicles animations.
//!
//! Two analyses over one mono PCM signal:
//! - [`features`]: the `icicles_generators` `AudioFeatures` matrix (one row of
//!   22 values every 10 ms), a port of the Dart `AudioFeatureExtractor`;
//! - [`beats`]: Beat This! beat and downbeat tracking through the `beat-this`
//!   crate (pure-Rust `rten` inference, no system libraries).
//!
//! [`ffi`] exposes both behind a C ABI for `dart:ffi`; the Rust API is usable
//! directly as well.

pub mod beats;
pub mod features;
pub mod ffi;

/// The crate version, also returned by `icicles_dsp_version()`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
