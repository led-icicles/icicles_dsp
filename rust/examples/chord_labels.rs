//! Prints the chord model's label per feature column (one every ~93 ms) of a
//! raw PCM file, as a JSON array — for comparing this crate's front end with
//! the librosa/PyTorch reference on real music:
//!
//! ```sh
//! ffmpeg -i song.mp3 -ac 1 -ar 44100 -f s16le song.pcm
//! cargo run --release --example chord_labels -- song.pcm [models dir]
//! ```
//!
//! Labels: `root × 14 + quality` (see `icicles_dsp::chords`), 255 = no chord.

use std::path::PathBuf;

use icicles_dsp::chords::{frame_labels, load_model, resample, Cqt};
use icicles_dsp::model::ModelSource;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: chord_labels <mono 16-bit 44.1 kHz PCM> [models dir]");
    let models = args.next().map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models"),
        PathBuf::from,
    );
    let bytes = std::fs::read(&path).expect("readable PCM file");
    let pcm: Vec<f32> = bytes
        .chunks_exact(2)
        .map(|b| f32::from(i16::from_le_bytes([b[0], b[1]])) / 32768.0)
        .collect();
    let mut model = load_model(ModelSource::Dir(&models)).expect("chord model");
    let audio = resample(&pcm, 44_100).expect("resampled audio");
    let labels = frame_labels(&mut model, &Cqt::new().features(&audio)).expect("labels");
    println!(
        "[{}]",
        labels
            .iter()
            .map(u8::to_string)
            .collect::<Vec<_>>()
            .join(",")
    );
}
