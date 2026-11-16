//! Decoding audio files (WAV, AIFF, FLAC, MP3, Ogg Vorbis, AAC/M4A) into
//! f32 PCM with `symphonia`, pure Rust — the same on the desktop and in the
//! browser build.

use std::io::Cursor;

use anyhow::{bail, Context, Result};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Decoded PCM: `samples` interleaved, `channels` 1 or 2.
pub struct Decoded {
    pub samples: Vec<f32>,
    pub channels: usize,
    pub sample_rate: u32,
}

/// The audio file `bytes` decoded from `start` seconds for `length` seconds
/// (`length <= 0`: to the end). More than two channels keep the first two.
/// A span past the end gives what is there (an error when nothing).
pub fn decode_audio(bytes: &[u8], start: f64, length: f64) -> Result<Decoded> {
    let source = MediaSourceStream::new(Box::new(Cursor::new(bytes.to_vec())), Default::default());
    let probed = symphonia::default::get_probe()
        .format(&Hint::new(), source, &FormatOptions::default(), &MetadataOptions::default())
        .context("unknown audio format")?;
    let mut format = probed.format;
    let track = format
        .default_track()
        .filter(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .context("no audio track")?;
    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.context("unknown sample rate")?;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("unsupported codec")?;

    let first = (start.max(0.0) * sample_rate as f64).round() as u64;
    let last = if length > 0.0 { first + (length * sample_rate as f64).round() as u64 } else { u64::MAX };
    let mut channels = 0usize;
    let mut samples = Vec::new();
    let mut frame = 0u64; // frames decoded so far
    let mut buffer: Option<SampleBuffer<f32>> = None;
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(Error::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(Error::ResetRequired) => break,
            Err(e) => return Err(e).context("reading the audio file"),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            // A damaged packet is skipped, like a player does.
            Err(Error::DecodeError(_)) => continue,
            Err(e) => return Err(e).context("decoding the audio"),
        };
        let spec = *decoded.spec();
        let count = spec.channels.count();
        if channels == 0 {
            channels = count.min(2);
        }
        let buf = buffer.get_or_insert_with(|| SampleBuffer::new(decoded.capacity() as u64, spec));
        if (buf.capacity() as usize) < decoded.capacity() * count {
            *buf = SampleBuffer::new(decoded.capacity() as u64, spec);
        }
        buf.copy_interleaved_ref(decoded);
        let frames = buf.samples().len() / count;
        for (i, chunk) in buf.samples().chunks_exact(count).enumerate() {
            let at = frame + i as u64;
            if at >= first && at < last {
                samples.extend_from_slice(&chunk[..channels]);
            }
        }
        frame += frames as u64;
        if frame >= last {
            break;
        }
    }
    if samples.is_empty() {
        bail!("no audio in the requested span");
    }
    Ok(Decoded { samples, channels, sample_rate })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A 16-bit PCM WAV of `frames` frames.
    pub(crate) fn wav(frames: usize, channels: u16, rate: u32) -> Vec<u8> {
        let data = frames * channels as usize * 2;
        let mut out = Vec::new();
        out.extend(b"RIFF");
        out.extend(((36 + data) as u32).to_le_bytes());
        out.extend(b"WAVEfmt ");
        out.extend(16u32.to_le_bytes());
        out.extend(1u16.to_le_bytes());
        out.extend(channels.to_le_bytes());
        out.extend(rate.to_le_bytes());
        out.extend((rate * channels as u32 * 2).to_le_bytes());
        out.extend((channels * 2).to_le_bytes());
        out.extend(16u16.to_le_bytes());
        out.extend(b"data");
        out.extend((data as u32).to_le_bytes());
        for i in 0..frames {
            let s = (0.4 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / rate as f32).sin() * 32767.0) as i16;
            for _ in 0..channels {
                out.extend(s.to_le_bytes());
            }
        }
        out
    }

    #[test]
    fn decodes_a_wav_span() {
        let bytes = wav(44_100 * 2, 2, 44_100);
        let all = decode_audio(&bytes, 0.0, 0.0).unwrap();
        assert_eq!((all.channels, all.sample_rate, all.samples.len()), (2, 44_100, 44_100 * 2 * 2));
        let span = decode_audio(&bytes, 0.5, 1.0).unwrap();
        assert_eq!(span.samples.len(), 44_100 * 2);
        assert_eq!(&span.samples[..2], &all.samples[22_050 * 2..22_050 * 2 + 2]);
        assert!(decode_audio(&bytes, 5.0, 1.0).is_err());
        assert!(decode_audio(b"not audio", 0.0, 0.0).is_err());
    }
}
