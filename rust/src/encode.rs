//! Ogg Vorbis encoding of PCM (the music of a show), pure Rust
//! (`rusty_vorbis` + the `ogg` muxer), the same on the desktop and in the
//! browser build.

use std::io::Cursor;

use anyhow::{bail, Context, Result};
use ogg::{PacketWriteEndInfo, PacketWriter};
use rusty_vorbis::{quality01_from_vorbis_q, Error, VorbisEncoder, VorbisEncoderConfig};

/// Highest channel count [`encode_vorbis`] accepts.
pub const MAX_CHANNELS: usize = 2;

/// Silence appended before encoding so the encoder's last block covers the
/// real end; the final granule position trims it off again.
const PAD_FRAMES: usize = 4096;

/// Stream serial of the single logical stream.
const SERIAL: u32 = 0x1c1c_1e5;

/// `interleaved` f32 PCM (`-1..1`, `channels` interleaved) at `sample_rate`
/// as an Ogg Vorbis file at Vorbis quality `quality` (`-1..=10`, clamped).
/// Always a stereo stream (mono is duplicated). The decoded length is the
/// input length (to the decoder's block granularity).
pub fn encode_vorbis(interleaved: &[f32], channels: usize, sample_rate: u32, quality: f32) -> Result<Vec<u8>> {
    if channels == 0 || channels > MAX_CHANNELS {
        bail!("channels must be 1..={MAX_CHANNELS}, got {channels}");
    }
    if !(8_000..=192_000).contains(&sample_rate) {
        bail!("unsupported sample rate {sample_rate}");
    }
    if interleaved.len() % channels != 0 {
        bail!("{} samples are not a whole number of {channels}-channel frames", interleaved.len());
    }
    let frames = interleaved.len() / channels;
    if frames == 0 {
        bail!("no samples");
    }
    // The encoder's setup is stereo (coupled): mono is written as two
    // equal channels.
    let clean = |s: &f32| if s.is_finite() { s.clamp(-1.0, 1.0) } else { 0.0 };
    let mut pcm = Vec::with_capacity((frames + PAD_FRAMES) * 2);
    if channels == 1 {
        pcm.extend(interleaved.iter().flat_map(|s| [clean(s), clean(s)]));
    } else {
        pcm.extend(interleaved.iter().map(clean));
    }
    pcm.resize((frames + PAD_FRAMES) * 2, 0.0);

    let q = if quality.is_finite() { quality.clamp(-1.0, 10.0) } else { 5.0 };
    let mut encoder = VorbisEncoder::new(VorbisEncoderConfig {
        quality: quality01_from_vorbis_q(q as f64),
        ..VorbisEncoderConfig::default()
    });
    encoder.push_pcm_f32(&pcm, 2, sample_rate).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    encoder.finish();
    let mut packets = Vec::new();
    loop {
        match encoder.next_packet() {
            Ok(packet) => packets.push(packet),
            Err(Error::Eof) => break,
            Err(e) => bail!("encoding failed: {e:?}"),
        }
    }
    if packets.len() < 4 {
        bail!("the encoder produced no audio");
    }
    // Drop audio packets that start past the real end; the last kept one
    // carries the exact length as its granule position.
    let total = frames as i64;
    let mut last = packets.len() - 1;
    while last > 3 && packets[last - 1].pts >= total {
        last -= 1;
    }
    packets.truncate(last + 1);

    let mut out = Vec::new();
    {
        let mut writer = PacketWriter::new(Cursor::new(&mut out));
        let count = packets.len();
        for (i, packet) in packets.into_iter().enumerate() {
            let (end, granule) = match i {
                // The identification header alone on the first page, the
                // comment and setup headers on the second.
                0 | 2 => (PacketWriteEndInfo::EndPage, 0),
                1 => (PacketWriteEndInfo::NormalPacket, 0),
                _ if i == count - 1 => (PacketWriteEndInfo::EndStream, total.min(packet.pts)),
                _ => (PacketWriteEndInfo::NormalPacket, packet.pts),
            };
            writer.write_packet(packet.data, SERIAL, end, granule.max(0) as u64).context("writing the Ogg stream")?;
        }
    }
    Ok(out)
}

/// The audio file `bytes` (any format [`crate::decode::decode_audio`]
/// reads) from `start` seconds for `length` seconds (`<= 0`: to the end)
/// as an Ogg Vorbis file at quality `quality`.
pub fn transcode_vorbis(bytes: &[u8], start: f64, length: f64, quality: f32) -> Result<Vec<u8>> {
    let pcm = crate::decode::decode_audio(bytes, start, length)?;
    encode_vorbis(&pcm.samples, pcm.channels, pcm.sample_rate, quality)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(bytes: &[u8]) -> (usize, Vec<f32>, u32, usize) {
        let mut reader = lewton::inside_ogg::OggStreamReader::new(Cursor::new(bytes)).unwrap();
        let channels = reader.ident_hdr.audio_channels as usize;
        let rate = reader.ident_hdr.audio_sample_rate;
        let mut samples = Vec::new();
        while let Some(packet) = reader.read_dec_packet_itl().unwrap() {
            samples.extend(packet.into_iter().map(|s| s as f32 / 32768.0));
        }
        (samples.len() / channels, samples, rate, channels)
    }

    fn tone(frames: usize, channels: usize, rate: u32) -> Vec<f32> {
        (0..frames)
            .flat_map(|i| {
                let s = 0.4 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / rate as f32).sin();
                std::iter::repeat_n(s, channels)
            })
            .collect()
    }

    fn rms(samples: &[f32]) -> f32 {
        (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
    }

    #[test]
    fn stereo_round_trips_with_its_length_and_level() {
        let pcm = tone(44_100 * 3 + 123, 2, 44_100);
        let bytes = encode_vorbis(&pcm, 2, 44_100, 5.0).unwrap();
        let (frames, decoded, rate, channels) = decode(&bytes);
        assert_eq!((rate, channels), (44_100, 2));
        assert!(frames.abs_diff(44_100 * 3 + 123) < 2048, "decoded {frames} frames");
        assert!((rms(&decoded) - rms(&pcm)).abs() < 0.03, "rms {} vs {}", rms(&decoded), rms(&pcm));
        assert!(bytes.len() * 8 < pcm.len() * 16, "smaller than 16-bit PCM");
    }

    #[test]
    fn mono_at_48k() {
        let pcm = tone(48_000, 1, 48_000);
        let (frames, _, rate, channels) = decode(&encode_vorbis(&pcm, 1, 48_000, 3.0).unwrap());
        assert_eq!((rate, channels), (48_000, 2), "mono is written as stereo");
        assert!(frames.abs_diff(48_000) < 2048);
    }

    #[test]
    fn transcodes_a_wav_span() {
        let wav = crate::decode::tests::wav(44_100 * 3, 2, 44_100);
        let bytes = transcode_vorbis(&wav, 1.0, 1.5, 5.0).unwrap();
        let (frames, _, rate, channels) = decode(&bytes);
        assert_eq!((rate, channels), (44_100, 2));
        assert!(frames.abs_diff(66_150) < 2048, "decoded {frames} frames");
        assert!(bytes.len() < wav.len() / 4);
    }

    #[test]
    fn rejects_bad_input() {
        assert!(encode_vorbis(&[], 2, 44_100, 5.0).is_err());
        assert!(encode_vorbis(&[0.0; 6], 3, 44_100, 5.0).is_err());
        assert!(encode_vorbis(&[0.0; 3], 2, 44_100, 5.0).is_err());
        assert!(encode_vorbis(&[0.0; 4], 2, 100, 5.0).is_err());
    }
}
