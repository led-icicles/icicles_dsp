//! Functional song structure with All-In-One (Kim & Nam, WASPAA 2023, MIT):
//! section boundaries and labels (intro, verse, chorus, bridge…) from the
//! four Demucs stems.
//!
//! Input: per stem (bass, drums, other, vocals — All-In-One's order) the
//! [`LogSpectrogram::allin1`](crate::spectrum::LogSpectrogram::allin1) of the
//! stem's mono signal, `[1, 4, frames, 81]`. Each fold model
//! (`allin1_fold{k}.onnx`, the eight Harmonix cross-validation folds) gives
//! logits per frame; the ensemble averages them (`allin1.models.ensemble`).
//! Post-processing is `allin1.postprocessing.functional` verbatim:
//! sigmoid boundary probabilities → local maxima within ±48 frames →
//! peak picking against the mean of the 12 s before and after → boundaries;
//! the label of a segment is the argmax of the mean softmax of its frames.

use beat_this::{Model, Tensor};

use crate::spectrum;

/// Frames per second of the activations.
pub const FPS: f64 = 100.0;
/// Number of function classes.
pub const LABEL_COUNT: usize = 10;
/// All-In-One's `HARMONIX_LABELS`, in class order.
pub const LABELS: [&str; LABEL_COUNT] = [
    "start", "end", "intro", "outro", "break", "bridge", "inst", "solo", "verse", "chorus",
];

/// `min_hops_per_beat` of the config (60 / 250 BPM at 100 fps ≈ 24).
const MIN_HOPS_PER_BEAT: usize = 24;
/// Peak-picking windows (12 s each side).
const WINDOW: usize = 12 * 100;

/// One labelled section: `[start, end)` in seconds, `label` an index into
/// [`LABELS`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Segment {
    pub start: f32,
    pub end: f32,
    pub label: u8,
}

/// The ensemble's mean logits: `section[frames]`, `function[LABEL_COUNT × frames]`.
pub struct Logits {
    pub frames: usize,
    pub section: Vec<f32>,
    pub function: Vec<f32>,
}

/// `[4, frames, 81]` spectrograms of bass, drums, other, vocals (All-In-One's
/// stem order) of mono 44.1 kHz stems.
pub fn features(bass: &[f32], drums: &[f32], other: &[f32], vocals: &[f32]) -> (Vec<f32>, usize) {
    let front = spectrum::LogSpectrogram::allin1();
    let specs = [bass, drums, other, vocals].map(|s| front.compute(s));
    let frames = specs
        .iter()
        .map(|s| s.len() / front.bands())
        .min()
        .unwrap_or(0);
    let mut out = Vec::with_capacity(4 * frames * front.bands());
    for spec in &specs {
        out.extend_from_slice(&spec[..frames * front.bands()]);
    }
    (out, frames)
}

/// Runs one fold over `features` (`[4, frames, 81]`), adding its logits to
/// `sum` (created on the first fold).
pub fn run_fold<M: Model>(
    model: &mut M,
    features: &[f32],
    frames: usize,
    sum: &mut Option<Logits>,
) -> anyhow::Result<()> {
    let bands = features.len() / (4 * frames.max(1));
    let result = model.run(&[(
        "features",
        &Tensor {
            shape: vec![1, 4, frames, bands],
            data: features.to_vec(),
        },
    )])?;
    let section = result
        .get("section")
        .ok_or_else(|| anyhow::anyhow!("all-in-one: no 'section' output"))?;
    let function = result
        .get("function")
        .ok_or_else(|| anyhow::anyhow!("all-in-one: no 'function' output"))?;
    anyhow::ensure!(
        section.data.len() == frames && function.data.len() == LABEL_COUNT * frames,
        "all-in-one: output shapes {:?} / {:?}",
        section.shape,
        function.shape
    );
    match sum {
        None => {
            *sum = Some(Logits {
                frames,
                section: section.data.clone(),
                function: function.data.clone(),
            })
        }
        Some(total) => {
            for (a, b) in total.section.iter_mut().zip(&section.data) {
                *a += b;
            }
            for (a, b) in total.function.iter_mut().zip(&function.data) {
                *a += b;
            }
        }
    }
    Ok(())
}

/// The segments of the ensemble mean of `folds` summed logits.
pub fn segments(sum: &Logits, folds: usize) -> Vec<Segment> {
    let frames = sum.frames;
    if frames == 0 {
        return Vec::new();
    }
    let k = folds.max(1) as f32;
    let probability: Vec<f32> = sum
        .section
        .iter()
        .map(|&x| 1.0 / (1.0 + (-x / k).exp()))
        .collect();
    let maxima = local_maxima(&probability, 4 * MIN_HOPS_PER_BEAT + 1);
    let strength = peak_picking(&maxima, WINDOW, WINDOW);
    let boundaries: Vec<usize> = (0..frames).filter(|&i| strength[i] > 0.0).collect();

    // Softmax over the classes, per frame.
    let mut function = vec![0.0f32; LABEL_COUNT * frames];
    for t in 0..frames {
        let column = |c: usize| sum.function[c * frames + t] / k;
        let max = (0..LABEL_COUNT)
            .map(column)
            .fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = (0..LABEL_COUNT).map(|c| (column(c) - max).exp()).collect();
        let total: f32 = exps.iter().sum();
        for c in 0..LABEL_COUNT {
            function[c * frames + t] = exps[c] / total;
        }
    }

    let mut cuts: Vec<usize> = boundaries.iter().copied().filter(|&i| i > 0).collect();
    cuts.push(frames);
    let duration = frames as f32 / FPS as f32;
    let mut times: Vec<f32> = boundaries.iter().map(|&i| i as f32 / FPS as f32).collect();
    if times.first() != Some(&0.0) {
        times.insert(0, 0.0);
    }
    if times.last() != Some(&duration) {
        times.push(duration);
    }
    let mut out = Vec::new();
    let mut from = 0;
    for (index, &to) in cuts.iter().enumerate() {
        if index + 1 >= times.len() {
            break;
        }
        let label = (0..LABEL_COUNT)
            .map(|c| {
                let values =
                    &function[c * frames + from..c * frames + to.max(from + 1).min(frames)];
                values.iter().sum::<f32>() / values.len().max(1) as f32
            })
            .enumerate()
            .fold((0, f32::NEG_INFINITY), |best, (c, v)| {
                if v > best.1 {
                    (c, v)
                } else {
                    best
                }
            })
            .0;
        out.push(Segment {
            start: times[index],
            end: times[index + 1],
            label: label as u8,
        });
        from = to;
    }
    out
}

/// `allin1.postprocessing.helpers.local_maxima`: values equal to the
/// maximum of their centered window (padded with −∞), zero elsewhere.
fn local_maxima(x: &[f32], size: usize) -> Vec<f32> {
    let half = size / 2;
    (0..x.len())
        .map(|i| {
            let from = i.saturating_sub(half);
            let to = (i + half + 1).min(x.len());
            let max = x[from..to]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            if x[i] == max {
                x[i]
            } else {
                0.0
            }
        })
        .collect()
}

/// `allin1.postprocessing.helpers.peak_picking`: the strength of each local
/// maximum of the `past + future + 1` window (zero padded) against the mean of
/// the `past` frames before and `future` frames after; zero elsewhere.
fn peak_picking(a: &[f32], past: usize, future: usize) -> Vec<f32> {
    let n = a.len();
    let at = |i: isize| -> f32 {
        if i >= 0 && (i as usize) < n {
            a[i as usize]
        } else {
            0.0
        }
    };
    // Prefix sums for the window means.
    let mut prefix = vec![0.0f64; n + 1];
    for i in 0..n {
        prefix[i + 1] = prefix[i] + f64::from(a[i]);
    }
    let range_sum = |from: isize, to: isize| -> f64 {
        let from = from.clamp(0, n as isize) as usize;
        let to = to.clamp(0, n as isize) as usize;
        if to > from {
            prefix[to] - prefix[from]
        } else {
            0.0
        }
    };
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        if a[i] <= 0.0 {
            continue;
        }
        let ii = i as isize;
        let max = (ii - past as isize..=ii + future as isize)
            .map(at)
            .fold(f32::NEG_INFINITY, f32::max);
        if a[i] != max {
            continue;
        }
        let past_mean = range_sum(ii - past as isize, ii) / past as f64;
        let future_mean = range_sum(ii + 1, ii + 1 + future as isize) / future as f64;
        out[i] = (f64::from(a[i]) - (past_mean + future_mean) / 2.0) as f32;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_maxima_keeps_window_peaks() {
        let x = [0.1, 0.5, 0.2, 0.2, 0.9, 0.1];
        assert_eq!(local_maxima(&x, 3), vec![0.0, 0.5, 0.0, 0.0, 0.9, 0.0]);
    }

    #[test]
    fn peak_picking_scores_against_the_neighbourhood() {
        let mut a = vec![0.0f32; 40];
        a[20] = 1.0;
        a[5] = 0.2;
        let s = peak_picking(&a, 4, 4);
        assert!((s[20] - 1.0).abs() < 1e-6);
        assert!(s[5] > 0.0);
        assert_eq!(s[6], 0.0);
    }
}
