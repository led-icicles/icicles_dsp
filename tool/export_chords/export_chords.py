# /// script
# requires-python = ">=3.10,<3.12"
# dependencies = [
#   "torch>=2.4",
#   "numpy>=1.26,<2",
#   "librosa==0.10.1",
#   "setuptools<81",
#   "scipy>=1.11,<1.14",
#   "soxr>=0.3.7",
#   "onnx>=1.16",
#   "onnxruntime>=1.18",
#   "onnxscript>=0.1",
# ]
# ///
"""Exports the ChordMini BTC chord recognition model to ONNX.

Run manually (not part of the build): `uv run tool/export_chords/export_chords.py`.

1. Downloads `checkpoints/btc_model_best.pth` of ptnghia-j/ChordMini (MIT; the
   BTC architecture of Park et al., ISMIR 2019, retrained with pseudo-labels
   and continual learning) at a pinned commit and checks its SHA-256.
2. Rebuilds the network (the repository's `btc_model.py` and
   `transformer_modules.py`, condensed; the output layer's LSTM is never used
   by the forward pass and is left out).
3. Exports `models/btc_chords.onnx`: input `features` `[batch, 108, 144]` (the
   normalized log-CQT below, 108 frames = 10 s), output `logits`
   `[batch, 108, 170]`, dynamic batch, and checks onnxruntime against PyTorch.
4. Writes the Rust parity fixture `rust/tests/fixtures/chords/`: a synthetic
   chord progression (16-bit PCM, 44.1 kHz), its log-CQT and the frame labels.

The front end is ChordMini's (`src/evaluation/utils/common.py`): mono audio at
22 050 Hz, `librosa.cqt` (0.10.1) over the whole song with 144 bins, 24 per
octave from C1, hop 2048, then `log(|cqt| + 1e-6)` and
`(x - mean) / (std + 1e-8)` with the checkpoint's scalars. Inference
(`reference_labels`) runs windows of 108 frames with a stride of 54 over the
features zero-padded to a multiple of 108, averages the logits of the windows
covering a frame, smooths them over time with a Gaussian of 9 frames and takes
the argmax. `rust/src/chords.rs` implements the same steps; its resampling
filters are its own, so the features match closely, not bit for bit.

Class index `i < 168`: root `i // 14` (C = 0), quality `i % 14` of
min, maj, dim, aug, min6, maj6, min7, minmaj7, maj7, 7, dim7, hdim7, sus2,
sus4; 168 = X (unknown chord), 169 = N (no chord).
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

COMMIT = "aa6e3a8d7b017f082fd2aaff9329d5c26af49c03"
WEIGHTS_URL = f"https://raw.githubusercontent.com/ptnghia-j/ChordMini/{COMMIT}/checkpoints/btc_model_best.pth"
WEIGHTS_SHA256 = "e0a12ca6d881f81e01dfb459b4836ec7dac8d8f87b0170a412b5807256b3a1fc"

AUDIO_RATE = 44100
SAMPLE_RATE = 22050
HOP = 2048
BINS = 144
BINS_PER_OCTAVE = 24
SEQ_LEN = 108
STRIDE = 54
CLASSES = 170
HIDDEN = 128
LAYERS = 8
HEADS = 4
KERNEL = 9

PACKAGE = Path(__file__).resolve().parents[2]
MODEL_OUT = PACKAGE / "models" / "btc_chords.onnx"
FIXTURE_DIR = PACKAGE / "rust" / "tests" / "fixtures" / "chords"


# --- network (ChordMini, condensed) ------------------------------------------


def bias_mask(length: int) -> torch.Tensor:
    mask = np.triu(np.full([length, length], -np.inf), 1)
    return torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(1)


def timing_signal(length: int, channels: int) -> torch.Tensor:
    position = np.arange(length)
    timescales = channels // 2
    increment = np.log(1.0e4) / max(float(timescales) - 1, 1)
    inverse = np.exp(np.arange(timescales).astype(np.float64) * -increment)
    scaled = np.expand_dims(position, 1) * np.expand_dims(inverse, 0)
    signal = np.concatenate([np.sin(scaled), np.cos(scaled)], axis=1)
    return torch.from_numpy(signal.reshape([1, length, channels])).float()


class LayerNorm(nn.Module):
    """The repository's own layer norm: unbiased std, eps outside the root."""

    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        centered = x - mean
        # x.std(-1) spelled out, so the export needs no ReduceL2/Std operator.
        std = torch.sqrt((centered * centered).sum(-1, keepdim=True) / (x.shape[-1] - 1))
        return self.gamma * centered / (std + self.eps) + self.beta


class MultiHeadAttention(nn.Module):
    def __init__(self, depth: int, heads: int, mask: torch.Tensor):
        super().__init__()
        self.heads = heads
        self.query_scale = (depth // heads) ** -0.5
        self.register_buffer("mask", mask, persistent=False)
        self.query_linear = nn.Linear(depth, depth, bias=False)
        self.key_linear = nn.Linear(depth, depth, bias=False)
        self.value_linear = nn.Linear(depth, depth, bias=False)
        self.output_linear = nn.Linear(depth, depth, bias=False)

    def split(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(-1, SEQ_LEN, self.heads, HIDDEN // self.heads).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        queries = self.split(self.query_linear(x)) * self.query_scale
        keys = self.split(self.key_linear(x))
        values = self.split(self.value_linear(x))
        weights = F.softmax(torch.matmul(queries, keys.permute(0, 1, 3, 2)) + self.mask, dim=-1)
        contexts = torch.matmul(weights, values).permute(0, 2, 1, 3).reshape(-1, SEQ_LEN, HIDDEN)
        return self.output_linear(contexts)


class Conv(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.conv = nn.Conv1d(size, size, kernel_size=3, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x.permute(0, 2, 1), (1, 1))).permute(0, 2, 1)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.layers = nn.ModuleList([Conv(size), Conv(size)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = F.relu(layer(x))  # upstream applies the ReLU after the last layer too
        return x


class SelfAttentionBlock(nn.Module):
    def __init__(self, mask: torch.Tensor):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(HIDDEN, HEADS, mask)
        self.positionwise_convolution = PositionwiseFeedForward(HIDDEN)
        self.layer_norm_mha = LayerNorm(HIDDEN)
        self.layer_norm_ffn = LayerNorm(HIDDEN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.multi_head_attention(self.layer_norm_mha(x))
        return x + self.positionwise_convolution(self.layer_norm_ffn(x))


class BiDirSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        mask = bias_mask(SEQ_LEN)
        self.attn_block = SelfAttentionBlock(mask)
        self.backward_attn_block = SelfAttentionBlock(torch.transpose(mask, 2, 3))
        self.linear = nn.Linear(HIDDEN * 2, HIDDEN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.cat((self.attn_block(x), self.backward_attn_block(x)), dim=2))


class BiDirSelfAttentionLayers(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("timing", timing_signal(SEQ_LEN, HIDDEN), persistent=False)
        self.embedding_proj = nn.Linear(BINS, HIDDEN, bias=False)
        self.self_attn_layers = nn.Sequential(*[BiDirSelfAttention() for _ in range(LAYERS)])
        self.layer_norm = LayerNorm(HIDDEN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(self.self_attn_layers(self.embedding_proj(x) + self.timing))


class OutputLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_projection = nn.Linear(HIDDEN, CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_projection(x)


class Btc(nn.Module):
    """`features` [batch, 108, 144] (normalized) -> `logits` [batch, 108, 170]."""

    def __init__(self):
        super().__init__()
        self.self_attn_layers = BiDirSelfAttentionLayers()
        self.output_layer = OutputLayer()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.self_attn_layers(features))


# --- front end and inference (the reference of rust/src/chords.rs) ------------


def log_cqt(audio_22k: np.ndarray) -> np.ndarray:
    """[frames, 144] float32 log-CQT of mono 22.05 kHz audio."""
    import librosa

    cqt = librosa.cqt(
        audio_22k,
        sr=SAMPLE_RATE,
        n_bins=BINS,
        bins_per_octave=BINS_PER_OCTAVE,
        hop_length=HOP,
        fmin=librosa.note_to_hz("C1"),
    )
    return np.log(np.abs(cqt) + 1e-6).T.astype(np.float32)


def frame_logits(model: nn.Module, features: np.ndarray, mean: float, std: float) -> np.ndarray:
    """[frames, 170]: the averaged logits of the overlapping windows."""
    frames = features.shape[0]
    pad = (-frames) % SEQ_LEN
    padded = np.pad(features, ((0, pad), (0, 0)))
    starts = [STRIDE * i for i in range((padded.shape[0] - SEQ_LEN) // STRIDE + 1)]
    windows = np.stack([padded[s : s + SEQ_LEN] for s in starts])
    with torch.no_grad():
        logits = model((torch.from_numpy(windows) - mean) / (std + 1e-8)).numpy()
    total = np.zeros((frames, CLASSES), dtype=np.float32)
    count = np.zeros(frames, dtype=np.float32)
    for window, start in zip(logits, starts):
        valid = min(SEQ_LEN, frames - start)
        if valid <= 0:
            continue
        total[start : start + valid] += window[:valid]
        count[start : start + valid] += 1
    return total / count[:, None]


def smooth(logits: np.ndarray) -> np.ndarray:
    """A Gaussian of 9 frames (sigma 1.5) over time, edges replicated."""
    offsets = np.arange(KERNEL) - KERNEL // 2
    kernel = np.exp(-0.5 * (offsets / (KERNEL / 6.0)) ** 2)
    kernel = (kernel / kernel.sum()).astype(np.float32)
    padded = np.pad(logits, ((KERNEL // 2, KERNEL // 2), (0, 0)), mode="edge")
    out = np.zeros_like(logits)
    for i, weight in enumerate(kernel):
        out += weight * padded[i : i + logits.shape[0]]
    return out


def reference_labels(model: nn.Module, features: np.ndarray, mean: float, std: float) -> np.ndarray:
    return smooth(frame_logits(model, features, mean, std)).argmax(axis=1)


# --- fixture signal ----------------------------------------------------------

# (root pitch class, semitones above the root), two seconds each.
PROGRESSION = [
    (0, (0, 4, 7)),  # C
    (9, (0, 3, 7)),  # A:min
    (5, (0, 4, 7)),  # F
    (7, (0, 4, 7)),  # G
    (2, (0, 3, 7, 10)),  # D:min7
    (7, (0, 4, 7, 10)),  # G:7
    (0, (0, 4, 7, 11)),  # C:maj7
    (4, (0, 3, 7)),  # E:min
    (9, (0, 3, 7)),  # A:min
    (5, (0, 4, 7)),  # F
    (0, (0, 4, 7)),  # C
    (7, (0, 4, 7)),  # G
]
CHORD_SECONDS = 2.0


def chord_progression() -> np.ndarray:
    """A deterministic organ-like progression: a bass note and a triad or
    seventh chord of decaying harmonics per chord, re-struck every half
    second. 16-bit quantized so the fixture is exact."""
    per_chord = int(CHORD_SECONDS * AUDIO_RATE)
    out = np.zeros(per_chord * len(PROGRESSION))
    t = np.arange(per_chord) / AUDIO_RATE
    strike = np.exp(-(t % 0.5) / 0.6) * np.minimum(1.0, (t % 0.5) / 0.01)
    fade = np.minimum(1.0, (CHORD_SECONDS - t) / 0.02)
    for k, (root, intervals) in enumerate(PROGRESSION):
        notes = [(36 + root, 0.9)] + [(60 + root + interval, 0.5) for interval in intervals]
        chord = np.zeros(per_chord)
        for midi, gain in notes:
            hz = 440.0 * 2.0 ** ((midi - 69) / 12)
            for harmonic in range(1, 7):
                chord += gain / harmonic**1.5 * np.sin(2 * np.pi * hz * harmonic * t)
        out[k * per_chord : (k + 1) * per_chord] = chord * strike * fade
    out = out / (np.abs(out).max() * 1.1)
    return np.round(out * 32767).astype(np.int16)


def label_name(index: int) -> str:
    if index == 169:
        return "N"
    if index == 168:
        return "X"
    roots = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    qualities = ["min", "maj", "dim", "aug", "min6", "maj6", "min7", "minmaj7", "maj7", "7", "dim7", "hdim7", "sus2", "sus4"]
    return f"{roots[index // 14]}:{qualities[index % 14]}"


# --- main --------------------------------------------------------------------


def load_checkpoint() -> dict:
    cache = Path.home() / ".cache" / "icicles_dsp" / f"chordmini_btc_{COMMIT[:12]}.pth"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {WEIGHTS_URL}")
        urllib.request.urlretrieve(WEIGHTS_URL, cache)
    digest = hashlib.sha256(cache.read_bytes()).hexdigest()
    if digest != WEIGHTS_SHA256:
        sys.exit(f"SHA-256 mismatch for {cache}: {digest}")
    return torch.load(cache, map_location="cpu", weights_only=True)


def main() -> None:
    import librosa

    torch.manual_seed(0)
    checkpoint = load_checkpoint()
    mean = float(checkpoint["normalization"]["mean"])
    std = float(checkpoint["normalization"]["std"])
    print(f"normalization: mean {mean!r}, std {std!r}")
    assert checkpoint["idx_to_chord"][1] == "C" and checkpoint["idx_to_chord"][169] == "N"

    model = Btc()
    weights = {k: v for k, v in checkpoint["model_state_dict"].items() if ".lstm." not in k}
    model.load_state_dict(weights, strict=True)
    model.eval()
    print(f"parameters: {sum(p.numel() for p in model.parameters())}")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (torch.randn(2, SEQ_LEN, BINS),),
        str(MODEL_OUT),
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"wrote {MODEL_OUT} ({MODEL_OUT.stat().st_size / 1e6:.1f} MB)")

    import onnx
    import onnxruntime as ort

    graph = onnx.load(str(MODEL_OUT))
    print(f"ops: {sorted({node.op_type for node in graph.graph.node})}")
    session = ort.InferenceSession(str(MODEL_OUT))
    for batch in (1, 3, 7):
        features = np.random.default_rng(batch).normal(0, 1, (batch, SEQ_LEN, BINS)).astype(np.float32)
        with torch.no_grad():
            expected = model(torch.from_numpy(features)).numpy()
        actual = session.run(None, {"features": features})[0]
        diff = np.abs(expected - actual).max()
        print(f"onnxruntime vs torch, batch {batch}: max diff {diff:.2e}")
        assert diff < 1e-4

    pcm = chord_progression()
    audio = pcm.astype(np.float32) / 32768.0
    audio_22k = librosa.resample(audio, orig_sr=AUDIO_RATE, target_sr=SAMPLE_RATE, res_type="soxr_hq")
    features = log_cqt(audio_22k)
    labels = reference_labels(model, features, mean, std)
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    (FIXTURE_DIR / "progression_44100.pcm").write_bytes(pcm.astype("<i2").tobytes())
    (FIXTURE_DIR / "progression.json").write_text(
        json.dumps(
            {
                "sampleRate": AUDIO_RATE,
                "samples": int(len(pcm)),
                "frames": int(features.shape[0]),
                "bins": BINS,
                "mean": mean,
                "std": std,
                "chordSeconds": CHORD_SECONDS,
                "progression": [[root, 1 if 4 in intervals else 0] for root, intervals in PROGRESSION],
                "features": [round(float(v), 4) for v in features.reshape(-1)],
                "labels": [int(v) for v in labels],
            }
        )
    )
    seconds = HOP / SAMPLE_RATE
    runs = []
    for i, label in enumerate(labels):
        if not runs or runs[-1][1] != label:
            runs.append([i * seconds, label])
    print(f"fixture: {features.shape[0]} frames; " + ", ".join(f"{t:.1f}s {label_name(l)}" for t, l in runs))


if __name__ == "__main__":
    main()
