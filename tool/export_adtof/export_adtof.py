# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#   "torch>=2.4",
#   "numpy>=1.26",
#   "onnx>=1.16",
#   "onnxruntime>=1.18",
#   "onnxscript>=0.1",
# ]
# ///
"""Exports the ADTOF Frame_RNN drum transcription model to ONNX.

Run manually (not part of the build): `uv run tool/export_adtof/export_adtof.py`.

1. Downloads the PyTorch weights of xavriley/ADTOF-pytorch (a port of
   MZehren/ADTOF, CC BY-NC-SA 4.0) at a pinned commit and checks their SHA-256.
2. Rebuilds the Frame_RNN network (the port's `model.py`, condensed) with
   `nn.GRU` — the same maths as the Keras GRU with `reset_after=True`, which
   ONNX expresses as `GRU` with `linear_before_reset=1` (supported by rten);
   the script asserts the equivalence against the port's explicit cell.
3. Exports `models/adtof_frame_rnn.onnx`: input `spectrogram` `[1, T, 84]`
   (the log filterbank below), output `activations` `[1, T, 5]` (sigmoid, at
   100 fps; classes kick, snare, toms, hi-hat, cymbals), dynamic `T`, and
   checks onnxruntime against PyTorch.
4. Writes the Rust parity fixture `rust/tests/fixtures/adtof/`: a synthetic
   2 s drum loop (16-bit PCM), its front-end spectrogram and activations.

The front end (`spectrogram`) mirrors the port's `audio.py` (itself matching
madmom's `LogarithmicFilteredSpectrogram` as ADTOF trained with it): 44.1 kHz
mono, symmetric Hann window of 2048, hop 441 (100 fps), centered frames with
zero padding, magnitude of bins 0..1023, a triangular log filterbank of 12
bands per octave between 20 Hz and 20 kHz on unique FFT bins (each filter
normalized to sum 1), then `log10(1 + x)`. `rust/src/drums.rs` implements the
same steps.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

COMMIT = "85c192e78f716ea0b111cc8a5ee4a8f6a3a4f8a9"
WEIGHTS_URL = (
    f"https://raw.githubusercontent.com/xavriley/ADTOF-pytorch/{COMMIT}"
    "/src/adtof_pytorch/data/adtof_frame_rnn_pytorch_weights.pth"
)
WEIGHTS_SHA256 = "1bc986e596ec47ba0b44916f87cd4a39f0b2bec23596df3fb5d0e87749217320"

SAMPLE_RATE = 44100
FRAME_SIZE = 2048
HOP = 441
BANDS_PER_OCTAVE = 12
FMIN = 20.0
FMAX = 20000.0

PACKAGE = Path(__file__).resolve().parents[2]
MODEL_OUT = PACKAGE / "models" / "adtof_frame_rnn.onnx"
FIXTURE_DIR = PACKAGE / "rust" / "tests" / "fixtures" / "adtof"


# --- front end --------------------------------------------------------------


def log_frequencies() -> np.ndarray:
    freqs = []
    factor = 2.0 ** (1.0 / BANDS_PER_OCTAVE)
    f = FMIN
    while f <= FMAX * (1.0 + 1e-12):
        freqs.append(f)
        f *= factor
    return np.array(freqs)


def filterbank() -> np.ndarray:
    fft_freqs = np.fft.fftfreq(FRAME_SIZE, 1 / SAMPLE_RATE)[: FRAME_SIZE // 2]
    bins = []
    last = -1
    for f in log_frequencies():
        b = int(np.argmin(np.abs(fft_freqs - f)))
        if b > last:
            bins.append(b)
            last = b
    n_fft_bins = len(fft_freqs)
    fb = np.zeros((len(bins) - 2, n_fft_bins), dtype=np.float32)
    for i in range(len(bins) - 2):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if right - left < 2:
            fb[i, left] = 1.0
            continue
        for b in range(left, center):
            fb[i, b] = (b - left) / float(center - left)
        fb[i, center] = 1.0
        for b in range(center + 1, min(right, n_fft_bins)):
            fb[i, b] = (right - b) / float(right - center)
    sums = fb.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1
    return (fb / sums).astype(np.float32)


def spectrogram(audio: np.ndarray) -> np.ndarray:
    """[T, 84] float32 log filtered spectrogram of mono 44.1 kHz audio."""
    window = np.hanning(FRAME_SIZE).astype(np.float32)
    padded = np.pad(audio.astype(np.float32), FRAME_SIZE // 2)
    frames = 1 + len(audio) // HOP
    fb = filterbank()
    out = np.empty((frames, fb.shape[0]), dtype=np.float32)
    for t in range(frames):
        frame = padded[t * HOP : t * HOP + FRAME_SIZE]
        if len(frame) < FRAME_SIZE:
            frame = np.pad(frame, (0, FRAME_SIZE - len(frame)))
        mag = np.abs(np.fft.rfft(frame * window))[: FRAME_SIZE // 2].astype(np.float32)
        out[t] = np.log10(1.0 + fb @ mag)
    return out


# --- network ----------------------------------------------------------------


class SamePadWidthMaxPool(nn.Module):
    """Keras 'same' max pooling over the frequency axis (kernel 3, stride 3)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freq = x.shape[-1]
        out_w = (freq + 2) // 3
        pad = max(0, (out_w - 1) * 3 + 3 - freq)
        if pad:
            x = torch.nn.functional.pad(x, (pad // 2, pad - pad // 2, 0, 0))
        return torch.nn.functional.max_pool2d(x, kernel_size=(1, 3), stride=(1, 3))


def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(cout, eps=1e-3),
        nn.Conv2d(cout, cout, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(cout, eps=1e-3),
        SamePadWidthMaxPool(),
        nn.Dropout2d(0.3),
    )


class FrameRnn(nn.Module):
    """ADTOF Frame_RNN (context 9 = the CNN's receptive field, so no context layer)."""

    def __init__(self, n_bins: int):
        super().__init__()
        self.cnn_blocks = nn.ModuleList([conv_block(1, 32), conv_block(32, 64)])
        w = n_bins
        for _ in range(2):
            w = (w + 2) // 3
        self.features = 64 * w
        self.gru_layers = nn.ModuleList(
            [nn.GRU(self.features if i == 0 else 120, 60, batch_first=True, bidirectional=True) for i in range(3)]
        )
        self.output_layer = nn.Linear(120, 5)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        # spec: [1, T, bins] -> [1, 1, T, bins]
        x = spec.unsqueeze(1)
        for block in self.cnn_blocks:
            x = block(x)
        # [1, C, T, W] -> [1, T, W, C] -> [1, T, W*C] (the port's feature order)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(1, -1, self.features)
        for gru in self.gru_layers:
            x, _ = gru(x)
        return torch.sigmoid(self.output_layer(x))


def keras_gru_reference(gru: nn.GRU, x: torch.Tensor) -> torch.Tensor:
    """The port's explicit KerasGRUCell (reset_after=True), bidirectional."""

    def run(prefix: str, seq):
        w_ih, w_hh = getattr(gru, f"weight_ih_l0{prefix}"), getattr(gru, f"weight_hh_l0{prefix}")
        b_ih, b_hh = getattr(gru, f"bias_ih_l0{prefix}"), getattr(gru, f"bias_hh_l0{prefix}")
        h_size = gru.hidden_size
        h = x.new_zeros(1, h_size)
        outs = {}
        for t in seq:
            xt = x[:, t, :]
            gi = xt @ w_ih.T + b_ih
            gh = h @ w_hh.T + b_hh
            r = torch.sigmoid(gi[:, :h_size] + gh[:, :h_size])
            z = torch.sigmoid(gi[:, h_size : 2 * h_size] + gh[:, h_size : 2 * h_size])
            n = torch.tanh(gi[:, 2 * h_size :] + r * gh[:, 2 * h_size :])
            h = (1 - z) * n + z * h
            outs[t] = h
        return torch.stack([outs[t] for t in range(x.shape[1])], dim=1)

    steps = range(x.shape[1])
    return torch.cat([run("", steps), run("_reverse", reversed(steps))], dim=2)


# --- fixture signal ---------------------------------------------------------


def drum_loop(seconds: float = 2.0) -> np.ndarray:
    """A deterministic 120 BPM loop: kick on every beat, snare on 2 and 4,
    closed hi-hat on eighths. 16-bit quantized so the fixture is exact."""
    rng = np.random.default_rng(1234)
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    out = np.zeros(n)
    beat = 0.5
    for k in range(int(seconds / beat) + 1):
        start = 0.1 + k * beat
        i0 = int(start * SAMPLE_RATE)
        if i0 >= n:
            break
        d = t[i0:] - start
        # kick: pitch sweep 150 -> 50 Hz
        phase = 2 * np.pi * (50 * d + (100 / 30) * (1 - np.exp(-30 * d)))
        out[i0:] += 0.9 * np.exp(-d / 0.15) * np.sin(phase)
        if k % 2 == 1:
            noise = rng.uniform(-1, 1, n - i0)
            out[i0:] += 0.5 * np.exp(-d / 0.08) * (0.6 * noise + 0.4 * np.sin(2 * np.pi * 190 * d))
    for k in range(int(seconds / (beat / 2)) + 1):
        start = 0.1 + k * beat / 2
        i0 = int(start * SAMPLE_RATE)
        if i0 >= n:
            break
        d = t[i0:] - start
        noise = rng.uniform(-1, 1, n - i0)
        hiss = np.diff(noise, prepend=0.0)  # crude high-pass
        out[i0:] += 0.25 * np.exp(-d / 0.02) * hiss
    out = np.clip(out / max(1.0, np.abs(out).max() * 1.1), -1, 1)
    return np.round(out * 32767).astype(np.int16)


# --- main -------------------------------------------------------------------


def load_weights() -> dict:
    cache = Path.home() / ".cache" / "icicles_dsp" / f"adtof_{COMMIT[:12]}.pth"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {WEIGHTS_URL}")
        urllib.request.urlretrieve(WEIGHTS_URL, cache)
    digest = hashlib.sha256(cache.read_bytes()).hexdigest()
    if digest != WEIGHTS_SHA256:
        sys.exit(f"SHA-256 mismatch for {cache}: {digest}")
    checkpoint = torch.load(cache, map_location="cpu", weights_only=True)
    return checkpoint.get("model_weights", checkpoint)


def main() -> None:
    torch.manual_seed(0)
    n_bins = filterbank().shape[0]
    print(f"filterbank: {n_bins} bands")
    model = FrameRnn(n_bins)
    weights = load_weights()
    missing, unexpected = model.load_state_dict(weights, strict=False)
    missing = [k for k in missing if "num_batches_tracked" not in k]
    if missing or unexpected:
        sys.exit(f"weights do not fit: missing {missing}, unexpected {unexpected}")
    model.eval()

    # nn.GRU == the port's Keras cell.
    with torch.no_grad():
        x = torch.randn(1, 37, model.features)
        ours, _ = model.gru_layers[0](x)
        ref = keras_gru_reference(model.gru_layers[0], x)
        gru_diff = (ours - ref).abs().max().item()
    print(f"nn.GRU vs Keras cell: max diff {gru_diff:.2e}")
    assert gru_diff < 1e-5

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 300, n_bins)
    torch.onnx.export(
        model,
        (dummy,),
        str(MODEL_OUT),
        input_names=["spectrogram"],
        output_names=["activations"],
        dynamic_axes={"spectrogram": {1: "time"}, "activations": {1: "time"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"wrote {MODEL_OUT} ({MODEL_OUT.stat().st_size / 1e6:.1f} MB)")

    import onnx
    import onnxruntime as ort

    graph = onnx.load(str(MODEL_OUT))
    ops = sorted({node.op_type for node in graph.graph.node})
    print(f"ops: {ops}")
    for node in graph.graph.node:
        if node.op_type == "GRU":
            lbr = [a.i for a in node.attribute if a.name == "linear_before_reset"]
            assert lbr == [1], f"GRU linear_before_reset {lbr}"
    session = ort.InferenceSession(str(MODEL_OUT))
    for frames in (50, 431, 1200):
        spec = np.random.default_rng(frames).uniform(0, 1.5, (1, frames, n_bins)).astype(np.float32)
        with torch.no_grad():
            expected = model(torch.from_numpy(spec)).numpy()
        actual = session.run(None, {"spectrogram": spec})[0]
        diff = np.abs(expected - actual).max()
        print(f"onnxruntime vs torch, T={frames}: max diff {diff:.2e}")
        assert diff < 1e-4

    pcm = drum_loop()
    audio = pcm.astype(np.float32) / 32768.0
    spec = spectrogram(audio)
    with torch.no_grad():
        act = model(torch.from_numpy(spec[None])).numpy()[0]
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    (FIXTURE_DIR / "drum_loop_44100.pcm").write_bytes(pcm.astype("<i2").tobytes())
    (FIXTURE_DIR / "drum_loop.json").write_text(
        json.dumps(
            {
                "sampleRate": SAMPLE_RATE,
                "samples": int(len(pcm)),
                "frames": int(spec.shape[0]),
                "bins": int(spec.shape[1]),
                "classes": 5,
                "spectrogram": [round(float(v), 6) for v in spec.reshape(-1)],
                "activations": [round(float(v), 6) for v in act.reshape(-1)],
            }
        )
    )
    peaks = {c: [round(i / 100, 2) for i in range(1, len(act) - 1)
                 if act[i, c] > 0.3 and act[i, c] >= act[i - 1, c] and act[i, c] > act[i + 1, c]]
             for c in range(5)}
    print(f"fixture: {spec.shape[0]} frames; activation peaks > 0.3 per class: {peaks}")


if __name__ == "__main__":
    main()
