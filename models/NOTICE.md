# Model files

The ONNX models in this directory come from
[beat-this-rs](https://github.com/danigb/beat-this-rs) (`models/` of the
repository, copied unchanged), the Rust port of
["Beat This!"](https://github.com/CPJKU/beat_this) by the Institute of
Computational Perception, JKU Linz (ISMIR 2024). Both the port and the
original work, including the model weights, are distributed under the MIT
License; the copyright notices below are retained as that license requires.

| File | Size | Role |
| --- | --- | --- |
| `mel_spectrogram.onnx` | ~270 KB | log-mel front end (required) |
| `beat_this_small.onnx` | ~10 MB | small beat model (`small1` checkpoint), bundled |
| `beat_this.onnx` | ~83 MB | full-accuracy FP32 beat model (`final0`), **not bundled**: `tool/download-models.sh` |

SHA-256 of the bundled files:

```
a5f8d39d989f31859454ba27afe61c5317ca95e4d9373e6853e5361b8937172f  beat_this_small.onnx
fdd59e65c515331308e4c8841edf99972deca646bdf6197744c2a5b7755e3de9  mel_spectrogram.onnx
```

```
MIT License

Copyright (c) 2025 danigb (Rust port)
Copyright (c) 2024 Institute of Computational Perception, JKU Linz, Austria ("Beat This!" original work)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Drum model

`adtof_frame_rnn.onnx` (~1.8 MB) is the Frame_RNN drum transcription model of
[ADTOF](https://github.com/MZehren/ADTOF) — M. Zehren, M. Alunno, P. Bientinesi,
"ADTOF: A large dataset of non-synthetic music for automatic drum transcription"
(ISMIR 2021) and "High-quality and reproducible automatic drum transcription
from crowdsourced data" (Signals, 2023) — exported to ONNX by
`tool/export_adtof/export_adtof.py` from the PyTorch weights of
[xavriley/ADTOF-pytorch](https://github.com/xavriley/ADTOF-pytorch) (commit
`85c192e78f716ea0b111cc8a5ee4a8f6a3a4f8a9`, SHA-256 of
`adtof_frame_rnn_pytorch_weights.pth`:
`1bc986e596ec47ba0b44916f87cd4a39f0b2bec23596df3fb5d0e87749217320`). The
network and its weights are unchanged; only the GRU layers are expressed as
ONNX `GRU` operators.

The ADTOF work, including the model weights, is licensed under the
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
License](https://creativecommons.org/licenses/by-nc-sa/4.0/) (CC BY-NC-SA 4.0):
non-commercial use only, with attribution, and adaptations under the same
license. The rest of this package stays MIT.

## Chord model

`btc_chords.onnx` (~13 MB) is the BTC chord recognition model
(J. Park, K. Choi, S. Jeon, D. Kim, J. Park, "A Bi-Directional Transformer for
Musical Chord Recognition", ISMIR 2019) as retrained by
[ChordMini](https://github.com/ptnghia-j/ChordMini) with pseudo-labels and
continual learning — exported to ONNX by `tool/export_chords/export_chords.py`
from `checkpoints/btc_model_best.pth` (commit
`aa6e3a8d7b017f082fd2aaff9329d5c26af49c03`, SHA-256
`e0a12ca6d881f81e01dfb459b4836ec7dac8d8f87b0170a412b5807256b3a1fc`). The
network and its weights are unchanged; the output layer's unused LSTM is left
out. SHA-256 of the ONNX file:
`fef3841996727fb32ed053f57a2284944a9e6cd7f8c4c7fcd1207bf049c4c6bb`.

ChordMini, including its checkpoints, is published under the MIT License
(Copyright (c) 2026 ChordMini contributors), which excludes the datasets the
models were trained on. Some of those annotations are themselves licensed for
non-commercial use, and whether that reaches the weights is disputed; treat
the file as non-commercial when in doubt.

## Deep analysis models (downloaded, not committed)

`tool/download-deep-models.sh` fetches ONNX conversions published by LightCut
(`azstorm/lightcut-music-analysis-models` on Hugging Face, MIT, pinned to
commit `971366ce07f76f7236b6bca636e162ecc12e985f`, SHA-256 checked) and stores
them under these names:

| File | Size | Role |
| --- | --- | --- |
| `htdemucs_core.onnx` | ~174 MB | Hybrid Transformer Demucs (`htdemucs`) network, 4 stems |
| `allin1_fold0..7.onnx` | ~3.4 MB each | All-In-One (`harmonix-all`) fold models: song structure |

Hybrid Transformer Demucs, Alexandre Défossez (Meta), MIT License:

```
MIT License

Copyright (c) Meta Platforms, Inc. and affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

All-In-One Music Structure Analyzer, Taejun Kim and Juhan Nam (WASPAA 2023), MIT License:

```
MIT License

Copyright (c) 2023 Taejun Kim

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
