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
