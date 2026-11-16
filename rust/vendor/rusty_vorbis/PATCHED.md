
Patched copy of rusty_vorbis 0.1.3 (Apache-2.0, see LICENSE): `produce_all` encodes on the calling thread when
only one is available, so the encoder runs in wasm32-wasip1 (no threads). Used through `[patch.crates-io]`.
