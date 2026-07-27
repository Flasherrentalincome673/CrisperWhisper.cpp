# CrisperWhisper.cpp Benchmarks

Generated 2026-07-27T15:18:22.063248+00:00 on Windows with NVIDIA GeForce RTX 5090, 32607 MiB.

| Scenario | C++/ggml | Python/Transformers | C++ speedup | C++ RTF | Python RTF | Transcript agreement |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Cold start + 11s audio | 3.586 s | 14.297 s | 3.99x | — | — | Exact |
| Warm 11s audio | 0.462 s | 1.952 s | 4.23x | 0.0420 | 0.1774 | Exact |
| Warm synthetic long-form 55s audio | 2.065 s | 6.190 s | 3.00x | 0.0375 | 0.1125 | Exact |

## Model loading

| Engine | Median model-load time |
| --- | ---: |
| C++ / ggml CUDA | 2.759 s |
| Python / Transformers CUDA | 8.236 s |

## Methodology

- Model: `nyralabs/CrisperWhisper2.0_large`, FP16, verbatim mode.
- C++: this repository's ggml CUDA runtime with flash attention.
- Python: Nyra's public `CrisperWhisperModel`, `backend="transformers"`,
  PyTorch CUDA, eager attention as configured by the upstream engine.
- Matching settings: greedy decoding, 256 maximum new tokens, no speculative
  decoding, hallucination repair disabled, temperature fallback disabled,
  no word timestamps, and fixed two-word continuation boundary handling.
- Cold start is median fresh-process wall time over 3 runs. It
  includes program/import startup, model loading, audio loading, and inference.
  The operating-system file cache is not purged between runs.
- Warm short audio is median of 5 measured calls after one
  unmeasured warm-up with the model kept loaded.
- Long-form is a synthetic 55-second repetition of the bundled
  [`samples/jfk.wav`](samples/jfk.wav) from whisper.cpp, used for reproducible
  timing rather than as a natural-speech quality test. It uses 30-second
  chunks, 26-second stride, 12 context words, and two boundary words. It is
  the median of 3 calls after one warm-up.
- RTF is call wall time divided by audio duration; lower is better.
- The Windows Python comparison uses Transformers because Nyra's custom
  CTranslate2 wheel is currently published for Linux only.

Raw measurements: [`benchmarks/results/comparison.json`](benchmarks/results/comparison.json).
