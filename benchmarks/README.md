# Benchmarking C++ against Python

The benchmark suite compares this repository's native ggml/CUDA runtime with
Nyra's public Python API under matching decoding settings.

Windows uses the Transformers backend because the custom
`ctranslate2-crisperwhisper` wheel is Linux-only. Install the benchmark
environment:

```powershell
py -3.12 -m venv .venv-benchmark
.\.venv-benchmark\Scripts\python.exe -m pip install `
  torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
.\.venv-benchmark\Scripts\python.exe -m pip install -e ".[transformers]"
```

Build the CUDA benchmark executable:

```powershell
.\scripts\build.ps1 -BuildDirectory build-universal
```

Run cold-start, warm, and long-form tests:

```powershell
.\.venv-benchmark\Scripts\python.exe benchmarks\run_comparison.py `
  --cpp-exe build-universal\bin\crisper-whisper-bench.exe `
  --ggml-model models\ggml-crisperwhisper-large-f16.bin `
  --hf-model nyralabs/CrisperWhisper2.0_large `
  --audio C:\path\to\speech.wav
```

Without `--long-audio`, the runner repeats the short input WAV to 55 seconds.
That synthetic clip is useful for reproducible performance timing, but it is
not a natural long-form quality sample. To test an uninterrupted meeting,
interview, or podcast excerpt instead:

```powershell
.\.venv-benchmark\Scripts\python.exe benchmarks\run_comparison.py `
  --cpp-exe build-universal\bin\crisper-whisper-bench.exe `
  --ggml-model models\ggml-crisperwhisper-large-f16.bin `
  --hf-model nyralabs/CrisperWhisper2.0_large `
  --audio C:\path\to\short-speech.wav `
  --long-audio C:\path\to\real-long-recording.wav
```

The runner writes `BENCHMARKS.md` and the raw measurements in
`benchmarks/results/comparison.json`.
