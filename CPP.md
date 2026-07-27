# CrisperWhisper.cpp

This repository now includes a native C++ inference path for the open
CrisperWhisper 2.0 checkpoints.  It uses ggml and whisper.cpp for CPU/CUDA
tensor execution, while keeping CrisperWhisper's custom mode prompts and
continuation-style long-audio decoding.

Python is used only once to download and convert a Hugging Face safetensors
checkpoint.  The resulting CLI and C++ library have no Python runtime
dependency.

## What works

- Verbatim and intended transcription.
- WAV, MP3, FLAC, and Ogg input, decoded and resampled to mono 16 kHz.
- CUDA acceleration, including RTX 50-series GPUs.
- CPU fallback.
- Experimental Linux ARM64 portable CPU release with native compilation,
  unit tests, and a Turbo-model timestamped inference smoke test.
- Audio longer than 30 seconds using 30-second windows, 26-second stride,
  12-word continuation context, and boundary re-coverage.
- Verbatimize prompts.
- Hotword prompts (effective on Pro checkpoints).
- Reusable `crisperwhisper::Model` C++ API.
- Plain-text and JSON CLI output.
- Opt-in supervised cross-attention word timestamps.

The port does not yet reproduce the Python package's speculative draft-model
decoding, dual-mode batching, or its full temperature/hallucination recovery
stack. Core transcription is native and uses the same checkpoint weights.

## License

The C++ and Python inference/conversion code is covered by this repository's
MIT license.  The model weights are separate: the open weights use Nyra's
non-commercial research license.  Read the checkpoint's `LICENSE.md` before
using it.

## Windows quick start

Prerequisites:

- Visual Studio 2022 C++ Build Tools.
- CMake.
- Python 3.10+ for the one-time conversion.
- For GPU builds: an NVIDIA driver and CUDA toolkit.

The Windows build script loads the Visual Studio 2022 compiler environment
and uses its bundled Ninja generator. This lets CMake drive `nvcc` directly,
even when CUDA's optional Visual Studio extension was not installed.

Download and convert the default large model:

```powershell
.\scripts\setup-model.ps1
```

The large download is about 3.1 GB.  Conversion produces an approximately
2.9 GB F16 ggml file in `models/`.

Build for the detected NVIDIA GPU:

```powershell
.\scripts\build.ps1
```

The default CUDA build embeds architectures `86;89;120`, covering RTX 3090,
RTX 4090, and RTX 5090-class GPUs in one executable. This universal target
requires CUDA Toolkit 12.8 or newer. To target only one:

```powershell
.\scripts\build.ps1 -CudaArchitectures "120"
```

Or build CPU-only:

```powershell
.\scripts\build.ps1 -Cpu
```

On x86-64, the default `balance` CPU profile targets AVX2 without build-machine-specific
tuning. Use `-CpuProfile portable` for baseline x86-64 compatibility or
`-CpuProfile fast` for native tuning on the machine that will run the binary:

```powershell
.\scripts\build.ps1 -Cpu -CpuProfile portable -BuildDirectory build-cpu-portable
.\scripts\build.ps1 -Cpu -CpuProfile fast -BuildDirectory build-cpu-fast
```

Transcribe:

```powershell
.\build\bin\crisper-whisper.exe `
  --model .\models\ggml-crisperwhisper-large-f16.bin `
  --file .\meeting.wav `
  --mode verbatim `
  --language en
```

Run `crisper-whisper.exe --help` to see every CLI option.

Clean intended text:

```powershell
.\build\bin\crisper-whisper.exe `
  -m .\models\ggml-crisperwhisper-large-f16.bin `
  -f .\meeting.wav `
  --mode intended
```

JSON output:

```powershell
.\build\bin\crisper-whisper.exe `
  -m .\models\ggml-crisperwhisper-large-f16.bin `
  -f .\meeting.wav `
  --json
```

Word timestamps are optional:

```powershell
.\build\bin\crisper-whisper.exe `
  -m .\models\ggml-crisperwhisper-large-f16.bin `
  -f .\meeting.wav `
  --word-timestamps `
  --json
```

Flash Attention remains active for transcription. Timestamp requests lazily
load a second context with Flash Attention disabled and perform one
teacher-forced alignment pass, since the fused kernel does not expose the
cross-attention probabilities needed by CrisperWhisper's supervised aligner.
Omitting `--word-timestamps` avoids that pass and its additional model memory.

## Linux quick start

Prerequisites on Debian/Ubuntu:

```bash
sudo apt update
sudo apt install -y build-essential cmake git python3-venv
```

Download and convert the model:

```bash
./scripts/setup-model.sh
```

Build automatically with CUDA when `nvcc` and an NVIDIA GPU are available,
otherwise build for CPU:

```bash
./scripts/build.sh
```

You can make the choice explicit:

```bash
./scripts/build.sh --cuda
./scripts/build.sh --cpu
./scripts/build.sh --cpu --cpu-profile portable
./scripts/build.sh --cpu --cpu-profile fast
```

Linux ARM64 CPU builds are experimental. Use the portable profile; the
repository builds and unit-tests it on a native GitHub ARM64 runner. ARM64
release jobs also run the Turbo model against `samples/jfk.wav` with
supervised word timestamps before packaging. Release binaries target Ubuntu
22.04-era GLIBC 2.35 and statically link the GNU C++ runtime:

```bash
./scripts/build.sh --cpu --cpu-profile portable
```

Transcribe:

```bash
./build/bin/crisper-whisper \
  --model ./models/ggml-crisperwhisper-large-f16.bin \
  --file ./meeting.wav \
  --mode verbatim \
  --language en
```

Install the CLI into the configured CMake prefix:

```bash
cmake --install build --config Release
```

## Portable manual CMake build

CPU:

```text
cmake -S . -B build \
  -DCRISPERWHISPER_CUDA=OFF \
  -DCRISPERWHISPER_CPU_PROFILE=balance
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

CUDA:

```text
cmake -S . -B build \
  -DCRISPERWHISPER_CUDA=ON \
  -DCRISPERWHISPER_CPU_PROFILE=balance
cmake --build build --config Release --parallel
```

The default wrapper targets `86;89;120`. For a manual RTX 5090-only build,
setting `-DCMAKE_CUDA_ARCHITECTURES=120` is supported.

## C++ API

```cpp
#include <crisperwhisper.h>
#include <iostream>

int main() {
    crisperwhisper::Model model(
        "models/ggml-crisperwhisper-large-f16.bin"
    );

    crisperwhisper::TranscriptionOptions options;
    options.mode = crisperwhisper::Mode::Verbatim;
    options.language = "en";
    options.word_timestamps = true;

    const auto result = model.transcribe_file("meeting.wav", options);
    std::cout << result.text << '\n';
    for (const auto & word : result.words) {
        std::cout << word.start_seconds << " -> "
                  << word.end_seconds << ": "
                  << word.word << '\n';
    }
}
```

## Why this is not a stock whisper.cpp conversion

CrisperWhisper 2.0 adds 32 vocabulary entries after Whisper's timestamp
tokens.  Stock whisper.cpp currently derives some control-token positions
from total vocabulary size, which makes its high-level decoder choose the
wrong task and timestamp IDs for this checkpoint.

This port serializes the complete Hugging Face vocabulary and uses
whisper.cpp's low-level encoder/decoder API with control IDs discovered from
the actual token strings.  That preserves CrisperWhisper's mode tokens while
retaining ggml's optimized CUDA and CPU kernels.
