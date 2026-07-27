# Building CrisperWhisper.cpp on Linux

![Linux](https://img.shields.io/badge/Linux-x86--64-FCC624)
![Compiler](https://img.shields.io/badge/compiler-GCC%20%7C%20Clang-00599C)
![CUDA](https://img.shields.io/badge/CUDA-optional-76B900)
![Build](https://img.shields.io/badge/build-CMake%203.20%2B-success)

This guide covers native CPU and NVIDIA CUDA builds on Linux. Runtime
transcription does not require Python. Python is only needed when converting
an original checkpoint yourself; preconverted models are available from
[`drbaph/CrisperWhisper2.0-GGML`](https://huggingface.co/drbaph/CrisperWhisper2.0-GGML).

Prebuilt x86-64 portable and AVX2 CPU ZIPs, plus an experimental ARM64
portable CPU ZIP, are published on
[GitHub Releases](https://github.com/Saganaki22/CrisperWhisper.cpp/releases).
Linux CUDA remains a native source build until the Linux CUDA archive has been
built and verified on Linux. No Docker workflow is required by this guide.

## Supported targets

Primary target:

- Linux x86-64
- GCC or Clang with C++17
- CMake 3.20+
- CPU inference with ggml
- NVIDIA CUDA inference when the CUDA toolkit is installed

Experimental target:

- Linux ARM64 CPU with the `portable` profile
- Native compile/unit-test CI on `ubuntu-22.04-arm`
- Turbo-model timestamped inference smoke test on the native ARM64 runner
- Prebuilt `linux-arm64-cpu-portable` release archive
- GLIBC 2.35 compatibility floor for Ubuntu 22.04-era systems

Linux ARM64 remains experimental pending testing on a wider range of boards,
including Orange Pi. Windows ARM64 and ARM CUDA/Jetson are not v1.1.0 targets.
The repository's primary CPU CI target is Ubuntu x86-64; other modern
distributions should work through the same CMake build.

## 1. Install build requirements

<details open>
<summary><strong>Ubuntu / Debian</strong></summary>

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  git \
  python3 \
  python3-venv \
  python3-pip
```

</details>

<details>
<summary><strong>Fedora</strong></summary>

```bash
sudo dnf install -y \
  gcc \
  gcc-c++ \
  cmake \
  git \
  python3 \
  python3-pip
```

Fedora normally includes the `venv` module in its Python package.

</details>

<details>
<summary><strong>Arch Linux</strong></summary>

```bash
sudo pacman -S --needed \
  base-devel \
  cmake \
  git \
  python \
  python-pip
```

</details>

## 2. Download and convert the model

From the repository root:

```bash
./scripts/setup-model.sh
```

This creates `.venv/`, installs only NumPy and `huggingface_hub`, downloads
`nyralabs/CrisperWhisper2.0_large`, and writes:

```text
models/ggml-crisperwhisper-large-f16.bin
```

The checkpoint download is approximately 3.1 GB. The converted model is
approximately 2.9 GB, so allow at least 7 GB of free disk space for the model,
cache, and conversion output.

Choose another checkpoint:

```bash
./scripts/setup-model.sh \
  --model nyralabs/CrisperWhisper2.0_medium \
  --output models/ggml-crisperwhisper-medium-f16.bin
```

If a previous conversion was interrupted, remove only the exact `.partial`
file after confirming it is not needed. The converter will not overwrite a
finished model unless `--force` is passed.

## 3. Build

<details open>
<summary><strong>Automatic backend selection</strong></summary>

```bash
./scripts/build.sh
```

The script enables CUDA only when both `nvcc` and `nvidia-smi` are available.
Otherwise it builds the CPU backend.

Output:

```text
build/bin/crisper-whisper
```

</details>

<details>
<summary><strong>CPU-only build</strong></summary>

```bash
./scripts/build.sh --cpu
```

The CPU profile can be selected independently:

| Profile | CPU target | Recommended use |
| --- | --- | --- |
| `balance` | AVX2, native tuning off | Default for modern x86-64 systems |
| `portable` | Baseline x86-64, optional SIMD off | Older or unknown CPUs |
| `fast` | Native build-machine tuning | The same machine or equivalent CPU |

```bash
./scripts/build.sh --cpu --cpu-profile portable --build-dir build-cpu-portable
./scripts/build.sh --cpu --cpu-profile balance  --build-dir build-cpu-balance
./scripts/build.sh --cpu --cpu-profile fast     --build-dir build-cpu-fast
```

On Linux ARM64, use `portable`. It is natively compiled and unit-tested in CI,
and the release workflow runs a Turbo-model timestamped inference smoke test.
Release binaries are built on Ubuntu 22.04 and reject dependencies newer than
GLIBC 2.35. Board-specific performance and compatibility remain experimental:

```bash
./scripts/build.sh --cpu --cpu-profile portable
```

Manual equivalent:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCRISPERWHISPER_CUDA=OFF \
  -DCRISPERWHISPER_CPU_PROFILE=balance \
  -DCRISPERWHISPER_BUILD_TESTS=ON

cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

</details>

<details>
<summary><strong>NVIDIA CUDA build</strong></summary>

Install a CUDA toolkit compatible with your NVIDIA driver using NVIDIA's
distribution-specific instructions. Confirm:

```bash
nvidia-smi
nvcc --version
```

Then build:

```bash
./scripts/build.sh --cuda
```

CUDA builds also include a CPU fallback. Select its compatibility profile with
`--cpu-profile`; `balance` is the default.

The default CUDA build embeds all three common RTX architecture targets so the
same executable can move between RTX 30, 40, and 50-series systems:

| GPU family | Typical compute capability | CMake architecture |
| --- | ---: | ---: |
| RTX 30 series | 8.6 | `86` |
| RTX 40 series | 8.9 | `89` |
| RTX 50 series | 12.0 | `120` |

Building this universal target requires
[CUDA Toolkit 12.8](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html)
or newer because 12.8 introduced `sm_120` compiler support. With an older
toolkit on a 3090 or 4090 machine, select only that card's architecture:

```bash
CUDA_ARCHITECTURES=86 ./scripts/build.sh --cuda  # RTX 3090
CUDA_ARCHITECTURES=89 ./scripts/build.sh --cuda  # RTX 4090
```

Manual RTX 5090 build:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCRISPERWHISPER_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCRISPERWHISPER_BUILD_TESTS=ON

cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

For a smaller single-architecture wrapper build:

```bash
CUDA_ARCHITECTURES=120 ./scripts/build.sh --cuda
```

</details>

<details>
<summary><strong>Clang build</strong></summary>

CPU:

```bash
CC=clang CXX=clang++ \
cmake -S . -B build-clang \
  -DCMAKE_BUILD_TYPE=Release \
  -DCRISPERWHISPER_CUDA=OFF

cmake --build build-clang --parallel
```

For CUDA, NVIDIA's supported host-compiler versions depend on the installed
toolkit. GCC is the least surprising default unless your CUDA release
explicitly supports the installed Clang version.

</details>

## 4. Test

```bash
ctest --test-dir build --output-on-failure
```

Print CLI help:

```bash
./build/bin/crisper-whisper --help
```

## 5. Transcribe

Input does not have to be preconverted. WAV, MP3, FLAC, and Ogg Vorbis are
decoded, downmixed to mono, and resampled to 16 kHz in memory by the native
runtime. No temporary file or FFmpeg subprocess is used.

Verbatim:

```bash
./build/bin/crisper-whisper \
  --model ./models/ggml-crisperwhisper-large-f16.bin \
  --file ./meeting.wav \
  --mode verbatim \
  --language en
```

Intended:

```bash
./build/bin/crisper-whisper \
  -m ./models/ggml-crisperwhisper-large-f16.bin \
  -f ./meeting.mp3 \
  --mode intended \
  --language en
```

JSON:

```bash
./build/bin/crisper-whisper \
  -m ./models/ggml-crisperwhisper-large-f16.bin \
  -f ./meeting.flac \
  --json > transcript.json
```

Optional supervised word timestamps:

```bash
./build/bin/crisper-whisper \
  -m ./models/ggml-crisperwhisper-large-f16.bin \
  -f ./meeting.flac \
  --word-timestamps \
  --json > transcript-with-words.json
```

The normal transcription pass keeps Flash Attention enabled. The timestamp
flag loads a separate no-Flash alignment context because fused Flash Attention
does not expose the cross-attention probability matrix.

Force CPU at runtime even when the binary includes CUDA:

```bash
./build/bin/crisper-whisper \
  -m ./models/ggml-crisperwhisper-large-f16.bin \
  -f ./meeting.wav \
  --cpu
```

## 6. Install

Default prefix:

```bash
sudo cmake --install build --config Release
```

Custom user-writable prefix:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HOME/.local" \
  -DCRISPERWHISPER_CUDA=OFF

cmake --build build --parallel
cmake --install build
```

Make sure `$HOME/.local/bin` is on `PATH`.

## Troubleshooting

<details>
<summary><strong><code>Permission denied</code> for a script</strong></summary>

If executable mode was lost while copying the source:

```bash
chmod +x scripts/build.sh scripts/setup-model.sh
```

Or run the scripts explicitly through Bash:

```bash
bash scripts/build.sh
bash scripts/setup-model.sh
```

</details>

<details>
<summary><strong><code>nvcc is not on PATH</code></strong></summary>

The NVIDIA driver alone provides `nvidia-smi`, but source builds also need the
CUDA toolkit and `nvcc`.

Either install/configure the toolkit or build CPU-only:

```bash
./scripts/build.sh --cpu
```

</details>

<details>
<summary><strong>CUDA host compiler is unsupported</strong></summary>

Each CUDA toolkit supports a bounded set of GCC versions. Check:

```bash
gcc --version
nvcc --version
```

Install a supported GCC version and select it explicitly, for example:

```bash
CC=gcc-13 CXX=g++-13 ./scripts/build.sh --cuda
```

Avoid `--allow-unsupported-compiler` for release binaries unless you have
validated the resulting executable carefully.

</details>

<details>
<summary><strong>CMake detects the wrong CUDA architecture</strong></summary>

Configure manually:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCRISPERWHISPER_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120
```

Replace `120` with the target GPU architecture.

</details>

<details>
<summary><strong>The process is killed while loading the model</strong></summary>

Check kernel logs and available memory:

```bash
free -h
dmesg | tail -n 50
```

The F16 large checkpoint itself is around 2.9 GB and inference requires
additional working memory. Close other memory-heavy processes or convert/use a
smaller CrisperWhisper checkpoint.

</details>

<details>
<summary><strong>Model loads but required custom tokens are missing</strong></summary>

Do not use the stock whisper.cpp Hugging Face converter for this model.
Recreate the file with:

```bash
./scripts/setup-model.sh --force
```

CrisperWhisper's appended vocabulary requires this repository's converter.

</details>

<details>
<summary><strong>Commercial use</strong></summary>

The C++ code is MIT licensed, but the open checkpoint is not. Review
`LICENSE.md` in the Hugging Face model repository and contact Nyra for
commercial model licensing.

</details>
