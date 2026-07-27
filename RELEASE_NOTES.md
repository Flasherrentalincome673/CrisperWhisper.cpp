## CrisperWhisper.cpp v1.1.1

This patch release fixes long-form transcription when the same speech appears
more than once near a 30-second chunk boundary.

### Fixed

- Timestamp-aware stitching now requires both matching text and matching
  absolute time before removing a word at a chunk seam.
- Repeated passages are no longer mistaken for duplicate overlap.
- The 55-second five-copy JFK regression now retains all five repetitions:
  110 timed words across two chunks, with no invalid ranges, overlaps,
  adjacent duplicates, or missing seam passage.

### Native ZIPs

- Windows x64 CUDA for RTX 30, 40, and 50 series
- Windows x64 CPU portable
- Windows x64 CPU AVX2
- Linux x64 CPU portable
- Linux x64 CPU AVX2
- Linux ARM64 CPU portable

The Linux packages are built on Ubuntu 22.04. The GNU C++ runtime is linked
statically, and the release rejects binaries requiring GLIBC newer than 2.35.
The ARM64 package has been validated on an Orange Pi 5 Pro running Ubuntu
22.04.

### Models and license

Model weights are not included in these ZIPs. Download converted GGML models
from [drbaph/CrisperWhisper2.0-GGML](https://huggingface.co/drbaph/CrisperWhisper2.0-GGML).

The code in this repository is MIT licensed. CrisperWhisper 2.0 models,
configuration, tokenizers, parameters, and model outputs remain governed by
their own [Nyra Health Non-Commercial Research License](https://huggingface.co/nyralabs/CrisperWhisper2.0_large/blob/main/LICENSE.md).
The gated Pro models require their applicable commercial license and access
approval.
