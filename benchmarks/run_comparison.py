#!/usr/bin/env python3
"""Run cold, warm, and long-form C++ versus Python benchmarks."""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
import wave
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpp-exe", type=Path, required=True)
    parser.add_argument("--ggml-model", type=Path, required=True)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument(
        "--long-audio",
        type=Path,
        help=(
            "Optional real long-form WAV. When omitted, the short WAV is "
            "repeated to --long-seconds for a deterministic synthetic test."
        ),
    )
    parser.add_argument("--mode", choices=("verbatim", "intended"), default="verbatim")
    parser.add_argument("--cold-runs", type=int, default=3)
    parser.add_argument("--warm-runs", type=int, default=5)
    parser.add_argument("--long-runs", type=int, default=3)
    parser.add_argument("--long-seconds", type=float, default=55.0)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=root / "benchmarks" / "results" / "comparison.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=root / "BENCHMARKS.md",
    )
    return parser.parse_args()


def run_json(command: list[str]) -> tuple[dict, float]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stderr[-4000:]}"
        )
    try:
        return json.loads(completed.stdout), elapsed
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "benchmark command did not emit JSON:\n"
            + completed.stdout[-4000:]
            + "\n"
            + completed.stderr[-4000:]
        ) from error


def make_repeated_wav(source: Path, seconds: float) -> Path:
    destination = (
        Path(tempfile.gettempdir())
        / f"crisperwhisper-benchmark-{seconds:g}s.wav"
    )
    with wave.open(str(source), "rb") as reader:
        params = reader.getparams()
        frames = reader.readframes(params.nframes)
    target_frames = int(round(params.framerate * seconds))
    bytes_per_frame = params.sampwidth * params.nchannels
    repeats = math.ceil(target_frames / params.nframes)
    payload = (frames * repeats)[: target_frames * bytes_per_frame]
    with wave.open(str(destination), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(payload)
    return destination


def normalize_words(text: str) -> list[str]:
    return re.findall(r"\w+(?:'\w+)?", text.casefold())


def word_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for index, ref_word in enumerate(reference, start=1):
        current = [index]
        for position, hyp_word in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[position] + 1,
                    previous[position - 1] + (ref_word != hyp_word),
                )
            )
        previous = current
    return previous[-1]


def median_run(payload: dict) -> dict:
    runs = payload["runs"]
    return {
        "seconds": statistics.median(run["call_seconds"] for run in runs),
        "processing_seconds": statistics.median(
            run["processing_seconds"] for run in runs
        ),
        "rtf": statistics.median(run["rtf"] for run in runs),
        "duration": statistics.median(
            run["duration_seconds"] for run in runs
        ),
        "text": runs[-1]["text"],
        "chunks": runs[-1]["chunks"],
    }


def format_seconds(value: float) -> str:
    return f"{value:.3f} s"


def main() -> int:
    args = parse_args()
    python = Path(sys.executable)
    python_bench = Path(__file__).with_name("python_benchmark.py")
    if args.long_audio:
        long_audio = args.long_audio.resolve()
        long_audio_kind = "user-supplied recording"
        long_method = (
            f"- Long-form uses the user-supplied `{long_audio.name}` recording. "
            f"It is the median of {args.long_runs} calls\n"
            "  after one warm-up."
        )
        long_label = "long-form"
    else:
        long_audio = make_repeated_wav(args.audio.resolve(), args.long_seconds)
        long_audio_kind = f"synthetic repetition of {args.audio.name}"
        long_method = (
            f"- Long-form is a synthetic {args.long_seconds:g}-second repetition "
            f"of `{args.audio.name}`,\n"
            "  used for reproducible timing rather than as a natural-speech "
            "quality test. It uses\n"
            "  30-second chunks, 26-second stride, 12 context words, and two "
            f"boundary words.\n"
            f"  It is the median of {args.long_runs} calls after one warm-up."
        )
        long_label = "synthetic long-form"
    try:
        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
            encoding="utf-8",
        ).splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        gpu = "NVIDIA GPU"

    common_cpp = [
        str(args.cpp_exe.resolve()),
        "--model",
        str(args.ggml_model.resolve()),
        "--mode",
        args.mode,
        "--language",
        "en",
        "--threads",
        "8",
    ]
    common_python = [
        str(python),
        str(python_bench),
        "--model",
        args.hf_model,
        "--backend",
        "transformers",
        "--mode",
        args.mode,
        "--language",
        "en",
    ]

    cold: dict[str, list] = {"cpp": [], "python": []}
    for index in range(args.cold_runs):
        order = ("cpp", "python") if index % 2 == 0 else ("python", "cpp")
        for engine in order:
            if engine == "cpp":
                payload, elapsed = run_json(
                    common_cpp
                    + ["--file", str(args.audio.resolve()), "--warmup", "0", "--runs", "1"]
                )
            else:
                payload, elapsed = run_json(
                    common_python
                    + ["--file", str(args.audio.resolve()), "--warmup", "0", "--runs", "1"]
                )
            cold[engine].append(
                {
                    "process_seconds": elapsed,
                    "model_load_seconds": payload["model_load_seconds"],
                    "payload": payload,
                }
            )

    cpp_warm, _ = run_json(
        common_cpp
        + [
            "--file",
            str(args.audio.resolve()),
            "--warmup",
            "1",
            "--runs",
            str(args.warm_runs),
        ]
    )
    py_warm, _ = run_json(
        common_python
        + [
            "--file",
            str(args.audio.resolve()),
            "--warmup",
            "1",
            "--runs",
            str(args.warm_runs),
        ]
    )

    cpp_long, _ = run_json(
        common_cpp
        + [
            "--file",
            str(long_audio),
            "--warmup",
            "1",
            "--runs",
            str(args.long_runs),
        ]
    )
    py_long, _ = run_json(
        common_python
        + [
            "--file",
            str(long_audio),
            "--warmup",
            "1",
            "--runs",
            str(args.long_runs),
        ]
    )

    cold_cpp_seconds = statistics.median(
        item["process_seconds"] for item in cold["cpp"]
    )
    cold_py_seconds = statistics.median(
        item["process_seconds"] for item in cold["python"]
    )
    cold_cpp_load = statistics.median(
        item["model_load_seconds"] for item in cold["cpp"]
    )
    cold_py_load = statistics.median(
        item["model_load_seconds"] for item in cold["python"]
    )
    warm_cpp = median_run(cpp_warm)
    warm_py = median_run(py_warm)
    long_cpp = median_run(cpp_long)
    long_py = median_run(py_long)

    def agreement(cpp: dict, python_result: dict) -> dict:
        cpp_words = normalize_words(cpp["text"])
        py_words = normalize_words(python_result["text"])
        distance = word_distance(py_words, cpp_words)
        return {
            "normalized_exact": cpp_words == py_words,
            "word_error_rate": distance / max(1, len(py_words)),
        }

    short_agreement = agreement(warm_cpp, warm_py)
    long_agreement = agreement(long_cpp, long_py)
    generated = datetime.now(timezone.utc).isoformat()
    result = {
        "generated_utc": generated,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "configuration": {
            "gpu": gpu,
            "model": "nyralabs/CrisperWhisper2.0_large",
            "dtype": "FP16",
            "mode": args.mode,
            "short_audio_seconds": warm_cpp["duration"],
            "long_audio_seconds": long_cpp["duration"],
            "long_audio_kind": long_audio_kind,
            "cold_runs": args.cold_runs,
            "warm_runs": args.warm_runs,
            "long_runs": args.long_runs,
            "warmup_runs": 1,
        },
        "cold_start": {
            "cpp_process_seconds": cold_cpp_seconds,
            "python_process_seconds": cold_py_seconds,
            "cpp_model_load_seconds": cold_cpp_load,
            "python_model_load_seconds": cold_py_load,
            "cpp_speedup": cold_py_seconds / cold_cpp_seconds,
            "raw": cold,
        },
        "warm_short": {
            "cpp": warm_cpp,
            "python": warm_py,
            "cpp_speedup": warm_py["seconds"] / warm_cpp["seconds"],
            "agreement": short_agreement,
            "raw": {"cpp": cpp_warm, "python": py_warm},
        },
        "warm_longform": {
            "cpp": long_cpp,
            "python": long_py,
            "cpp_speedup": long_py["seconds"] / long_cpp["seconds"],
            "agreement": long_agreement,
            "raw": {"cpp": cpp_long, "python": py_long},
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    short_match = (
        "Exact" if short_agreement["normalized_exact"]
        else f"WER {short_agreement['word_error_rate']:.1%}"
    )
    long_match = (
        "Exact" if long_agreement["normalized_exact"]
        else f"WER {long_agreement['word_error_rate']:.1%}"
    )
    markdown = f"""# CrisperWhisper.cpp Benchmarks

Generated {generated} on {platform.system()} with {gpu}.

| Scenario | C++/ggml | Python/Transformers | C++ speedup | C++ RTF | Python RTF | Transcript agreement |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Cold start + {warm_cpp['duration']:.0f}s audio | {format_seconds(cold_cpp_seconds)} | {format_seconds(cold_py_seconds)} | {cold_py_seconds / cold_cpp_seconds:.2f}x | — | — | {short_match} |
| Warm {warm_cpp['duration']:.0f}s audio | {format_seconds(warm_cpp['seconds'])} | {format_seconds(warm_py['seconds'])} | {warm_py['seconds'] / warm_cpp['seconds']:.2f}x | {warm_cpp['rtf']:.4f} | {warm_py['rtf']:.4f} | {short_match} |
| Warm {long_label} {long_cpp['duration']:.0f}s audio | {format_seconds(long_cpp['seconds'])} | {format_seconds(long_py['seconds'])} | {long_py['seconds'] / long_cpp['seconds']:.2f}x | {long_cpp['rtf']:.4f} | {long_py['rtf']:.4f} | {long_match} |

## Model loading

| Engine | Median model-load time |
| --- | ---: |
| C++ / ggml CUDA | {format_seconds(cold_cpp_load)} |
| Python / Transformers CUDA | {format_seconds(cold_py_load)} |

## Methodology

- Model: `nyralabs/CrisperWhisper2.0_large`, FP16, {args.mode} mode.
- C++: this repository's ggml CUDA runtime with flash attention.
- Python: Nyra's public `CrisperWhisperModel`, `backend="transformers"`,
  PyTorch CUDA, eager attention as configured by the upstream engine.
- Matching settings: greedy decoding, 256 maximum new tokens, no speculative
  decoding, hallucination repair disabled, temperature fallback disabled,
  no word timestamps, and fixed two-word continuation boundary handling.
- Cold start is median fresh-process wall time over {args.cold_runs} runs. It
  includes program/import startup, model loading, audio loading, and inference.
  The operating-system file cache is not purged between runs.
- Warm short audio is median of {args.warm_runs} measured calls after one
  unmeasured warm-up with the model kept loaded.
{long_method}
- RTF is call wall time divided by audio duration; lower is better.
- The Windows Python comparison uses Transformers because Nyra's custom
  CTranslate2 wheel is currently published for Linux only.

Raw measurements: [`benchmarks/results/comparison.json`](benchmarks/results/comparison.json).
"""
    args.output_markdown.write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
