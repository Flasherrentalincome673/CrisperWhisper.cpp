#!/usr/bin/env python3
"""Benchmark Nyra's Python CrisperWhisper API in a persistent process."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=("transformers", "ct2"), default="transformers"
    )
    parser.add_argument(
        "--mode", choices=("verbatim", "intended"), default="verbatim"
    )
    parser.add_argument("--language", default="en")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.runs < 1:
        raise ValueError("--warmup must be non-negative and --runs positive")

    import torch

    from crisperwhisper import CrisperWhisperModel

    load_started = time.perf_counter()
    model = CrisperWhisperModel(
        args.model,
        backend=args.backend,
        compute_type="float16",
        device="cuda",
    )
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    kwargs = dict(
        language=args.language,
        mode=args.mode,
        longform_strategy="continuation",
        chunk_duration=30.0,
        stride=26.0,
        context_words=12,
        drop_words=2,
        timestamp_aware_drop=False,
        temperature_fallback=False,
        max_new_tokens=args.max_tokens,
        speculative_decoding=False,
        hallucination_mitigation=False,
        early_eot_recovery=False,
        word_timestamps=False,
    )

    def transcribe_once() -> dict:
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = model.transcribe(args.file, **kwargs)
        torch.cuda.synchronize()
        call_seconds = time.perf_counter() - started
        return {
            "call_seconds": call_seconds,
            "processing_seconds": result.processing_time,
            "duration_seconds": result.duration,
            "rtf": call_seconds / result.duration,
            "chunks": len(result.chunks or []),
            "text": result.text,
        }

    for _ in range(args.warmup):
        transcribe_once()
    runs = [transcribe_once() for _ in range(args.runs)]

    print(
        json.dumps(
            {
                "engine": f"python-{args.backend}-cuda",
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "model_load_seconds": load_seconds,
                "warmup_runs": args.warmup,
                "runs": runs,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
