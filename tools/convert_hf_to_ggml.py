#!/usr/bin/env python3
"""Convert a Hugging Face CrisperWhisper 2.0 checkpoint to whisper.cpp ggml.

The converter streams safetensors directly and does not import PyTorch or
Transformers.  It also serializes *all* added tokens.  That last detail is
required for CrisperWhisper's mode, event, context, hotword, and verbatimize
tokens and is the main reason the stock whisper.cpp HF converter is not used.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import tempfile
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO, NamedTuple

import numpy as np


MEL_FILTERS_URL = (
    "https://raw.githubusercontent.com/openai/whisper/"
    "248b6cb124225dd263bb9bd32d060b6517e067f8/"
    "whisper/assets/mel_filters.npz"
)

CONVERSION_MAP = {
    "self_attn.k_proj": "attn.key",
    "self_attn.q_proj": "attn.query",
    "self_attn.v_proj": "attn.value",
    "self_attn.out_proj": "attn.out",
    "self_attn_layer_norm": "attn_ln",
    "encoder_attn.q_proj": "cross_attn.query",
    "encoder_attn.v_proj": "cross_attn.value",
    "encoder_attn.out_proj": "cross_attn.out",
    "encoder_attn_layer_norm": "cross_attn_ln",
    "fc1": "mlp.0",
    "fc2": "mlp.2",
    "final_layer_norm": "mlp_ln",
    "encoder.layer_norm.bias": "encoder.ln_post.bias",
    "encoder.layer_norm.weight": "encoder.ln_post.weight",
    "encoder.embed_positions.weight": "encoder.positional_embedding",
    "decoder.layer_norm.bias": "decoder.ln.bias",
    "decoder.layer_norm.weight": "decoder.ln.weight",
    "decoder.embed_positions.weight": "decoder.positional_embedding",
    "decoder.embed_tokens.weight": "decoder.token_embedding.weight",
    "proj_out.weight": "decoder.proj.weight",
}


class TensorRecord(NamedTuple):
    source_path: Path
    source_name: str
    dtype: str
    shape: tuple[int, ...]
    offset_start: int
    offset_end: int
    data_start: int


def bytes_to_unicode() -> dict[int, str]:
    values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    chars = values[:]
    extra = 0
    for value in range(256):
        if value not in values:
            values.append(value)
            chars.append(256 + extra)
            extra += 1
    return dict(zip(values, map(chr, chars), strict=True))


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_model(model: str, cache_dir: str | None) -> Path:
    local = Path(model).expanduser()
    if local.is_dir():
        return local.resolve()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface_hub is required for a remote model. "
            "Install requirements-convert.txt in a virtual environment."
        ) from error

    print(f"Downloading {model} from Hugging Face (weights are about 3.1 GB)...")
    return Path(
        snapshot_download(
            repo_id=model,
            cache_dir=cache_dir,
            allow_patterns=[
                "*.json",
                "*.txt",
                "*.md",
                "LICENSE*",
                "model.safetensors",
                "model-*.safetensors",
                "model.safetensors.index.json",
            ],
        )
    )


def read_safetensors_header(path: Path) -> tuple[int, dict]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"invalid safetensors file: {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        header = json.loads(handle.read(header_length))
    return 8 + header_length, header


def tensor_records(model_dir: Path) -> Iterator[TensorRecord]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = load_json(index_path)
        filenames = sorted(set(index["weight_map"].values()))
    else:
        filenames = ["model.safetensors"]

    found_any = False
    for filename in filenames:
        path = model_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"missing weight shard: {path}")
        data_start, header = read_safetensors_header(path)
        for name, metadata in header.items():
            if name == "__metadata__":
                continue
            found_any = True
            offsets = metadata["data_offsets"]
            yield TensorRecord(
                source_path=path,
                source_name=name,
                dtype=metadata["dtype"],
                shape=tuple(int(value) for value in metadata["shape"]),
                offset_start=int(offsets[0]),
                offset_end=int(offsets[1]),
                data_start=data_start,
            )
    if not found_any:
        raise ValueError("checkpoint contains no tensors")


def map_tensor_name(source_name: str) -> str | None:
    # The supervised blank head is used by Nyra's Python word aligner, not by
    # the Whisper encoder/decoder forward pass.
    if source_name.startswith("encoder_blank_head."):
        return None
    # The output projection is tied to decoder.embed_tokens in Whisper.
    if source_name == "proj_out.weight":
        return None
    if not source_name.startswith("model."):
        raise ValueError(f"unrecognized non-Whisper tensor: {source_name}")

    parts = source_name.split(".")[1:]
    if len(parts) >= 5 and parts[1] == "layers":
        side = parts[0]
        layer = parts[2]
        component = ".".join(parts[3:-1])
        suffix = parts[-1]
        if component == "encoder_attn.k_proj":
            mapped_component = (
                "attn.key" if side == "encoder" else "cross_attn.key"
            )
        else:
            try:
                mapped_component = CONVERSION_MAP[component]
            except KeyError as error:
                raise ValueError(
                    f"unrecognized layer tensor: {source_name}"
                ) from error
        return f"{side}.blocks.{layer}.{mapped_component}.{suffix}"

    stripped = ".".join(parts)
    return CONVERSION_MAP.get(stripped, stripped)


def read_tensor(record: TensorRecord) -> np.ndarray:
    length = record.offset_end - record.offset_start
    with record.source_path.open("rb") as handle:
        handle.seek(record.data_start + record.offset_start)
        raw = handle.read(length)
    if len(raw) != length:
        raise OSError(f"short read for tensor {record.source_name}")

    if record.dtype == "BF16":
        bf16 = np.frombuffer(raw, dtype="<u2")
        fp32_bits = bf16.astype("<u4") << np.uint32(16)
        array = fp32_bits.view("<f4")
    elif record.dtype == "F16":
        array = np.frombuffer(raw, dtype="<f2")
    elif record.dtype == "F32":
        array = np.frombuffer(raw, dtype="<f4")
    else:
        raise ValueError(
            f"unsupported dtype {record.dtype} for {record.source_name}"
        )
    return array.reshape(record.shape)


def full_vocabulary(model_dir: Path, vocabulary_size: int) -> list[bytes]:
    base = load_json(model_dir / "vocab.json")
    added_path = model_dir / "added_tokens.json"
    added = load_json(added_path) if added_path.exists() else {}

    byte_decoder = {
        char: value for value, char in bytes_to_unicode().items()
    }
    tokens: list[bytes | None] = [None] * vocabulary_size

    for text, token_id in base.items():
        token_bytes = bytes(byte_decoder[char] for char in text)
        tokens[int(token_id)] = token_bytes
    for text, token_id in added.items():
        tokens[int(token_id)] = text.encode("utf-8")

    missing = [index for index, token in enumerate(tokens) if token is None]
    if missing:
        preview = ", ".join(map(str, missing[:20]))
        raise ValueError(
            f"tokenizer has {len(missing)} missing IDs (first: {preview})"
        )
    return [token for token in tokens if token is not None]


def ensure_mel_filters(cache_dir: Path) -> Path:
    path = cache_dir / "mel_filters.npz"
    if path.exists():
        return path
    cache_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading OpenAI Whisper mel filters...")
    with urllib.request.urlopen(MEL_FILTERS_URL) as response:
        payload = response.read()
    temporary = path.with_suffix(".npz.partial")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return path


def write_int(output: BinaryIO, value: int) -> None:
    output.write(struct.pack("<i", int(value)))


def write_model(
    model_dir: Path,
    output_path: Path,
    output_dtype: str,
    mel_filters_path: Path,
) -> None:
    config = load_json(model_dir / "config.json")
    vocabulary_size = int(config["vocab_size"])
    max_length = config.get("max_length")
    if max_length is None:
        max_length = config.get("max_target_positions", 448)

    tokens = full_vocabulary(model_dir, vocabulary_size)
    with np.load(mel_filters_path) as archive:
        mel_filters = np.asarray(
            archive[f"mel_{int(config['num_mel_bins'])}"], dtype="<f4"
        )

    records = list(tensor_records(model_dir))
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mapped_count = 0
    mapped_names: set[str] = set()
    skipped: list[str] = []
    print(f"Converting {len(records)} safetensors tensors -> {output_path}")
    try:
        with partial.open("wb") as output:
            write_int(output, 0x67676D6C)
            for value in (
                vocabulary_size,
                config["max_source_positions"],
                config["d_model"],
                config["encoder_attention_heads"],
                config["encoder_layers"],
                max_length,
                config["d_model"],
                config["decoder_attention_heads"],
                config["decoder_layers"],
                config["num_mel_bins"],
                1 if output_dtype == "f16" else 0,
            ):
                write_int(output, value)

            write_int(output, mel_filters.shape[0])
            write_int(output, mel_filters.shape[1])
            mel_filters.tofile(output)

            write_int(output, len(tokens))
            for token in tokens:
                write_int(output, len(token))
                output.write(token)

            for index, record in enumerate(records, start=1):
                destination_name = map_tensor_name(record.source_name)
                if destination_name is None:
                    skipped.append(record.source_name)
                    continue

                data = read_tensor(record)
                original_dimensions = data.ndim
                if destination_name in (
                    "encoder.conv1.bias",
                    "encoder.conv2.bias",
                ):
                    data = data.reshape(data.shape[0], 1)

                force_f32 = (
                    output_dtype == "f32"
                    or original_dimensions < 2
                    or destination_name
                    in (
                        "encoder.positional_embedding",
                        "decoder.positional_embedding",
                    )
                )
                if force_f32:
                    encoded = np.asarray(data, dtype="<f4")
                    ftype = 0
                else:
                    encoded = np.asarray(data, dtype="<f2")
                    ftype = 1

                encoded_name = destination_name.encode("utf-8")
                output.write(
                    struct.pack(
                        "<iii", encoded.ndim, len(encoded_name), ftype
                    )
                )
                for dimension in reversed(encoded.shape):
                    write_int(output, dimension)
                output.write(encoded_name)
                encoded.tofile(output)
                mapped_count += 1
                mapped_names.add(destination_name)

                if index == 1 or index % 50 == 0 or index == len(records):
                    print(
                        f"  [{index:4d}/{len(records)}] "
                        f"{record.source_name} -> {destination_name}",
                        flush=True,
                    )
    except BaseException:
        print(f"Conversion interrupted; partial file kept at {partial}")
        raise

    required = {
        "encoder.conv1.weight",
        "decoder.token_embedding.weight",
        f"decoder.blocks.{int(config['decoder_layers']) - 1}.mlp_ln.weight",
    }
    # The names and count catch both a partially downloaded checkpoint and
    # accidental converter regressions without hard-coding every tensor.
    missing_required = sorted(required - mapped_names)
    if missing_required:
        raise ValueError(
            "checkpoint is missing required inference tensors: "
            + ", ".join(missing_required)
        )
    if mapped_count < 100:
        raise ValueError(
            f"only mapped {mapped_count} tensors; checkpoint is incompatible"
        )

    os.replace(partial, output_path)
    print(
        f"Done: {mapped_count} tensors, {len(tokens)} tokens, "
        f"{output_path.stat().st_size / (1024 ** 3):.2f} GiB"
    )
    print("Skipped non-inference tensors: " + ", ".join(skipped))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="nyralabs/CrisperWhisper2.0_large",
        help="Hugging Face model ID or a local snapshot directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/ggml-crisperwhisper-large-f16.bin"),
    )
    parser.add_argument(
        "--type", choices=("f16", "f32"), default="f16"
    )
    parser.add_argument("--cache-dir")
    parser.add_argument(
        "--mel-filters",
        type=Path,
        help="Optional local OpenAI Whisper mel_filters.npz",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output model",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.force:
        print(
            f"Refusing to replace existing model: {output}\n"
            "Pass --force if replacement is intentional.",
            file=sys.stderr,
        )
        return 2

    print(
        "CrisperWhisper model weights use Nyra's non-commercial research "
        "license; see the model repository's LICENSE.md."
    )
    model_dir = resolve_model(args.model, args.cache_dir)
    filters = (
        args.mel_filters.expanduser().resolve()
        if args.mel_filters
        else ensure_mel_filters(
            Path(tempfile.gettempdir()) / "crisperwhisper_cpp"
        )
    )
    write_model(model_dir, output, args.type, filters)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
