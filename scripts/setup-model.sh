#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model="nyralabs/CrisperWhisper2.0_large"
output="$project_root/models/ggml-crisperwhisper-large-f16.bin"
force=()

while (($#)); do
    case "$1" in
        --model)
            shift
            model="$1"
            ;;
        --output)
            shift
            output="$1"
            ;;
        --force)
            force=(--force)
            ;;
        -h|--help)
            echo "Usage: $0 [--model HF_ID] [--output PATH] [--force]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
    esac
    shift
done

venv="$project_root/.venv"
python="$venv/bin/python"
if [[ ! -x "$python" ]]; then
    python3 -m venv "$venv"
fi

"$python" -m pip install --upgrade pip
"$python" -m pip install -r "$project_root/requirements-convert.txt"
"$python" "$project_root/tools/convert_hf_to_ggml.py" \
    --model "$model" \
    --output "$output" \
    --type f16 \
    "${force[@]}"
