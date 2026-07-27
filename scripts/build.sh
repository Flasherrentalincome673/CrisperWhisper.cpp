#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-"$project_root/build"}"
build_type="${BUILD_TYPE:-Release}"
cuda="auto"
cpu_profile="balance"

while (($#)); do
    case "$1" in
        --cpu)
            cuda="off"
            ;;
        --cuda)
            cuda="on"
            ;;
        --build-dir)
            shift
            build_dir="$1"
            ;;
        --cpu-profile)
            shift
            cpu_profile="$1"
            ;;
        -h|--help)
            echo "Usage: $0 [--cpu|--cuda] [--cpu-profile portable|balance|fast] [--build-dir PATH]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
    esac
    shift
done

case "$cpu_profile" in
    portable|balance|fast) ;;
    *)
        echo "Unknown CPU profile: $cpu_profile" >&2
        exit 2
        ;;
esac

if [[ "$cuda" == "auto" ]]; then
    if command -v nvcc >/dev/null 2>&1 &&
       command -v nvidia-smi >/dev/null 2>&1; then
        cuda="on"
    else
        cuda="off"
    fi
fi

configure=(
    -S "$project_root"
    -B "$build_dir"
    "-DCMAKE_BUILD_TYPE=$build_type"
    -DCRISPERWHISPER_BUILD_TESTS=ON
    "-DCRISPERWHISPER_CPU_PROFILE=$cpu_profile"
)

if [[ "$cuda" == "on" ]]; then
    if ! command -v nvcc >/dev/null 2>&1; then
        echo "CUDA build requested, but nvcc is not on PATH." >&2
        exit 1
    fi
    configure+=(-DCRISPERWHISPER_CUDA=ON)
    # One release binary covers Ampere RTX 30, Ada RTX 40, and Blackwell
    # RTX 50 cards. Override for a narrower build with CUDA_ARCHITECTURES=...
    configure+=(
        "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES:-86;89;120}"
    )
else
    configure+=(-DCRISPERWHISPER_CUDA=OFF)
fi

cmake "${configure[@]}"
cmake --build "$build_dir" --config "$build_type" --parallel
ctest --test-dir "$build_dir" -C "$build_type" --output-on-failure

if [[ -x "$build_dir/bin/crisper-whisper" ]]; then
    executable="$build_dir/bin/crisper-whisper"
else
    executable="$build_dir/bin/$build_type/crisper-whisper"
fi
echo "Built ($cpu_profile CPU profile): $executable"
