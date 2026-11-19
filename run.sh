#!/usr/bin/env bash
# Compile a CUDA file to <dir>/bin/<name>.o
# and run if compilation succeeds
# Run with: ./run.sh path/to/file.cu [program args]

set -euo pipefail

if [[ $# -lt 1 || "$1" != *.cu ]]; then
    echo "Usage: $0 path/to/file.cu [program args...]" >&2
    exit 1
fi

src=$1
shift
dir=$(dirname "$src")
name=$(basename "$src" .cu)
out="$dir/bin/$name.o"

if ! command -v nvcc >/dev/null 2>&1; then
    module load cuda12.3/toolkit/12.3.2
fi

mkdir -p "$dir/bin"
nvcc -arch=sm_70 -o "$out" "$src" && "$out" "$@"
