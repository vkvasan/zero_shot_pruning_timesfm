#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
SCRIPT_DIR="$ROOT_DIR/scripts"

source "$SCRIPT_DIR/cuda_env.sh" >/dev/null

# L40S = compute capability 8.9. Override if needed, e.g. CUDA_ARCH=sm_90.
CUDA_ARCH="${CUDA_ARCH:-sm_89}"

OUT_DIR="$ROOT_DIR/.cuda_build"
mkdir -p "$OUT_DIR"

BIN="$OUT_DIR/cuda_vector_add"
SRC="$SCRIPT_DIR/cuda_vector_add.cu"

echo "[build] $SRC -> $BIN"
nvcc -O2 -lineinfo -arch="$CUDA_ARCH" -o "$BIN" "$SRC"

echo "[run] $BIN"
"$BIN"

echo
echo "Nsight commands:"
echo "  nsys profile -o $OUT_DIR/nsys_report $BIN"
echo "  ncu --set full $BIN"
