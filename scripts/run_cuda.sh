#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  cat >&2 <<'EOF'
Usage:
  scripts/run_cuda.sh <file.cu> [--profile nsys|ncu|none] [--] [program args...]

Examples:
  scripts/run_cuda.sh add.cu
  scripts/run_cuda.sh scripts/cuda_vector_add.cu --profile nsys
  scripts/run_cuda.sh add.cu -- --size 1048576
EOF
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
SCRIPT_DIR="$ROOT_DIR/scripts"
source "$SCRIPT_DIR/cuda_env.sh" >/dev/null

# L40S = compute capability 8.9. Override if needed, e.g. CUDA_ARCH=sm_90.
CUDA_ARCH="${CUDA_ARCH:-sm_89}"

SRC_ARG="$1"
shift

PROFILE_MODE="none"
PROGRAM_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --profile)
      shift
      PROFILE_MODE="${1:-}"
      if [ -z "$PROFILE_MODE" ]; then
        echo "Missing value for --profile" >&2
        exit 1
      fi
      shift
      ;;
    --)
      shift
      PROGRAM_ARGS=("$@")
      break
      ;;
    *)
      PROGRAM_ARGS+=("$1")
      shift
      ;;
  esac
done

case "$PROFILE_MODE" in
  none|nsys|ncu) ;;
  *)
    echo "Unsupported --profile mode: $PROFILE_MODE (use none|nsys|ncu)" >&2
    exit 1
    ;;
esac

if [[ "$SRC_ARG" = /* ]]; then
  SRC="$SRC_ARG"
else
  SRC="$PWD/$SRC_ARG"
fi

if [ ! -f "$SRC" ]; then
  echo "CUDA source file not found: $SRC" >&2
  exit 1
fi

OUT_DIR="$ROOT_DIR/.cuda_build"
mkdir -p "$OUT_DIR"

BASE_NAME="$(basename "$SRC" .cu)"
BIN="$OUT_DIR/$BASE_NAME"

echo "[build] $SRC -> $BIN"
nvcc -O2 -lineinfo -arch="$CUDA_ARCH" -o "$BIN" "$SRC"

echo "[run] profile=$PROFILE_MODE"
case "$PROFILE_MODE" in
  none)
    "$BIN" "${PROGRAM_ARGS[@]}"
    ;;
  nsys)
    REPORT="$OUT_DIR/${BASE_NAME}_nsys"
    nsys profile -o "$REPORT" "$BIN" "${PROGRAM_ARGS[@]}"
    echo "Nsight Systems report: ${REPORT}.qdrep / ${REPORT}.nsys-rep"
    ;;
  ncu)
    ncu --set full "$BIN" "${PROGRAM_ARGS[@]}"
    ;;
esac
