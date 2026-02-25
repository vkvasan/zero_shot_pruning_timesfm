#!/usr/bin/env bash
# Source this file to enable CUDA toolkit commands in the current shell.

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

if [ ! -d "$CUDA_HOME" ]; then
  echo "CUDA_HOME does not exist: $CUDA_HOME" >&2
  return 1 2>/dev/null || exit 1
fi

case ":$PATH:" in
  *":$CUDA_HOME/bin:"*) ;;
  *) export PATH="$CUDA_HOME/bin:$PATH" ;;
esac

case ":${LD_LIBRARY_PATH:-}:" in
  *":$CUDA_HOME/lib64:"*) ;;
  *) export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac

export CUDA_HOME

echo "CUDA enabled:"
echo "  CUDA_HOME=$CUDA_HOME"
echo "  nvcc=$(command -v nvcc || echo not-found)"
echo "  nsys=$(command -v nsys || echo not-found)"
echo "  ncu=$(command -v ncu || echo not-found)"

