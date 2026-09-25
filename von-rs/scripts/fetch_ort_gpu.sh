#!/usr/bin/env bash
# Fetch the official onnxruntime-gpu wheel and vendor its shared libraries
# under von-rs/third_party/ort-gpu (gitignored).
#
# Why: the CUDA execution provider needs a GPU-build ORT dylib. ort-sys's
# downloaded prebuilt CUDA kernels lack sm_120/Blackwell SASS and abort with
# cudaErrorNoKernelImageForDevice on this RTX 5070 Ti; the official wheel
# works (proven on laya, same GPU). The CPU path keeps using the oracle
# venv's CPU wheel dylib — the oracle environment is never mutated.
#
# Usage: bash von-rs/scripts/fetch_ort_gpu.sh [version]
set -euo pipefail

VERSION="${1:-1.30.0}"
DEST="$(dirname "$0")/../third_party/ort-gpu"

if [ -n "$(find "$DEST" -name 'libonnxruntime.so*' 2>/dev/null | head -1)" ]; then
    echo "onnxruntime-gpu already vendored at $DEST"
    exit 0
fi

mkdir -p "$DEST"
echo "downloading onnxruntime-gpu==${VERSION} (no deps; only the shared libs are used)"
uv pip install --python "${VIRTUAL_ENV:-.venv}/bin/python" \
    --target "$DEST" --no-deps "onnxruntime-gpu==${VERSION}"

echo "vendored:"
find "$DEST" -name "libonnxruntime.so*" -o -name "libonnxruntime_providers_*.so" | sed 's/^/  /'
echo
echo "The Rust runtime resolves this automatically (VON_ORT_GPU_DYLIB or the"
echo "third_party/ort-gpu tree); CUDA runtime libs come from the system ldconfig."
