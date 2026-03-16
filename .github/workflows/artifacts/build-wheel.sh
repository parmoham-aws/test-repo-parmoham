#!/bin/bash
# ============================================================
# Repo-owned wheel build script for private-neuron-integration-dev
#
# Called by CodeBuild when neuron.build-nightly-whl label is present.
# Builds the torch-neuronx wheel inside the base image container.
#
# Environment variables (set by CodeBuild):
#   BASE_IMAGE  — ECR URI of the nightly base image (has torch, neuronxcc, cmake, etc.)
#   REPO_DIR    — Path to the cloned repo on the CodeBuild host
#   OUTPUT_DIR  — Directory where the .whl file must be placed
#
# Contract: This script must produce exactly one .whl file in $OUTPUT_DIR.
# ============================================================
set -e

echo "=== Building torch-neuronx wheel ==="
echo "BASE_IMAGE: $BASE_IMAGE"
echo "REPO_DIR:   $REPO_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"

# Validate required inputs
if [ -z "$BASE_IMAGE" ]; then
  echo "ERROR: BASE_IMAGE is required (neuron.base-image: label must be set)"
  exit 1
fi

if [ -z "$REPO_DIR" ] || [ ! -d "$REPO_DIR" ]; then
  echo "ERROR: REPO_DIR is not set or does not exist: $REPO_DIR"
  exit 1
fi

# Pull the base image (has torch headers, neuronxcc, cmake, etc.)
docker pull "$BASE_IMAGE"

# Build wheel inside the base image container
# - Mount repo read-only (setup.py writes _build_config.py, so we copy to /tmp)
# - Output wheel to /output which maps to OUTPUT_DIR
# - Activate the venv so pip/python find torch headers + Neuron deps
# - USE_CMAKE=1 triggers CMake-based C++ extension compilation (_C.so)
docker run --rm \
  -v "$REPO_DIR:/src:ro" \
  -v "$OUTPUT_DIR:/output" \
  --entrypoint "" \
  "$BASE_IMAGE" \
  bash -c '
    set -e
    echo "Activating venv for build deps..."
    source /opt/torch-neuronx/.venv/bin/activate
    echo "Python: $(which python)"
    echo "Pip: $(which pip)"

    # Verify torch is available (needed for C++ extension headers)
    python -c "import torch; print(f\"torch {torch.__version__} at {torch.__path__[0]}\")" \
      || { echo "ERROR: torch not found in venv"; exit 1; }

    # Install MLIR dev headers if not present (needed for IrConcatStrategy C++ compilation)
    if [ ! -f /usr/lib/llvm-18/include/mlir/IR/BuiltinOps.h ]; then
      apt-get update -y && apt-get install -y libmlir-18-dev 2>/dev/null || true
    fi

    echo "Copying repo to writable temp dir..."
    cp -r /src /tmp/build
    cd /tmp/build

    pip install --quiet wheel setuptools grpcio-tools
    USE_CMAKE=1 pip wheel --no-deps --no-build-isolation -w /output .

    echo "=== Wheel built successfully ==="
    ls -la /output/*.whl
  '

echo "=== build-wheel.sh complete ==="
