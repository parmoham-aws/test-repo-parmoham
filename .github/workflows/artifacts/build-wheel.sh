#!/bin/bash
# ============================================================
# Repo-owned wheel build script for private-torch-neuronx
# (Temporarily modified to build private-torch-neuronx wheel
#  instead of private-neuron-integration-dev wheel for testing)
#
# Called by CodeBuild when neuron.build-nightly-whl label is present.
# Builds the torch-neuron wheel inside the base image container.
#
# Environment variables (set by CodeBuild):
#   BASE_IMAGE  — ECR URI of the nightly base image (has torch, neuronxcc, cmake, etc.)
#   REPO_DIR    — Path to the cloned repo on the CodeBuild host
#   OUTPUT_DIR  — Directory where the .whl file must be placed
#
# Contract: This script must produce exactly one .whl file in $OUTPUT_DIR.
# ============================================================
set -e

# ── Configuration: which repo to build the wheel from ────────
# Override: build private-torch-neuronx instead of the cloned repoN
TORCH_NEURON_REPO="aws-neuron/private-torch-neuronx"
TORCH_NEURON_BRANCH="${TORCH_NEURON_BRANCH:-main}"

echo "=== Building torch-neuron wheel (from private-torch-neuronx) ==="
echo "BASE_IMAGE:         $BASE_IMAGE"
echo "REPO_DIR:           $REPO_DIR"
echo "OUTPUT_DIR:         $OUTPUT_DIR"
echo "TORCH_NEURON_REPO:  $TORCH_NEURON_REPO"
echo "TORCH_NEURON_BRANCH: $TORCH_NEURON_BRANCH"

# Validate required inputs
if [ -z "$BASE_IMAGE" ]; then
  echo "ERROR: BASE_IMAGE is required (neuron.base-image: label must be set)"
  exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
  echo "ERROR: OUTPUT_DIR is not set"
  exit 1
fi

# Clone private-torch-neuronx repo on the CodeBuild host with fallback logic
TORCH_NEURON_DIR="$(pwd)/private-torch-neuronx"
CLONE_SUCCESS=false

# Method 1: Try GitHub App token first (BUILD_ARG_GITHUB_AUTH_B64)
if [ -n "$BUILD_ARG_GITHUB_AUTH_B64" ]; then
  echo "=== Attempting clone using GitHub App token (BUILD_ARG_GITHUB_AUTH_B64) ==="
  echo '#!/bin/sh' > /tmp/git-askpass-torch.sh
  echo 'echo "$BUILD_ARG_GITHUB_AUTH_B64" | base64 -d | cut -d: -f2-' >> /tmp/git-askpass-torch.sh
  chmod +x /tmp/git-askpass-torch.sh
  
  if GIT_ASKPASS=/tmp/git-askpass-torch.sh GIT_TERMINAL_PROMPT=0 \
    git clone --depth=50 --branch "$TORCH_NEURON_BRANCH" \
    "https://x-access-token@github.com/${TORCH_NEURON_REPO}.git" "$TORCH_NEURON_DIR" 2>&1; then
    echo "SUCCESS: Cloned $TORCH_NEURON_REPO using GitHub App token"
    CLONE_SUCCESS=true
  else
    echo "FAILED: GitHub App token clone failed (exit $?)"
    echo "GitHub App may not have access to $TORCH_NEURON_REPO"
    rm -rf "$TORCH_NEURON_DIR" 2>/dev/null || true
  fi
  rm -f /tmp/git-askpass-torch.sh
else
  echo "INFO: BUILD_ARG_GITHUB_AUTH_B64 not available, skipping GitHub App token method"
fi

# Method 2: Fallback to config table token (BUILD_ARG_GITHUB_TOKEN)
if [ "$CLONE_SUCCESS" = "false" ] && [ -n "$BUILD_ARG_GITHUB_TOKEN" ]; then
  echo "=== Attempting fallback clone using config table token (BUILD_ARG_GITHUB_TOKEN) ==="
  if git clone --depth=50 --branch "$TORCH_NEURON_BRANCH" \
    "https://${BUILD_ARG_GITHUB_TOKEN}@github.com/${TORCH_NEURON_REPO}.git" "$TORCH_NEURON_DIR" 2>&1; then
    echo "SUCCESS: Cloned $TORCH_NEURON_REPO using config table token"
    CLONE_SUCCESS=true
  else
    echo "FAILED: Config table token clone also failed (exit $?)"
    echo "Config table token may not have access to $TORCH_NEURON_REPO"
    rm -rf "$TORCH_NEURON_DIR" 2>/dev/null || true
  fi
else
  if [ "$CLONE_SUCCESS" = "false" ]; then
    echo "INFO: BUILD_ARG_GITHUB_TOKEN not available, skipping config table token method"
  fi
fi

if [ "$CLONE_SUCCESS" = "true" ]; then
  echo "Cloned private-torch-neuronx at $(cd "$TORCH_NEURON_DIR" && git rev-parse HEAD)"
else
  echo "ERROR: Failed to clone $TORCH_NEURON_REPO using any available method"
  exit 1
fi

# Pull the base image (has torch headers, neuronxcc, cmake, etc.)
docker pull "$BASE_IMAGE"

# Build wheel inside the base image container
# - Mount the private-torch-neuronx repo read-only
# - Output wheel to /output which maps to OUTPUT_DIR
# - Activate the venv so pip/python find torch headers + Neuron deps
# - private-torch-neuronx uses PyTorch CppExtension (no USE_CMAKE needed)
docker run --rm \
  -v "$TORCH_NEURON_DIR:/src:ro" \
  -v "$OUTPUT_DIR:/output" \
  --entrypoint "" \
  "$BASE_IMAGE" \
  bash -c '
    set -e
    echo "=== Inside container: building private-torch-neuronx wheel ==="

    # Activate venv if it exists (base image from private-neuron-integration-dev)
    if [ -f /opt/torch-neuronx/.venv/bin/activate ]; then
      echo "Activating venv for build deps..."
      source /opt/torch-neuronx/.venv/bin/activate
    fi

    echo "Python: $(which python)"
    echo "Pip: $(which pip)"

    # Verify torch is available (needed for C++ extension headers)
    # Disable backend autoload to avoid broken torch_neuronx import in base image
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -c "import torch; print(f\"torch {torch.__version__} at {torch.__path__[0]}\")" \
      || { echo "ERROR: torch not found"; exit 1; }

    echo "Copying repo to writable temp dir..."
    cp -r /src /tmp/build
    cd /tmp/build

    # Install build dependencies
    pip install --quiet wheel setuptools grpcio-tools

    # Build the wheel using setup.py (PyTorch CppExtension for C++ code)
    # No USE_CMAKE needed — private-torch-neuronx only supports CppExtension
    # Disable backend autoload to avoid broken torch_neuronx import during build
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 pip wheel --no-deps --no-build-isolation -w /output .

    echo "=== Wheel built successfully ==="
    ls -la /output/*.whl
  '

echo "=== build-wheel.sh complete ==="
