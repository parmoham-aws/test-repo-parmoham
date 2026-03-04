# ============================================================
# Multi-stage Dockerfile for Neuron CI (torch-neuronx)
#
# Stages:
#   nightly-base  — Dependencies only (OS, Python, Neuron runtime/tools,
#                   compiler, EFA, cmake, bazel, uv). No repo clones.
#                   Built nightly by the Nightly Image Trigger Lambda.
#   complete      — Full build (base + torch-neuronx from source + torch-mlir)
#                   Default target for standalone/PR builds.
#
# Usage:
#   Nightly base:  docker build --target nightly-base -t base .
#   Full build:    docker build -t complete .
#
# Build args (resolved by nightly Lambda from DDB config):
#   NEURON_APT_REPO_URL  — Full authenticated APT repo URL (https://user:pass@host)
#   NEURON_PIP_REPO_URL  — Full authenticated PIP repo URL (https://user:pass@host)
#   GITHUB_TOKEN         — GitHub token for cloning private repos (complete stage only)
# ============================================================

# ============================================================
# Stage 1: nightly-base — all dependencies, no repo clones
# ============================================================
FROM public.ecr.aws/ubuntu/ubuntu:22.04_stable AS nightly-base

# Build arguments — full authenticated URLs constructed by nightly Lambda
# Format: https://user:pass@hostname (no credentials in Dockerfile)
ARG NEURON_APT_REPO_URL=""
ARG NEURON_PIP_REPO_URL=""

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# ── Python 3.10 ──────────────────────────────────────────────
RUN apt-get update -y && \
    apt-get install -y \
    wget curl gnupg lsb-release software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update -y && \
    apt-get install -y \
    python3.10 python3.10-dev python3.10-distutils python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# ── AWS CLI ──────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    less unzip jq \
    && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip && ./aws/install \
    && rm -rf awscliv2.zip aws && apt-get clean

# ── Neuron APT repository ───────────────────────────────────
# Add authenticated repo if URL provided, otherwise use public
RUN . /etc/os-release && \
    if [ -n "${NEURON_APT_REPO_URL}" ]; then \
      echo "deb ${NEURON_APT_REPO_URL} ${VERSION_CODENAME} main" \
        > /etc/apt/sources.list.d/neuron.list && \
      REPO_HOST=$(echo "${NEURON_APT_REPO_URL}" | sed 's|https://[^@]*@||') && \
      REPO_CREDS=$(echo "${NEURON_APT_REPO_URL}" | sed 's|https://||;s|@.*||') && \
      wget -qO - "https://${REPO_CREDS}@${REPO_HOST}/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB" \
        | gpg --dearmor -o /usr/share/keyrings/neuron-keyring.gpg && \
      echo "deb [signed-by=/usr/share/keyrings/neuron-keyring.gpg] ${NEURON_APT_REPO_URL} ${VERSION_CODENAME} main" \
        > /etc/apt/sources.list.d/neuron.list; \
    else \
      echo "deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main" \
        > /etc/apt/sources.list.d/neuron.list && \
      wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB \
        | gpg --dearmor -o /usr/share/keyrings/neuron-keyring.gpg && \
      echo "deb [signed-by=/usr/share/keyrings/neuron-keyring.gpg] https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main" \
        > /etc/apt/sources.list.d/neuron.list; \
    fi

# Also add public repo as fallback for tools
RUN . /etc/os-release && \
    wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB \
      | gpg --dearmor -o /usr/share/keyrings/neuron-public-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/neuron-public-keyring.gpg] https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main" \
      > /etc/apt/sources.list.d/neuron-public.list

RUN apt-get update -y

# ── Build dependencies ───────────────────────────────────────
RUN apt-get install -y \
    git build-essential python3.10-venv python3-numpy \
    python3-setuptools python3-wheel \
    && rm -rf /var/lib/apt/lists/*

# ── LLVM/Clang 18 ───────────────────────────────────────────
RUN apt-get update -y && \
    apt-get install -y python3-apt && \
    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
    echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-18 main" >> /etc/apt/sources.list && \
    apt-get update -y && \
    apt-get install -y clang-18 && \
    rm -rf /var/lib/apt/lists/*

# ── Neuron Runtime & Collectives ─────────────────────────────
RUN apt-get update -y && \
    apt-get install -y \
    aws-neuronx-collectives \
    aws-neuronx-runtime-lib \
    aws-neuronx-dkms

RUN echo "=== Neuron Runtime & Collectives ===" && dpkg -l | grep aws-neuronx

# ── Neuron Tools ─────────────────────────────────────────────
RUN apt-get update && apt-get install -y aws-neuronx-tools=2.*
RUN echo "=== Neuron Tools ===" && dpkg -l | grep aws-neuronx-tools

# ── EFA Driver ───────────────────────────────────────────────
RUN curl -O https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz && \
    wget https://efa-installer.amazonaws.com/aws-efa-installer.key && \
    gpg --import aws-efa-installer.key && \
    wget https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz.sig && \
    gpg --verify ./aws-efa-installer-latest.tar.gz.sig && \
    tar -xvf aws-efa-installer-latest.tar.gz && \
    cd aws-efa-installer && bash efa_installer.sh --yes --skip-kmod && \
    cd / && rm -rf aws-efa-installer-latest.tar.gz aws-efa-installer

# ── CMake, Ninja, patchelf ───────────────────────────────────
RUN apt-get update -y && \
    apt-get install -y cmake ninja-build patchelf \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install cmake ninja

# ── MKL (for PyTorch CPU build) ──────────────────────────────
RUN python -m pip install mkl-static mkl-include

# ── Bazelisk ─────────────────────────────────────────────────
RUN cd /tmp && \
    wget https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-amd64 && \
    chmod +x bazelisk-linux-amd64 && \
    mv bazelisk-linux-amd64 /usr/local/bin/bazel && \
    bazel --version

# ── uv (fast pip) ───────────────────────────────────────────
RUN pip install uv

# ── Neuron PIP repository ───────────────────────────────────
# Configure authenticated pip repo if URL provided, otherwise use public
RUN if [ -n "${NEURON_PIP_REPO_URL}" ]; then \
      python -m pip config set global.extra-index-url \
        "${NEURON_PIP_REPO_URL} ${NEURON_PIP_REPO_URL}/private"; \
    else \
      python -m pip config set global.extra-index-url \
        "https://pip.repos.neuron.amazonaws.com"; \
    fi

# ── Neuron Compiler ──────────────────────────────────────────
RUN pip install --force-reinstall neuronx_cc neuronx_cc_stubs nki && \
    echo "=== neuronxcc ===" && pip list | grep -i neuron && \
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -c "import neuronxcc; print('neuronxcc OK')"

# ── torch-mlir build dependencies ────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ccache clang lld llvm libstdc++-12-dev \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

# ── Environment ──────────────────────────────────────────────
ENV PATH="/opt/aws/neuron/bin:$PATH"
ENV BAZELISK_BASE_URL=https://github.com/bazelbuild/bazel/releases/download

# Clean up authenticated repo URLs from apt sources (security)
RUN if [ -n "${NEURON_APT_REPO_URL}" ]; then \
      rm -f /etc/apt/sources.list.d/neuron.list; \
    fi

RUN mkdir -p /workspace
WORKDIR /workspace
CMD ["/bin/bash"]


# ============================================================
# Stage 2: complete — full build with torch-neuronx + torch-mlir
# ============================================================
FROM nightly-base AS complete

ARG GITHUB_TOKEN=""

# ── Clone torch-neuronx ──────────────────────────────────────
RUN cd /opt && \
    if [ -n "${GITHUB_TOKEN}" ]; then \
      git clone https://${GITHUB_TOKEN}@github.com/aws-neuron/torch-neuronx.git; \
    else \
      git clone https://github.com/aws-neuron/torch-neuronx.git; \
    fi

RUN chmod +x /opt/torch-neuronx/tools/* && \
    chmod +x /opt/torch-neuronx/tests/pytorch_tests/*.sh || true

# ── Set up uv venv ───────────────────────────────────────────
RUN uv venv /opt/torch-neuronx/.venv
ENV UV_PROJECT_ENVIRONMENT=/opt/torch-neuronx/.venv
ENV PATH="/opt/torch-neuronx/.venv/bin:/opt/aws/neuron/bin:$PATH"
ENV NRT_LOCAL_PATH="/opt/aws/neuron"

RUN uv pip install -U pip

# ── Install test dependencies ────────────────────────────────
RUN uv pip install einops boto3

# ── Clone and build torch-mlir ───────────────────────────────
RUN cd /opt && \
    if [ -n "${GITHUB_TOKEN}" ]; then \
      git clone https://${GITHUB_TOKEN}@github.com/aws-neuron/torch-mlir.git; \
    else \
      git clone https://github.com/aws-neuron/torch-mlir.git; \
    fi && \
    cd torch-mlir && \
    git submodule update --init --recursive && \
    pip3 install --no-cache-dir -r build-requirements.txt && \
    mkdir -p /opt/torch-mlir-wheels && \
    CMAKE_GENERATOR=Ninja \
    TORCH_MLIR_PYTHON_PACKAGE_VERSION=0.0.1 \
    TORCH_MLIR_ENABLE_LTC=0 \
    TORCH_MLIR_ENABLE_JIT_IR_IMPORTER=0 \
    TORCH_MLIR_ENABLE_ONLY_MLIR_PYTHON_BINDINGS=1 \
    python3 -m pip wheel -v --no-build-isolation -w /opt/torch-mlir-wheels . && \
    uv pip install --no-cache-dir --no-deps /opt/torch-mlir-wheels/neuron_torch_mlir-*.whl

# Clean up torch-mlir build artifacts
RUN cd /opt/torch-mlir && rm -rf build/ .git/ externals/ && pip3 cache purge

# ── Build torch-neuronx ──────────────────────────────────────
RUN cd /opt/torch-neuronx && \
    uv pip install -e .[dev] && \
    chmod +x ./tools/* && \
    ./tools/build

# ── Verify installation ──────────────────────────────────────
RUN TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" && \
    ls -la /opt/torch-neuronx/torch_neuronx/_C*.so && \
    pip list | grep -i neuron

RUN mkdir -p /workspace
WORKDIR /workspace
CMD ["/bin/bash"]