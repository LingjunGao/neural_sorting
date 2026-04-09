# =============================================================================
# Dockerfile for Neural Sorting
# 
# Build:
#   docker build -t neural-sorting:cu118 .
#
# Run (interactive, with GPU):
#   docker run --gpus all -it --rm \
#       -v /path/to/data:/workspace/neural_sorting/examples/data \
#       -v /path/to/results:/workspace/neural_sorting/examples/results \
#       neural-sorting:cu118 bash
#
# Run evaluation directly:
#   docker run --gpus all --rm \
#       -v /path/to/data:/workspace/neural_sorting/examples/data \
#       -v /path/to/results:/workspace/neural_sorting/examples/results \
#       neural-sorting:cu118 \
#       bash examples/benchmarks/Eval_mlp.sh
# =============================================================================

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# ------------------------------------------------------------------
# 1. System packages
# ------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl unzip \
        build-essential gcc-11 g++-11 \
        libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Pin GCC-11 as default
ENV CC=/usr/bin/gcc-11
ENV CXX=/usr/bin/g++-11
ENV CUDAHOSTCXX=/usr/bin/g++-11

# CUDA environment
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# No GPU at build time — tell PyTorch which architectures to compile for.
# 7.0=V100, 7.5=T4/RTX2080, 8.0=A100, 8.6=RTX3090/A40, 8.9=RTX4090/L40, 9.0=H100
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

# ------------------------------------------------------------------
# 2. Python (system-level, no conda needed inside Docker)
# ------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3.10-venv python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

RUN python -m pip install --no-cache-dir --upgrade pip

# ------------------------------------------------------------------
# 3. PyTorch + core Python deps
# ------------------------------------------------------------------
RUN python -m pip install --no-cache-dir \
        torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
        --extra-index-url https://download.pytorch.org/whl/cu118

RUN python -m pip install --no-cache-dir \
        "numpy==1.26.4" "setuptools==69.5.1" "wheel<0.43" \
        ninja packaging nerfacc

# ------------------------------------------------------------------
# 4. Copy source and install neural_sorting from source
# ------------------------------------------------------------------
WORKDIR /workspace/neural_sorting
COPY . /workspace/neural_sorting

RUN python -m pip install --no-cache-dir -e . --no-build-isolation

# ------------------------------------------------------------------
# 5. Install example requirements
# ------------------------------------------------------------------
RUN python -m pip install --no-cache-dir \
        -r examples/requirements.txt --no-build-isolation

# ------------------------------------------------------------------
# 6. Default entrypoint
# ------------------------------------------------------------------
# Data and results are expected to be mounted at runtime:
#   examples/data/360_v2/   (mipnerf360 dataset)
#   examples/results/       (benchmark + mlp_checkpoint)
#
# To download them inside the container, run:
#   cd examples && python datasets/download_dataset.py
#   bash -c "source ../setup.sh"   (for Zenodo checkpoints only)

CMD ["bash"]
