# ProCompNav reproduction image.
#
# Build:
#     docker build -t procompnav:latest .
#
# Run (driven by run_experiments.py):
#     export PROCOMPNAV_DATA_DIR=/path/to/host/data    # has scene_datasets/, instancenav_datasets/
#     export PROCOMPNAV_VIDEO_DIR=/path/to/host/videos
#     export HF_HOME=/path/to/host/hf_cache
#     python run_experiments.py --task_type coin --split val_seen \
#         --vllm_gpu 0 --vision_gpu 1 \
#         --shard_size 50 --shard0 0 --shard1 1 \
#         --eval_folder_name procompnav_coin_val_seen
#
# Every Python dependency is pinned in requirements.lock (frozen from a
# verified working environment); base/system deps are version-pinned where
# they materially affect behavior.

ARG CUDA_VERSION=12.6.0
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility,display \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH=/opt/conda/bin:/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
    CUDA_HOME=/usr/local/cuda

# --- system packages ---------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake pkg-config git curl wget unzip \
    ca-certificates sudo less vim tmux htop bash-completion \
    libgl1-mesa-dev libgl1-mesa-glx libgl1-mesa-dri \
    libsm6 libxext6 libxrender-dev libglvnd-dev libx11-dev \
    libglu1-mesa-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
    libomp-dev libegl1-mesa-dev libglm-dev libjpeg-dev \
    libvulkan1 libvulkan-dev libglfw3-dev vulkan-tools \
    freeglut3-dev mesa-utils mesa-utils-extra xorg-dev xvfb \
    ffmpeg libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
    libswscale-dev libswresample-dev libavfilter-dev \
    libcurl4-openssl-dev \
    iproute2 net-tools \
 && rm -rf /var/lib/apt/lists/*

# Vulkan / EGL ICD shims for headless Habitat-Sim rendering.
RUN mkdir -p /usr/share/vulkan/icd.d/ /usr/share/glvnd/egl_vendor.d/ \
 && echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.205"}}' \
        > /usr/share/vulkan/icd.d/nvidia_icd.json \
 && echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
        > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

ENV XDG_RUNTIME_DIR=/tmp/xdg_runtime_dir
RUN mkdir -p ${XDG_RUNTIME_DIR} && chmod 777 ${XDG_RUNTIME_DIR}

# --- llama.cpp (CUDA build, pinned to tag b7628 = commit 8e3a761189...) ------
ARG LLAMA_CPP_TAG=b7628
RUN git clone --depth 1 --branch ${LLAMA_CPP_TAG} \
        https://github.com/ggerganov/llama.cpp /opt/llama.cpp \
 && cd /opt/llama.cpp \
 && cmake -S . -B build \
        -DGGML_CUDA=ON -DLLAMA_CURL=ON -DBUILD_SHARED_LIBS=OFF \
        -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build --config Release -j$(nproc) --target llama-server \
 && cp build/bin/llama-server /usr/local/bin/llama-server \
 && rm -rf build .git

# --- Miniconda + Python 3.10 -------------------------------------------------
RUN wget -qO /tmp/miniconda.sh \
        https://repo.anaconda.com/miniconda/Miniconda3-py310_25.5.1-0-Linux-x86_64.sh \
 && bash /tmp/miniconda.sh -b -p /opt/conda \
 && rm /tmp/miniconda.sh \
 && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
 && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
 && conda clean -afy

RUN pip install --no-cache-dir --upgrade pip

# --- Python deps: locked install --------------------------------------------
# Two-pass install: (1) torch + the small set of packages needed to build
# packages that opt out of build isolation (flash_attn, habitat-sim, ...);
# (2) the full lockfile with build isolation disabled so source builds see torch.
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu126 \
        torch==2.9.0+cu126 torchvision==0.24.0+cu126 torchaudio==2.9.0+cu126 \
        setuptools wheel packaging ninja numpy==1.25.0 cython==3.2.4 \
        psutil einops pybind11 cmake

COPY requirements.lock /tmp/requirements.lock
# `--no-deps` is intentional: the lockfile is a pip-freeze snapshot of a
# verified-working environment, so it already pins every transitive dep.
# Skipping pip's resolver avoids spurious "ResolutionImpossible" errors that
# appear because some packages declare stricter ranges than what coexists in
# the real environment.
#
# `CMAKE_POLICY_VERSION_MINIMUM=3.5` keeps habitat-sim's older Corrade submodule
# compatible with modern CMake (>=4.0) which dropped support for
# `cmake_minimum_required(VERSION < 3.5)`.
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5
RUN pip install --no-cache-dir --no-build-isolation --no-deps \
        --retries 10 --timeout 120 \
        --index-url https://pypi.org/simple \
        --extra-index-url https://download.pytorch.org/whl/cu126 \
        -r /tmp/requirements.lock

WORKDIR /workspace/CoIN
ENTRYPOINT []
CMD ["/bin/bash"]
