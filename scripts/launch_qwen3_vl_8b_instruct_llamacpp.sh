#!/usr/bin/env bash
# Launch a llama.cpp server for Qwen3-VL-8B-Instruct-GGUF.
#
# The GGUF weights are resolved automatically from the HuggingFace cache
# rooted at $HF_HOME (default: /workspace/hf_cache inside the container).
#
# Pre-requisite (one-time, on the host):
#     huggingface-cli download Qwen/Qwen3-VL-8B-Instruct-GGUF \
#         "Qwen3VL-8B-Instruct-F16.gguf" "mmproj-Qwen3VL-8B-Instruct-F16.gguf"
# That populates $HF_HOME/hub/models--Qwen--Qwen3-VL-8B-Instruct-GGUF/snapshots/<rev>/.

set -euo pipefail

: "${HF_HOME:?HF_HOME must point to your HuggingFace cache dir}"

MODEL_REPO_DIR="${HF_HOME}/hub/models--Qwen--Qwen3-VL-8B-Instruct-GGUF/snapshots"
if [ ! -d "$MODEL_REPO_DIR" ]; then
    echo "[ERROR] Qwen3-VL-8B-Instruct-GGUF not found under $MODEL_REPO_DIR." >&2
    echo "        Run: huggingface-cli download Qwen/Qwen3-VL-8B-Instruct-GGUF" >&2
    exit 1
fi

SNAPSHOT_DIR=$(ls -1dt "${MODEL_REPO_DIR}"/*/ 2>/dev/null | head -n 1)
SNAPSHOT_DIR="${SNAPSHOT_DIR%/}"

MODEL_FILE=$(ls "${SNAPSHOT_DIR}"/Qwen3VL-8B-Instruct-*.gguf 2>/dev/null | head -n 1 || true)
MMPROJ_FILE=$(ls "${SNAPSHOT_DIR}"/mmproj-Qwen3VL-8B-Instruct-*.gguf 2>/dev/null | head -n 1 || true)

if [ -z "$MODEL_FILE" ] || [ -z "$MMPROJ_FILE" ]; then
    echo "[ERROR] Could not locate model / mmproj GGUF in $SNAPSHOT_DIR" >&2
    exit 1
fi

LLAMA_PORT="${LLAMA_PORT:-8000}"

echo "[INFO] Launching llama-server"
echo "       model:  $MODEL_FILE"
echo "       mmproj: $MMPROJ_FILE"
echo "       port:   $LLAMA_PORT"
echo "       cuda:   ${CUDA_VISIBLE_DEVICES:-unset}"

exec llama-server \
    -m  "$MODEL_FILE" \
    --mmproj "$MMPROJ_FILE" \
    --n_gpu_layers -1 \
    --host 0.0.0.0 \
    --port "$LLAMA_PORT" \
    --jinja \
    -c 12000 \
    -np 2
