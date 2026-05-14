#!/usr/bin/env bash

# 1. Environment Variables Configuration
export VLFM_PYTHON=${VLFM_PYTHON:-$(command -v python)}
export MOBILE_SAM_CHECKPOINT=${MOBILE_SAM_CHECKPOINT:-data/mobile_sam.pt}
export GROUNDING_DINO_CONFIG=${GROUNDING_DINO_CONFIG:-GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py}
export GROUNDING_DINO_WEIGHTS=${GROUNDING_DINO_WEIGHTS:-data/groundingdino_swint_ogc.pth}
export CLASSES_PATH=${CLASSES_PATH:-vlfm/vlm/classes.txt}

export CUDA_DEVICE=${CUDA_DEVICE:-1}
export HF_HOME=${HF_HOME:-"$HOME/.cache/huggingface"} # HF Cache Directory

# 2. Port Assignment for various models (20000 ~ 30000)
GROUNDING_DINO_PORT=${GROUNDING_DINO_PORT:?GROUNDING_DINO_PORT must be set by the parent script}
BLIP2ITM_PORT=${BLIP2ITM_PORT:?BLIP2ITM_PORT must be set by the parent script}
SAM_PORT=${SAM_PORT:?SAM_PORT must be set by the parent script}

# 3. Validation
CUDA_DEVICE=${CUDA_DEVICE:?CUDA_DEVICE must be set}

# 4. Tmux Session Setup
session_name=vlm_servers_CUDA_DEVICE_${CUDA_DEVICE}_GDINO_PORT_${GROUNDING_DINO_PORT}

# Create a detached tmux session
tmux new-session -d -s ${session_name}

# Split the window vertically and horizontally
tmux split-window -v -t ${session_name}:0
tmux split-window -h -t ${session_name}:0.0
tmux split-window -h -t ${session_name}:0.2
tmux split-window -h -t ${session_name}:0.3

# 5. Run Server Commands in each pane
tmux send-keys -t ${session_name}:0.0 "CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} ${VLFM_PYTHON} -m vlfm.vlm.grounding_dino --port ${GROUNDING_DINO_PORT}" C-m
tmux send-keys -t ${session_name}:0.1 "CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} ${VLFM_PYTHON} -m vlfm.vlm.blip2itm --port ${BLIP2ITM_PORT}" C-m
tmux send-keys -t ${session_name}:0.2 "CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} ${VLFM_PYTHON} -m vlfm.vlm.sam --port ${SAM_PORT}" C-m

# 6. Output Information
echo "========================================"
echo "Assigned Sequential Ports:"
echo "GROUNDING_DINO_PORT : ${GROUNDING_DINO_PORT}"
echo "BLIP2ITM_PORT       : ${BLIP2ITM_PORT}"
echo "SAM_PORT            : ${SAM_PORT}"
echo "========================================"
echo "Created tmux session '${session_name}'. You must wait up to 90 seconds for the model weights to finish being loaded."
echo "Run the following to monitor all the server commands:"
echo "tmux attach-session -t ${session_name}"
