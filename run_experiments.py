#!/usr/bin/env python3
"""ProCompNav experiment launcher.

Generates a Docker run script that spawns:
- one llama.cpp server (Qwen3-VL-8B-Instruct-GGUF) on `--vllm_gpu`
- two vision-model server groups (GroundingDINO + BLIP2 + MobileSAM) on `--vision_gpu`
- two evaluation processes (two shards) on `--vision_gpu`

All host-side paths and the Docker image tag are configurable via CLI flags or
environment variables, so the same script runs unmodified on any workstation.

Examples
--------
CoIN val_seen, shards 0 and 1, on host GPUs 0 (LLM) and 1 (policy+vision):

    python run_experiments.py \\
        --task_type coin --split val_seen \\
        --vllm_gpu 0 --vision_gpu 1 \\
        --shard_size 50 --shard0 0 --shard1 1 \\
        --eval_folder_name procompnav_coin_val_seen

Text-goal val, on GPUs 2,3:

    python run_experiments.py \\
        --task_type text_goal --split val \\
        --vllm_gpu 2 --vision_gpu 3 \\
        --shard_size 50 --shard0 0 --shard1 1 \\
        --eval_folder_name procompnav_textnav_val
"""

from pathlib import Path
import argparse
import os
import random
import stat
import string


###############################################################################
# Docker run template. All host paths are injected via .format(...).
###############################################################################
RUN_SERVER_DOCKER = r"""#!/bin/bash
set -euo pipefail

rand_container_id="procompnav_$(openssl rand -hex 2)"

echo "===== Running Docker Container: $rand_container_id ====="

if ! docker images | awk '{{print $1":"$2}}' | grep -q '^{docker_image}$'; then
    echo "[ERROR] Docker image '{docker_image}' not found." >&2
    echo "        Build it first: docker build -t {docker_image} ." >&2
    exit 1
fi

docker run --rm \
    --gpus '"device={device_ids}"' \
    --device /dev/dri:/dev/dri \
    --shm-size=64G \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -e HF_HOME=/workspace/hf_cache \
    -v {host_repo}:/workspace/CoIN \
    -v {host_data_dir}:{host_data_dir} \
    -v {host_video_dir}:/workspace/CoIN_video \
    -v {host_hf_cache}:/workspace/hf_cache \
    --mount type=bind,source={host_scene_dir},target=/workspace/CoIN/data/scene_datasets \
    --mount type=bind,source={host_instancenav_dir},target=/workspace/CoIN/data/instancenav_datasets \
    --name $rand_container_id \
    {docker_image} \
    /bin/bash -c "cd /workspace/CoIN && {script_path}"
"""


###############################################################################
# Inside-container script template.
###############################################################################
SCRIPT = r"""#!/bin/bash
set -uo pipefail

# Inputs ----------------------------------------------------------------------
save_panoramas={save_panoramas}
save_video={save_video}
save_logging_images={save_logging_images}
shard_size={shard_size}
shard0={shard0}
shard1={shard1}
trigger_step={trigger_step}
enable_multi_view={enable_multi_view}
enable_multi_view_optimization={enable_multi_view_optimization}
enable_loop_value={enable_loop_value}
enable_NLI_based={enable_NLI_based}
instance_grouping_method="{instance_grouping_method}"
enable_pbp_refinement={enable_pbp_refinement}
pbp_refinement_thres={pbp_refinement_thres}
split="{split}"
task_type="{task_type}"
dataset_type="{dataset_type}"
dataset_data_path="{dataset_data_path}"
dataset_scenes_dir="{dataset_scenes_dir}"
min_num_instances_for_pbp_trigger={min_num_instances_for_pbp_trigger}
eval_folder_name="{eval_folder_name}"
vllm_gpu={vllm_gpu_in_container}
vision_gpu={vision_gpu_in_container}

RUN_TS=$(date +%Y%m%d_%H%M%S)

cd /workspace/CoIN
mkdir -p logs "logs/$eval_folder_name" "/workspace/CoIN_video/$eval_folder_name"

# Ensure runtime python packages are present (idempotent) ----------------------
python -c "import sentence_transformers" 2>/dev/null || pip install --quiet sentence_transformers

# 1. Launch the LLM (llama.cpp) server -----------------------------------------
export LLava_PORT={vllm_port}
export USER_SIMULATOR_PORT={vllm_port}
VLLM_LOGFILE="/workspace/CoIN/logs/$eval_folder_name/${{RUN_TS}}_qwen3vl8b_llamacpp_{vllm_port}.log"

echo "[INFO] Starting LLM server on GPU $vllm_gpu (port {vllm_port})"
( cd /workspace/CoIN && \
  CUDA_VISIBLE_DEVICES=$vllm_gpu LLAMA_PORT={vllm_port} \
  bash ./scripts/launch_qwen3_vl_8b_instruct_llamacpp.sh > "$VLLM_LOGFILE" 2>&1 ) &
VLLM_PID=$!

sleep 20

# 2. Launch vision model servers (two groups on the vision GPU) ---------------
export CUDA_DEVICE=$vision_gpu

export GD_PORT_0=$(( (RANDOM % 5000) + 20000 ))
export BLIP_PORT_0=$(( GD_PORT_0 + 1 ))
export SAM_PORT_0=$(( GD_PORT_0 + 2 ))

echo "[INFO] Vision Group 0: GDino=$GD_PORT_0 BLIP=$BLIP_PORT_0 SAM=$SAM_PORT_0 on GPU $vision_gpu"
( cd /workspace/CoIN && \
  GROUNDING_DINO_PORT=$GD_PORT_0 BLIP2ITM_PORT=$BLIP_PORT_0 SAM_PORT=$SAM_PORT_0 \
  bash ./scripts/new_launch_vlm_servers.sh > /dev/null 2>&1 ) &
VISION_PID_0=$!

export GD_PORT_1=$(( (RANDOM % 5000) + 25000 ))
export BLIP_PORT_1=$(( GD_PORT_1 + 1 ))
export SAM_PORT_1=$(( GD_PORT_1 + 2 ))

echo "[INFO] Vision Group 1: GDino=$GD_PORT_1 BLIP=$BLIP_PORT_1 SAM=$SAM_PORT_1 on GPU $vision_gpu"
( cd /workspace/CoIN && \
  GROUNDING_DINO_PORT=$GD_PORT_1 BLIP2ITM_PORT=$BLIP_PORT_1 SAM_PORT=$SAM_PORT_1 \
  bash ./scripts/new_launch_vlm_servers.sh > /dev/null 2>&1 ) &
VISION_PID_1=$!

sleep 60
nvidia-smi || true

# 3. Wait for LLM server to be ready ------------------------------------------
echo "[INFO] Waiting for LLM server to listen..."
if timeout 600 bash -c '
    while ! grep -q "server is listening on http" "'"$VLLM_LOGFILE"'" 2>/dev/null; do
        sleep 2
    done
'; then
    echo "[INFO] LLM server ready. Log: $VLLM_LOGFILE"
else
    echo "[ERROR] LLM server did not start within 600s. See $VLLM_LOGFILE" >&2
    exit 1
fi

# 4. Run the two eval shards in parallel --------------------------------------
run_one_shard() {{
    local shard_idx=$1
    local gd_port=$2
    local blip_port=$3
    local sam_port=$4

    local uuid="q$(openssl rand -hex 2)"
    local label="$((shard_idx * shard_size))_$(((shard_idx + 1) * shard_size))"
    local logfile="logs/$eval_folder_name/${{label}}_${{RUN_TS}}_${{uuid}}.log"

    echo "[$(date +'%T')] Shard $shard_idx -> $logfile (uuid=$uuid)"

    local run_args=(
      habitat_baselines.num_environments=1
      habitat_baselines.num_processes=1
      habitat.task.measurements.success.success_distance=1.0
      habitat_baselines.eval.split=$split
      habitat.dataset.data_path=$dataset_data_path
      habitat.dataset.scenes_dir=$dataset_scenes_dir
      habitat.simulator.scene_dataset=/workspace/CoIN/data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json
      habitat_baselines.eval.video_option="[disk]"
      habitat_baselines.video_dir=/workspace/CoIN_video/$eval_folder_name
      habitat_baselines.test_episode_count=1200
      habitat.environment.max_episode_steps={max_episode_steps}

      habitat_baselines.rl.policy.save_panoramas=$save_panoramas
      habitat_baselines.rl.policy.save_video=$save_video
      habitat_baselines.rl.policy.save_logging_images=$save_logging_images

      habitat_baselines.rl.policy.shard_episode.shard_size=$shard_size
      habitat_baselines.rl.policy.shard_episode.shard=$shard_idx

      habitat_baselines.rl.policy.explore_only=False
      habitat_baselines.rl.policy.enable_rotate=True
      habitat_baselines.rl.policy.rotate_radius=1.0

      habitat_baselines.rl.policy.enable_multi_view=$enable_multi_view
      habitat_baselines.rl.policy.enable_multi_view_optimization=$enable_multi_view_optimization
      habitat_baselines.rl.policy.pbp.trigger_step=$trigger_step
      habitat_baselines.rl.policy.pbp.enable_sufficient_exploration_trigger=True

      habitat_baselines.rl.policy.enable_loop_value=$enable_loop_value

      habitat_baselines.rl.policy.pbp.enable_NLI_based=$enable_NLI_based
      habitat_baselines.rl.policy.pbp.instance_grouping_method="$instance_grouping_method"
      habitat_baselines.rl.policy.pbp.NLI_kmeans_modal="image"
      habitat_baselines.rl.policy.pbp.enable_pbp_refinement=$enable_pbp_refinement
      habitat_baselines.rl.policy.pbp.pbp_refinement_thres=$pbp_refinement_thres
      habitat_baselines.rl.policy.pbp.min_num_instances_for_pbp_trigger=$min_num_instances_for_pbp_trigger
    )
    if [ "$task_type" = "text_goal" ]; then
        run_args+=(
            habitat_baselines.rl.policy.task_type=$task_type
            habitat.dataset.type=$dataset_type
        )
    fi

    GROUNDING_DINO_PORT=$gd_port BLIP2ITM_PORT=$blip_port SAM_PORT=$sam_port \
    EVAL_FOLDER_NAME=$eval_folder_name WANDB_MODE=disabled \
    HABITAT_ENV_DEBUG=0 HYDRA_FULL_ERROR=1 \
    CUDA_VISIBLE_DEVICES=$vision_gpu \
    python -m vlfm.run "${{run_args[@]}}" > "$logfile" 2>&1
}}

run_one_shard $shard0 $GD_PORT_0 $BLIP_PORT_0 $SAM_PORT_0 &
MAIN0_PID=$!
sleep 3
run_one_shard $shard1 $GD_PORT_1 $BLIP_PORT_1 $SAM_PORT_1 &
MAIN1_PID=$!

wait "$MAIN0_PID" "$MAIN1_PID" || true

# 5. Cleanup ------------------------------------------------------------------
kill -SIGINT  $VLLM_PID $VISION_PID_0 $VISION_PID_1 2>/dev/null || true
sleep 5
tmux kill-session -t "vlm_servers_CUDA_DEVICE_${{vision_gpu}}_GDINO_PORT_${{GD_PORT_0}}" 2>/dev/null || true
tmux kill-session -t "vlm_servers_CUDA_DEVICE_${{vision_gpu}}_GDINO_PORT_${{GD_PORT_1}}" 2>/dev/null || true
kill -9 $VLLM_PID $VISION_PID_0 $VISION_PID_1 2>/dev/null || true

echo "[INFO] All runs completed."
"""


###############################################################################
# Argument parsing
###############################################################################
def _str2bool(v: str) -> bool:
    return v.lower() in {"1", "true", "yes", "y", "t"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    # Host-side configuration ------------------------------------------------
    p.add_argument("--docker_image", type=str,
                   default=os.environ.get("PROCOMPNAV_DOCKER_IMAGE", "procompnav:latest"),
                   help="Docker image tag (env: PROCOMPNAV_DOCKER_IMAGE)")
    p.add_argument("--host_repo", type=str, default=os.getcwd(),
                   help="Host path of this repo (default: $PWD)")
    p.add_argument("--host_data_dir", type=str,
                   default=os.environ.get("PROCOMPNAV_DATA_DIR",
                                          os.path.expanduser("~/procompnav_data")),
                   help="Host root containing scene_datasets/ and the instance-nav datasets. "
                        "Also bind-mounted at the same path inside the container so absolute "
                        "symlinks inside scene_datasets resolve (env: PROCOMPNAV_DATA_DIR).")
    p.add_argument("--host_scene_dir", type=str,
                   default=os.environ.get("PROCOMPNAV_SCENE_DIR", ""),
                   help="Host dir for HM3D scenes (contains hm3d/ and "
                        "hm3d_annotated_basis.scene_dataset_config.json). "
                        "Default: $PROCOMPNAV_DATA_DIR/scene_datasets "
                        "(env: PROCOMPNAV_SCENE_DIR)")
    p.add_argument("--host_instancenav_dir", type=str,
                   default=os.environ.get("PROCOMPNAV_INSTANCENAV_DIR", ""),
                   help="Host dir for instance-nav datasets (contains instancenav/, "
                        "instance_imagenav_hm3d_v3/). Default: $PROCOMPNAV_DATA_DIR/datasets "
                        "(env: PROCOMPNAV_INSTANCENAV_DIR)")
    p.add_argument("--host_video_dir", type=str,
                   default=os.environ.get("PROCOMPNAV_VIDEO_DIR",
                                          os.path.expanduser("~/procompnav_videos")),
                   help="Host dir where episode videos will be written "
                        "(env: PROCOMPNAV_VIDEO_DIR)")
    p.add_argument("--host_hf_cache", type=str,
                   default=os.environ.get("HF_HOME",
                                          os.path.expanduser("~/.cache/huggingface")),
                   help="Host HuggingFace cache directory (env: HF_HOME)")

    # GPU layout -------------------------------------------------------------
    p.add_argument("--vllm_gpu", type=int, default=0,
                   help="Host GPU id for the LLM (llama.cpp) server")
    p.add_argument("--vision_gpu", type=int, default=1,
                   help="Host GPU id for vision servers + policy")
    p.add_argument("--vllm_port", type=int, default=8000,
                   help="Port for the LLM server (default 8000)")

    # Eval payload -----------------------------------------------------------
    p.add_argument("--task_type", type=str, default="text_goal",
                   choices=["coin", "text_goal"])
    p.add_argument("--split", type=str, default=None,
                   help="coin: val_seen|val_unseen|val_seen_synonyms; text_goal: val")
    p.add_argument("--shard_size", type=int, default=100)
    p.add_argument("--shard0", type=int, default=0)
    p.add_argument("--shard1", type=int, default=1)
    p.add_argument("--max_episode_steps", type=int, default=None)
    p.add_argument("--trigger_step", type=int, default=None)
    p.add_argument("--min_num_instances_for_pbp_trigger", type=int, default=None)

    # Ablation knobs (defaults reproduce the main ProCompNav numbers) --------
    p.add_argument("--enable_multi_view", type=_str2bool, default=True)
    p.add_argument("--enable_multi_view_optimization", type=_str2bool, default=True)
    p.add_argument("--enable_loop_value", type=_str2bool, default=True)
    p.add_argument("--enable_NLI_based", type=_str2bool, default=True)
    p.add_argument("--enable_pbp_refinement", type=_str2bool, default=True)
    p.add_argument("--pbp_refinement_thres", type=float, default=0.9)
    p.add_argument("--instance_grouping_method", type=str,
                   default="dense_average_image_text")
    p.add_argument("--dataset_type", type=str, default="InstanceNavTextGoalDataset")

    # Logging knobs ----------------------------------------------------------
    p.add_argument("--save_panoramas", type=_str2bool, default=False)
    p.add_argument("--save_video", type=_str2bool, default=False)
    p.add_argument("--save_logging_images", type=_str2bool, default=False)

    # Output -----------------------------------------------------------------
    p.add_argument("--eval_folder_name", type=str, required=True,
                   help="Subdirectory name under logs/ and videos/")
    p.add_argument("--save_dir", type=Path, default=Path("server_experiments"))

    return p


def main() -> None:
    args = build_parser().parse_args()

    if not args.host_scene_dir:
        args.host_scene_dir = os.path.join(args.host_data_dir, "scene_datasets")
    if not args.host_instancenav_dir:
        # The community release of Instance-ImageNav text-goal episodes ships
        # under a directory named ``datasets``; older drops used
        # ``instancenav_datasets``. Auto-detect either.
        for cand in ("datasets", "instancenav_datasets"):
            p = os.path.join(args.host_data_dir, cand)
            if os.path.isdir(p):
                args.host_instancenav_dir = p
                break
        if not args.host_instancenav_dir:
            args.host_instancenav_dir = os.path.join(args.host_data_dir, "datasets")

    if args.split is None:
        args.split = "val_seen" if args.task_type == "coin" else "val"
    if args.task_type == "text_goal" and args.split != "val":
        raise ValueError("task_type='text_goal' requires --split=val")
    if args.task_type == "coin" and args.split not in {
            "val_seen", "val_unseen", "val_seen_synonyms"}:
        raise ValueError(
            "task_type='coin' requires --split in {val_seen, val_unseen, val_seen_synonyms}")
    if args.max_episode_steps is None:
        args.max_episode_steps = 500 if args.task_type == "coin" else 1000
    if args.trigger_step is None:
        args.trigger_step = 400 if args.task_type == "coin" else 600
    if args.min_num_instances_for_pbp_trigger is None:
        args.min_num_instances_for_pbp_trigger = 5 if args.task_type == "coin" else 3

    if args.task_type == "coin":
        dataset_data_path = f"CoIN-Bench/{args.split}/{args.split}.json.gz"
        dataset_scenes_dir = f"CoIN-Bench/{args.split}/content"
    else:
        dataset_data_path = (
            f"/workspace/CoIN/data/instancenav_datasets/instancenav/"
            f"{args.split}/{args.split}_text.json.gz"
        )
        dataset_scenes_dir = (
            f"/workspace/CoIN/data/instancenav_datasets/instance_imagenav_hm3d_v3/"
            f"{args.split}/content"
        )

    # `--gpus device=A,B` remaps to container-side ids 0 and 1.
    device_ids = f"{args.vllm_gpu},{args.vision_gpu}"
    vllm_gpu_in_container = 0
    vision_gpu_in_container = 1

    args.save_dir.mkdir(exist_ok=True, parents=True)
    (args.save_dir / "scripts").mkdir(exist_ok=True, parents=True)
    (args.save_dir / "run_docker").mkdir(exist_ok=True, parents=True)

    rand = "".join(random.choices(string.ascii_lowercase + string.digits, k=7))
    file = f"run_{rand}.sh"
    inside_script = args.save_dir / "scripts" / file
    docker_script = args.save_dir / "run_docker" / file

    inside_script.write_text(SCRIPT.format(
        save_panoramas=args.save_panoramas,
        save_video=args.save_video,
        save_logging_images=args.save_logging_images,
        shard_size=args.shard_size,
        shard0=args.shard0,
        shard1=args.shard1,
        trigger_step=args.trigger_step,
        enable_multi_view=args.enable_multi_view,
        enable_multi_view_optimization=args.enable_multi_view_optimization,
        enable_loop_value=args.enable_loop_value,
        enable_NLI_based=args.enable_NLI_based,
        instance_grouping_method=args.instance_grouping_method,
        enable_pbp_refinement=args.enable_pbp_refinement,
        pbp_refinement_thres=args.pbp_refinement_thres,
        eval_folder_name=args.eval_folder_name,
        split=args.split,
        task_type=args.task_type,
        dataset_type=args.dataset_type,
        dataset_data_path=dataset_data_path,
        dataset_scenes_dir=dataset_scenes_dir,
        min_num_instances_for_pbp_trigger=args.min_num_instances_for_pbp_trigger,
        max_episode_steps=args.max_episode_steps,
        vllm_gpu_in_container=vllm_gpu_in_container,
        vision_gpu_in_container=vision_gpu_in_container,
        vllm_port=args.vllm_port,
    ))
    inside_script.chmod(inside_script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    docker_script.write_text(RUN_SERVER_DOCKER.format(
        docker_image=args.docker_image,
        host_repo=args.host_repo,
        host_data_dir=args.host_data_dir,
        host_scene_dir=args.host_scene_dir,
        host_instancenav_dir=args.host_instancenav_dir,
        host_video_dir=args.host_video_dir,
        host_hf_cache=args.host_hf_cache,
        device_ids=device_ids,
        script_path=f"/workspace/CoIN/{inside_script}",
    ))
    docker_script.chmod(docker_script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    print(f"[INFO] Inside-container script: {inside_script}")
    print(f"[INFO] Launching: bash {docker_script}")
    os.execvp("bash", ["bash", str(docker_script)])


if __name__ == "__main__":
    main()
