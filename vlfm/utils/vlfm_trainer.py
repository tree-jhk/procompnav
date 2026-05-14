# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
import logging
import warnings; warnings.filterwarnings("ignore")
import gc
import os
import cv2
from collections import defaultdict
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import tqdm
import gym
from transformers import AutoModel, AutoImageProcessor, AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
from habitat import VectorEnv, logger
from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat_baselines import PPOTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.common.tensorboard_utils import (
    TensorboardWriter,
)

from habitat_baselines.rl.ppo.policy import PolicyActionData
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    get_action_space_info,
    inference_mode,
    is_continuous_action_space,
)
from habitat_baselines.utils.info_dict import (
    extract_scalars_from_info as extract_scalars_from_info_habitat,
)
from habitat.utils.visualizations import maps as habitat_maps
from omegaconf import OmegaConf
import gzip, json
from frontier_exploration.utils.general_utils import xyz_to_habitat
from vlfm.utils.habitat_visualizer import overlay_rotate_history_on_map


def extract_scalars_from_info(info: Dict[str, Any]) -> Dict[str, float]:
    info_filtered = {k: v for k, v in info.items() if not isinstance(v, list)}
    return extract_scalars_from_info_habitat(info_filtered)


from colorama import Fore
from colorama import init as init_colorama

init_colorama()

ALLOWED_INSTANCE_GROUPING_METHODS = (
    "dense_half_image",
    "dense_half_text",
    "dense_half_image_text",
    "dense_average_image",
    "dense_average_text",
    "dense_average_image_text",
    "kmeans",
)

import copy

import wandb
import time

if not hasattr(gym.Env, "geodesic_distance"):
    def geodesic_distance(self, *args, **kwargs) -> float:
        if kwargs:
            position_a = kwargs.get("position_a")
            position_b = kwargs.get("position_b")
        elif len(args) == 1 and isinstance(args[0], dict):
            position_a = args[0].get("position_a")
            position_b = args[0].get("position_b")
        elif len(args) == 2:
            position_a, position_b = args
        else:
            raise TypeError("geodesic_distance expects (position_a, position_b) or kwargs position_a/position_b.")

        if position_a is None or position_b is None:
            raise TypeError("geodesic_distance requires both position_a and position_b.")

        sim = self.habitat_env.sim
        return float(sim.geodesic_distance(np.array(position_a), np.array(position_b)))

    gym.Env.geodesic_distance = geodesic_distance  # type: ignore[attr-defined]

def log_vlm_connection_info() -> None:
    """Print the currently configured VLM ports for traceability."""
    ports = (
        "GROUNDING_DINO_PORT",
        "BLIP2ITM_PORT",
        "SAM_PORT",
        "LLava_PORT",
        "LLAMA_PORT",
    )
    print("VLM port configuration:")
    for key in ports:
        print(f"  {key}={os.environ.get(key, 'unset')}")


@baseline_registry.register_trainer(name="vlfm")
class VLFMTrainer(PPOTrainer):
    envs: VectorEnv

    def _infer_failure_reason(self, episode_info: Dict[str, Any]) -> str:
        success_flag = episode_info.get("success", 0)
        if success_flag:
            return "success"

        total_detected = episode_info.get("total_detected_objects", 0)
        if total_detected == 0:
            return "failure_no_detection"

        num_candidates = episode_info.get("num_candidate_objects", 0)
        if num_candidates == 0:
            return "failure_no_candidates"

        pbp_results = episode_info.get("pbp_results") or []
        if pbp_results:
            final_status = pbp_results[-1].get("status", "")
            if final_status in {"PBP_FALLBACK_MULTIPLE_CANDIDATES", "PBP_NO_SELECTION"}:
                return "failure_pbp"

        if episode_info.get("target_detected", 0) and not success_flag:
            return "failure_nav"

        return "failure_else"

    def _save_step_maps(
        self,
        episode_dir: str,
        info: Dict[str, Any],
        policy_info: Dict[str, Any],
    ) -> None:
        os.makedirs(episode_dir, exist_ok=True)

        top_down_map = info.get("top_down_map", {})
        original_map = top_down_map.get("map")
        if original_map is not None:
            top_down_map["map"] = original_map.copy()

        overlay_rotate_history_on_map(info, policy_info)
        top_down_img = habitat_maps.colorize_draw_agent_and_fit_to_height(
            info["top_down_map"],
            480,
        )

        if original_map is not None:
            top_down_map["map"] = original_map
        cv2.imwrite(
            os.path.join(episode_dir, "top_down_map.png"),
            cv2.cvtColor(top_down_img, cv2.COLOR_RGB2BGR),
        )

        if original_map is not None:
            cell_size = float(policy_info.get("sufficient_exp_cell_size", 0.25))
            scaled_r_score = float(policy_info.get("scaled_r_score", policy_info.get("sufficient_exp_weight", 0.0)))
            soft_TSS = float(policy_info.get("soft_TSS", policy_info.get("sufficient_exp_soft_TSS", 0.0)))
            hard_TSS = float(policy_info.get("hard_TSS", policy_info.get("sufficient_exp_hard_TSS", 0.0)))
            trigger_threshold = float(policy_info.get("sufficient_exp_trigger_threshold", 0.0))
            r_score_min = float(policy_info.get("r_score_min", policy_info.get("sufficient_exp_r_score_min", 0.0)))
            r_score_max = float(policy_info.get("r_score_max", policy_info.get("sufficient_exp_r_score_max", 0.0)))
            cnt_revisit = int(policy_info.get("cnt_revisit", policy_info.get("sufficient_exp_cnt_revisit", 0)))
            num_frontiers_max = int(policy_info.get("num_frontiers_max", policy_info.get("sufficient_exp_num_frontiers_max", 0)))
            num_frontiers = int(policy_info.get("num_frontiers", policy_info.get("sufficient_exp_num_frontiers", 0)))
            threshold_delta = float(policy_info.get("sufficient_exp_trigger_threshold_delta", 0.0))
            cell_visits = policy_info.get("cell_visits", policy_info.get("sufficient_exp_cell_visits", []))
            loop_blacklist_cells = policy_info.get("loop_blacklist_cells", [])
            loop_blacklist_added_cells = policy_info.get("loop_blacklist_added_cells", [])
            td = info["top_down_map"]
            tf = np.asarray(td["tf_episodic_to_global"], dtype=np.float32)
            if tf.size == 16:
                tf = tf.reshape(4, 4)
            lower = np.asarray(td["lower_bound"], dtype=np.float32)
            upper = np.asarray(td["upper_bound"], dtype=np.float32)
            grid_resolution = np.asarray(td["grid_resolution"], dtype=np.float32)
            grid_size = np.asarray(
                [
                    abs(float(upper[1] - lower[1])) / float(grid_resolution[0]),
                    abs(float(upper[0] - lower[0])) / float(grid_resolution[1]),
                ],
                dtype=np.float32,
            )

            def _epi_xy_to_rc(xy: np.ndarray) -> Tuple[int, int]:
                pt = np.asarray([float(xy[0]), float(xy[1]), 0.0, 1.0], dtype=np.float32)
                g = (tf @ pt)[:3]
                hab = xyz_to_habitat(g.reshape(1, 3))[0]
                sim_xy = hab[[2, 0]]
                rc = ((sim_xy - lower[::-1]) / grid_size).astype(int)
                return int(rc[0]), int(rc[1])

            h0, w0 = int(original_map.shape[0]), int(original_map.shape[1])
            heat = np.zeros((h0, w0, 3), dtype=np.uint8)
            if scaled_r_score < 0.0:
                scaled_r_score = 0.0
            if scaled_r_score > 1.0:
                scaled_r_score = 1.0
            color_bgr = cv2.applyColorMap(np.uint8([[int(round(255.0 * scaled_r_score))]]), cv2.COLORMAP_JET)[0, 0]
            color_rgb = np.array([int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0])], dtype=np.uint8)
            gray_rgb = np.array([0, 200, 0], dtype=np.uint8)
            row_step = max(1, int(round(float(cell_size) / float(grid_size[0]))))
            col_step = max(1, int(round(float(cell_size) / float(grid_size[1]))))
            r_origin, c_origin = _epi_xy_to_rc(np.zeros(2, dtype=np.float32))
            revisit_boxes = []
            blacklist_boxes = []
            blacklist_added_boxes = []
            for ijv in cell_visits:
                cx, cy = float(ijv[0] + 0.5) * float(cell_size), float(ijv[1] + 0.5) * float(cell_size)
                rr, cc = _epi_xy_to_rc(np.array([cx, cy], dtype=np.float32))
                ri = int(np.floor(float(rr - r_origin) / float(row_step)))
                ci = int(np.floor(float(cc - c_origin) / float(col_step)))
                r0, c0 = int(np.clip(r_origin + ri * row_step, 0, h0 - 1)), int(np.clip(c_origin + ci * col_step, 0, w0 - 1))
                r1, c1 = int(np.clip(r0 + row_step - 1, 0, h0 - 1)), int(np.clip(c0 + col_step - 1, 0, w0 - 1))
                if int(ijv[2]) >= 2:
                    heat[r0 : r1 + 1, c0 : c1 + 1] = color_rgb
                    revisit_boxes.append((r0, c0, r1, c1))
                else:
                    heat[r0 : r1 + 1, c0 : c1 + 1] = gray_rgb
            for ij in loop_blacklist_cells:
                cx, cy = float(ij[0] + 0.5) * float(cell_size), float(ij[1] + 0.5) * float(cell_size)
                rr, cc = _epi_xy_to_rc(np.array([cx, cy], dtype=np.float32))
                ri = int(np.floor(float(rr - r_origin) / float(row_step)))
                ci = int(np.floor(float(cc - c_origin) / float(col_step)))
                r0, c0 = int(np.clip(r_origin + ri * row_step, 0, h0 - 1)), int(np.clip(c_origin + ci * col_step, 0, w0 - 1))
                r1, c1 = int(np.clip(r0 + row_step - 1, 0, h0 - 1)), int(np.clip(c0 + col_step - 1, 0, w0 - 1))
                blacklist_boxes.append((r0, c0, r1, c1))
            for ij in loop_blacklist_added_cells:
                cx, cy = float(ij[0] + 0.5) * float(cell_size), float(ij[1] + 0.5) * float(cell_size)
                rr, cc = _epi_xy_to_rc(np.array([cx, cy], dtype=np.float32))
                ri = int(np.floor(float(rr - r_origin) / float(row_step)))
                ci = int(np.floor(float(cc - c_origin) / float(col_step)))
                r0, c0 = int(np.clip(r_origin + ri * row_step, 0, h0 - 1)), int(np.clip(c_origin + ci * col_step, 0, w0 - 1))
                r1, c1 = int(np.clip(r0 + row_step - 1, 0, h0 - 1)), int(np.clip(c0 + col_step - 1, 0, w0 - 1))
                blacklist_added_boxes.append((r0, c0, r1, c1))

            if heat.shape[0] > heat.shape[1]:
                heat = np.rot90(heat, 1)
            heat = cv2.resize(heat, (top_down_img.shape[1], top_down_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            cell_img = np.ascontiguousarray(top_down_img.copy())
            mask = (heat[:, :, 0] | heat[:, :, 1] | heat[:, :, 2]) > 0
            cell_img[mask] = (0.45 * cell_img[mask] + 0.55 * heat[mask]).astype(np.uint8)
            rotate = h0 > w0
            if rotate:
                r_origin, c_origin = w0 - 1 - c_origin, r_origin
                row_step, col_step = col_step, row_step
                h1, w1 = w0, h0
            else:
                h1, w1 = h0, w0
            row_step_out = max(1, int(round(float(row_step) * float(cell_img.shape[0]) / float(h1))))
            col_step_out = max(1, int(round(float(col_step) * float(cell_img.shape[1]) / float(w1))))
            r0 = int(round(float(int(r_origin % row_step)) * float(cell_img.shape[0]) / float(h1)))
            c0 = int(round(float(int(c_origin % col_step)) * float(cell_img.shape[1]) / float(w1)))
            if blacklist_boxes:
                overlay = np.zeros_like(cell_img, dtype=np.uint8)
                for br0, bc0, br1, bc1 in blacklist_boxes:
                    if rotate:
                        br0, bc0, br1, bc1 = w0 - 1 - bc1, br0, w0 - 1 - bc0, br1
                    br0o = int(round(float(br0) * float(cell_img.shape[0]) / float(h1)))
                    br1o = int(round(float(br1) * float(cell_img.shape[0]) / float(h1)))
                    bc0o = int(round(float(bc0) * float(cell_img.shape[1]) / float(w1)))
                    bc1o = int(round(float(bc1) * float(cell_img.shape[1]) / float(w1)))
                    cv2.rectangle(overlay, (bc0o, br0o), (bc1o, br1o), (140, 0, 0), -1)
                red_mask = (overlay[:, :, 0] | overlay[:, :, 1] | overlay[:, :, 2]) > 0
                cell_img[red_mask] = (0.7 * cell_img[red_mask] + 0.3 * overlay[red_mask]).astype(np.uint8)
            if blacklist_added_boxes:
                overlay_new = np.zeros_like(cell_img, dtype=np.uint8)
                for br0, bc0, br1, bc1 in blacklist_added_boxes:
                    if rotate:
                        br0, bc0, br1, bc1 = w0 - 1 - bc1, br0, w0 - 1 - bc0, br1
                    br0o = int(round(float(br0) * float(cell_img.shape[0]) / float(h1)))
                    br1o = int(round(float(br1) * float(cell_img.shape[0]) / float(h1)))
                    bc0o = int(round(float(bc0) * float(cell_img.shape[1]) / float(w1)))
                    bc1o = int(round(float(bc1) * float(cell_img.shape[1]) / float(w1)))
                    cv2.rectangle(overlay_new, (bc0o, br0o), (bc1o, br1o), (128, 0, 128), -1)
                purple_mask = (overlay_new[:, :, 0] | overlay_new[:, :, 1] | overlay_new[:, :, 2]) > 0
                cell_img[purple_mask] = (0.6 * cell_img[purple_mask] + 0.4 * overlay_new[purple_mask]).astype(np.uint8)
            for r in range(r0, cell_img.shape[0], row_step_out):
                cv2.line(cell_img, (0, r), (cell_img.shape[1] - 1, r), (255, 0, 0), 1)
            for c in range(c0, cell_img.shape[1], col_step_out):
                cv2.line(cell_img, (c, 0), (c, cell_img.shape[0] - 1), (255, 0, 0), 1)

            for r0, c0, r1, c1 in revisit_boxes:
                if rotate:
                    r0, c0, r1, c1 = w0 - 1 - c1, r0, w0 - 1 - c0, r1
                r0o = int(round(float(r0) * float(cell_img.shape[0]) / float(h1)))
                r1o = int(round(float(r1) * float(cell_img.shape[0]) / float(h1)))
                c0o = int(round(float(c0) * float(cell_img.shape[1]) / float(w1)))
                c1o = int(round(float(c1) * float(cell_img.shape[1]) / float(w1)))
                cv2.rectangle(cell_img, (c0o, r0o), (c1o, r1o), (255, 0, 0), 1)

            cv2.putText(cell_img, f"soft_TSS={soft_TSS:.3f} hard_TSS={hard_TSS:.3f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(cell_img, f"soft_TSS={soft_TSS:.3f} hard_TSS={hard_TSS:.3f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(cell_img, f"scaled_r_score={scaled_r_score:.3f} r_min={r_score_min:.3f} r_max={r_score_max:.3f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(cell_img, f"scaled_r_score={scaled_r_score:.3f} r_min={r_score_min:.3f} r_max={r_score_max:.3f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(cell_img, f"cnt_revisit={cnt_revisit} n_f={num_frontiers} n_f_max={num_frontiers_max} threshold_delta={threshold_delta:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(cell_img, f"cnt_revisit={cnt_revisit} n_f={num_frontiers} n_f_max={num_frontiers_max} threshold_delta={threshold_delta:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            bar_h = 50
            grad = np.tile(np.linspace(0, 255, cell_img.shape[1], dtype=np.uint8), (bar_h, 1))
            bar = cv2.applyColorMap(grad, cv2.COLORMAP_JET)[:, :, ::-1]
            bar = np.ascontiguousarray(bar)
            x = int(round(scaled_r_score * float(cell_img.shape[1] - 1)))
            cv2.line(bar, (x, 0), (x, bar_h - 1), (255, 255, 255), 2)
            cv2.putText(bar, "0", (5, bar_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(bar, "1", (bar.shape[1] - 20, bar_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(
                bar,
                f"scaled_r_score={scaled_r_score:.2f}",
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            out = np.vstack([cell_img, bar])
            out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(episode_dir, "cell_top_down_map.png"), out_bgr)
        for map_key, file_name in (
            ("obstacle_map", "obstacle_map.png"),
            ("value_map", "value_map.png"),
        ):
            if map_key not in policy_info:
                continue

            map_img = np.array(policy_info[map_key])

            if map_img.dtype != np.uint8:
                map_img = np.clip(map_img, 0, 255).astype(np.uint8)

            if map_img.ndim == 2:
                map_img = cv2.cvtColor(map_img, cv2.COLOR_GRAY2BGR)
            elif map_img.ndim == 3 and map_img.shape[2] == 4:
                map_img = cv2.cvtColor(map_img, cv2.COLOR_RGBA2BGR)
            elif map_img.ndim == 3 and map_img.shape[2] == 3:
                map_img = cv2.cvtColor(map_img, cv2.COLOR_RGB2BGR)

            cv2.imwrite(os.path.join(episode_dir, file_name), map_img)

    # This runs because of execute_exp(cfg, "eval") at vlfm/run.py -> The "eval" mode.
    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        if self._is_distributed:
            raise RuntimeError("Evaluation does not support distributed mode")

        # Some configurations require not to load the checkpoint, like when using
        # a hierarchial policy
        if self.config.habitat_baselines.eval.should_load_ckpt:
            # map_location="cpu" is almost always better than mapping to a CUDA device.
            try:
                ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu", weights_only=True)
            except:
                ckpt_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            step_id = ckpt_dict["extra_state"]["step"]
            print(step_id)
        else:
            ckpt_dict = {"config": None}

        config = self._get_resume_state_config_or_new_config(ckpt_dict["config"])

        with read_write(config):
            config.habitat.dataset.split = config.habitat_baselines.eval.split
            config.habitat.seed = 2025
            config.habitat.environment.iterator_options.shuffle = True

        ds_json_data_path = config.habitat.dataset.data_path.format(split=config.habitat.dataset.split)
        # open with gzip json
        with gzip.open(ds_json_data_path, "rt") as f:
            ds_json_data = json.load(f)
        integer_to_string_mapping = ds_json_data.get("category_to_task_category_id")
        if integer_to_string_mapping is None:
            scenes_dir = str(config.habitat.dataset.scenes_dir).format(split=config.habitat.dataset.split)
            if not os.path.isdir(scenes_dir):
                raise RuntimeError(
                    "Missing category_to_task_category_id and invalid scenes_dir for fallback mapping: "
                    f"{scenes_dir}"
                )
            categories = set()
            scene_files = sorted(
                fname for fname in os.listdir(scenes_dir) if fname.endswith(".json.gz")
            )
            if len(scene_files) == 0:
                raise RuntimeError(f"No scene content files found for fallback mapping under {scenes_dir}")
            for fname in scene_files:
                with gzip.open(os.path.join(scenes_dir, fname), "rt") as sf:
                    scene_data = json.load(sf)
                for ep in scene_data.get("episodes", []):
                    object_category = ep.get("object_category")
                    if object_category is not None:
                        categories.add(str(object_category))
            integer_to_string_mapping = {c: i for i, c in enumerate(sorted(categories))}
            ds_json_data["category_to_task_category_id"] = integer_to_string_mapping

        text_goal_by_episode: Dict[int, str] = {}
        task_type = str(getattr(config.habitat_baselines.rl.policy, "task_type", "")).strip().lower()
        if task_type == "text_goal":
            attr_data = ds_json_data.get("attribute_data")
            if not isinstance(attr_data, dict):
                attr_data = {}
            scenes_dir = str(config.habitat.dataset.scenes_dir).format(split=config.habitat.dataset.split)
            if not os.path.isdir(scenes_dir):
                raise RuntimeError(f"Invalid scenes_dir for text_goal mapping: {scenes_dir}")
            scene_files = sorted(fname for fname in os.listdir(scenes_dir) if fname.endswith(".json.gz"))
            for fname in scene_files:
                scene_token = fname.split(".")[0]
                with gzip.open(os.path.join(scenes_dir, fname), "rt") as sf:
                    scene_data = json.load(sf)
                for ep in scene_data.get("episodes", []):
                    ep_id = int(ep["episode_id"])
                    goal_object_id = ep.get("goal_object_id")
                    if goal_object_id is None:
                        continue
                    goal_key = f"{scene_token}_{goal_object_id}"
                    text_goal_dict = attr_data.get(goal_key)
                    if not isinstance(text_goal_dict, dict):
                        scene_attr = attr_data.get(scene_token)
                        if isinstance(scene_attr, dict):
                            text_goal_dict = scene_attr.get(str(goal_object_id)) or scene_attr.get(goal_object_id)
                    if not isinstance(text_goal_dict, dict):
                        text_goal_dict = {}
                    intrinsic = str(
                        text_goal_dict.get("intrinsic_attributes")
                        or text_goal_dict.get("intrinsic_attribute")
                        or text_goal_dict.get("intrinsic")
                        or ""
                    ).strip()
                    extrinsic = str(
                        text_goal_dict.get("extrinsic_attributes")
                        or text_goal_dict.get("extrinsic_attribute")
                        or text_goal_dict.get("extrinsic")
                        or text_goal_dict.get("relation")
                        or ""
                    ).strip()
                    instruction = (intrinsic + " " + extrinsic).strip()
                    if not instruction:
                        instruction = str(text_goal_dict.get("instruction") or text_goal_dict.get("text_goal") or "").strip()
                    text_goal_by_episode[ep_id] = instruction

        max_episode_steps = int(getattr(config.habitat.environment, "max_episode_steps", 0))
        save_video = bool(config.habitat_baselines.rl.policy.save_video)
        save_logging_images = bool(config.habitat_baselines.rl.policy.save_logging_images)
        video_enabled = len(self.config.habitat_baselines.eval.video_option) > 0 and save_video
        if not save_video and len(self.config.habitat_baselines.eval.video_option) > 0:
            logger.info(
                "Disabling video recording because habitat_baselines.rl.policy.save_video=false "
                f"(habitat.environment.max_episode_steps={max_episode_steps})."
            )
        elif len(self.config.habitat_baselines.eval.video_option) == 0:
            logger.info(
                "Video recording disabled because habitat_baselines.eval.video_option is empty "
                f"(habitat.environment.max_episode_steps={max_episode_steps})."
            )
        else:
            logger.info(
                "Enabling video recording with video_option="
                f"{self.config.habitat_baselines.eval.video_option} "
                f"(habitat.environment.max_episode_steps={max_episode_steps})."
            )

        if video_enabled:
            agent_config = get_agent_config(config.habitat.simulator)
            agent_sensors = agent_config.sim_sensors
            extra_sensors = config.habitat_baselines.eval.extra_sim_sensors
            with read_write(agent_sensors):
                agent_sensors.update(extra_sensors)
            with read_write(config):
                if config.habitat.gym.obs_keys is not None:
                    for render_view in extra_sensors.values():
                        if render_view.uuid not in config.habitat.gym.obs_keys:
                            config.habitat.gym.obs_keys.append(render_view.uuid)
                config.habitat.simulator.debug_render = True

        if config.habitat_baselines.verbose:
            logger.info(f"env config: {OmegaConf.to_yaml(config)}")
        # print("=" * 100)
        # print(config)
        # print(config.keys())
        # print(config.values())
        # print("=" * 100)
        # breakpoint()
        self._init_envs(config, is_eval=True)

        self._agent = self._create_agent(None)
        self._agent.actor_critic.set_obj_id_to_name(integer_to_string_mapping=integer_to_string_mapping)
        action_shape, discrete_actions = get_action_space_info(self._agent.policy_action_space)

        if self._agent.actor_critic.should_load_agent_state:
            self._agent.load_state_dict(ckpt_dict)

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device="cpu")

        test_recurrent_hidden_states = torch.zeros(
            (
                self.config.habitat_baselines.num_environments,
                *self._agent.hidden_state_shape,
            ),
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.habitat_baselines.num_environments,
            *action_shape,
            device=self.device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        not_done_masks = torch.zeros(
            self.config.habitat_baselines.num_environments,
            1,
            device=self.device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[Any, Any] = {}  # dict of dicts that stores stats per episode
        ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0)

        # Get the current timestamp and format it as a string
        from datetime import datetime
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        rgb_frames: List[List[np.ndarray]] = [[] for _ in range(self.config.habitat_baselines.num_environments)]

        number_of_eval_episodes = self.config.habitat_baselines.test_episode_count
        evals_per_ep = self.config.habitat_baselines.eval.evals_per_ep
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            # if total_num_eps is negative, it means the number of evaluation episodes is unknown
            if total_num_eps < number_of_eval_episodes and total_num_eps > 1:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes, dataset only has {{total_num_eps}}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps
            else:
                assert evals_per_ep == 1
        assert number_of_eval_episodes > 0, "You must specify a number of evaluation episodes with test_episode_count"

        # Optional episode sharding: split the full set of evaluation episodes into contiguous shards.
        # Prefer policy.shard_episode if present; otherwise fall back to top-level habitat_baselines fields.
        policy_cfg = self.config.habitat_baselines.rl.policy
        if hasattr(policy_cfg, "pbp") and hasattr(policy_cfg.pbp, "instance_grouping_method"):
            instance_grouping_method = str(policy_cfg.pbp.instance_grouping_method).lower()
            if instance_grouping_method not in ALLOWED_INSTANCE_GROUPING_METHODS:
                raise ValueError(
                    f"[vlfm_trainer] Invalid instance_grouping_method: {instance_grouping_method}. "
                    f"Allowed: {list(ALLOWED_INSTANCE_GROUPING_METHODS)}"
                )
        if hasattr(policy_cfg, "shard_episode"):
            shard_size = policy_cfg.shard_episode.shard_size
            shard_index = policy_cfg.shard_episode.shard
        else:
            shard_size = getattr(self.config.habitat_baselines, "shard_size", 0)
            shard_index = getattr(self.config.habitat_baselines, "shard", 0)
        print(Fore.CYAN + f"Shard evaluation config: size={shard_size}, index={shard_index}")
        shard_enabled = shard_size is not None and shard_size > 0
        shard_start = 0
        shard_end = number_of_eval_episodes
        if shard_enabled:
            shard_start = shard_index * shard_size
            shard_end = min(shard_start + shard_size, number_of_eval_episodes)
            if shard_start >= number_of_eval_episodes:
                logger.warn(
                    f"Shard start {shard_start} is >= total eval episodes {number_of_eval_episodes}. Nothing to run."
                )
                return
            # For this run, only evaluate the episodes in [shard_start, shard_end).
            number_of_eval_episodes = shard_end - shard_start
            print(
                Fore.CYAN
                + f"[SHARD] Evaluating shard {shard_index} with size {shard_size}: "
                f"episodes [{shard_start}, {shard_end}) out of {self.config.habitat_baselines.test_episode_count}"
            )

        episodes_to_run = set(map(str, policy_cfg.episodes_to_run))
        episodes_to_skip = set(map(str, policy_cfg.episodes_to_skip))

        episode_folder_suffix = timestamp_str
        if shard_enabled:
            episode_folder_suffix = f"{timestamp_str}_{shard_start}_{shard_end}"

        if video_enabled:
            os.makedirs(self.config.habitat_baselines.video_dir + f"/{episode_folder_suffix}", exist_ok=True)

        expected_eval_episodes = len(episodes_to_run) if len(episodes_to_run) > 0 else number_of_eval_episodes
        pbar = tqdm.tqdm(total=expected_eval_episodes * evals_per_ep)
        self._agent.eval()

        from vlfm.utils.habitat_visualizer import HabitatVis

        split = config.habitat.dataset.split

        eval_folder_name = os.environ.get("EVAL_FOLDER_NAME", "")
        if eval_folder_name:
            data_backup_path = f"eval_folder/{split}/{eval_folder_name}/{episode_folder_suffix}"
        else:
            data_backup_path = f"eval_folder/{split}/{episode_folder_suffix}"
        if not os.path.exists(data_backup_path):
            os.makedirs(data_backup_path)
        self._agent._actor_critic.set_folder_for_data_backup(data_backup_path)
        if getattr(self._agent._actor_critic, "_enable_multi_view", False) and getattr(
            self._agent._actor_critic, "_enable_multi_view_optimization", False
        ):
            image_embedding_model = "facebook/dinov2-large"
            cuda_device_id = 0 if os.environ.get("CUDA_VISIBLE_DEVICES") else int(os.environ.get("CUDA_DEVICE", 0))
            device = f"cuda:{cuda_device_id}"
            image_only_model = AutoModel.from_pretrained(
                image_embedding_model,
                device_map=device,
                dtype=torch.bfloat16,
            ).eval()
            image_only_processor = AutoImageProcessor.from_pretrained(image_embedding_model)
            self._agent._actor_critic.image_only_model = image_only_model
            self._agent._actor_critic.image_only_processor = image_only_processor
            text_only_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
            text_only_model = text_only_model.to(torch.bfloat16)
            self._agent._actor_critic.text_only_model = text_only_model
            nli_model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
            nli_only_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
            nli_only_model = AutoModelForSequenceClassification.from_pretrained(
                nli_model_name
            ).to(device).to(torch.bfloat16).eval()
            self._agent._actor_critic.nli_only_model = nli_only_model
            self._agent._actor_critic.nli_only_tokenizer = nli_only_tokenizer

        print(Fore.CYAN + "=" * 80)
        print(Fore.CYAN + "[EVALUATION] Model Configuration Summary:")
        try:
            print(Fore.GREEN + f"  VLM: {self._agent._actor_critic.vlm_agent_brain.model_name}")
            print(Fore.GREEN + f"  LLM: {self._agent._actor_critic.llm_agent_brain.model_name}")
        except:
            pass
        print(Fore.CYAN + "=" * 80)
     
        hab_vis = HabitatVis() if video_enabled else None
        print(Fore.GREEN + f"Evaluating a total of {expected_eval_episodes * evals_per_ep} episodes")
        cnt_episode = 0
        check_first_observation = True
        episode_start_gps = None
        agent_episode_distance = 0.0
        prev_position = None
        cnt_step = 0
        pose_history_by_episode: Dict[Any, Dict[int, Dict[str, Any]]] = defaultdict(dict)
        episode_min_goal_distance: Dict[Any, float] = {}
        episode_proximity_state: Dict[Any, str] = {}

        PROXIMITY_BANDS: List[Tuple[float, str]] = [
            (0.25, "reached"),
            (0.50, "adjacent"),
            (0.75, "nearby"),
            (1.00, "close"),
        ]

        def update_min_goal_distance(ep_id: Any, distance: float) -> float:
            previous = episode_min_goal_distance.get(ep_id, float("inf"))
            new_value = distance if np.isinf(previous) else min(distance, previous)
            episode_min_goal_distance[ep_id] = new_value
            return new_value

        def get_min_goal_distance(ep_id: Any) -> float:
            return episode_min_goal_distance.get(ep_id, float("inf"))

        def compute_proximity_state(distance: float) -> str:
            for threshold, label in PROXIMITY_BANDS:
                if distance <= threshold:
                    return label
            return "far"

        def save_observation_image(
            observation: Dict[str, Any],
            save_dir: str,
            filename: str = "last_observation",
        ) -> None:
            rgb_obs = observation.get("rgb")
            if rgb_obs is None:
                return

            if isinstance(rgb_obs, torch.Tensor):
                rgb = rgb_obs.detach().cpu().numpy()
            else:
                rgb = np.asarray(rgb_obs)

            if rgb.ndim == 3 and rgb.shape[0] in (3, 4) and rgb.shape[0] != rgb.shape[-1]:
                rgb = np.transpose(rgb[:3], (1, 2, 0))
            if rgb.ndim == 3 and rgb.shape[-1] > 3:
                rgb = rgb[..., :3]

            if rgb.dtype != np.uint8:
                max_val = float(np.max(rgb)) if rgb.size > 0 else 0.0
                scale = 255.0 if max_val <= 1.0 else 1.0
                rgb = np.clip(rgb * scale, 0, 255).astype(np.uint8)
            else:
                rgb = rgb.copy()

            os.makedirs(save_dir, exist_ok=True)
            safe_filename = filename.replace(" ", "_")
            save_path = os.path.join(save_dir, f"{safe_filename}.png")
            try:
                cv2.imwrite(save_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            except cv2.error as exc:
                print(Fore.RED + f"[WARN] Failed to save last observation snapshot: {exc}")

        def update_goal_tracking(
            env_idx: int,
            episode: Any,
            episode_id: Any,
            observation: Dict[str, Any],
        ) -> None:
            goals = getattr(episode, "goals", None)
            if goals:
                # NOTE: Ground-truth goal coordinates; strict read-only logging to avoid leaking answers into policy updates.
                primary_goal = goals[0]
                goal_position = getattr(primary_goal, "position", None)
                if goal_position is not None:
                    goal_position = np.asarray(goal_position, dtype=np.float32)
                    infos[env_idx]["goal_xyz_gt"] = goal_position
                    if goal_position.size >= 3:
                        infos[env_idx]["goal_xy_gt"] = goal_position[[0, 2]]
                    elif goal_position.size >= 2:
                        infos[env_idx]["goal_xy_gt"] = goal_position[:2]

            # updates based on geodiesic distance (not euclidean distance)
            dist_to_goal = infos[env_idx].get("distance_to_goal")
            if dist_to_goal is None:
                return
            dist_to_goal = float(dist_to_goal)
            if not np.isfinite(dist_to_goal):
                return

            current_min_distance = update_min_goal_distance(episode_id, dist_to_goal)
            print(
                f"[Episode {episode_id}] Step {cnt_step}: Distance to goal: {dist_to_goal:.3f}, "
                f"Min distance so far: {current_min_distance:.3f}"
            )

            current_proximity_label = compute_proximity_state(dist_to_goal)
            episode_proximity_state[episode_id] = compute_proximity_state(current_min_distance)

            if current_proximity_label in ("adjacent", "reached"):
                observation_dir = os.path.join(
                    data_backup_path,
                    str(episode_id),
                    "observation",
                )
                filename = f"{cnt_step:04d}_{current_proximity_label}_{dist_to_goal:.3f}"
                save_observation_image(
                    observation,
                    observation_dir,
                    filename=filename,
                )
        category_id_inverse_mapping = {v: k for k, v in integer_to_string_mapping.items()}
        skip_state_sync_enabled = task_type in {"text_goal", "image_goal"}

        def sync_policy_inputs_after_skip(
            observations_after_skip: List[Dict[str, Any]],
            dones_after_skip: List[bool],
        ) -> Tuple[Any, torch.Tensor]:
            if not skip_state_sync_enabled:
                return batch, not_done_masks
            synced_batch = batch_obs(observations_after_skip, device=self.device)  # type: ignore
            synced_batch = apply_obs_transforms_batch(synced_batch, self.obs_transforms)  # type: ignore
            synced_not_done_masks = torch.tensor(
                [[not done] for done in dones_after_skip],
                dtype=torch.bool,
                device="cpu",
            )
            prev_actions.zero_()
            return synced_batch, synced_not_done_masks

        # wandb! wandb! wandb! wandb! wandb! wandb! wandb! wandb! wandb! wandb! wandb! wandb! wandb! wandb! wandb! wandb! wandb! wandb!
        if os.environ.get("WANDB_MODE", "") != "disabled":
            wandb.init(
                project="Fight_AIUTA",
                entity="tree-jhk",
                name=f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                config={
                    "split": config.habitat.dataset.split,
                    "model": self._agent.__class__.__name__,
                    "ckpt_path": checkpoint_path,
                    "num_eval_episodes": expected_eval_episodes,
                    "timestamp": timestamp_str
                }
            )
        target_total_episode_evals = expected_eval_episodes * evals_per_ep
        while len(stats_episodes) < target_total_episode_evals and self.envs.num_envs > 0:
            current_episodes_info = self.envs.current_episodes()
            current_episode_id = str(current_episodes_info[0].episode_id)
            # Shard-based episode skipping: if shard_size > 0, only evaluate episodes in [shard_start, shard_end).
            if shard_enabled:
                if cnt_episode < shard_start:
                    print(
                        Fore.YELLOW
                        + f"[SKIP] Pre-shard episode {current_episode_id} (idx {cnt_episode})"
                    )
                    print("episode_id before step:", self.envs.current_episodes()[0].episode_id)
                    # dummy step to continue environment rollout
                    observations, rewards_l, dones, infos = [
                        list(x) for x in zip(*self.envs.step([0] * self.envs.num_envs))
                    ]
                    batch, not_done_masks = sync_policy_inputs_after_skip(observations, dones)
                    print("episode_id after step:", self.envs.current_episodes()[0].episode_id)
                    print(Fore.YELLOW + f"objectgoal after step for episode {self.envs.current_episodes()[0].episode_id}: {category_id_inverse_mapping[observations[0]['objectgoal'][0]]}")
                    cnt_episode += 1
                    check_first_observation = True
                    cnt_step = 0
                    continue
                if cnt_episode >= shard_end:
                    print(
                        Fore.YELLOW
                        + f"[DONE] Reached end of shard episodes [{shard_start}, {shard_end}); stopping evaluation."
                    )
                    break
            if episodes_to_skip and current_episode_id in episodes_to_skip:
                print(Fore.YELLOW + f"[SKIP] Skipping episode {current_episode_id} (in episodes_to_skip)")
                print("episode_id before step:", self.envs.current_episodes()[0].episode_id)
                # dummy step to continue environment rollout
                observations, rewards_l, dones, infos = [list(x) for x in zip(*self.envs.step([0]*self.envs.num_envs))]
                batch, not_done_masks = sync_policy_inputs_after_skip(observations, dones)
                print("episode_id after step:", self.envs.current_episodes()[0].episode_id)
                print(Fore.YELLOW + f"objectgoal after step for episode {self.envs.current_episodes()[0].episode_id}: {category_id_inverse_mapping[observations[0]['objectgoal'][0]]}")
                cnt_episode += 1
                check_first_observation = True
                cnt_step = 0
                continue
            if episodes_to_run and current_episode_id not in episodes_to_run:
                print(Fore.YELLOW + f"[SKIP] Skipping episode {current_episode_id} (not in episodes_to_run)")
                print("episode_id before step:", self.envs.current_episodes()[0].episode_id)
                # dummy step to continue environment rollout
                observations, rewards_l, dones, infos = [list(x) for x in zip(*self.envs.step([0]*self.envs.num_envs))]
                batch, not_done_masks = sync_policy_inputs_after_skip(observations, dones)
                print("episode_id after step:", self.envs.current_episodes()[0].episode_id)
                print(Fore.YELLOW + f"objectgoal after step for episode {self.envs.current_episodes()[0].episode_id}: {category_id_inverse_mapping[observations[0]['objectgoal'][0]]}")
                cnt_episode += 1
                check_first_observation = True
                cnt_step = 0
                continue
            try:
                cnt_step += 1
                with torch.no_grad():
                    with inference_mode():
                        ep_obj = current_episodes_info[0]
                        raw_instruction = getattr(ep_obj, "instruction", None)
                        if task_type == "text_goal" and not raw_instruction:
                            raw_instruction = text_goal_by_episode.get(int(ep_obj.episode_id), "")
                        instruction_to_set = str(raw_instruction or "").strip()
                        self._agent._actor_critic.set_ep_id(ep_obj.episode_id)
                        self._agent._actor_critic.set_text_goal(instruction_to_set)
                        try:
                            action_data = self._agent.actor_critic.act(
                                batch,
                                test_recurrent_hidden_states,
                                prev_actions,
                                not_done_masks,
                                deterministic=False,
                                current_step=cnt_step,
                            )
                        except:
                            print(Fore.RED + "[ERROR] Exception in act - stop episode and load the next one.")
                            action_data = PolicyActionData(
                                rnn_hidden_states=test_recurrent_hidden_states,
                                actions=torch.zeros_like(prev_actions),
                                policy_info=[
                                    {"target_point_cloud": np.array([])} for _ in range(self.envs.num_envs)
                                ],
                            )
                            logging.exception(
                                "Execption during act function for episode: " + str(current_episodes_info[0].episode_id)
                            )

                        if "VLFM_RECORD_ACTIONS_DIR" in os.environ:
                            action_id = action_data.actions.cpu()[0].item()
                            filepath = os.path.join(
                                os.environ["VLFM_RECORD_ACTIONS_DIR"],
                                "actions.txt",
                            )
                            # If the file doesn't exist, create it
                            if not os.path.exists(filepath):
                                open(filepath, "w").close()
                            with open(filepath, "a") as f:
                                f.write(f"{action_id}\n")

                        if action_data.should_inserts is None:
                            test_recurrent_hidden_states = action_data.rnn_hidden_states
                            prev_actions.copy_(action_data.actions)  # type: ignore
                        else:
                            for i, should_insert in enumerate(action_data.should_inserts):
                                if should_insert.item():
                                    test_recurrent_hidden_states[i] = action_data.rnn_hidden_states[i]
                                    prev_actions[i].copy_(action_data.actions[i])  # type: ignore
                # NB: Move actions to CPU.  If CUDA tensors are
                # sent in to env.step(), that will create CUDA contexts
                # in the subprocesses.
                if is_continuous_action_space(self._env_spec.action_space):
                    # Clipping actions to the specified limits
                    step_data = [
                        np.clip(
                            a.numpy(),
                            self._env_spec.action_space.low,
                            self._env_spec.action_space.high,
                        )
                        for a in action_data.env_actions.cpu()
                    ]
                else:
                    step_data = [a.item() for a in action_data.env_actions.cpu()]

                outputs = self.envs.step(step_data)

                observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]

                if check_first_observation:
                    episode_start_gps = np.array(observations[0]["gps"])
                    check_first_observation = False
                    agent_episode_distance = 0.0
                    prev_position = np.array(observations[0]["gps"])

                    episode_start_time = time.time()  # used to measure per-episode inference time
                else:
                    current_position = np.array(observations[0]["gps"])
                    if prev_position is not None:
                        step_distance = np.linalg.norm(current_position - prev_position)
                        agent_episode_distance += step_distance
                    prev_position = current_position
                
                policy_infos = self._agent.actor_critic.get_extra(action_data, infos, dones)
                for i in range(len(policy_infos)):
                    # Keep large binary blobs out of `infos` (used for overlay/metrics),
                    # while still allowing the visualizer to consume them via `action_data.policy_info`.
                    filtered_policy_info = {
                        k: v for k, v in policy_infos[i].items() if k not in {"instance_imagegoal"}
                    }
                    infos[i].update(filtered_policy_info)
                # Track distance-to-target and proximity state.
                for env_idx in range(len(infos)):
                    episode = current_episodes_info[env_idx]
                    episode_id = episode.episode_id
                    update_goal_tracking(env_idx, episode, episode_id, observations[env_idx])
    
                batch = batch_obs(  # type: ignore
                    observations,
                    device=self.device,
                )
                batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

                not_done_masks = torch.tensor(
                    [[not done] for done in dones],
                    dtype=torch.bool,
                    device="cpu",
                )

                rewards = torch.tensor(rewards_l, dtype=torch.float, device="cpu").unsqueeze(1)
                current_episode_reward += rewards
                next_episodes_info = self.envs.current_episodes()
                envs_to_pause = []
                n_envs = self.envs.num_envs
                for i in range(n_envs):
                    if (
                        ep_eval_count[
                            (
                                next_episodes_info[i].scene_id,
                                next_episodes_info[i].episode_id,
                            )
                        ]
                        == evals_per_ep
                    ):
                        envs_to_pause.append(i)
                    # try:
                    if video_enabled and hab_vis is not None:
                        infos[i]["episode_id"] = current_episodes_info[i].episode_id
                        infos[i]["data_backup_path"] = data_backup_path
                        infos[i]["step"] = cnt_step
                        hab_vis.collect_data(batch, infos, policy_infos)

                    episode_dir = os.path.join(
                        data_backup_path,
                        str(current_episodes_info[i].episode_id),
                        "self_questioner",
                    )
                    if save_logging_images:
                        self._save_step_maps(episode_dir, infos[i], action_data.policy_info[i])
                    # Episode ended.
                    episode_dir = os.path.join(
                        data_backup_path,
                        str(current_episodes_info[i].episode_id),
                        "self_questioner",
                    )
                    if save_logging_images:
                        save_observation_image(observations[i], episode_dir, filename="current_observation")
                    if not not_done_masks[i].item():
                        pbar.update()
                        episode_stats = {"reward": current_episode_reward[i].item()}
                        episode_stats.update(extract_scalars_from_info(infos[i]))

                        episode_end_time = time.time()
                        elapsed_time = episode_end_time - episode_start_time

                        # breakpoint()
                        start_gps = np.array(episode_start_gps) if episode_start_gps is not None else None
                        gps_value = infos[i].get(
                            "gps",
                            observations[i].get("gps") if isinstance(observations[i], dict) else None,
                        )
                        if isinstance(gps_value, str):
                            end_gps = np.fromstring(gps_value.strip("[]"), sep=" ")
                        else:
                            end_gps = np.array(gps_value) if gps_value is not None else np.array([])

                        nav_goal_value = infos[i].get("nav_goal", None)
                        if nav_goal_value is None and getattr(action_data, "policy_info", None):
                            if len(action_data.policy_info) > i and isinstance(action_data.policy_info[i], dict):
                                nav_goal_value = action_data.policy_info[i].get("nav_goal", None)
                        target_gps = np.array(nav_goal_value) if nav_goal_value is not None else np.array([])

                        geodesic_distance_end_to_target = float(
                            infos[i].get("distance_to_goal", float("nan"))
                        )

                        episode_stats["agent_episode_distance"] = agent_episode_distance
                        episode_stats["geodesic_distance_end_to_target"] = geodesic_distance_end_to_target
                        episode_stats["total_steps"] = cnt_step
                        episode_stats["elapsed_time_sec"] = elapsed_time

                        current_episode_reward[i] = 0
                        k = (
                            current_episodes_info[i].scene_id,
                            current_episodes_info[i].episode_id,
                        )
                        ep_eval_count[k] += 1
                        # use scene_id + episode_id as unique id for storing stats
                        stats_episodes[(k, ep_eval_count[k])] = episode_stats

                        gc.collect()

                        # save infos to disk
                        ep_id_path = os.path.join(data_backup_path, str(current_episodes_info[i].episode_id))
                        if not os.path.exists(ep_id_path):
                            os.makedirs(ep_id_path)

                        infos_to_save = extract_scalars_from_info(infos[i])
                        print(Fore.GREEN + self._agent._actor_critic._target_object.split("|")[0])
                        infos_to_save["episode_id"] = current_episodes_info[i].episode_id
                        infos_to_save["scene_id"] = current_episodes_info[i].scene_id
                        if task_type == "text_goal":
                            ep_obj = current_episodes_info[i]
                            raw_instruction = getattr(ep_obj, "instruction", None)
                            if not raw_instruction:
                                raw_instruction = text_goal_by_episode.get(int(ep_obj.episode_id), "")
                            infos_to_save["text_goal"] = str(raw_instruction or "").strip()
                        
                        # Extra logs for AIUTA-style analysis.
                        llm_agent_brain = self._agent._actor_critic._object_map.llm_agent_brain
                        vlm_agent_brain = self._agent._actor_critic._object_map.vlm_agent_brain
                        vlm_oracle = self._agent._actor_critic._object_map.vlm_oracle
                        target_object_informations = llm_agent_brain.target_object_informations # facts of target object obtained by user(simulator)
                        ep_id = current_episodes_info[i].episode_id
                        if ep_id is not None:
                            num_obtained_facts = vlm_oracle.how_many_question_to_the_user(ep_id)
                        
                        # detected_objects = llm_agent_brain.objects_graph_informations
                        detected_objects = self._agent._actor_critic._object_map.object_final_status

                        questions_for_target_object_list = vlm_oracle.questions_for_target_object_list
                        answers_for_target_object_list = vlm_oracle.answers_for_target_object_list
                        facts_for_target_object_list = vlm_oracle.facts_for_target_object_list

                        cnt_ask = self._agent._actor_critic._object_map.cnt_ask
                        total_detected_objects = self._agent._actor_critic._object_map.object_unique_id

                        visited_target_by_detection = False
                        possibly_target = None

                        # Annotate each detected object with distance metrics and verification results.
                        for object_id, object_info in detected_objects.items():
                            # object_cloud = object_info['object_map_position']
                            # if object_cloud.shape[1] != 4:
                            #     continue  # skip malformed entries
                            # object_mean_pos = np.mean(object_cloud[:, :2], axis=0)  # (x, y)
                            # dist = float(np.linalg.norm(target_gps - object_mean_pos))
                            # detected_objects[object_id]["straight_line_candidate_to_target"] = dist
                            # detected_objects[object_id]["object_mean_pos"] = object_mean_pos

                            if object_info.get("is_verified_target_by_user"):
                                visited_target_by_detection = True
                                possibly_target = object_id
                                break
                        
                        num_candidate_objects = len(detected_objects)

                        # you can save other infos as well here
                        infos_to_save["object_category"] = self._agent._actor_critic._target_object.split("|")[0]
                        infos_to_save["agent_episode_distance"] = agent_episode_distance
                        infos_to_save["geodesic_distance_end_to_target"] = geodesic_distance_end_to_target
                        infos_to_save["video_dir"] = (
                            self.config.habitat_baselines.video_dir + f"/{episode_folder_suffix}" if video_enabled else ""
                        )
                        infos_to_save["total_steps"] = cnt_step
                        infos_to_save["total_asks"] = cnt_ask
                        infos_to_save["total_questions_to_human"] = int(vlm_oracle.ask_to_human_episode_counter.get(ep_id, 0))
                        infos_to_save["visited_target_by_detection"] = int(visited_target_by_detection)
                        min_distance_recorded = get_min_goal_distance(ep_id)
                        proximity_state_recorded = episode_proximity_state.get(ep_id, "far")
                        infos_to_save["target_min_distance"] = float(min_distance_recorded)
                        infos_to_save["target_proximity_state"] = proximity_state_recorded
                        infos_to_save["visited_target"] = int(proximity_state_recorded in ("success", "adjacent"))
                        infos_to_save["num_candidate_objects"] = num_candidate_objects
                        infos_to_save["questions_for_target_object_list"] = questions_for_target_object_list.get(ep_id, [])
                        infos_to_save["answers_for_target_object_list"] = answers_for_target_object_list.get(ep_id, [])
                        infos_to_save["facts_for_target_object_list"] = facts_for_target_object_list.get(ep_id, [])
                        infos_to_save["total_detected_objects"] = total_detected_objects
                        rotate_controller = getattr(self._agent._actor_critic, "_rotate_controller", None)
                        rotate_used = getattr(rotate_controller, "rotate_used", 0)
                        openness = getattr(rotate_controller, "last_openness_score", -1.0)
                        infos_to_save["rotate_used"] = int(rotate_used)
                        infos_to_save["openness"] = float(openness)
                        infos_to_save["num_detected_objects"] = num_candidate_objects
                        infos_to_save["elapsed_time_sec"] = elapsed_time
                        if task_type == "coin":
                            response_len_total_tokens = int(vlm_oracle.response_len_total_tokens_episode.get(ep_id, 0))
                            response_len_num_valid_responses = int(
                                vlm_oracle.response_len_num_valid_responses_episode.get(ep_id, 0)
                            )
                            infos_to_save["response_len_total_tokens"] = response_len_total_tokens
                            infos_to_save["response_len_num_valid_responses"] = response_len_num_valid_responses
                            infos_to_save["response_len_avg_tokens"] = (
                                float(response_len_total_tokens / response_len_num_valid_responses)
                                if response_len_num_valid_responses > 0
                                else 0.0
                            )
                            success_value = float(infos_to_save.get("success", 0.0))
                            nq_value = int(infos_to_save["total_questions_to_human"])
                            infos_to_save["snq_including_nq0"] = (
                                float(success_value / nq_value) if nq_value > 0 else float(1.0 if success_value > 0.0 else 0.0)
                            )
                            infos_to_save["snq_excluding_nq0"] = float(success_value / nq_value) if nq_value > 0 else None
                        coin_meta = getattr(self._agent._actor_critic, "_coin_meta", None)
                        if coin_meta is not None:
                            infos_to_save["target_object_instance_id"] = coin_meta.object_instance_id
                            infos_to_save["target_object_position"] = coin_meta.target_position
                            infos_to_save["target_object_distractors"] = coin_meta.distractor_positions
                            infos_to_save["target_object_camera_spec"] = coin_meta.camera_spec
                        
                        # Add compact PBP results (omit verbose per-round logs).
                        if hasattr(self._agent._actor_critic._object_map, "pbp_results"):
                            raw_pbp_results = self._agent._actor_critic._object_map.pbp_results
                            compact_pbp_results = []
                            for pbp_result in raw_pbp_results:
                                if isinstance(pbp_result, dict):
                                    compact_result = dict(pbp_result)
                                    compact_result.pop("logs", None)
                                    compact_pbp_results.append(compact_result)
                                else:
                                    compact_pbp_results.append(pbp_result)
                            infos_to_save["pbp_results"] = compact_pbp_results
                        else:
                            infos_to_save["pbp_results"] = []
                        infos_to_save["pbp_avg_time_sec"] = float(
                            self._agent._actor_critic._object_map.get_pbp_avg_time_sec()
                        )
                        if "failure_reason" not in infos_to_save:
                            infos_to_save["failure_reason"] = self._infer_failure_reason(infos_to_save)
                        object_map = getattr(self._agent._actor_critic, "_object_map", None)
                        episode_total_time_sec = float(elapsed_time)
                        def _timing(key: str):
                            if object_map is None:
                                return 0.0, 0.0, 0
                            return (
                                float(object_map.get_timing_avg_sec(key)),
                                float(object_map.get_timing_total_sec(key)),
                                int(object_map.get_timing_count(key)),
                            )

                        reid_multiview_avg_time_sec, reid_multiview_total_time_sec, reid_multiview_count = _timing(
                            "reid_multiview"
                        )
                        pbp_avg_time_sec, pbp_total_time_sec, pbp_count = _timing("pbp")
                        pbp_round_avg_time_sec, _, pbp_round_count = _timing("pbp_round")
                        pbp_depth_avg_time_sec, _, pbp_depth_count = _timing("pbp_depth")
                        preproc_keys = (
                            "merge_instances_into_groups",
                            "select_diverse_view_for_each_group",
                            "generate_caption_for_each_group",
                        )
                        preproc_stats = {}
                        preproc_total_time_sec = 0.0
                        for key in preproc_keys:
                            avg_sec, total_sec, count = _timing(key)
                            preproc_stats[key] = (avg_sec, total_sec, count)
                            preproc_total_time_sec += float(total_sec)

                        exploration_time_sec = max(
                            0.0, episode_total_time_sec - pbp_total_time_sec - preproc_total_time_sec
                        )
                        pbp_total_time_per_instance = pbp_total_time_sec / max(1, total_detected_objects)
                        infos_to_save["reid_multiview_avg_time_sec"] = float(reid_multiview_avg_time_sec)
                        infos_to_save["pbp_round_avg_time_sec"] = float(pbp_round_avg_time_sec)
                        infos_to_save["pbp_depth_avg_time_sec"] = float(pbp_depth_avg_time_sec)
                        infos_to_save["pbp_total_time_sec"] = float(pbp_total_time_sec)
                        infos_to_save["reid_multiview_total_time_sec"] = float(reid_multiview_total_time_sec)
                        infos_to_save["exploration_time_sec"] = float(exploration_time_sec)
                        infos_to_save["pbp_total_time_per_instance"] = float(pbp_total_time_per_instance)
                        infos_to_save["pbp_preprocess_total_time_sec"] = float(preproc_total_time_sec)
                        for key in preproc_keys:
                            avg_sec, total_sec, count = preproc_stats[key]
                            infos_to_save[f"{key}_avg_time_sec"] = float(avg_sec)
                            infos_to_save[f"{key}_total_time_sec"] = float(total_sec)
                            infos_to_save[f"{key}_count"] = int(count)
                        print(
                            Fore.MAGENTA
                            + f"[MULTI_VIEW] Episode {ep_id} reID+multi-view avg: {reid_multiview_avg_time_sec:.6f}s total: {reid_multiview_total_time_sec:.6f}s (n={reid_multiview_count})"
                        )
                        print(
                            Fore.MAGENTA
                            + f"[MULTI_VIEW] Episode {ep_id} merge_instances_into_groups total: {preproc_stats['merge_instances_into_groups'][1]:.6f}s (n={preproc_stats['merge_instances_into_groups'][2]})"
                        )
                        print(
                            Fore.MAGENTA
                            + f"[MULTI_VIEW] Episode {ep_id} select_diverse_view_for_each_group total: {preproc_stats['select_diverse_view_for_each_group'][1]:.6f}s (n={preproc_stats['select_diverse_view_for_each_group'][2]})"
                        )
                        print(
                            Fore.MAGENTA
                            + f"[MULTI_VIEW] Episode {ep_id} generate_caption_for_each_group total: {preproc_stats['generate_caption_for_each_group'][1]:.6f}s (n={preproc_stats['generate_caption_for_each_group'][2]})"
                        )
                        print(
                            Fore.MAGENTA
                            + f"[MULTI_VIEW] Episode {ep_id} PBP preproc total (merge+select+caption): {preproc_total_time_sec:.6f}s"
                        )
                        if object_map is not None:
                            print(Fore.MAGENTA + f"[PBP] Episode {ep_id} PBP avg: {pbp_avg_time_sec:.6f}s (n={pbp_count})")
                            print(
                                Fore.MAGENTA
                                + f"[PBP] Episode {ep_id} PBP round avg: {pbp_round_avg_time_sec:.6f}s (n={pbp_round_count})"
                            )
                            print(
                                Fore.MAGENTA
                                + f"[PBP] Episode {ep_id} PBP depth avg: {pbp_depth_avg_time_sec:.6f}s (n={pbp_depth_count})"
                            )
                        reid_pct = (reid_multiview_total_time_sec / episode_total_time_sec * 100.0) if episode_total_time_sec > 0 else 0.0
                        pbp_pct = (pbp_total_time_sec / episode_total_time_sec * 100.0) if episode_total_time_sec > 0 else 0.0
                        exploration_pct = (exploration_time_sec / episode_total_time_sec * 100.0) if episode_total_time_sec > 0 else 0.0
                        preproc_pct = (preproc_total_time_sec / episode_total_time_sec * 100.0) if episode_total_time_sec > 0 else 0.0
                        print(Fore.MAGENTA + f"[TIME] Episode {ep_id} summary")
                        print(
                            Fore.MAGENTA
                            + f"[TIME] reID+multi-view avg: {reid_multiview_avg_time_sec:.6f}s total: {reid_multiview_total_time_sec:.6f}s ({reid_pct:.2f}%)"
                        )
                        print(
                            Fore.MAGENTA
                            + f"[TIME] PBP preproc total (merge+select+caption): {preproc_total_time_sec:.6f}s ({preproc_pct:.2f}%)"
                        )
                        print(
                            Fore.MAGENTA
                            + f"[TIME] PBP round avg: {pbp_round_avg_time_sec:.6f}s (n={pbp_round_count}) | PBP depth avg: {pbp_depth_avg_time_sec:.6f}s (n={pbp_depth_count})"
                        )
                        print(Fore.MAGENTA + f"[TIME] PBP total: {pbp_total_time_sec:.6f}s ({pbp_pct:.2f}%)")
                        print(Fore.MAGENTA + f"[TIME] PBP total per # of instances: {pbp_total_time_per_instance:.6f}s for {total_detected_objects} instances")
                        print(Fore.MAGENTA + f"[TIME] Exploration total: {exploration_time_sec:.6f}s ({exploration_pct:.2f}%)")
                        print(Fore.MAGENTA + f"[TIME] Episode total: {episode_total_time_sec:.6f}s")
                        infos_to_save["instance_image_description"] = object_map.vlm_oracle.get_description_of_the_image()
                        with open(os.path.join(ep_id_path, "info.json"), "w") as f:
                            json.dump(infos_to_save, f)
                    
                        jsonl_path = os.path.join(ep_id_path, "detected_objects.jsonl")
                        all_views = getattr(object_map, "views", None) if object_map is not None else None
                        with open(jsonl_path, "w") as f:
                            if not isinstance(all_views, dict) or not all_views:
                                print(
                                    Fore.YELLOW
                                    + "[WARN] object_map.views missing/empty; detected_objects.jsonl will be empty."
                                )
                            else:
                                for _, view_meta in sorted(all_views.items(), key=lambda kv: kv[0]):
                                    if not isinstance(view_meta, dict):
                                        continue
                                    instance_id = view_meta.get("instance_id")
                                    view_id = view_meta.get("view_id")
                                    if not instance_id or not view_id:
                                        continue
                                    out = dict(view_meta)
                                    out["representation_view"] = bool(view_meta.get("view_num") == 1)

                                    step_key = out.get("step")
                                    if isinstance(step_key, int):
                                        pose = pose_history_by_episode.get(ep_id, {}).get(step_key)
                                        if isinstance(pose, dict):
                                            out["robot_xy"] = pose.get("robot_xy", out.get("robot_xy"))
                                            out["robot_xyz"] = pose.get("robot_xyz")
                                            if "robot_heading" in pose:
                                                out["robot_heading"] = pose["robot_heading"]

                                    if out.get("geodesic_distance") is None:
                                        tf_flat = out.get("tf")
                                        end = out.get("closest_object_position")
                                        if (
                                            isinstance(tf_flat, list)
                                            and len(tf_flat) == 16
                                            and isinstance(end, (list, tuple))
                                            and len(end) == 3
                                        ):
                                            tf_mat = np.array(tf_flat, dtype=np.float32).reshape(4, 4)
                                            start = tf_mat[:3, 3]
                                            end_arr = np.array(end, dtype=np.float32)
                                            geo = float(
                                                self.envs.call_at(
                                                    i,
                                                    "geodesic_distance",
                                                    {"position_a": start.tolist(), "position_b": end_arr.tolist()},
                                                )
                                            )
                                            out["geodesic_distance"] = geo if np.isfinite(geo) else -1.0

                                    for key in (
                                        "tf",
                                        "tf_shape",
                                        "instance_num",
                                        "view_num",
                                        "closest_object_position",
                                        "is_possible_target",
                                        "rgb_image_description",
                                    ):
                                        out.pop(key, None)

                                    json.dump(out, f)
                                    f.write("\n")

                        episode_min_goal_distance.pop(ep_id, None)
                        episode_proximity_state.pop(ep_id, None)

                        from vlfm.utils.episode_stats_logger import (
                            log_episode_stats,
                        )

                        try:
                            failure_cause = log_episode_stats(
                                current_episodes_info[i].episode_id,
                                current_episodes_info[i].scene_id,
                                infos[i],
                            )
                        except Exception:
                            failure_cause = "Unknown"

    # wandb logging! wandb logging! wandb logging! wandb logging! wandb logging! wandb logging! wandb logging! wandb logging! wandb logging!
                        log_dict = {
                            "episode_id": current_episodes_info[i].episode_id,
                            "scene_id": current_episodes_info[i].scene_id,
                            "success": infos_to_save["success"],
                            "spl": infos_to_save["spl"],
                            "soft_spl": infos_to_save["soft_spl"],
                            "distance_to_goal": infos_to_save["distance_to_goal"],
                            "agent_episode_distance": infos_to_save["agent_episode_distance"],
                            "geodesic_distance_end_to_target": infos_to_save["geodesic_distance_end_to_target"],
                            "visited_target": infos_to_save["visited_target"],
                            "total_asks": infos_to_save["total_asks"],
                            "total_questions_to_human": infos_to_save["total_questions_to_human"],
                            "num_candidate_objects": infos_to_save["num_candidate_objects"],
                            "object_category": infos_to_save["object_category"],
                            "total_detected_objects": infos_to_save["total_detected_objects"],
                            "fail_reason": failure_cause,
                        }
                        if os.environ.get("WANDB_MODE", "") != "disabled":
                            wandb.log(log_dict)
                        if video_enabled and hab_vis is not None:
                            rgb_frames[i] = hab_vis.flush_frames(failure_cause)
                            generate_video(
                                video_option=self.config.habitat_baselines.eval.video_option,
                                # video_dir=self.config.habitat_baselines.video_dir,
                                video_dir=self.config.habitat_baselines.video_dir + f"/{episode_folder_suffix}",
                                images=rgb_frames[i],
                                episode_id=current_episodes_info[i].episode_id,
                                checkpoint_idx=checkpoint_index,
                                metrics=extract_scalars_from_info(infos[i]),
                                fps=self.config.habitat_baselines.video_fps,
                                tb_writer=writer,
                                keys_to_include_in_name=self.config.habitat_baselines.eval_keys_to_include_in_name,
                            )
                        # try:
                        gfx_str = infos[i].get(GfxReplayMeasure.cls_uuid, "")
                        if gfx_str != "":
                            write_gfx_replay(
                                gfx_str,
                                self.config.habitat.task,
                                current_episodes_info[i].episode_id,
                            )
                        # except Exception as e:
                        #     print(e)
                        #     continue
                        

                        
                        cnt_episode += 1
                        print(f"Episode current_episodes_info[0].episode_id ends: {cnt_episode} episodes out of {number_of_eval_episodes} done.")
                        check_first_observation = True
                        cnt_step = 0
                    gc.collect()
                    del action_data, observations, infos
                    torch.cuda.empty_cache()
            # except Exception as e:
            #     cnt_episode += 1
            #     print(f"Episode current_episodes_info[0].episode_id ends: {cnt_episode} episodes out of {number_of_eval_episodes} done.")
            #     check_first_observation = True
            #     cnt_step = 0
            #     continue
            except Exception as e:
                import traceback
                print(f"[ERROR] Exception occurred during episode {current_episodes_info[0].episode_id}: {e}")
                traceback.print_exc()
                cnt_episode += 1
                check_first_observation = True
                cnt_step = 0
                continue

            not_done_masks = not_done_masks.to(device=self.device)
            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        pbar.close()

        assert (
            len(ep_eval_count) >= expected_eval_episodes
        ), f"Expected {expected_eval_episodes} episodes, got {len(ep_eval_count)}."

        aggregated_stats = {}
        if not stats_episodes:
            logger.error("No episode stats collected; skipping aggregation.")
            aggregated_stats["reward"] = float("nan")
        else:
            for stat_key in next(iter(stats_episodes.values())).keys():
                values = [v[stat_key] for v in stats_episodes.values() if stat_key in v]
                aggregated_stats[stat_key] = np.mean(values) if values else float("nan")

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalar("eval_reward/average_reward", aggregated_stats.get("reward", float("nan")), step_id)

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)

        self.envs.close()

        if os.environ.get("WANDB_MODE", "") != "disabled":
            wandb.log({
                "avg_success": aggregated_stats.get("success", 0.0),
                "avg_spl": aggregated_stats.get("spl", 0.0),
                "avg_agent_episode_distance": aggregated_stats.get("agent_episode_distance", 0.0),
            })

            wandb.finish()
