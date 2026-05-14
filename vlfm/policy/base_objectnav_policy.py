# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
import json
import math
import re
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
import time
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from torch import Tensor

from vlfm.mapping.object_point_cloud_map import ObjectPointCloudMap
from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.obs_transformers.utils import image_resize
from vlfm.policy.utils.pointnav_policy import WrappedPointNavResNetPolicy
from vlfm.utils.geometry_utils import get_fov, rho_theta
from vlfm.vlm.blip2 import BLIP2Client
from vlfm.vlm.coco_classes import COCO_CLASSES
from vlfm.vlm.grounding_dino import GroundingDINOClient, ObjectDetections
from vlfm.vlm.sam import MobileSAMClient
from vlfm.vlm.yolov7 import YOLOv7Client
import vlfm.vlm.llava_next as LLaVA
from vlfm.vlm.openai_llm import OpenAILLMClient
from vlfm.oracle.oracle import VLMOracle
from vlfm.brain.vlm_brain_history import VLM_History
from vlfm.brain.llm_brain_history import LLM_History

try:
    from habitat_baselines.common.tensor_dict import TensorDict

    from vlfm.policy.base_policy import BasePolicy
except Exception:

    class BasePolicy:  # type: ignore
        pass


from colorama import Fore
from colorama import init as init_colorama
from pathlib import Path

init_colorama(autoreset=True)

from vlfm.utils.prompts import LLaVa_TARGET_OBJECT_IS_DETECTED, LLaVa_TARGET_OBJECT_IS_DETECTED_NEARBY
from vlfm.utils.coin_bench_episode_resolver import CoinBenchEpisodeMeta, CoinBenchEpisodeResolver
from vlfm.utils.rotate import RotateController

from vlfm.policy.process_multi_view import (
    merge_instances_into_groups,
    select_diverse_view_for_each_group,
    generate_caption_for_each_group,
    merge_instances_into_groups_caption,
)


class BaseObjectNavPolicy(BasePolicy):
    _target_object: str = ""
    _policy_info: Dict[str, Any] = {}
    _object_masks: Union[np.ndarray, Any] = None  # set by ._update_object_map()
    _stop_action: Union[Tensor, Any] = None  # MUST BE SET BY SUBCLASS
    _observations_cache: Dict[str, Any] = {}
    _non_coco_caption = ""
    _load_yolo: bool = False

    def __init__(
        self,
        pointnav_policy_path: str,
        depth_image_shape: Tuple[int, int],
        pointnav_stop_radius: float,
        object_map_erosion_size: float,
        visualize: bool = True,
        compute_frontiers: bool = True,
        explore_only: bool = False,
        goal_point_method: str = "goal_point_multiple",
        min_obstacle_height: float = 0.15,
        max_obstacle_height: float = 0.88,
        agent_radius: float = 0.18,
        obstacle_map_area_threshold: float = 1.5,
        hole_area_thresh: int = 100000,
        use_vqa: bool = False,
        vqa_prompt: str = "Is this ",
        coco_threshold: float = 0.8,
        non_coco_threshold: float = 0.4,
        rotate_enabled: Optional[bool] = None,
        enable_rotate: Optional[bool] = None,
        rotate_radius: float = 1.0,
        robot_rotate_openness_threshold: float = 0.1,
        num_angle_bin: int = 360,
        save_panoramas: bool = False,
        panorama_output_dir: str = "panoramas",
        enable_multi_view: bool = True,
        enable_multi_view_optimization: bool = False,
        coin_bench_data_path: Optional[str] = None,
        coin_bench_content_dir: Optional[str] = None,
        coin_bench_scene_dataset_config: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._task_type: str = str(kwargs.pop("task_type", os.environ.get("VLFM_TASK_TYPE", "coin")))
        if self._task_type not in ("text_goal", "coin", "image_goal", "object_goal"):
            raise ValueError(
                f"Unsupported task_type='{self._task_type}'. Expected one of "
                "['text_goal', 'coin', 'image_goal', 'object_goal']."
            )
        if self._task_type in ("image_goal", "object_goal"):
            raise NotImplementedError(f"task_type='{self._task_type}' is not connected yet.")
        self._object_detector = GroundingDINOClient(port=int(os.environ.get("GROUNDING_DINO_PORT")))
        # self._coco_object_detector = YOLOv7Client(port=int(os.environ.get("YOLOV7_PORT")))
        self._mobile_sam = MobileSAMClient(port=int(os.environ.get("SAM_PORT")))
        self._use_vqa = use_vqa and (not explore_only)
        # self._use_vqa = False

        ##### LLM and VLM
        vlm_connector = LLaVA.LLavaNextClient(port=int(os.environ.get("LLava_PORT", "8000")))
        self.vlm_connector = vlm_connector

        test_with_groq = False
        llm_client_params = {
            "model": "",
            "base_url": "http://localhost:8000/v1",  # vLLM endpoint
        }
        LLM_CONNECTOR = OpenAILLMClient(llm_client_params)

        self.VLM_ORACLE = VLMOracle(vlm_connector, LLM_CONNECTOR)  # only accessible to the llm oracle

        self.vlm_agent_brain = VLM_History(vlm_connector)
        self.llm_agent_brain = LLM_History(LLM_CONNECTOR)
        ###### End LLM and VLM
        print(Fore.CYAN + "=" * 60)
        print(Fore.CYAN + "[MODEL INFO] Initialized Models:")
        print(Fore.GREEN + f"  - VLM (MLLM): {vlm_connector.model_name} (port {os.environ.get('LLava_PORT')})")
        print(Fore.GREEN + f"  - LLM: {LLM_CONNECTOR.model_name}")
        print(Fore.GREEN + f"  - Object Detector: GroundingDINO (port {os.environ.get('GROUNDING_DINO_PORT')})")
        # print(Fore.GREEN + f"  - COCO Detector: YOLOv7 (port {os.environ.get('YOLOV7_PORT', '12184')})")
        print(Fore.GREEN + f"  - Segmentation: MobileSAM (port {os.environ.get('SAM_PORT')})")
        print(Fore.CYAN + "=" * 60)

        self._pointnav_policy = WrappedPointNavResNetPolicy(pointnav_policy_path)

        # Extract PBP config from kwargs if present
        pbp_config = kwargs.get('pbp_config', None)
        
        self._object_map: ObjectPointCloudMap = ObjectPointCloudMap(
            erosion_size=object_map_erosion_size,
            vlm_agent_brain=self.vlm_agent_brain,
            llm_agent_brain=self.llm_agent_brain,
            vlm_oracle=self.VLM_ORACLE,
            pbp_config=pbp_config,
            enable_multi_view=enable_multi_view,
            enable_multi_view_optimization=enable_multi_view_optimization,
        )
        # Configure exploration-only behavior for the object map.
        setattr(self._object_map, "explore_only", explore_only)
        self._enable_multi_view = bool(enable_multi_view)
        self._enable_multi_view_optimization = bool(enable_multi_view_optimization)
        if (not self._enable_multi_view) and self._enable_multi_view_optimization:
            print(Fore.MAGENTA + "[MULTI_VIEW] WARNING: enable_multi_view=False overrides enable_multi_view_optimization=True.")
        self._depth_image_shape = tuple(depth_image_shape)
        self._pointnav_stop_radius = pointnav_stop_radius
        self._visualize = visualize
        self._vqa_prompt = vqa_prompt
        self._coco_threshold = coco_threshold
        self._non_coco_threshold = non_coco_threshold
        self._num_steps = 0
        self._did_reset = False
        self._last_goal = np.zeros(2)
        self._done_initializing = False
        self._called_stop = False
        self._stop_rho_min = None
        self._stop_rho_stall_steps = 0
        self._false_positive_counter = 0
        self._target_image_saved = False
        self._compute_frontiers = compute_frontiers
        if compute_frontiers:
            self._obstacle_map = ObstacleMap(
                min_height=min_obstacle_height,
                max_height=max_obstacle_height,
                area_thresh=obstacle_map_area_threshold,
                agent_radius=agent_radius,
                hole_area_thresh=hole_area_thresh,
                size=1500,
                pixels_per_meter=30,
            )
        print(Fore.YELLOW + "[INFO]: Non COCO Obj detector thresh: " + str(non_coco_threshold))
        self.folder_for_backup = None
        self.ep_id = None
        rotate_flag = enable_rotate if enable_rotate is not None else rotate_enabled
        if rotate_flag is None:
            rotate_flag = False
        self._rotate_controller = RotateController(
            enabled=bool(rotate_flag),
            rotate_radius=rotate_radius,
            rotate_openness_threshold=robot_rotate_openness_threshold,
            num_angle_bin=num_angle_bin,
            save_panoramas=save_panoramas,
            panorama_output_dir=panorama_output_dir,
        )

        self._coin_resolver: Optional[CoinBenchEpisodeResolver] = (
            CoinBenchEpisodeResolver(
                data_path=coin_bench_data_path,
                content_dir=coin_bench_content_dir,
                scene_dataset_config=coin_bench_scene_dataset_config,
            )
            if (self._task_type == "coin" and (coin_bench_data_path or coin_bench_content_dir))
            else None
        )
        self._coin_meta: Optional[CoinBenchEpisodeMeta] = None
        self._coin_target_image: Optional[np.ndarray] = None
        self._goal_point_method = goal_point_method
        self._text_goal: str = ""
        self._invalid_text_goal_episode: bool = False

        self._suffexp_last_robot_xy: Optional[np.ndarray] = None
        self._suffexp_last_action_id: Optional[int] = None
        self._suffexp_last_step: Optional[int] = None
        self._suffexp_num_frontiers_max = 0
        self._suffexp_cnt_revisit = 0
        self._suffexp_r_score_min = 0.0
        self._suffexp_r_score_max = 0.0
        self._suffexp_cell_visits: Dict[Tuple[int, int], int] = {}
        self._suffexp_scaled_r_score = 0.0
        self._suffexp_soft_TSS = 0.0
        self._suffexp_hard_TSS = 0.0
        self._suffexp_instance_count = 0
        self._suffexp_consecutive_condition_steps = 0
        self._suffexp_hard_condition_met = False
        self._trigger_heavy_next_mature_count = 0
        self._trigger_last_failed_merge_key = ""
        self._trigger_last_failed_merged_candidate_count = 0

    def set_folder_for_data_backup(self, folder) -> None:
        self.folder_for_backup = folder
        self._object_map.logging_root = Path(folder)

    def set_ep_id(self, ep_id) -> None:
        self.ep_id = ep_id

    def set_text_goal(self, text_goal: str) -> None:
        self._text_goal = str(text_goal or "").strip()
        if self._task_type == "text_goal" and not self._text_goal:
            if not self._invalid_text_goal_episode:
                print(Fore.RED + f"[TEXT_GOAL] Invalid episode {self.ep_id}: empty text_goal(instruction). Episode will stop.")
            self._invalid_text_goal_episode = True

    def _reset(self) -> None:
        self._finalize_rotate_panorama()
        self._target_object = ""
        self._invalid_text_goal_episode = False
        self._pointnav_policy.reset()
        self._object_map.reset(self.ep_id, self._target_object.split("|")[0])
        self._last_goal = np.zeros(2)
        self._num_steps = 0
        self._done_initializing = False
        self._called_stop = False
        self._stop_rho_min = None
        self._stop_rho_stall_steps = 0
        if self._compute_frontiers:
            self._obstacle_map.reset()
        self.llm_agent_brain.reset()
        self.VLM_ORACLE.reset()
        self._false_positive_counter = 0
        self._target_image_saved = False
        self._did_reset = True
        self._rotate_controller.reset()
        self._coin_meta = None
        self._coin_target_image = None

        self._suffexp_last_robot_xy = None
        self._suffexp_last_action_id = None
        self._suffexp_last_step = None
        self._suffexp_num_frontiers_max = 0
        self._suffexp_cnt_revisit = 0
        self._suffexp_r_score_min = 0.0
        self._suffexp_r_score_max = 0.0
        self._suffexp_cell_visits = {}
        self._suffexp_scaled_r_score = 0.0
        self._suffexp_soft_TSS = 0.0
        self._suffexp_hard_TSS = 0.0
        self._suffexp_instance_count = 0
        self._suffexp_consecutive_condition_steps = 0
        self._suffexp_hard_condition_met = False
        self._trigger_heavy_next_mature_count = 0
        self._trigger_last_failed_merge_key = ""
        self._trigger_last_failed_merged_candidate_count = 0

    def _get_mature_instance_count(self, instance_confirming_additional_steps: int) -> int:
        instances = getattr(self._object_map, "instances", {})
        current_step = int(self._num_steps)
        mature_count = 0
        for inst in instances.values():
            first_seen_step = int(inst.get("first_seen_step", current_step))
            if current_step >= first_seen_step + int(instance_confirming_additional_steps):
                mature_count += 1
        return int(mature_count)

    def _update_sufficient_exploration(self) -> None:
        cfg = getattr(self._object_map, "pbp_config", None)
        if not cfg or not bool(cfg.get("enable_sufficient_exploration_trigger", False)):
            return
        cell_size = float(cfg.get("sufficient_exp_cell_size", 0.25))
        min_num_instances = int(cfg.get("min_num_instances_for_pbp_trigger", 2))
        hard_tss_threshold = float(cfg.get("hard_tss_threshold_for_pbp_trigger", 0.55))
        consecutive_steps_for_pbp_trigger = int(cfg.get("consecutive_steps_for_pbp_trigger", 10))
        instance_confirming_additional_steps = int(cfg.get("instance_confirming_additional_steps", 20))
        frontiers = self._observations_cache.get("frontier_sensor", np.array([]))
        num_frontiers = int(len(frontiers))
        if num_frontiers > self._suffexp_num_frontiers_max:
            self._suffexp_num_frontiers_max = num_frontiers
        num_frontiers_max = int(self._suffexp_num_frontiers_max)
        if num_frontiers_max <= 0:
            r_score = 0.0
        else:
            promisingness_score = float(num_frontiers) / float(num_frontiers_max)
            r_score = 1.0 - promisingness_score # r_score == (1 - promisingness_score)
        r_score = min(1.0, max(0.0, r_score))
        if not self._suffexp_cell_visits:
            self._suffexp_r_score_min = float(r_score)
            self._suffexp_r_score_max = float(r_score)
        else:
            if float(r_score) < float(self._suffexp_r_score_min):
                self._suffexp_r_score_min = float(r_score)
            if float(r_score) > float(self._suffexp_r_score_max):
                self._suffexp_r_score_max = float(r_score)
        scaled_r_score = (
            0.0
            if float(self._suffexp_r_score_max) <= float(self._suffexp_r_score_min)
            else float((float(r_score) - float(self._suffexp_r_score_min)) / (float(self._suffexp_r_score_max) - float(self._suffexp_r_score_min)))
        )
        scaled_r_score = min(1.0, max(0.0, scaled_r_score))
        self._suffexp_scaled_r_score = float(scaled_r_score)
        curr_xy = self._observations_cache["robot_xy"]
        curr_cell = tuple(np.floor(curr_xy[:2] / cell_size).astype(int).tolist())
        if self._suffexp_last_robot_xy is None and not self._suffexp_cell_visits:
            self._suffexp_cell_visits[curr_cell] = 1
        if self._suffexp_last_action_id == 1 and self._suffexp_last_robot_xy is not None:
            prev_cell = tuple(np.floor(self._suffexp_last_robot_xy[:2] / cell_size).astype(int).tolist())
            if curr_cell != prev_cell:
                visits = int(self._suffexp_cell_visits.get(curr_cell, 0) + 1)
                self._suffexp_cell_visits[curr_cell] = visits
                if visits == 2:
                    self._suffexp_cnt_revisit += 1
        step_count = 1 if self._suffexp_last_step is None else int(self._suffexp_last_step) + 1
        visited_cells = int(len(self._suffexp_cell_visits))
        cnt_visited = max(visited_cells, 1)

        # Equation code for soft TSS - self._suffexp_scaled_r_score will be 0~1, depending on the promisingness of frontier situation.
        sum_soft_revisit_score = float(self._suffexp_cnt_revisit) * float(self._suffexp_scaled_r_score) # revisit * scaled_r_score
        self._suffexp_soft_TSS = float(sum_soft_revisit_score / float(cnt_visited)) # TSS = sum_soft_revisit_score / visited_cells == Soft Gated Trajectory Saturation Score

        # Equation code for hard TSS - the weight of the revisit is always 1.0, regardless of the promisingness of frontier situation. Fits for methods one frontier at a time (e.g., a method that obtain frontiers considering the continuous space of the boundary between known area and unknown area).
        sum_hard_revisit_score = float(self._suffexp_cnt_revisit) * 1.0
        self._suffexp_hard_TSS = float(sum_hard_revisit_score / float(cnt_visited)) # TSS = sum_hard_revisit_score / visited_cells == Hard Gated Trajectory Saturation Score
        instance_count = self._get_mature_instance_count(instance_confirming_additional_steps)
        hard_condition_met = (
            int(instance_count) >= int(min_num_instances)
            and float(self._suffexp_hard_TSS) >= float(hard_tss_threshold)
        )
        if hard_condition_met:
            self._suffexp_consecutive_condition_steps = int(self._suffexp_consecutive_condition_steps) + 1
        else:
            self._suffexp_consecutive_condition_steps = 0
        self._suffexp_instance_count = int(instance_count)
        self._suffexp_hard_condition_met = bool(hard_condition_met)

        revisit_cells = [list(k) for k, v in self._suffexp_cell_visits.items() if v >= 2]
        cell_visits = [[int(k[0]), int(k[1]), int(v)] for k, v in self._suffexp_cell_visits.items()]
        self._policy_info.update(
            {
                "sufficient_exp_cell_size": float(cell_size),
                "sufficient_exp_step_count": int(step_count),
                "sufficient_exp_visited_cells": int(visited_cells),
                "sufficient_exp_num_frontiers": int(num_frontiers),
                "sufficient_exp_num_frontiers_max": int(self._suffexp_num_frontiers_max),
                "sufficient_exp_cnt_revisit": int(self._suffexp_cnt_revisit),
                "sufficient_exp_r_score_min": float(self._suffexp_r_score_min),
                "sufficient_exp_r_score_max": float(self._suffexp_r_score_max),
                "sufficient_exp_weight": float(self._suffexp_scaled_r_score),
                "sufficient_exp_soft_TSS": float(self._suffexp_soft_TSS),
                "sufficient_exp_hard_TSS": float(self._suffexp_hard_TSS),
                "sufficient_exp_cell_visits": cell_visits,
                "sufficient_exp_revisit_cells": revisit_cells,
                "num_frontiers_max": int(self._suffexp_num_frontiers_max),
                "num_frontiers": int(num_frontiers),
                "cnt_revisit": int(self._suffexp_cnt_revisit),
                "cell_visits": cell_visits,
                "r_score_min": float(self._suffexp_r_score_min),
                "r_score_max": float(self._suffexp_r_score_max),
                "scaled_r_score": float(self._suffexp_scaled_r_score),
                "soft_TSS": float(self._suffexp_soft_TSS),
                "hard_TSS": float(self._suffexp_hard_TSS),
                "instance_count": int(self._suffexp_instance_count),
                "consecutive_condition_steps": int(self._suffexp_consecutive_condition_steps),
                "hard_condition_met": bool(self._suffexp_hard_condition_met),
            }
        )
        # print(
        #     Fore.MAGENTA
        #     + f"[SUFF_EXP] step_count={step_count} visited_cells={visited_cells} num_frontiers={num_frontiers} num_frontiers_max={num_frontiers_max} "
        #     + f"cnt_revisit={self._suffexp_cnt_revisit} r_score={r_score:.4f} r_score_min={self._suffexp_r_score_min:.4f} r_score_max={self._suffexp_r_score_max:.4f} scaled_r_score={self._suffexp_scaled_r_score:.4f} "
        #     + f"soft_TSS={self._suffexp_soft_TSS:.4f} "
        #     + f"hard_TSS={self._suffexp_hard_TSS:.4f} instance_count={self._suffexp_instance_count} "
        #     + f"consecutive_condition_steps={self._suffexp_consecutive_condition_steps} hard_condition_met={self._suffexp_hard_condition_met}"
        # )

    def _sufficient_exploration_trigger_enabled(self) -> bool:
        cfg = getattr(self._object_map, "pbp_config", None)
        return bool(cfg and cfg.get("enable_sufficient_exploration_trigger", False))

    def _are_all_frontiers_blacklisted(self, frontiers: np.ndarray, suffexp_trigger_enabled: bool) -> bool:
        if not suffexp_trigger_enabled or len(frontiers) == 0 or np.array_equal(frontiers, np.zeros((1, 2))):
            return False
        blacklist_cells = getattr(self, "_loop_blacklist_cells", None)
        if not blacklist_cells:
            return False
        cell_size = float(self._object_map.pbp_config.get("sufficient_exp_cell_size", 0.25))
        frontier_cells = [
            tuple(np.floor(np.asarray(frontier[:2], dtype=np.float32) / cell_size).astype(int).tolist())
            for frontier in frontiers
        ]
        return len(frontier_cells) > 0 and all(cell in blacklist_cells for cell in frontier_cells)

    def _get_pbp_trigger_context(self, suffexp_trigger_enabled: bool) -> Tuple[int, int, int, int, int, bool]:
        trigger_step = int(self._object_map.pbp_config.get("trigger_step", 400))
        min_num_instances = int(self._object_map.pbp_config.get("min_num_instances_for_pbp_trigger", 2))
        instance_confirming_additional_steps = int(self._object_map.pbp_config.get("instance_confirming_additional_steps", 20))
        mature_instance_count = int(self._get_mature_instance_count(instance_confirming_additional_steps))
        frontiers_now = self._observations_cache.get("frontier_sensor", np.array([]))
        num_frontiers_now = int(len(frontiers_now))
        all_frontiers_blacklisted = self._are_all_frontiers_blacklisted(frontiers_now, suffexp_trigger_enabled)
        no_frontiers_available = (num_frontiers_now == 0) or all_frontiers_blacklisted
        return (
            trigger_step,
            min_num_instances,
            instance_confirming_additional_steps,
            mature_instance_count,
            num_frontiers_now,
            no_frontiers_available,
        )

    def _apply_task_pbp_trigger_conditions(
        self,
        should_run_pbp: bool,
        raw_mature_instance_count: int,
        min_num_instances: int,
        instance_confirming_additional_steps: int,
    ) -> bool:
        mature_count = int(raw_mature_instance_count)
        required_raw_mature_instance_count = max(
            int(min_num_instances),
            int(self._trigger_heavy_next_mature_count),
        )
        return bool(should_run_pbp or (mature_count >= required_raw_mature_instance_count))

    def _build_merge_key(self, refinement_result: Dict) -> str:
        member_groups = []
        for group_info in refinement_result.values():
            if not isinstance(group_info, dict):
                continue
            members = group_info.get("members", [])
            if not isinstance(members, list):
                continue
            member_groups.append(",".join(sorted(str(m) for m in members)))
        return "|".join(sorted(member_groups))

    def _task_pbp_merge_gate_after_refinement(self, refinement_result, min_num_instances: int, mature_instance_count: int):
        if self._task_type not in ("text_goal", "coin") or refinement_result is None:
            return refinement_result
        if not isinstance(refinement_result, dict):
            return refinement_result
        trigger_step = int(getattr(self._object_map, "pbp_config", {}).get("trigger_step", 400))
        if int(self._num_steps) >= trigger_step:
            return refinement_result

        self._trigger_heavy_next_mature_count = max(
            int(self._trigger_heavy_next_mature_count),
            int(mature_instance_count) + 1,
        )
        current_merge_key = self._build_merge_key(refinement_result)
        if bool(self._trigger_last_failed_merge_key) and current_merge_key == str(self._trigger_last_failed_merge_key):
            task_tag = self._task_type
            print(
                Fore.YELLOW
                + f"[PBP][{task_tag}] pbp-gate blocked: same merged layout as previous failed run "
                + f"(merged_candidate_count={len(refinement_result)})."
            )
            return None
        merged_candidate_count = int(len(refinement_result))
        if merged_candidate_count < int(min_num_instances):
            self._trigger_last_failed_merge_key = current_merge_key
            self._trigger_last_failed_merged_candidate_count = merged_candidate_count
            task_tag = self._task_type
            print(
                Fore.YELLOW
                + f"[PBP][{task_tag}] heavy-stage failed: merged_candidate_count={merged_candidate_count} "
                + f"< min_num_instances={min_num_instances}. "
                + f"next_required_mature_count={self._trigger_heavy_next_mature_count}"
            )
            return None
        return refinement_result

    def _run_multi_view_refinement(self, episode_root: str, episode_goal_category: str) -> Optional[bool]:
        result = True
        if not (self._enable_multi_view and self._enable_multi_view_optimization):
            return result
        merge_start = time.perf_counter()
        result = merge_instances_into_groups(
            episode_root,
            episode_goal_category=episode_goal_category,
            eps=1,
            min_samples=1,
        )
        merge_elapsed = float(time.perf_counter() - merge_start)
        self._object_map.add_timing("merge_instances_into_groups", merge_elapsed)
        print(
            Fore.MAGENTA
            + f"[MULTI_VIEW] merge_instances_into_groups: {merge_elapsed:.6f}s "
            + f"(goal={episode_goal_category})"
        )
        if result is None:  # instances.jsonl is not ready yet.
            return result
        diverse_view_start = time.perf_counter()
        select_diverse_view_for_each_group(
            episode_root,
            episode_goal_category=episode_goal_category,
            image_embedding_model="facebook/dinov2-large",
            cuda_device_id=0 if os.environ.get("CUDA_VISIBLE_DEVICES") else int(os.environ.get("CUDA_DEVICE", 0)),
            desired_num_clusters=6,  # TODO: hyperparameter
            default_num_views=6,
            vlm_connector=self.vlm_connector,
            image_only_model=getattr(self, "image_only_model", None),
            image_only_processor=getattr(self, "image_only_processor", None),
        )
        diverse_view_elapsed = float(time.perf_counter() - diverse_view_start)
        self._object_map.add_timing("select_diverse_view_for_each_group", diverse_view_elapsed)
        print(
            Fore.MAGENTA
            + f"[MULTI_VIEW] select_diverse_view_for_each_group: {diverse_view_elapsed:.6f}s "
            + f"(goal={episode_goal_category})"
        )
        caption_start = time.perf_counter()
        generate_caption_for_each_group(
            self.vlm_connector,
            episode_root,
            episode_goal_category=episode_goal_category,
        )
        caption_elapsed = float(time.perf_counter() - caption_start)
        self._object_map.add_timing("generate_caption_for_each_group", caption_elapsed)
        print(
            Fore.MAGENTA
            + f"[MULTI_VIEW] generate_caption_for_each_group: {caption_elapsed:.6f}s "
            + f"(goal={episode_goal_category})"
        )
        caption_merge_start = time.perf_counter()
        result = merge_instances_into_groups_caption(
            episode_root,
            episode_goal_category=episode_goal_category,
            llm_connector=self.llm_agent_brain.LLM_CLIENT,
            text_only_model=self.text_only_model,
            similarity_threshold=0.9,
        )
        caption_merge_elapsed = float(time.perf_counter() - caption_merge_start)
        self._object_map.add_timing("merge_instances_into_groups_caption", caption_merge_elapsed)
        print(
            Fore.MAGENTA
            + f"[MULTI_VIEW] merge_instances_into_groups_caption: {caption_merge_elapsed:.6f}s "
            + f"(goal={episode_goal_category})"
        )
        return result

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
        current_step: int = 0,
    ) -> Any:
        """
        Starts the episode by 'initializing' and allowing robot to get its bearings
        (e.g., rotating in place to get a good view of the scene).
        Then, explores the scene until it finds the target object.
        Once the target object is found, it navigates to the object.
        """
        self._prev_actions = prev_actions
        self._pre_step(observations, masks)
        if self._invalid_text_goal_episode:
            self._called_stop = True
            action_numpy = self._stop_action.detach().cpu().numpy()[0]
            if len(action_numpy) == 1:
                action_numpy = action_numpy[0]
            print(Fore.RED + f"[TEXT_GOAL] Episode: {self.ep_id} | Step: {self._num_steps} | Mode: stop_invalid_text_goal | Action: {action_numpy}")
            self._num_steps += 1
            self._observations_cache = {}
            self._did_reset = False
            return self._stop_action, rnn_hidden_states
        suffexp_trigger_enabled = self._sufficient_exploration_trigger_enabled()
        pbp_cfg = getattr(self._object_map, "pbp_config", {}) if hasattr(self, "_object_map") else {}
        self._policy_info["enable_sufficient_exploration_trigger"] = suffexp_trigger_enabled
        self._policy_info["sufficient_exp_cell_size"] = float(pbp_cfg.get("sufficient_exp_cell_size", 0.25))
        if suffexp_trigger_enabled:
            self._update_sufficient_exploration()

        skip_instance_image = getattr(self._object_map, "explore_only", False)
        if self._num_steps == 0:
            obs_instance_image = observations["instance_imagegoal"].cpu().squeeze().numpy()
            height, width = int(obs_instance_image.shape[0]), int(obs_instance_image.shape[1])
            instance_image = obs_instance_image
            if self._coin_resolver is not None and self.ep_id is not None:
                meta = self._coin_meta or self._coin_resolver.get_episode_meta(int(self.ep_id))
                self._coin_meta = meta
                instance_image = self._coin_resolver.render_target_image(
                    meta,
                    resolution=(height, width),
                )
                self._coin_target_image = instance_image

                print(
                    Fore.LIGHTCYAN_EX
                    + "[EP_META] "
                    + f"ep={meta.episode_id} scene={meta.scene_id} "
                    + f"category={meta.object_category} instance={meta.object_instance_id} "
                    + f"target_pos={meta.target_position} "
                    + f"distractors={meta.distractor_positions} "
                    + f"camera_pos={meta.camera_spec.get('position')} "
                    + f"camera_rot={meta.camera_spec.get('rotation')} "
                    + f"hfov={meta.camera_spec.get('hfov')}"
                )
            if not skip_instance_image:
                print(Fore.LIGHTCYAN_EX + "[INFO] Setting the Instance Image - accessible to the VLM-Simulated user (oracle)")
                self.VLM_ORACLE.set_instance_image(
                    instance_image=instance_image,
                    target_object=self._target_object,
                    task_type=self._task_type,
                    text_goal=self._text_goal,
                    ep_id=self.ep_id,
                )
            else:
                print(Fore.CYAN + "[INFO] Exploration-only mode: skipping instance image setup.")
            self._save_target_object_image(instance_image)

        del observations["instance_imagegoal"]

        object_map_rgbd = self._observations_cache["object_map_rgbd"]

        # thus we don't want to run the LLm brain if we detect an object candidate and we reasoned about it
        should_detected_while_exploring = None
        if self._num_steps > 10:
            robot_xy = self._observations_cache["robot_xy"]

        detections = [
            self._update_object_map(
                rgb,
                depth,
                tf,
                min_depth,
                max_depth,
                fx,
                fy,
                should_detected_while_exploring=should_detected_while_exploring,
                current_step=current_step,
            )
            for (rgb, depth, tf, min_depth, max_depth, fx, fy) in object_map_rgbd
        ]
        robot_xy = self._observations_cache["robot_xy"]
        
        # PBP trigger: Run property-based binary partitioning at configured step
        if (
            hasattr(self._object_map, "pbp_config")
            and self._object_map.pbp_config
            and not getattr(self._object_map, "explore_only", False)
        ):
            (
                trigger_step,
                min_num_instances,
                instance_confirming_additional_steps,
                mature_instance_count,
                num_frontiers_now,
                no_frontiers_available,
            ) = self._get_pbp_trigger_context(suffexp_trigger_enabled)
            # If instances.jsonl isn’t ready, skip instead of raising an error, and set the trigger condition to >= trigger_step so it retries at the next step.
            should_run_pbp = self._num_steps >= trigger_step
            if suffexp_trigger_enabled:
                should_run_pbp = (
                    should_run_pbp  # Trigger condition: number of steps
                    or no_frontiers_available  # Trigger condition: no frontiers (zero or all blacklisted)
                )
            raw_instance_count = int(len(getattr(self._object_map, "instances", {})))
            hard_deadline_instance_bypass = False
            if self._task_type == "text_goal":
                hard_deadline_instance_bypass = bool(
                    int(self._num_steps) >= int(trigger_step)
                    and int(raw_instance_count) < int(min_num_instances)
                )
                if hard_deadline_instance_bypass:
                    print(
                        Fore.YELLOW
                        + f"[PBP][text_goal] hard-deadline bypass enabled "
                        + f"(step={self._num_steps}, raw_instance_count={raw_instance_count}, min_num_instances={min_num_instances})."
                    )
                else:
                    should_run_pbp = self._apply_task_pbp_trigger_conditions(
                        should_run_pbp=should_run_pbp,
                        raw_mature_instance_count=mature_instance_count,
                        min_num_instances=min_num_instances,
                        instance_confirming_additional_steps=instance_confirming_additional_steps,
                    )
            elif self._task_type == "coin":
                hard_deadline_instance_bypass = bool(
                    int(self._num_steps) >= int(trigger_step)
                    and int(raw_instance_count) < int(min_num_instances)
                )
                if hard_deadline_instance_bypass:
                    print(
                        Fore.YELLOW
                        + f"[PBP][coin] hard-deadline bypass enabled "
                        + f"(step={self._num_steps}, raw_instance_count={raw_instance_count}, min_num_instances={min_num_instances})."
                    )
                else:
                    should_run_pbp = self._apply_task_pbp_trigger_conditions(
                        should_run_pbp=should_run_pbp,
                        raw_mature_instance_count=mature_instance_count,
                        min_num_instances=min_num_instances,
                        instance_confirming_additional_steps=instance_confirming_additional_steps,
                    )
            trigger_force_run = bool(int(self._num_steps) >= int(trigger_step))
            first_trigger_forced_run = bool(trigger_force_run and int(self._object_map.get_pbp_count()) == 0)
            if should_run_pbp and not self._object_map._pbp_target_found:
                if (
                    (not first_trigger_forced_run)
                    and bool(self._trigger_last_failed_merge_key)
                    and int(mature_instance_count) < int(self._trigger_heavy_next_mature_count)
                ):
                    print(
                        Fore.YELLOW
                        + f"[PBP][{self._task_type}] pre-gate blocked before multi-view: "
                        + f"mature_count={mature_instance_count} < next_required_mature_count={self._trigger_heavy_next_mature_count}."
                    )
                    result = None
                else:
                    episode_root = str(self._object_map.current_episode_dir)
                    episode_goal_category = self._target_object.split("|")[0]
                    result = self._run_multi_view_refinement(
                        episode_root=episode_root,
                        episode_goal_category=episode_goal_category,
                    )
                if (
                    trigger_force_run
                    and (not first_trigger_forced_run)
                    and isinstance(result, dict)
                    and bool(self._trigger_last_failed_merge_key)
                    and int(self._trigger_last_failed_merged_candidate_count) > 0
                ):
                    merged_candidate_count = int(len(result))
                    if merged_candidate_count <= int(self._trigger_last_failed_merged_candidate_count):
                        self._trigger_heavy_next_mature_count = max(
                            int(self._trigger_heavy_next_mature_count),
                            int(mature_instance_count) + 1,
                        )
                        self._trigger_last_failed_merge_key = self._build_merge_key(result)
                        self._trigger_last_failed_merged_candidate_count = merged_candidate_count
                        print(
                            Fore.YELLOW
                            + f"[PBP][{self._task_type}] trigger-stage blocked: merged_candidate_count={merged_candidate_count} "
                            + f"did not increase from last_failed_merged_candidate_count={self._trigger_last_failed_merged_candidate_count}. "
                            + f"next_required_mature_count={self._trigger_heavy_next_mature_count}"
                        )
                        result = None
                if (not hard_deadline_instance_bypass) and (not trigger_force_run):
                    if self._task_type == "text_goal":
                        result = self._task_pbp_merge_gate_after_refinement(
                            refinement_result=result,
                            min_num_instances=min_num_instances,
                            mature_instance_count=mature_instance_count,
                        )
                    elif self._task_type == "coin":
                        result = self._task_pbp_merge_gate_after_refinement(
                            refinement_result=result,
                            min_num_instances=min_num_instances,
                            mature_instance_count=mature_instance_count,
                        )

                if result is not None or first_trigger_forced_run: # Ensure one forced PBP attempt at/after trigger_step.
                    # then, let's run PBP

                    print(Fore.CYAN + f"[PBP] Triggering PBP at step {self._num_steps}")
                    target_image = self.VLM_ORACLE.get_instance_image()
                    category = self._target_object.split("|")[0]
                    self._object_map.text_only_model = getattr(self, "text_only_model", None)
                    self._object_map.nli_only_model = getattr(self, "nli_only_model", None)
                    self._object_map.nli_only_tokenizer = getattr(self, "nli_only_tokenizer", None)
                    self._object_map.image_only_model = getattr(self, "image_only_model", None)
                    self._object_map.image_only_processor = getattr(self, "image_only_processor", None)

                    pbp_start = time.perf_counter()
                    selected_obj_id, pbp_status = self._object_map.run_property_based_partitioning(
                        target_image=target_image,
                        category=category,
                        target_object_full=self._target_object,
                        ep_id=self.ep_id,
                        current_step=self._num_steps,
                        num_frontiers=num_frontiers_now,
                        task_type=self._task_type,
                        text_goal=self._text_goal,
                    )
                    pbp_success = pbp_status in {"PBP_TARGET_FOUND", "PBP_FALLBACK_SINGLE_CANDIDATE"}
                    if pbp_success:
                        self._trigger_last_failed_merge_key = ""
                        self._trigger_last_failed_merged_candidate_count = 0
                        self._trigger_heavy_next_mature_count = 0
                    if not pbp_success:
                        current_merge_key = self._build_merge_key(result) if isinstance(result, dict) else ""
                        self._trigger_last_failed_merge_key = current_merge_key
                        if isinstance(result, dict):
                            self._trigger_last_failed_merged_candidate_count = int(len(result))
                        self._trigger_heavy_next_mature_count = max(
                            int(self._trigger_heavy_next_mature_count),
                            int(mature_instance_count) + 1,
                        )
                        print(
                            Fore.YELLOW
                            + f"[PBP][{self._task_type}] failed with status={pbp_status}. "
                            + "blocked until merged layout changes."
                        )
                    pbp_elapsed = float(time.perf_counter() - pbp_start)
                    self._object_map.add_timing("pbp", pbp_elapsed)
                    print(
                        Fore.MAGENTA
                        + f"PBP: {pbp_elapsed:.6f}s "
                        + f"(avg={self._object_map.get_pbp_avg_time_sec():.6f}s, n={self._object_map.get_pbp_count()})"
                    )

                    if pbp_status == "PBP_NO_CANDIDATES_INCREASE_TRIGGER_THRESHOLD":
                        pass
                    elif pbp_status == "PBP_WAIT_FOR_NEW_INSTANCE":
                        pass
                    elif selected_obj_id is None: # System problem
                        # No object selected - trigger stop gracefully
                        print(Fore.RED + "[PBP] No object selected. Stopping episode.")
                        self._called_stop = True
                        action_numpy = self._stop_action.detach().cpu().numpy()[0]
                        if len(action_numpy) == 1:
                            action_numpy = action_numpy[0]
                        print(f"Step: {self._num_steps} | Mode: navigate | Action: {action_numpy}")
                        if detections and detections[0] is not None:
                            self._policy_info.update(self._get_policy_info(detections[0]))
                        self._suffexp_last_step = int(self._num_steps)
                        self._suffexp_last_action_id = int(self._stop_action[0].item())
                        self._suffexp_last_robot_xy = robot_xy.copy()
                        self._num_steps += 1
                        self._observations_cache = {}
                        self._did_reset = False
                        return self._stop_action, rnn_hidden_states
        
        if self._goal_point_method == "goal_point_single":
            goal = getattr(self._object_map, "pbp_goal_xy", None)
        else:
            goal = None
        if goal is None:
            goal = self._get_target_object_location(robot_xy)

        # If exploration-only mode is enabled on the object map,
        # keep exploring even when a goal exists.
        if getattr(self._object_map, "explore_only", False):
            print(Fore.CYAN + "[INFO] Exploration-only mode: continuing to explore.")
            if not self._done_initializing:  # Initialize
                mode = "initialize"
                self._record_rotate_frame("initialize")
                pointnav_action = self._initialize()
            else:
                mode = "explore"
                self._finalize_rotate_panorama("initialize")
                pointnav_action = self._explore(observations)
        else:
            if not self._done_initializing:  # Initialize
                mode = "initialize"
                self._record_rotate_frame("initialize")
                pointnav_action = self._initialize()
            elif goal is None:  # Haven't found target object yet
                mode = "explore"
                self._finalize_rotate_panorama("initialize")
                pointnav_action = self._explore(observations)
            else:
                mode = "navigate"
                # self._finalize_rotate_panorama("initialize")
                pointnav_action = self._pointnav(goal[:2], stop=True, is_navigate_mode=True)

        action_numpy = pointnav_action.detach().cpu().numpy()[0]
        if len(action_numpy) == 1:
            action_numpy = action_numpy[0]
        frontier_score_log = getattr(self, "_latest_frontier_score_log", None)
        if frontier_score_log:
            print(f"Episode: {self.ep_id} | Step: {self._num_steps} | {frontier_score_log}")
        print(f"Episode: {self.ep_id} | Step: {self._num_steps} | Mode: {mode} | Action: {action_numpy}")
        self._policy_info.update(self._get_policy_info(detections[0]))
        self._suffexp_last_step = int(self._num_steps)
        self._suffexp_last_action_id = int(pointnav_action[0].item())
        self._suffexp_last_robot_xy = robot_xy.copy()
        self._num_steps += 1

        self._observations_cache = {}
        self._did_reset = False

        return pointnav_action, rnn_hidden_states

    def how_many_question_to_the_user(self, ep_id):
        return self.VLM_ORACLE.how_many_question_to_the_user(ep_id)

    def _pre_step(self, observations: "TensorDict", masks: Tensor) -> None:
        assert masks.shape[1] == 1, "Currently only supporting one env at a time"
        if not self._did_reset and masks[0] == 0:
            self._reset()
            self._target_object = observations["objectgoal"]
            if self._coin_resolver is not None and self.ep_id is not None:
                self._coin_meta = self._coin_resolver.get_episode_meta(int(self.ep_id))
                self._target_object = self._coin_meta.object_category
        try:
            self._cache_observations(observations)
        except IndexError as e:
            print(e)
            print("Reached edge of map, stopping.")
            raise StopIteration
        self._record_initial_rotate_location()
        self._policy_info = {}

    def _initialize(self) -> Tensor:
        raise NotImplementedError

    def _explore(self, observations: "TensorDict") -> Tensor:
        raise NotImplementedError

    def _rotate(self) -> Tensor:
        """Default rotation behavior; subclasses/mixins should override."""
        raise NotImplementedError

    def _random_move(self) -> Tensor:
        """Default random-move behavior; subclasses/mixins should override."""
        raise NotImplementedError

    def _reset_rotation_mode(self) -> None:
        """Allows subclasses to reset any rotation state."""
        return None

    def _record_initial_rotate_location(self) -> None:
        self._rotate_controller.record_initial_location(self._observations_cache.get("robot_xy"))

    def _record_rotate_frame(self, mode: str) -> None:
        self._rotate_controller.record_panorama_frame(self._get_current_rgb_frame(), mode)

    def _finalize_rotate_panorama(self, mode: Optional[str] = None) -> None:
        output_dir = self._get_rotate_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        path = self._rotate_controller.finalize_panorama(mode, self._num_steps, output_dir)
        if path is not None:
            print(Fore.GREEN + f"[PANORAMA] Saved {mode} panorama to {path}")

    def _get_rotate_output_dir(self) -> str:
        if self.folder_for_backup is not None:
            episode = str(self.ep_id) if self.ep_id is not None else "episode"
            return os.path.join(self.folder_for_backup, episode, "panoramas")
        return self._rotate_controller.panorama_output_dir

    def _get_current_rgb_frame(self) -> Optional[np.ndarray]:
        frames = self._observations_cache.get("object_map_rgbd")
        if not frames:
            return None
        return frames[0][0]

    def _rotate_active(self) -> bool:
        return self._rotate_controller.active()

    def _rotate_action(self) -> Tensor:
        if not self._rotate_controller.in_progress:
            raise RuntimeError("Called _rotate_action() while rotate is not active.")
        print(Fore.BLUE + "[ROTATE] Continuing rotate mode.")
        action, path = self._rotate_controller.step_action(
            rotate_fn=self._rotate,
            frame=self._get_current_rgb_frame(),
            num_steps=self._num_steps,
            output_dir=self._get_rotate_output_dir(),
        )
        if path is not None:
            print(Fore.GREEN + f"[PANORAMA] Saved rotate panorama to {path}")
        return action

    def _try_start_rotate(self) -> bool:
        robot_xy = self._observations_cache.get("robot_xy")
        started, reason, openness_score, backend = self._rotate_controller.try_start(
            robot_xy=robot_xy,
            obstacle_map=getattr(self, "_obstacle_map", None),
            rotation_turns=int(getattr(self, "_rotation_turns", 12)),
        )
        if reason == "near_prev":
            print(Fore.YELLOW + "[ROTATE] Skipping: already rotated near this location.")
            return False
        if reason in ("low_openness", "started"):
            print(Fore.CYAN + f"[ROTATE] Openness backend: {backend}")
            print(
                Fore.CYAN
                + f"[ROTATE] openness_score={openness_score:.4f}, threshold={self._rotate_controller.rotate_openness_threshold:.4f}"
            )
        if reason == "low_openness":
            print(Fore.YELLOW + "[ROTATE] Skipping: openness below threshold.")
            return False
        if started:
            print(
                Fore.GREEN
                + f"[ROTATE] Entering rotate mode near location {robot_xy}, rotates_used={self._rotate_controller.rotate_used}"
            )
        return started

    def _get_target_object_location(self, position: np.ndarray) -> Union[None, np.ndarray]:
        target_key = self._target_object.split("|")[0]
        if self._object_map.has_object(target_key):
            return self._object_map.get_best_object(target_key, position)
        else:
            return None

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        target_key = self._target_object.split("|")[0]
        if self._object_map.has_object(target_key):
            target_point_cloud = self._object_map.get_target_cloud(target_key)
        else:
            target_point_cloud = np.array([])
        policy_info = {
            "target_object": self._target_object.split("|")[0],
            "gps": str(self._observations_cache["robot_xy"] * np.array([1, -1])),
            "yaw": np.rad2deg(self._observations_cache["robot_heading"]),
            "target_detected": self._object_map.has_object(target_key),
            "target_point_cloud": target_point_cloud,
            "nav_goal": self._last_goal,
            "stop_called": self._called_stop,
            # don't render these on egocentric images when making videos:
            "render_below_images": [
                "target_object",
            ],
        }
        if self._coin_meta is not None:
            policy_info["target_object_instance_id"] = self._coin_meta.object_instance_id
            policy_info["target_object_position"] = self._coin_meta.target_position
            policy_info["target_object_distractors"] = self._coin_meta.distractor_positions
            policy_info["target_object_camera_spec"] = self._coin_meta.camera_spec
        if self._coin_target_image is not None:
            policy_info["instance_imagegoal"] = self._coin_target_image

        if self._rotate_controller.rotate_locations:
            policy_info["rotate_locations"] = [loc.tolist() for loc in self._rotate_controller.rotate_locations]
            policy_info["rotate_radius_m"] = float(self._rotate_controller.rotate_radius)

        if not self._visualize:
            return policy_info

        annotated_depth = self._observations_cache["object_map_rgbd"][0][1] * 255
        annotated_depth = cv2.cvtColor(annotated_depth.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        if self._object_masks.sum() > 0:
            # If self._object_masks isn't all zero, get the object segmentations and
            # draw them on the rgb and depth images
            contours, _ = cv2.findContours(self._object_masks, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            annotated_rgb = cv2.drawContours(detections.annotated_frame, contours, -1, (255, 0, 0), 2)
            annotated_depth = cv2.drawContours(annotated_depth, contours, -1, (255, 0, 0), 2)
        else:
            annotated_rgb = self._observations_cache["object_map_rgbd"][0][0]
        policy_info["annotated_rgb"] = annotated_rgb
        policy_info["annotated_depth"] = annotated_depth

        if self._compute_frontiers:
            policy_info["obstacle_map"] = cv2.cvtColor(self._obstacle_map.visualize(), cv2.COLOR_BGR2RGB)

        if "DEBUG_INFO" in os.environ:
            policy_info["render_below_images"].append("debug")
            policy_info["debug"] = "debug: " + os.environ["DEBUG_INFO"]

        return policy_info

    def _get_object_detections(self, img: np.ndarray) -> ObjectDetections:
        target_classes = self._normalize_detector_classes(self._target_object)
        self._non_coco_caption = " . ".join(target_classes) + " ."
        has_coco = any(c in COCO_CLASSES for c in target_classes) and self._load_yolo
        has_non_coco = any(c not in COCO_CLASSES for c in target_classes)

        detections = (
            self._coco_object_detector.predict(img)
            if has_coco
            else self._object_detector.predict(img, caption=self._non_coco_caption)
        )
        # print(Fore.YELLOW + "target caption: " +  self._non_coco_caption)
        detections.filter_by_class(target_classes)
        det_conf_threshold = self._coco_threshold if has_coco else self._non_coco_threshold
        detections.filter_by_conf(det_conf_threshold)

        if has_coco and has_non_coco and detections.num_detections == 0:
            # Retry with non-coco object detector
            detections = self._object_detector.predict(img, caption=self._non_coco_caption)
            detections.filter_by_class(target_classes)
            detections.filter_by_conf(self._non_coco_threshold)

        return detections

    @staticmethod
    def _normalize_detector_label(label: str) -> str:
        """Normalize labels for detector prompts/classes by removing special chars."""
        normalized = re.sub(r"[^0-9a-zA-Z]+", " ", str(label)).strip().lower()
        return re.sub(r"\s+", " ", normalized)

    def _normalize_detector_classes(self, target_object: str) -> List[str]:
        raw_classes = [c.strip() for c in str(target_object).split("|")]
        normalized_classes = [self._normalize_detector_label(c) for c in raw_classes]
        normalized_classes = [c for c in normalized_classes if c]
        return normalized_classes if normalized_classes else ["object"]

    def _pointnav(self, goal: np.ndarray, stop: bool = False, is_navigate_mode: bool = True) -> Tensor:
        """
        Calculates rho and theta from the robot's current position to the goal using the
        gps and heading sensors within the observations and the given goal, then uses
        it to determine the next action to take using the pre-trained pointnav policy.

        Args:
            goal (np.ndarray): The goal to navigate to as (x, y), where x and y are in
                meters.
            stop (bool): Whether to stop if we are close enough to the goal.

        """
        if is_navigate_mode == True:
            print(Fore.MAGENTA + f"[PointNav] Navigating to goal at {goal}")
        masks = torch.tensor([self._num_steps != 0], dtype=torch.bool, device="cuda")
        if not np.array_equal(goal, self._last_goal):
            if np.linalg.norm(goal - self._last_goal) > 0.1:
                self._pointnav_policy.reset()
                masks = torch.zeros_like(masks)
            self._last_goal = goal
            self._stop_rho_min = None
            self._stop_rho_stall_steps = 0
        robot_xy = self._observations_cache["robot_xy"]
        heading = self._observations_cache["robot_heading"]
        rho, theta = rho_theta(robot_xy, heading, goal)
        rho_theta_tensor = torch.tensor([[rho, theta]], device="cuda", dtype=torch.float32)
        obs_pointnav = {
            "depth": image_resize(
                self._observations_cache["nav_depth"],
                (self._depth_image_shape[0], self._depth_image_shape[1]),
                channels_last=True,
                interpolation_mode="area",
            ),
            "pointgoal_with_gps_compass": rho_theta_tensor,
        }
        self._policy_info["rho_theta"] = np.array([rho, theta])
        print(Fore.BLUE + f"[PointNav] Rho: {rho:.2f} m, Theta: {np.rad2deg(theta):.2f} deg")
        if rho < self._pointnav_stop_radius and stop: # This is for STOP after navigation.
            if getattr(self._object_map, "explore_only", False):
                print(Fore.CYAN + "[INFO] Exploration-only mode: executing random move instead of STOP.")
                return self._random_move()
            print(Fore.RED + "[INFO] Within stop radius, calling STOP action. (This is for STOP after navigation.)")
            self._called_stop = True
            return self._stop_action
        if stop:
            if self._stop_rho_min is None or rho < self._stop_rho_min - 0.02:
                self._stop_rho_min = rho
                self._stop_rho_stall_steps = 0
            else:
                self._stop_rho_stall_steps += 1
                if self._stop_rho_stall_steps >= 20:
                    if getattr(self._object_map, "explore_only", False):
                        print(Fore.CYAN + "[INFO] Exploration-only mode: executing random move instead of STOP.")
                        return self._random_move()
                    print(Fore.RED + "[INFO] Rho not decreasing, probably an obstacle ahead. Calling STOP action.")
                    self._called_stop = True
                    return self._stop_action
        action = self._pointnav_policy.act(obs_pointnav, masks, deterministic=True)
        if action[0].item() == self._stop_action.item() and not is_navigate_mode:
            # if getattr(self._object_map, "explore_only", False):
            print(Fore.CYAN + "[INFO] Exploration-only mode: executing random move instead of STOP.")
            return self._random_move()
            # print(Fore.RED + "[INFO] PointNav policy called STOP action.")
            # self._called_stop = True
        return action

    def _update_object_map(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
        should_detected_while_exploring: None,
        current_step: int = 0,
    ) -> ObjectDetections:
        """
        Updates the object map with the given rgb and depth images, and the given
        transformation matrix from the camera to the episodic coordinate frame.

        Args:
            rgb (np.ndarray): The rgb image to use for updating the object map. Used for
                object detection and Mobile SAM segmentation to extract better object
                point clouds.
            depth (np.ndarray): The depth image to use for updating the object map. It
                is normalized to the range [0, 1] and has a shape of (height, width).
            tf_camera_to_episodic (np.ndarray): The transformation matrix from the
                camera to the episodic coordinate frame.
            min_depth (float): The minimum depth value (in meters) of the depth image.
            max_depth (float): The maximum depth value (in meters) of the depth image.
            fx (float): The focal length of the camera in the x direction.
            fy (float): The focal length of the camera in the y direction.
            should_detected_while_exploring: if not none, there is a candidate object in the scene, thus we should not detect anymore
        Returns:
            ObjectDetections: The object detections from the object detector.
        """
        if should_detected_while_exploring is not None:
            print("-------------------------------------------------------------------------------------------------")
            return

        # if self._object_map.has_object(self._target_object):
        #     print(Fore.GREEN + f"Object {self._target_object} already detected, navigating towards it.")
        #     return

        detections = self._get_object_detections(rgb)
        height, width = rgb.shape[:2]
        self._object_masks = np.zeros((height, width), dtype=np.uint8)
        if np.array_equal(depth, np.ones_like(depth)) and detections.num_detections > 0:
            depth = self._infer_depth(rgb, min_depth, max_depth)
            obs = list(self._observations_cache["object_map_rgbd"][0])
            obs[1] = depth
            self._observations_cache["object_map_rgbd"][0] = tuple(obs)
        for idx in range(len(detections.logits)):
            bbox_denorm = detections.boxes[idx] * np.array([width, height, width, height])
            object_mask = self._mobile_sam.segment_bbox(rgb, bbox_denorm.tolist())

            if self._use_vqa:
                ### we use our uncertainty estimation technique to filter out detection false positives
                ######
                contours, _ = cv2.findContours(object_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                annotated_rgb = cv2.drawContours(rgb.copy(), contours, -1, (255, 0, 0), 3)

                answer, logits = self.vlm_agent_brain.reduce_detector_false_positive(
                    detected_image=annotated_rgb, target_object=self._target_object.split("|")[0], get_logits=True
                )
                # uncertainty_est = self.llm_agent_brain.filter_self_questioner_answer_by_uncertainty(
                #     [dict(question="", answer=answer, logits_likelihood=logits)], tau=0.75, offset=0.05
                # )
                # uncertainty_est = uncertainty_est[0]["certainty_label"]

                if answer.lower().startswith("no"):
                    print(Fore.YELLOW + f"skipping detection as, probably it's a false positive")
                    continue
                # elif answer.lower().startswith("yes"):
                #     answer, logits = self.vlm_agent_brain.reduce_detector_obstructed_object(
                #         detected_image=annotated_rgb, target_object=self._target_object.split("|")[0], get_logits=True
                #     )
                #     if answer.lower().startswith("no"):
                #         print(Fore.YELLOW + f"skipping detection as, probably it's the candidate but obstructed")
                #         continue
                # if uncertainty_est == "uncertain":
                #     continue

                # i don't know
                # if "know" in answer.lower():
                #     continue
                # if "i" in answer.lower():
                #     continue
            
            #### The VLM return 'yes', and it is certain of it. Proceed.
            print(Fore.GREEN + f"Detected object: {detections.phrases[idx]}")
            contours, _ = cv2.findContours(object_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            annotated_rgb = cv2.drawContours(rgb.copy(), contours, -1, (255, 0, 0), 2)

            self._object_masks[object_mask > 0] = 1
            self._object_map.update_map(
                self._target_object.split("|")[0],
                depth,
                object_mask,
                tf_camera_to_episodic,
                min_depth,
                max_depth,
                fx,
                fy,
                rgb_image=rgb,
                target_object=self._target_object,
                total_num_steps=self._num_steps,
                ep_id=self.ep_id,
                current_step=self._num_steps,
                bbox=bbox_denorm,
                det_conf=float(detections.logits[idx].item()) if len(detections.logits) > idx else 0.0,
                robot_xy=self._observations_cache.get("robot_xy"),
                robot_heading=self._observations_cache.get("robot_heading"),
            )

        return detections

    def _cache_observations(self, observations: "TensorDict") -> None:
        """Extracts the rgb, depth, and camera transform from the observations.

        Args:
            observations ("TensorDict"): The observations from the current timestep.
        """
        raise NotImplementedError

    def _save_target_object_image(self, image: np.ndarray) -> None:
        if self._target_image_saved:
            return
        if getattr(self, "folder_for_backup", None) is None:
            return
        if getattr(self, "ep_id", None) is None:
            return
        target_name = self._target_object.split("|")[0]
        save_dir = os.path.join(self.folder_for_backup, str(self.ep_id), "self_questioner")
        os.makedirs(save_dir, exist_ok=True)
        filename = f"target_object_{target_name}.png"
        save_path = os.path.join(save_dir, filename)
        image_to_save = image
        if image_to_save.ndim == 3 and image_to_save.shape[2] == 3:
            try:
                image_to_save = cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR)
            except cv2.error:
                pass
        try:
            cv2.imwrite(save_path, image_to_save)
            self._target_image_saved = True
        except cv2.error as exc:
            print(Fore.RED + f"[WARN] Failed to save target object snapshot: {exc}")

    def _infer_depth(self, rgb: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
        """Infers the depth image from the rgb image.

        Args:
            rgb (np.ndarray): The rgb image to infer the depth from.

        Returns:
            np.ndarray: The inferred depth image.
        """
        raise NotImplementedError


@dataclass
class VLFMConfig:
    name: str = "HabitatITMPolicy"
    task_type: str = "coin"
    text_prompt: str = "Seems like there is a target_object ahead."
    pointnav_policy_path: str = "data/pointnav_weights.pth"
    depth_image_shape: Tuple[int, int] = (224, 224)
    pointnav_stop_radius: float = 0.75
    use_max_confidence: bool = False
    use_distance_value: bool = False
    distance_threshold: float = 3.0
    enable_loop_value: bool = False
    loop_alpha: float = 0.02
    loop_determine_threshold: float = 0.6
    loop_blacklist_count_threshold: int = 10
    object_map_erosion_size: int = 5
    explore_only: bool = False
    goal_point_method: str = "goal_point_multiple"
    exploration_thresh: float = 0.0
    enable_rotate: bool = False
    rotate_radius: float = 1.0
    robot_rotate_openness_threshold: float = 0.1
    num_angle_bin: int = 360
    save_panoramas: bool = False
    save_logging_images: bool = False
    panorama_output_dir: str = "panoramas"
    enable_multi_view: bool = True
    enable_multi_view_optimization: bool = False
    save_video: bool = MISSING

    # affect the Minimum unexplored area (in pixels) needed adjacent
    # to a frontier for that frontier to be valid. Defaults to -1.
    obstacle_map_area_threshold: float = 0.5  # in square meters

    min_obstacle_height: float = 0.61
    max_obstacle_height: float = 0.88
    hole_area_thresh: int = 100000
    use_vqa: bool = True
    vqa_prompt: str = "Is this "
    coco_threshold: float = 0.8
    non_coco_threshold: float = 0.45
    agent_radius: float = 0.18

    def __post_init__(self) -> None:
        print(
            Fore.YELLOW
            + f"[VLFMConfig] use_distance_value={self.use_distance_value}, "
            f"distance_threshold={self.distance_threshold}, "
            f"enable_loop_value={self.enable_loop_value}, "
            f"loop_alpha={self.loop_alpha}, "
            f"loop_determine_threshold={self.loop_determine_threshold}, "
            f"loop_blacklist_count_threshold={self.loop_blacklist_count_threshold}, "
            f"explore_only={self.explore_only}"
        )

    @classmethod  # type: ignore
    @property
    def kwaarg_names(cls) -> List[str]:
        # This returns all the fields listed above, except the name field
        return [f.name for f in fields(VLFMConfig) if f.name != "name"]


cs = ConfigStore.instance()
cs.store(group="policy", name="vlfm_config_base", node=VLFMConfig())
