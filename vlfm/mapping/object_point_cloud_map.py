# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Dict, Union, Optional, Tuple, List
from dataclasses import dataclass
import json
import os
import cv2
import numpy as np
import open3d as o3d
from pathlib import Path
import re
import requests
import sys
import math
from vlfm.utils.geometry_utils import (
    extract_yaw,
    get_point_cloud,
    transform_points,
    within_fov_cone,
)
from vlfm.vlm.server_wrapper import image_to_str
from colorama import Fore
from colorama import init as init_colorama

init_colorama(autoreset=True)
from vlfm.brain.llm_brain_history import LLM_History
from vlfm.brain.vlm_brain_history import VLM_History
from vlfm.oracle.oracle import VLMOracle
from vlfm.vlm.pbp_module import PropertyBasedBinaryPartitioning

import time
try:
    import psutil  # lightweight; used for debug logging
except Exception:
    psutil = None


@dataclass
class TimingStat:
    total_sec: float = 0.0
    count: int = 0

    def add(self, elapsed_sec: float, count: int = 1) -> None:
        self.total_sec += float(elapsed_sec)
        self.count += int(count)

    def avg_sec(self) -> float:
        if self.count <= 0:
            return 0.0
        return float(self.total_sec / self.count)


class ObjectPointCloudMap:
    clouds: Dict[str, np.ndarray] = {}
    detection_cloud: Dict[str, np.ndarray] = {}
    use_dbscan: bool = True

    def _save_instances(self) -> None:
        if self.current_episode_dir is None:
            return
        path = self.current_episode_dir / "instances.jsonl"
        excluded_keys = {
            "cloud",
            "view_keys",
            "rep_view_key",
            "2D_center_point_cloud_per_view",
            "2D_center_point_cloud_current",
            "2D_center_point_cloud_of_exact_instance",
            "2D_center_point_cloud",
            "instance_caption",
        }

        def _default(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (np.floating, np.integer)):
                return value.item()
            raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")

        with open(path, "w") as f:
            for instance_id, inst in self.instances.items():
                object_name = str(inst.get("object_name", "object"))
                instance_to_save = {
                    "instance_id": f"{object_name}_{int(instance_id):03d}",
                }
                center_2d = inst.get("2D_center_point_cloud")
                if isinstance(center_2d, list) and len(center_2d) == 2:
                    instance_to_save["instance_position"] = [float(center_2d[0]), float(center_2d[1])]
                instance_to_save.update({k: v for k, v in inst.items() if k not in excluded_keys})
                f.write(json.dumps(instance_to_save, default=_default) + "\n")

    def __init__(
        self,
        erosion_size: float,
        vlm_agent_brain,
        llm_agent_brain: LLM_History,
        vlm_oracle: VLMOracle,
        pbp_config: Dict = None,
        enable_multi_view: bool = True,
        enable_multi_view_optimization: bool = False,
    ) -> None:
        self._erosion_size = erosion_size
        self.last_target_coord: Union[np.ndarray, None] = None
        # Instance/view memory (episode has a single category).
        self.object_unique_id = 0  # counts instances; kept for compatibility with existing logs
        self.instances: Dict[int, Dict] = {}
        self.views: Dict[Tuple[int, int], Dict] = {}
        self.detection_cloud: Dict[str, np.ndarray] = {}
        self.enable_multi_view = bool(enable_multi_view)
        self.enable_multi_view_optimization = bool(enable_multi_view_optimization)

        self.vlm_agent_brain: VLM_History = vlm_agent_brain
        self.llm_agent_brain: LLM_History = llm_agent_brain
        self.vlm_oracle: VLMOracle = vlm_oracle

        self.cnt_ask = 0
        self.cnt_questions_asked_to_human = 0

        self.object_final_status = {}
        self.logging_root: Optional[Path] = None
        self.current_episode_dir: Optional[Path] = None
        self.pbp_goal_xy: Optional[np.ndarray] = None

        # Exploration-only mode: skip LLM/VLM calls.
        # Default is False; can be overridden by policy/config.
        self.explore_only: bool = False

        
        # PBP integration
        self.pbp_config = pbp_config or {}
        self.pbp_results = []
        self._pbp_target_found = False
        self._pbp_last_status: Optional[str] = None
        
        # Initialize PBP module if config provided
        if self.pbp_config:
            self.pbp = PropertyBasedBinaryPartitioning(
                llm_client=self.llm_agent_brain.LLM_CLIENT,
                mllm_client=self.vlm_agent_brain.llava_client,
                vlm_oracle=self.vlm_oracle,
                pbp_config=self.pbp_config
            )
        else:
            self.pbp = None
        self.vlm_oracle.pbp_module = self.pbp
        # Debugging aid: track memory hotspots when detections accumulate.
        self._last_logged_object_id = 0

        # Perf: re-ID + multi-view bookkeeping time (per detection).
        self._timings: Dict[str, TimingStat] = {}

    def reset(self, ep_id, target_obj) -> None:
        self.clouds = {}
        self.last_target_coord = None
        self.vlm_agent_brain.reset()
        self.llm_agent_brain.reset()
        self.vlm_oracle.reset()
        self.cnt_ask = 0
        self.cnt_questions_asked_to_human = 0
        self.object_unique_id = 0
        self.instances = {}
        self.views = {}
        self.detection_cloud = {}
        self.object_final_status = {}
        self.current_episode_dir = None
        self.pbp_goal_xy = None
        if self.logging_root is not None and ep_id is not None:
            self.current_episode_dir = self.logging_root / str(ep_id)
            self.current_episode_dir.mkdir(parents=True, exist_ok=True)
        # Reset PBP state
        self.pbp_results = []
        self._pbp_target_found = False
        self._pbp_last_status = None
        if ep_id in self.vlm_oracle.pbp_dialogue_history:
            del self.vlm_oracle.pbp_dialogue_history[ep_id]
        self._last_logged_object_id = 0
        self._timings = {}

    def add_timing(self, name: str, elapsed_sec: float, count: int = 1) -> None:
        stat = self._timings.get(name)
        if stat is None:
            stat = TimingStat()
            self._timings[name] = stat
        stat.add(elapsed_sec, count=count)

    def get_timing_total_sec(self, name: str) -> float:
        stat = self._timings.get(name)
        if stat is None:
            return 0.0
        return float(stat.total_sec)

    def get_timing_count(self, name: str) -> int:
        stat = self._timings.get(name)
        if stat is None:
            return 0
        return int(stat.count)

    def get_timing_avg_sec(self, name: str) -> float:
        stat = self._timings.get(name)
        if stat is None:
            return 0.0
        return float(stat.avg_sec())

    def get_reid_multiview_avg_time_sec(self) -> float:
        return float(self.get_timing_avg_sec("reid_multiview"))

    def get_reid_multiview_count(self) -> int:
        return int(self.get_timing_count("reid_multiview"))

    def get_pbp_avg_time_sec(self) -> float:
        return float(self.get_timing_avg_sec("pbp"))

    def get_pbp_count(self) -> int:
        return int(self.get_timing_count("pbp"))

    def get_pbp_round_avg_time_sec(self) -> float:
        return float(self.get_timing_avg_sec("pbp_round"))

    def get_pbp_round_count(self) -> int:
        return int(self.get_timing_count("pbp_round"))

    def get_pbp_depth_avg_time_sec(self) -> float:
        return float(self.get_timing_avg_sec("pbp_depth"))

    def get_pbp_depth_count(self) -> int:
        return int(self.get_timing_count("pbp_depth"))

    def has_object(self, target_class: str) -> bool:
        return target_class in self.clouds and len(self.clouds[target_class]) > 0

    def _get_rgb_image_description_for_detection(
        self,
        enable_multi_view: bool,
        current_obs_cloud: np.ndarray,
        object_name: str,
        rgb_image: np.ndarray,
    ) -> Optional[str]:
        if enable_multi_view:
            return ""
        if self.is_detection_seen(current_obs_cloud, object_name):
            print(Fore.RED + "Object is already seen, skipping")
            return None
        print(Fore.LIGHTBLUE_EX + "[INFO] This detection is new, continue with the LLM and LVLM logic")
        prompt = (
            f"You are given an image, which shows a {object_name}. You need to describe the {object_name}.\n"
            f"First, describe the {object_name} in the image, focusing on its appearance and distinctive features (use only: color, shape).\n"
            f"Then, describe the other objects (use only: color, shape) close to the {object_name}, and their spatial relationships "
            f"(use only: ['next to', 'top', 'under']) with the {object_name}.\n"
        )
        return self.vlm_agent_brain.get_description_of_the_image(rgb_image, prompt)

    def _merge_point_cloud_into_instance(
        self,
        enable_multi_view: bool,
        object_name: str,
        current_obs_cloud: np.ndarray,
        current_step: int,
    ) -> Tuple[int, int]:
        # Instance re-identification (re-ID) using point cloud overlap ratio.
        matched_instance_id: Optional[int] = None
        matched_score = -1.0
        best_score = -1.0
        matched_distances: Optional[np.ndarray] = None
        matched_center_distance = float("inf")
        fallback_instance_id: Optional[int] = None
        fallback_center_distance = float("inf")
        fallback_distances: Optional[np.ndarray] = None
        prev_num_pnt_cloud_before_view = 0
        if enable_multi_view:
            current_obs_center_2d = current_obs_cloud[:, :2].mean(axis=0)
            for instance_id, inst in self.instances.items():
                inst_cloud = inst.get("cloud")
                if not isinstance(inst_cloud, np.ndarray) or inst_cloud.size == 0:
                    continue
                distances = np.linalg.norm(inst_cloud[:, :3] - current_obs_cloud[:, :3][:, None], axis=2)
                r1 = float(np.mean(np.any(distances < 0.03, axis=1)))  # new->old
                r2 = float(np.mean(np.any(distances < 0.03, axis=0)))  # old->new
                score = max(r1, r2)
                score = 0.0 if score < 0.0 else score
                inst_center_2d = inst.get("2D_center_point_cloud")
                if isinstance(inst_center_2d, list) and len(inst_center_2d) == 2:
                    inst_center_2d = np.array(inst_center_2d, dtype=np.float32)
                else:
                    inst_center_2d = inst_cloud[:, :2].mean(axis=0)
                center_distance = float(np.linalg.norm(inst_center_2d - current_obs_center_2d))

                if score >= 0.3:
                    better_score = score > matched_score
                    tie_by_center = score == matched_score and center_distance < matched_center_distance
                    tie_by_id = (
                        score == matched_score
                        and center_distance == matched_center_distance
                        and (
                            matched_instance_id is None
                            or int(instance_id) < int(matched_instance_id)
                        )
                    )
                    if better_score or tie_by_center or tie_by_id:
                        matched_instance_id = int(instance_id)
                        matched_score = score
                        matched_distances = distances
                        matched_center_distance = center_distance
                elif center_distance <= 1.0:
                    better_distance = center_distance < fallback_center_distance
                    tie_by_id = (
                        center_distance == fallback_center_distance
                        and (
                            fallback_instance_id is None
                            or int(instance_id) < int(fallback_instance_id)
                        )
                    )
                    if better_distance or tie_by_id:
                        fallback_instance_id = int(instance_id)
                        fallback_center_distance = center_distance
                        fallback_distances = distances
                best_score = max(best_score, score)
            if matched_instance_id is None and fallback_instance_id is not None:
                matched_instance_id = int(fallback_instance_id)
                matched_score = 0.0
                matched_distances = fallback_distances

        if matched_instance_id is None:
            instance_id = int(self.object_unique_id) + 1
            self.object_unique_id = instance_id
            self.instances[instance_id] = {
                "object_name": object_name,
                "cloud": current_obs_cloud,
                "view_count": 0,
                "view_keys": [],
                "rep_view_key": (instance_id, 1),
                "pbp_selected": False,
                "first_seen_step": int(current_step),
                "2D_center_point_cloud_per_view": {},
            }
            print(Fore.LIGHTBLUE_EX + f"[REID] New instance: {object_name}_{instance_id:03d} (best_score={best_score:.3f})")
        else:
            instance_id = matched_instance_id
            inst_cloud = self.instances[instance_id]["cloud"]
            if matched_distances is None:
                matched_distances = np.linalg.norm(inst_cloud[:, :3] - current_obs_cloud[:, :3][:, None], axis=2)
            add_mask = ~np.any(matched_distances < 0.03, axis=1)
            before_pts = int(inst_cloud.shape[0])
            prev_num_pnt_cloud_before_view = before_pts
            added_pts = int(np.sum(add_mask))
            if np.any(add_mask):
                inst_cloud = np.concatenate((inst_cloud, current_obs_cloud[add_mask]), axis=0)
            after_concat_pts = int(inst_cloud.shape[0])
            inst_cloud = voxel_downsample_points(inst_cloud, voxel_size=0.05)
            after_down_pts = int(inst_cloud.shape[0])
            self.instances[instance_id]["cloud"] = inst_cloud
            print(Fore.LIGHTBLUE_EX + f"[REID] Matched instance: {object_name}_{instance_id:03d} (score={matched_score:.3f})")
            if added_pts > 0:
                print(
                    Fore.LIGHTBLUE_EX
                    + f"[CLOUD] Expand {object_name}_{instance_id:03d}: +{added_pts} pts "
                    + f"({before_pts}->{after_concat_pts}->{after_down_pts} after voxel. Increased ratio: {(after_down_pts - before_pts) / before_pts * 100:.2f}%)"
                )

        if not enable_multi_view:
            det_cloud = self.detection_cloud.get(object_name)
            if det_cloud is None:
                det_cloud = current_obs_cloud
            else:
                det_cloud = np.concatenate((det_cloud, current_obs_cloud), axis=0)
            self.detection_cloud[object_name] = voxel_downsample_points(det_cloud, voxel_size=0.05)

        return int(instance_id), int(prev_num_pnt_cloud_before_view)

    def update_map(
        self,
        object_name: str,
        depth_img: np.ndarray,
        object_mask: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
        rgb_image,
        target_object,
        total_num_steps: int,
        ep_id: int,
        force_trigger_interaction: bool = False, # skip directly to the Interaction module for debugging.
        current_step: int = 0,
        bbox=None,
        det_conf: float = 0.0,
        robot_xy: Optional[np.ndarray] = None,
        robot_heading: Optional[float] = None,
    ) -> None:
        """Updates the object map with the latest information from the agent."""
        local_cloud = self._extract_object_cloud(depth_img, object_mask, min_depth, max_depth, fx, fy)
        if len(local_cloud) == 0:
            return

        current_obs_cloud = transform_points(tf_camera_to_episodic, local_cloud)

        curr_position = tf_camera_to_episodic[:3, 3]
        closest_point = self._get_closest_point(current_obs_cloud, curr_position)
        dist = np.linalg.norm(closest_point[:3] - curr_position)
        '''Reminder on the coordinate system:
        curr_position.shape: (3, ) -> [x, y, z]
        -> z is the height from the ground
        
        current_obs_cloud.shape: (N, 3) -> [[x, y, z], ... ]
        -> z is the height from the ground
        -> Check CoIN/vlfm/utils/geometry_utils.py - get_point_cloud()

        P.S. CoIN/vlfm/visualizations/top_down_map.py - [x, y, z]
        -> y is the height from the ground
        '''

        if not force_trigger_interaction:
            if dist < 0.5:
                # Object is too close to trust as a valid object
                print(Fore.RED + "Object is too close to trust as a valid object")
                return False

        enable_multi_view = bool(getattr(self, "enable_multi_view", True))
        rgb_image_description = self._get_rgb_image_description_for_detection(
            enable_multi_view=enable_multi_view,
            current_obs_cloud=current_obs_cloud,
            object_name=object_name,
            rgb_image=rgb_image,
        )
        if rgb_image_description is None:
            return False

        reid_multiview_start = time.perf_counter()

        instance_id, prev_num_pnt_cloud_before_view = self._merge_point_cloud_into_instance(
            enable_multi_view=enable_multi_view,
            object_name=object_name,
            current_obs_cloud=current_obs_cloud,
            current_step=current_step,
        )

        inst = self.instances[instance_id]
        if enable_multi_view:
            view_idx = int(inst.get("view_count", 0)) + 1
            inst["view_count"] = view_idx
            view_key = (instance_id, view_idx)
            inst["view_keys"].append(view_key)
            view_id = f"{object_name}_{instance_id:03d}_{view_idx:03d}"
        else:
            view_idx = 1
            inst["view_count"] = 1
            view_key = (instance_id, 1)
            inst["view_keys"].append(view_key)
            view_id = f"{object_name}_{instance_id:03d}"
        
        instance_tag = f"{object_name}_{instance_id:03d}"

        inst.setdefault("2D_center_point_cloud_current", {})
        inst.setdefault("2D_center_point_cloud_of_exact_instance", {})
        inst_cloud = self.instances[instance_id]["cloud"]
        inst["2D_center_point_cloud"] = inst_cloud[:, :2].mean(axis=0).tolist()
        inst["2D_center_point_cloud_current"][view_id] = inst_cloud[:, :2].mean(axis=0).tolist()
        inst["2D_center_point_cloud_of_exact_instance"][view_id] = current_obs_cloud[:, :2].mean(axis=0).tolist()
        # Save the detected image for logging.
        image_path_str = self._save_instance_view_image(object_name, instance_id, view_idx, rgb_image)
        if image_path_str is None:
            image_path_str = ""

        bbox_list = None
        if bbox is not None:
            try:
                bbox_list = [float(x) for x in bbox]
            except Exception:
                bbox_list = None
        if robot_xy is None:
            robot_xy_list = [float(curr_position[0]), float(curr_position[1])]
        else:
            robot_xy_list = [float(robot_xy[0]), float(robot_xy[1])]
        robot_xy_list = [0.0 if abs(v) < 1e-9 else v for v in robot_xy_list]

        robot_heading_val = 0.0 if robot_heading is None else float(robot_heading)

        view_meta = {
            "step": int(current_step),
            "det_conf": float(det_conf),
            "dist": float(dist),
            "geodesic_distance": None,
            "tf": [float(x) for x in tf_camera_to_episodic.reshape(-1).tolist()],
            "tf_shape": [4, 4],
            "instance_num": int(instance_id),
            "view_num": int(view_idx),
            "instance_id": instance_tag,
            "view_id": view_id,
            "object_name": str(object_name),
            "bbox": bbox_list,
            "robot_xy": robot_xy_list,
            "robot_heading": robot_heading_val,
            "prev_num_pnt_cloud": int(prev_num_pnt_cloud_before_view),
            "closest_object_position": [float(closest_point[0]), float(closest_point[1]), float(closest_point[2])],
            "pbp_selected": bool(inst.get("pbp_selected", False)),
            "image_path": str(image_path_str),
            "rgb_image_path": str(image_path_str),
            "rgb_image_description": rgb_image_description,
        }
        self.views[view_key] = view_meta

        # Create/update per-instance summary (representative = first view).
        if view_idx == 1:
            self.object_final_status[instance_tag] = {
                "instance_id": instance_tag,
                "final_status": "FOUND_AND_CONTINUE",
                "object_stop_score": -1,
                "object_map_position": inst["cloud"],
                "rgb_image": rgb_image,
                "total_questions_to_human": self.cnt_questions_asked_to_human,
                "current_step": int(current_step),
                "step": int(current_step),
                "det_conf": float(det_conf),
                "dist": float(dist),
                "tf": view_meta["tf"],
                "tf_shape": [4, 4],
                "instance_num": int(instance_id),
                "view_num": int(view_idx),
                "view_id": view_id,
                "object_name": str(object_name),
                "bbox": bbox_list,
                "robot_xy": robot_xy_list,
                "robot_heading": robot_heading_val,
                "pbp_selected": False,
                "rgb_image_path": str(image_path_str),
                "is_verified_target_by_user": False,
                "target_verification_decision": "unknown",
                "target_verification_is_match": None,
            }
        if enable_multi_view and self.current_episode_dir is not None:
            self._save_instances()

        reid_multiview_elapsed = float(time.perf_counter() - reid_multiview_start)
        self.add_timing("reid_multiview", reid_multiview_elapsed)
        print(
            Fore.MAGENTA
            + f"[MULTI_VIEW] reID+multi-view: {reid_multiview_elapsed:.6f}s "
            + f"(avg={self.get_reid_multiview_avg_time_sec():.6f}s, n={self.get_reid_multiview_count()})"
        )
        self._log_memory_snapshot(reason="post_detection_save", target_object=target_object)
        # Return False to ensure the agent remains in 'explore' mode.
        return False

    def _save_instance_view_image(
        self,
        object_name: str,
        instance_id: int,
        view_idx: int,
        rgb_image: np.ndarray,
    ) -> Optional[str]:
        """Persist a view image at detected_objects/{category}_{instance}/{category}_{instance}_{view}.png."""
        if self.current_episode_dir is None:
            raise RuntimeError("Current episode directory not set.")

        image_dir = (
            self.current_episode_dir
            / "detected_objects"
            / f"{object_name}_{instance_id:03d}"
        )
        image_dir.mkdir(parents=True, exist_ok=True)
        safe_object_name = object_name.replace("/", "_").replace("\\", "_").replace("|", "_").replace(" ", "_")
        image_path = image_dir / f"{safe_object_name}_{instance_id:03d}_{view_idx:03d}.png"

        image_to_save = rgb_image
        if rgb_image.ndim == 3 and rgb_image.shape[2] == 3:
            try:
                image_to_save = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            except cv2.error:
                raise ValueError("Failed to convert RGB image to BGR format for saving.")
                image_to_save = rgb_image

        if cv2.imwrite(str(image_path), image_to_save):
            return str(image_path.relative_to(self.current_episode_dir))
        return None


    # ------------------------- Memory Debug Helpers ------------------------- #
    def _bytes_human(self, num_bytes: float) -> str:
        if num_bytes <= 0:
            return "0B"
        units = ["B", "KB", "MB", "GB", "TB"]
        idx = min(int(math.log(num_bytes, 1024)), len(units) - 1)
        scaled = num_bytes / (1024 ** idx)
        return f"{scaled:.2f}{units[idx]}"

    def _estimate_detection_buffers(self) -> Dict[str, float]:
        """Roughly estimate memory owned by detection buffers (numpy arrays only)."""
        total_cloud_bytes = 0.0
        total_rgb_bytes = 0.0
        total_cloud_points = 0
        total_rgb_images = 0

        for _, obj_data in self.object_final_status.items():
            cloud = obj_data.get("object_map_position")
            if isinstance(cloud, np.ndarray):
                total_cloud_bytes += cloud.nbytes
                total_cloud_points += int(cloud.shape[0])
            rgb = obj_data.get("rgb_image")
            if isinstance(rgb, np.ndarray):
                total_rgb_bytes += rgb.nbytes
                total_rgb_images += 1

        return {
            "object_cloud_bytes": total_cloud_bytes,
            "object_rgb_bytes": total_rgb_bytes,
            "object_cloud_points": float(total_cloud_points),
            "object_rgb_images": float(total_rgb_images),
            "detection_cloud_bytes": 0.0,
            "detection_cloud_points": 0.0,
        }

    def _log_memory_snapshot(self, reason: str, target_object: str) -> None:
        """
        Log process RSS and rough sizes of detection-related buffers.
        Designed to be lightweight; only runs when psutil is present.
        """
        if psutil is None:
            return
        # Avoid spamming every frame; log once per detection id.
        if self.object_unique_id == self._last_logged_object_id:
            return
        self._last_logged_object_id = self.object_unique_id

        proc = psutil.Process()
        mem_info = proc.memory_info()
        rss = mem_info.rss
        total_ram = psutil.virtual_memory().total
        rss_pct = (rss / total_ram) * 100 if total_ram > 0 else 0.0

        buf_stats = self._estimate_detection_buffers()
        obj_cloud = buf_stats["object_cloud_bytes"]
        obj_rgb = buf_stats["object_rgb_bytes"]
        det_cloud = buf_stats["detection_cloud_bytes"]
        owned_bytes = obj_cloud + obj_rgb + det_cloud
        owned_pct = (owned_bytes / rss) * 100 if rss > 0 else 0.0
        num_objs = len(self.object_final_status)
        num_det_classes = 0

        print(
            Fore.YELLOW
            + f"[MEM][{reason}] obj={target_object} id={self.object_unique_id} | "
            + f"rss={self._bytes_human(rss)} ({rss_pct:.1f}% of host) | "
            + f"buffers={self._bytes_human(owned_bytes)} ({owned_pct:.1f}% of RSS) | "
            + f"obj_cloud={self._bytes_human(obj_cloud)} pts={int(buf_stats['object_cloud_points'])} | "
            + f"obj_rgb={self._bytes_human(obj_rgb)} imgs={int(buf_stats['object_rgb_images'])} | "
            + f"detection_cloud={self._bytes_human(det_cloud)} pts={int(buf_stats['detection_cloud_points'])} | "
            + f"counts: object_final_status={num_objs}, detection_cloud_keys={num_det_classes}"
        )

    def get_best_object(self, target_class: str, curr_position: np.ndarray) -> np.ndarray:
        target_cloud = self.get_target_cloud(target_class)

        closest_point_2d = self._get_closest_point(target_cloud, curr_position)[:2]
        # return None

        if self.last_target_coord is None:
            self.last_target_coord = closest_point_2d
        else:
            # Do NOT update self.last_target_coord if:
            # 1. the closest point is only slightly different
            # 2. the closest point is a little different, but the agent is too far for
            #    the difference to matter much
            delta_dist = np.linalg.norm(closest_point_2d - self.last_target_coord)
            if delta_dist < 0.1:
                # closest point is only slightly different
                return self.last_target_coord
            elif delta_dist < 0.5 and np.linalg.norm(curr_position - closest_point_2d) > 2.0:
                # closest point is a little different, but the agent is too far for
                # the difference to matter much
                return self.last_target_coord
            else:
                self.last_target_coord = closest_point_2d

        return self.last_target_coord

    def get_target_cloud(self, target_class: str) -> np.ndarray:
        return self.clouds[target_class].copy()

    def get_target_cloud_for_checking_seen_detection(
        self, target_class: str, potential_target: bool
    ) -> np.ndarray:
        if potential_target:
            target_cloud = self.clouds.get(target_class, np.empty((0, 3))).copy()
        else:
            target_cloud = self.detection_cloud.get(target_class, np.empty((0, 3))).copy()
        if target_cloud.size == 0:
            return target_cloud
        if target_cloud.shape[1] > 3:
            within_range_exists = np.any(target_cloud[:, -1] == 1)
            if within_range_exists:
                target_cloud = target_cloud[target_cloud[:, -1] == 1]
        return target_cloud

    def is_detection_seen(self, new_detection, target_class, potential_target: bool = False):
        if target_class in self.detection_cloud:
            target_cloud = self.get_target_cloud_for_checking_seen_detection(target_class, potential_target)
            if target_cloud.size == 0:
                return False
            distances = np.linalg.norm(target_cloud[:, :3] - new_detection[:, :3][:, None], axis=2)
            seen_threshold = 1.5
            condition = np.any(distances < seen_threshold)
            if condition:
                return True
        return False

    def _extract_object_cloud(
        self,
        depth: np.ndarray,
        object_mask: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
    ) -> np.ndarray:
        final_mask = object_mask * 255
        final_mask = cv2.erode(final_mask, None, iterations=self._erosion_size)  # type: ignore

        valid_depth = depth.copy()
        valid_depth[valid_depth == 0] = 1  # set all holes (0) to just be far (1)
        valid_depth = valid_depth * (max_depth - min_depth) + min_depth
        cloud = get_point_cloud(valid_depth, final_mask, fx, fy)
        print(f"detected object cloud.shape (raw): {cloud.shape}")
        cloud = get_random_subarray(cloud, 5000)
        print(f"detected object cloud.shape (random 5000): {cloud.shape}")
        if self.use_dbscan:
            cloud = open3d_dbscan_filtering(cloud)
            print(f"detected object cloud.shape (dbscan): {cloud.shape}")
        cloud = voxel_downsample_points(cloud, voxel_size=0.05)
        print(f"detected object cloud.shape (voxel downsampled): {cloud.shape}")

        return cloud

    def _get_closest_point(self, cloud: np.ndarray, curr_position: np.ndarray) -> np.ndarray:
        ndim = curr_position.shape[0]
        if self.use_dbscan:
            # Return the point that is closest to curr_position, which is 2D
            closest_point = cloud[np.argmin(np.linalg.norm(cloud[:, :ndim] - curr_position, axis=1))]
        else:
            # Calculate the Euclidean distance from each point to the reference point
            if ndim == 2:
                ref_point = np.concatenate((curr_position, np.array([0.5])))
            else:
                ref_point = curr_position
            distances = np.linalg.norm(cloud[:, :3] - ref_point, axis=1)

            # Use argsort to get the indices that would sort the distances
            sorted_indices = np.argsort(distances)

            # Get the top 20% of points
            percent = 0.25
            top_percent = sorted_indices[: int(percent * len(cloud))]
            try:
                median_index = top_percent[int(len(top_percent) / 2)]
            except IndexError:
                median_index = 0
            closest_point = cloud[median_index]
        return closest_point

    def run_property_based_partitioning(
        self,
        target_image: np.ndarray,
        category: str,
        target_object_full: str,
        ep_id: int,
        current_step: int,
        num_frontiers: int,
        task_type: str = "coin",
        text_goal: str = "",
    ) -> Tuple[Optional[int], str]:
        if self.pbp is None:
            print(Fore.RED + "[PBP] PBP module not initialized. Skipping.")
            self._pbp_last_status = "PBP_NOT_INITIALIZED"
            return None, "PBP_NOT_INITIALIZED"
        
        if self._pbp_target_found:
            print(Fore.YELLOW + "[PBP] Target already selected in this episode. Skipping.")
            self._pbp_last_status = "PBP_TARGET_ALREADY_FOUND"
            return None, "PBP_TARGET_ALREADY_FOUND"
        
        if self.current_episode_dir is None:
            print(Fore.YELLOW + "[PBP] current_episode_dir not set. Skipping PBP.")
            self._pbp_last_status = "PBP_NO_EPISODE_DIR"
            return None, "PBP_NO_EPISODE_DIR"

        enable_multi_view = bool(getattr(self, "enable_multi_view", True))
        enable_multi_view_optimization = bool(getattr(self, "enable_multi_view_optimization", False))
        candidates: Dict[int, Dict] = {}
        group_members: Dict[int, List[int]] = {}
        group_centers: Dict[int, List[float]] = {}
        group_clouds: Dict[int, np.ndarray] = {}

        if not enable_multi_view:
            for (instance_id, _), view_meta in sorted(self.views.items(), key=lambda kv: kv[0]):
                if not isinstance(view_meta, dict):
                    continue
                image_path = view_meta.get("image_path") or view_meta.get("rgb_image_path")
                if not isinstance(image_path, str) or not image_path:
                    continue
                img_path = Path(image_path)
                if not img_path.is_absolute():
                    img_path = self.current_episode_dir / img_path
                diverse_views_image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if diverse_views_image is None:
                    continue
                inst = self.instances.get(int(instance_id))
                if not isinstance(inst, dict):
                    continue
                rep_cloud = inst.get("cloud")
                if not isinstance(rep_cloud, np.ndarray) or rep_cloud.size == 0:
                    continue
                center_2d = inst.get("2D_center_point_cloud")
                if isinstance(center_2d, list) and len(center_2d) == 2:
                    center_xy = [float(center_2d[0]), float(center_2d[1])]
                else:
                    center_xy = rep_cloud[:, :2].mean(axis=0).tolist()
                caption = view_meta.get("rgb_image_description", "")
                group_id = int(instance_id)
                candidates[group_id] = {
                    "diverse_views_image": diverse_views_image,
                    "representative_image": diverse_views_image,
                    "description": caption if isinstance(caption, str) else "",
                    "position": rep_cloud,
                }
                group_members[group_id] = [int(instance_id)]
                group_centers[group_id] = center_xy
                group_clouds[group_id] = rep_cloud
        elif not enable_multi_view_optimization:
            prompt = (
                f"You are given an image, which shows a {category}. You need to describe the {category}.\n"
                f"First, describe the {category} in the image, focusing on its appearance and distinctive features (use only: color, shape).\n"
                f"Then, describe the other objects (use only: color, shape) close to the {category}, and their spatial relationships "
                f"(use only: ['next to', 'top', 'under']) with the {category}.\n"
            )
            for instance_id, inst in sorted(self.instances.items(), key=lambda kv: kv[0]):
                rep_cloud = inst.get("cloud")
                if not isinstance(rep_cloud, np.ndarray) or rep_cloud.size == 0:
                    continue
                center_2d = inst.get("2D_center_point_cloud")
                if isinstance(center_2d, list) and len(center_2d) == 2:
                    center_xy = np.array(center_2d, dtype=np.float32)
                else:
                    center_xy = rep_cloud[:, :2].mean(axis=0).astype(np.float32)
                best_view_key = None
                best_dist = float("inf")
                for vk in inst.get("view_keys", []):
                    vm = self.views.get(vk)
                    if not isinstance(vm, dict):
                        continue
                    pos = vm.get("closest_object_position")
                    if not isinstance(pos, list) or len(pos) < 2:
                        continue
                    d = float(np.linalg.norm(np.array(pos[:2], dtype=np.float32) - center_xy))
                    if d < best_dist:
                        best_dist = d
                        best_view_key = vk
                if best_view_key is None:
                    continue
                rep_view = self.views.get(best_view_key)
                if not isinstance(rep_view, dict):
                    continue
                image_path = rep_view.get("image_path") or rep_view.get("rgb_image_path")
                if not isinstance(image_path, str) or not image_path:
                    continue
                img_path = Path(image_path)
                if not img_path.is_absolute():
                    img_path = self.current_episode_dir / img_path
                diverse_views_image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if diverse_views_image is None:
                    continue
                caption = self.vlm_agent_brain.get_description_of_the_image(diverse_views_image, prompt)
                for vk in inst.get("view_keys", []):
                    vm = self.views.get(vk)
                    if isinstance(vm, dict):
                        vm["rgb_image_description"] = caption
                group_id = int(instance_id)
                candidates[group_id] = {
                    "diverse_views_image": diverse_views_image,
                    "representative_image": diverse_views_image,
                    "description": caption if isinstance(caption, str) else "",
                    "position": rep_cloud,
                }
                group_members[group_id] = [int(instance_id)]
                group_centers[group_id] = center_xy.astype(float).tolist()
                group_clouds[group_id] = rep_cloud
        else:
            groups_root = self.current_episode_dir / "detected_objects"
            group_dirs = [
                d for d in os.listdir(str(groups_root))
                if d.startswith("group_") and d.endswith("_merged")
            ] if groups_root.exists() else []

            for group_dir in sorted(group_dirs):
                parts = group_dir.split("_")
                group_id_str = parts[1] if len(parts) >= 2 else "" # parts[1] == group id (0~N-1) (N == number of groups)
                if not group_id_str.isdigit():
                    raise ValueError(f"Invalid group directory name: {group_dir}")
                group_id = int(group_id_str) # group id (0~N-1)

                group_path = groups_root / group_dir
                group_info_path = group_path / "group_info.json"
                if not group_info_path.exists():
                    raise FileNotFoundError(f"Group info file not found: {group_info_path}")
                with open(group_info_path, "r") as f:
                    group_info = json.load(f)

                center_2d = group_info.get("position_group_center", None)
                if not (isinstance(center_2d, list) and len(center_2d) == 2):
                    raise ValueError(f"position_group_center should be a length-2 list: {group_info_path}")
                center_xy = [float(center_2d[0]), float(center_2d[1])]

                members = group_info.get("members", [])
                caption = group_info.get("caption", "")
                representative_view_filename = group_info.get("representative_view_filename", "")
                if not isinstance(members, list):
                    raise ValueError(f"members should be a list in group_info.json: {group_info_path}")
                member_ids: List[int] = [int(m.rsplit("_", 1)[1]) for m in members]
                rep_cloud = self.instances[member_ids[0]]["cloud"]

                diverse_path = group_path / "diverse_views_dino.png"
                if not diverse_path.exists():
                    continue
                diverse_views_image = cv2.imread(str(diverse_path), cv2.IMREAD_COLOR)
                if diverse_views_image is None:
                    continue
                representative_image = None
                if isinstance(representative_view_filename, str) and representative_view_filename:
                    representative_image_path = group_path / representative_view_filename
                    representative_image = cv2.imread(str(representative_image_path), cv2.IMREAD_COLOR)

                candidates[group_id] = {
                    "diverse_views_image": diverse_views_image,
                    "representative_image": representative_image,
                    "description": caption if isinstance(caption, str) else "",
                    "position": rep_cloud,
                }
                group_members[group_id] = member_ids
                group_centers[group_id] = center_xy
                group_clouds[group_id] = rep_cloud

        if len(candidates) == 0:
            trigger_step = int(self.pbp_config.get("trigger_step", 400))
            if int(current_step) < int(trigger_step):
                print(
                    Fore.RED
                    + f"[PBP] No candidates found before trigger_step={trigger_step}. Requesting trigger-threshold increase."
                )
                self._pbp_last_status = "PBP_NO_CANDIDATES_INCREASE_TRIGGER_THRESHOLD"
                return None, "PBP_NO_CANDIDATES_INCREASE_TRIGGER_THRESHOLD"
            print(Fore.YELLOW + "[PBP] No candidates found. Skipping PBP.")
            self._pbp_last_status = "PBP_FALLBACK_NO_CANDIDATES_REMAIN"
            return None, "PBP_FALLBACK_NO_CANDIDATES_REMAIN"
        
        print(Fore.CYAN + f"[PBP] Running with {len(candidates)} candidates at step {current_step}")
        
        # Run PBP
        self.pbp.text_only_model = getattr(self, "text_only_model", None)
        self.pbp.nli_only_model = getattr(self, "nli_only_model", None)
        self.pbp.nli_only_tokenizer = getattr(self, "nli_only_tokenizer", None)
        self.pbp.image_only_model = getattr(self, "image_only_model", None)
        self.pbp.image_only_processor = getattr(self, "image_only_processor", None)
        selected_object_id, pbp_run_log = self.pbp.run(
            ep_id=ep_id,
            category=category,
            target_image=target_image,
            candidates=candidates,
            current_step=current_step,
            num_frontiers=int(num_frontiers),
            task_type=task_type,
            text_goal=text_goal,
        )
        pbp_status = pbp_run_log.get("status", "")
        self._pbp_last_status = pbp_status
        if pbp_status in {"PBP_TARGET_FOUND", "PBP_FALLBACK_SINGLE_CANDIDATE"}:
            self._pbp_target_found = True
        
        # Add metadata to log
        pbp_run_log['execution_index'] = len(self.pbp_results)
        pbp_run_log['trigger_step'] = current_step
        
        # Store result
        self.pbp_results.append(pbp_run_log)
        self.add_timing(
            "pbp_round",
            float(pbp_run_log["round_time_total_sec"]),
            count=int(pbp_run_log["round_time_count"]),
        )
        self.add_timing(
            "pbp_depth",
            float(pbp_run_log["depth_time_total_sec"]),
            count=int(pbp_run_log["depth_time_count"]),
        )
        if self.pbp.enable_NLI_based and self.current_episode_dir is not None:
            discriminative_log_path = self.current_episode_dir / "discriminative_log.jsonl"
            with open(discriminative_log_path, "a") as f:
                for round_log in pbp_run_log.get("logs", []):
                    refinement_pas = list(round_log.get("refinement_pas", []))
                    if len(refinement_pas) == 0:
                        continue
                    refinement_before_groups = dict(round_log.get("refinement_before_groups", {}))
                    refinement_after_groups = dict(round_log.get("refinement_after_groups", {}))
                    extracted_group = str(refinement_before_groups.get("extracted_group", round_log.get("prop_group", "")))
                    non_extracted_group = "g2" if extracted_group == "g1" else "g1"
                    non_extracted_threshold = float(round_log.get("refinement_threshold", self.pbp.pbp_refinement_thres))
                    moved_object_ids = [int(item["object_id"]) for item in refinement_pas if item.get("group_before") != item.get("group_after")]
                    extracted_before = [item for item in refinement_pas if str(item.get("group_before", "")) == extracted_group]
                    non_extracted_before = [item for item in refinement_pas if str(item.get("group_before", "")) == non_extracted_group]
                    extracted_after = [item for item in refinement_pas if str(item.get("group_after", "")) == extracted_group]
                    non_extracted_after = [item for item in refinement_pas if str(item.get("group_after", "")) == non_extracted_group]
                    extracted_to_non_extracted = [
                        item for item in refinement_pas
                        if str(item.get("group_before", "")) == extracted_group and float(item.get("pas", 0.0)) < float(round_log.get("refinement_threshold", self.pbp.pbp_refinement_thres))
                    ]
                    non_extracted_to_extracted = [
                        item for item in refinement_pas
                        if str(item.get("group_before", "")) == non_extracted_group and float(item.get("pas", 0.0)) >= non_extracted_threshold
                    ]
                    entry = {
                        "ep_id": int(ep_id),
                        "execution_index": int(pbp_run_log["execution_index"]),
                        "trigger_step": int(pbp_run_log["trigger_step"]),
                        "round": int(round_log.get("round", 0)),
                        "depth": int(round_log.get("depth", 0)),
                        "pbp_round_status": str(round_log.get("pbp_round_status", "")),
                        "category": category,
                        "property": str(round_log.get("chosen_property", "")),
                        "prop_group": str(round_log.get("prop_group", "")),
                        "hypothesis": f"the {category} has the attribute {round_log.get('chosen_property', '')}",
                        "refinement_enabled": bool(round_log.get("refinement_enabled", False)),
                        "refinement_change": bool(round_log.get("refinement_change", False)),
                        "refinement_threshold": float(round_log.get("refinement_threshold", self.pbp.pbp_refinement_thres)),
                        "non_extracted_threshold": float(non_extracted_threshold),
                        "extracted_group": extracted_group,
                        "non_extracted_group": non_extracted_group,
                        "refinement_before_groups": refinement_before_groups,
                        "refinement_after_groups": refinement_after_groups,
                        "moved_object_ids": moved_object_ids,
                        "extracted_to_non_extracted": extracted_to_non_extracted,
                        "non_extracted_to_extracted": non_extracted_to_extracted,
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        for group_id in candidates.keys():
            for instance_id in group_members.get(group_id, []):
                instance_tag = f"{category}_{int(instance_id):03d}"
                if instance_tag not in self.object_final_status:
                    continue
                if "pbp_properties" not in self.object_final_status[instance_tag]:
                    self.object_final_status[instance_tag]["pbp_properties"] = []
                if "pbp_selected" not in self.object_final_status[instance_tag]:
                    self.object_final_status[instance_tag]["pbp_selected"] = False
                if "pbp_max_depth" not in self.object_final_status[instance_tag]:
                    self.object_final_status[instance_tag]["pbp_max_depth"] = 0
                if "pbp_eliminated_at_depth" not in self.object_final_status[instance_tag]:
                    self.object_final_status[instance_tag]["pbp_eliminated_at_depth"] = None

        for round_log in pbp_run_log.get("logs", []):
            if "yes_object_ids" not in round_log or "no_object_ids" not in round_log:
                continue
            prop = round_log.get("chosen_property", "")
            depth = round_log.get("depth", 0)

            for group_id in round_log["yes_object_ids"]:
                for instance_id in group_members.get(int(group_id), []):
                    instance_tag = f"{category}_{int(instance_id):03d}"
                    if instance_tag in self.object_final_status:
                        self.object_final_status[instance_tag]["pbp_properties"].append((prop, True, depth))
                        self.object_final_status[instance_tag]["pbp_max_depth"] = max(
                            self.object_final_status[instance_tag]["pbp_max_depth"], depth
                        )

            for group_id in round_log["no_object_ids"]:
                for instance_id in group_members.get(int(group_id), []):
                    instance_tag = f"{category}_{int(instance_id):03d}"
                    if instance_tag in self.object_final_status:
                        self.object_final_status[instance_tag]["pbp_properties"].append((prop, False, depth))
                        self.object_final_status[instance_tag]["pbp_max_depth"] = max(
                            self.object_final_status[instance_tag]["pbp_max_depth"], depth
                        )

        if selected_object_id is not None and int(selected_object_id) in group_centers:
            selected_group_id = int(selected_object_id)
            for instance_id in group_members.get(selected_group_id, []):
                inst = self.instances.get(int(instance_id))
                if isinstance(inst, dict):
                    inst["pbp_selected"] = True
                    for vk in inst.get("view_keys", []):
                        if vk in self.views:
                            self.views[vk]["pbp_selected"] = True
                instance_tag = f"{category}_{int(instance_id):03d}"
                if instance_tag in self.object_final_status:
                    self.object_final_status[instance_tag]["pbp_selected"] = True

            center_xy = group_centers[selected_group_id]
            self.pbp_goal_xy = np.array(center_xy, dtype=np.float32)
            self.clouds[category] = group_clouds[selected_group_id]

            print(Fore.GREEN + f"[PBP] Selected group {selected_group_id}. Navigating to center point of grouped instance {selected_group_id}. goal point: {self.pbp_goal_xy}.")

            for group_id in candidates.keys():
                if int(group_id) == selected_group_id:
                    continue
                for instance_id in group_members.get(int(group_id), []):
                    other_tag = f"{category}_{int(instance_id):03d}"
                    if other_tag in self.object_final_status:
                        max_depth = self.object_final_status[other_tag].get("pbp_max_depth", 0)
                        self.object_final_status[other_tag]["pbp_eliminated_at_depth"] = max_depth
        
        # Update total_questions_to_human for all objects
        total_questions = self.vlm_oracle.ask_to_human_episode_counter.get(ep_id, 0)
        for group_id in candidates.keys():
            for instance_id in group_members.get(int(group_id), []):
                tag = f"{category}_{int(instance_id):03d}"
                if tag in self.object_final_status:
                    self.object_final_status[tag]["total_questions_to_human"] = total_questions

        if self.current_episode_dir is not None:
            self._save_instances()
        
        return selected_object_id, pbp_status


def open3d_dbscan_filtering(points: np.ndarray, eps: float = 0.2, min_points: int = 100) -> np.ndarray:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Perform DBSCAN clustering
    labels = np.array(pcd.cluster_dbscan(eps, min_points))

    # Count the points in each cluster
    unique_labels, label_counts = np.unique(labels, return_counts=True)

    # Exclude noise points, which are given the label -1
    non_noise_labels_mask = unique_labels != -1
    non_noise_labels = unique_labels[non_noise_labels_mask]
    non_noise_label_counts = label_counts[non_noise_labels_mask]

    if len(non_noise_labels) == 0:  # only noise was detected
        return np.array([])

    # Find the label of the largest non-noise cluster
    largest_cluster_label = non_noise_labels[np.argmax(non_noise_label_counts)]

    # Get the indices of points in the largest non-noise cluster
    largest_cluster_indices = np.where(labels == largest_cluster_label)[0]

    # Get the points in the largest non-noise cluster
    largest_cluster_points = points[largest_cluster_indices]

    return largest_cluster_points


def voxel_downsample_points(points: np.ndarray, voxel_size: float = 0.05) -> np.ndarray:
    """
    Downsample an (N, 3) point cloud using voxel grid filtering via Open3D.

    Args:
        points: Array of xyz points with shape (N, 3).
        voxel_size: Voxel edge length in meters (e.g., 0.05 for 5cm).

    Returns:
        Downsampled xyz points with shape (M, 3).
    """
    if points is None or len(points) == 0:
        return np.array([])
    if voxel_size is None or voxel_size <= 0:
        return points

    xyz = np.asarray(points[:, :3], dtype=np.float64)
    finite_mask = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite_mask]
    if len(xyz) == 0:
        return np.array([])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    down = pcd.voxel_down_sample(voxel_size=float(voxel_size))
    return np.asarray(down.points)


def get_random_subarray(points: np.ndarray, size: int) -> np.ndarray:
    """
    This function returns a subarray of a given 3D points array. The size of the
    subarray is specified by the user. The elements of the subarray are randomly
    selected from the original array. If the size of the original array is smaller than
    the specified size, the function will simply return the original array.

    Args:
        points (numpy array): A numpy array of 3D points. Each element of the array is a
            3D point represented as a numpy array of size 3.
        size (int): The desired size of the subarray.

    Returns:
        numpy array: A subarray of the original points array.
    """
    if len(points) <= size:
        return points
    indices = np.random.choice(len(points), size, replace=False)
    return points[indices]


