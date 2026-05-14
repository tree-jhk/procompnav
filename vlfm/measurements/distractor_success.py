# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from typing import Any, List, Optional
import gzip
import json
import os
from pathlib import Path

import numpy as np
from habitat import registry
from habitat.config.default_structured_configs import (
    MeasurementConfig,
)
from habitat.core.embodied_task import Measure
from habitat.core.simulator import Simulator
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from colorama import Fore
from colorama import init as init_colorama

init_colorama(autoreset=True)


@registry.register_measure
class DistractorSuccess(Measure):
    cls_uuid: str = "distractor_success"

    def __init__(self, sim: Simulator, config: DictConfig, *args: Any, **kwargs: Any) -> None:
        self._sim = sim
        self._config = config
        self._current_episode = None
        self._coin_bench_content_dir: Optional[Path] = None
        self._coin_bench_distractors: Optional[List[List[float]]] = None
        self._success_distance = kwargs["task"]._config.measurements.success.success_distance
        super().__init__(*args, **kwargs)

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any) -> str:
        return DistractorSuccess.cls_uuid

    def reset_metric(self, *args: Any, **kwargs: Any) -> None:
        self._history = []
        self._current_episode = kwargs["episode"]
        assert len(self._current_episode.goals) == 1, "uncertainty lang task is an instance obj nav"

        if self._coin_bench_content_dir is None:
            content_dir = os.environ.get("COIN_BENCH_CONTENT_DIR", "").strip()
            if not content_dir:
                task_type = os.environ.get("VLFM_TASK_TYPE", "").strip()
                if task_type in ("image_goal", "object_goal"):
                    raise NotImplementedError(f"task_type='{task_type}' is not connected yet.")
                if task_type == "text_goal":
                    content_dir = "/workspace/CoIN/data/instancenav_datasets/instance_imagenav_hm3d_v3/val/content"
                else:
                    split = os.environ.get("VLFM_SPLIT", "").strip() or "val_seen"
                    content_dir = f"/workspace/CoIN/CoIN-Bench/{split}/content"
            self._coin_bench_content_dir = Path(content_dir)

        scene_id = str(self._current_episode.scene_id)
        scene_glb = Path(scene_id).name  # e.g. q5QZSEeHe5g.basis.glb
        scene_token = scene_glb.split(".")[0]  # e.g. q5QZSEeHe5g
        content_path = self._coin_bench_content_dir / f"{scene_token}.json.gz"

        object_instance_id = getattr(self._current_episode.goals[0], "object_id", None)
        task_type = os.environ.get("VLFM_TASK_TYPE", "").strip()
        if task_type in ("image_goal", "object_goal"):
            raise NotImplementedError(f"task_type='{task_type}' is not connected yet.")
        if task_type == "text_goal":
            goal_key = f"{scene_token}_{object_instance_id}"
        else:
            goal_key = f"{scene_glb}_{object_instance_id}"
        with gzip.open(content_path, "rt", encoding="utf-8") as f:
            scene_content = json.load(f)
        raw_goal = scene_content["goals"][goal_key]
        if isinstance(raw_goal, list):
            raw_goal = raw_goal[0]
        self._coin_bench_distractors = (raw_goal or {}).get("distractors", [])

        self.update_metric()

    def update_metric(self, *args: Any, **kwargs: Any) -> None:
        if "action" not in kwargs:
            self._metric = 0
            return

        action = kwargs["action"]["action"]
        if action != 0:
            # action is not 'stop'
            self._metric = 0
            return

        distractors = self._coin_bench_distractors
        if distractors is None:
            distractors = self._current_episode.goals[0].distractors
        current_position = self._sim.get_agent_state().position

        for distractor in distractors:
            geo_dist = self._sim.geodesic_distance(current_position, distractor)
            if np.isnan(geo_dist):
                continue

            if np.isinf(geo_dist):
                continue

            offset = 0.1  # certain picture or mirror are directly on the wall, thus not reachable

            if self._sim.geodesic_distance(current_position, distractor) < self._success_distance + offset:
                self._metric = 1
                return

        # othewise, FP of GroundingDINO
        self._metric = 0


@dataclass
class DistractorSuccessMeasurementConfig(MeasurementConfig):
    type: str = DistractorSuccess.__name__


cs = ConfigStore.instance()
cs.store(
    package="habitat.task.measurements.distractor_success",
    group="habitat/task/measurements",
    name="distractor_success",
    node=DistractorSuccessMeasurementConfig,
)
