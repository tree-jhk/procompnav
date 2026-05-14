#!/usr/bin/env python3

import gzip
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import attr
from habitat.core.registry import registry
from habitat.core.simulator import AgentState
from habitat.datasets.image_nav.instance_image_nav_dataset import InstanceImageNavDatasetV1
from habitat.datasets.pointnav.pointnav_dataset import DEFAULT_SCENE_PATH_PREFIX
from habitat.tasks.nav.instance_image_nav_task import (
    InstanceImageGoal,
    InstanceImageGoalNavEpisode,
    InstanceImageParameters,
)
from habitat.tasks.nav.object_nav_task import ObjectViewLocation

if TYPE_CHECKING:
    from omegaconf import DictConfig


def _load_json_gz(path: Path) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


@attr.s(auto_attribs=True, kw_only=True)
class InstanceNavTextGoalEpisode(InstanceImageGoalNavEpisode):
    instruction: str = ""
    instruction_input: str = ""
    camera_spec: Dict[str, Any] = attr.ib(factory=dict)
    object_instance_id: Optional[int] = None


@registry.register_dataset(name="InstanceNavTextGoalDataset")
class InstanceNavTextGoalDataset(InstanceImageNavDatasetV1):
    episodes: List[InstanceNavTextGoalEpisode] = []  # type: ignore

    def __init__(self, config: Optional["DictConfig"] = None) -> None:
        self.episodes = []
        self.goals = {}
        self._attribute_data: Dict[str, Dict[str, str]] = {}
        self._episode_field_names = {f.name for f in attr.fields(InstanceNavTextGoalEpisode)}
        super().__init__(config)
        self.episodes = list(self.episodes)

    @staticmethod
    def _deserialize_view_points(goal: InstanceImageGoal) -> InstanceImageGoal:
        for vidx, view in enumerate(goal.view_points):
            if isinstance(view, ObjectViewLocation):
                view_location = view
            elif isinstance(view, dict):
                view_location = ObjectViewLocation(**view)  # type: ignore[arg-type]
            else:
                raise TypeError(f"Unexpected view_points element type: {type(view)}")
            if not isinstance(view_location.agent_state, AgentState):
                view_location.agent_state = AgentState(**view_location.agent_state)  # type: ignore[arg-type]
            goal.view_points[vidx] = view_location
        return goal

    def from_json(self, json_str: str, scenes_dir: Optional[str] = None) -> None:
        deserialized = json.loads(json_str)
        attribute_data = deserialized.get("attribute_data") or {}
        if not isinstance(attribute_data, dict):
            raise ValueError("InstanceNavTextGoalDataset expects `attribute_data` to be a dict.")
        self._attribute_data = attribute_data

        cfg = getattr(self, "_config", None) or getattr(self, "config", None)
        cfg_scenes_dir = getattr(cfg, "scenes_dir", "") if cfg is not None else ""
        content_dir = Path(scenes_dir or cfg_scenes_dir).expanduser()
        if not content_dir.is_dir():
            raise RuntimeError(
                "InstanceNavTextGoalDataset requires `habitat.dataset.scenes_dir` to point to the InstanceNav content directory."
            )
        scene_files = sorted(content_dir.glob("*.json.gz"))
        if not scene_files:
            raise RuntimeError(f"No scene content files found under scenes_dir={content_dir}")

        episodes: List[Dict[str, Any]] = []
        scene_contents: Dict[str, Dict[str, Any]] = {}
        categories: set[str] = set()
        for p in scene_files:
            scene_data = _load_json_gz(p)
            scene_token = p.name.split(".")[0]
            scene_contents[scene_token] = scene_data
            for ep in scene_data.get("episodes", []):
                ep = dict(ep)
                ep["_scene_token_from_file"] = scene_token
                episodes.append(ep)
                if "object_category" in ep:
                    categories.add(str(ep["object_category"]))

        if "category_to_task_category_id" in deserialized:
            self.category_to_task_category_id = deserialized["category_to_task_category_id"]
        else:
            self.category_to_task_category_id = {c: i for i, c in enumerate(sorted(categories))}
        if "category_to_scene_annotation_category_id" in deserialized:
            self.category_to_scene_annotation_category_id = deserialized["category_to_scene_annotation_category_id"]
        else:
            self.category_to_scene_annotation_category_id = dict(self.category_to_task_category_id)

        for ep in episodes:
            object_category = str(ep.get("object_category") or "")
            goal_object_id = ep.get("goal_object_id")
            if goal_object_id is None:
                raise KeyError("Missing `goal_object_id` in InstanceNav episode entry.")
            scene_token = str(ep.get("_scene_token_from_file") or "").strip()
            if not scene_token:
                raise KeyError(f"Missing _scene_token_from_file for episode {ep.get('episode_id')}")
            goal_object_scene_id = f"{scene_token}_{goal_object_id}"

            text_goal_dict = self._attribute_data.get(goal_object_scene_id)
            if not isinstance(text_goal_dict, dict):
                scene_attr = self._attribute_data.get(scene_token)
                if isinstance(scene_attr, dict):
                    text_goal_dict = scene_attr.get(str(goal_object_id)) or scene_attr.get(goal_object_id)
            if not isinstance(text_goal_dict, dict):
                text_goal_dict = {}
            intrinsic = str(text_goal_dict.get("intrinsic_attributes") or "").strip()
            extrinsic = str(text_goal_dict.get("extrinsic_attributes") or "").strip()
            instruction = (intrinsic + " " + extrinsic).strip()
            if not instruction:
                instruction = str(text_goal_dict.get("instruction") or text_goal_dict.get("text_goal") or "").strip()
            if not instruction:
                raise ValueError(
                    f"Empty text_goal for episode_id={ep.get('episode_id')} key={goal_object_scene_id} "
                    f"attr_keys={list(text_goal_dict.keys())}"
                )

            scene_content_path = content_dir / f"{scene_token}.json.gz"
            scene_content = scene_contents.get(scene_token)
            if scene_content is None:
                raise KeyError(f"Missing scene content for scene_token={scene_token}")
            raw_goal = scene_content.get("goals", {}).get(goal_object_scene_id)
            if raw_goal is None:
                raise KeyError(f"Missing goals[{goal_object_scene_id}] in {scene_content_path}")
            if isinstance(raw_goal, list):
                raw_goal = raw_goal[0]

            object_position = raw_goal.get("position")
            if object_position is None:
                raise KeyError(f"Missing goals[{goal_object_scene_id}].position in {scene_content_path}")
            raw_image_goals = raw_goal.get("image_goals") or []
            if not raw_image_goals:
                raise KeyError(f"Missing goals[{goal_object_scene_id}].image_goals in {scene_content_path}")

            image_goal_keys = ("position", "rotation", "hfov", "image_dimensions", "frame_coverage", "object_coverage")
            image_goals = [InstanceImageParameters(**{k: g[k] for k in image_goal_keys if k in g}) for g in raw_image_goals]
            camera_spec = dict(raw_image_goals[0])

            goal = self._deserialize_view_points(
                InstanceImageGoal(
                    position=raw_goal.get("position"),
                    radius=raw_goal.get("radius", 0.0),
                    view_points=raw_goal.get("view_points", []),
                    object_id=raw_goal.get("object_id", goal_object_id),
                    object_name=raw_goal.get("object_name", object_category),
                    object_name_id=raw_goal.get("object_name_id", goal_object_id),
                    object_category=raw_goal.get("object_category", object_category),
                    room_id=raw_goal.get("room_id", None),
                    room_name=raw_goal.get("room_name", None),
                    image_goals=image_goals,
                    object_surface_area=raw_goal.get("object_surface_area", None),
                )
            )

            ep_fields = {k: v for k, v in ep.items() if k in self._episode_field_names}
            ep_fields["object_instance_id"] = goal_object_id
            ep_fields["goal_object_id"] = str(goal_object_id)
            ep_fields["goal_image_id"] = 0
            ep_fields["object_category"] = object_category
            ep_fields["instruction"] = instruction
            ep_fields["instruction_input"] = f"Find the {object_category}".strip()
            ep_fields["camera_spec"] = camera_spec
            ep_fields["goals"] = []
            episode = InstanceNavTextGoalEpisode(**ep_fields)
            episode.goals = [goal]
            episode.scene_id = DEFAULT_SCENE_PATH_PREFIX + episode.scene_id
            self.goals[episode.goal_key] = goal
            self.episodes.append(episode)


@registry.register_dataset(name="InstanceNavImageGoalDataset")
class InstanceNavImageGoalDataset(InstanceImageNavDatasetV1):
    episodes: List[InstanceNavTextGoalEpisode] = []  # type: ignore

    def __init__(self, config: Optional["DictConfig"] = None) -> None:
        self.episodes = []
        super().__init__(config)
        self.episodes = list(self.episodes)

    def from_json(self, json_str: str, scenes_dir: Optional[str] = None) -> None:
        raise NotImplementedError("InstanceNavImageGoalDataset is not connected yet.")
