from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class CoinBenchEpisodeMeta:
    episode_id: int
    scene_id: str
    object_category: str
    object_instance_id: str
    camera_spec: Dict[str, Any]
    target_position: Optional[List[float]]
    distractor_positions: List[List[float]]
    scene_path: Optional[Path]


def _load_json_gz(path: Path) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def resolve_coin_bench_content_dir(data_path: Optional[str], scenes_dir: Optional[str]) -> Optional[Path]:
    """
    CoIN-Bench stores per-scene metadata under `.../<split>/content/*.json.gz`.
    Prefer `scenes_dir` if it already points to that folder; otherwise infer from `data_path`.
    """
    if scenes_dir:
        p = Path(scenes_dir).expanduser()
        if p.is_dir() and any(p.glob("*.json.gz")):
            return p

    if data_path:
        p = Path(data_path).expanduser()
        if p.suffixes[-2:] == [".json", ".gz"]:
            inferred = p.parent / "content"
            if inferred.is_dir() and any(inferred.glob("*.json.gz")):
                return inferred
    return None


class CoinBenchEpisodeResolver:
    def __init__(
        self,
        *,
        data_path: Optional[str] = None,
        content_dir: Optional[str] = None,
        scene_dataset_config: Optional[str] = None,
    ) -> None:
        self._data_path = Path(data_path).expanduser() if data_path else None
        inferred_content = resolve_coin_bench_content_dir(data_path, content_dir)
        self._content_dir = inferred_content
        self._scene_dataset_config = Path(scene_dataset_config).expanduser() if scene_dataset_config else None
        self._scene_datasets_root = (
            self._scene_dataset_config.parent.parent if self._scene_dataset_config else None
        )

        self._episode_index: Optional[Dict[int, Dict[str, Any]]] = None
        self._scene_cache: Dict[str, Dict[str, Any]] = {}
        self._rendered_target_images: Dict[int, np.ndarray] = {}

    def _build_episode_index(self) -> None:
        if self._episode_index is not None:
            return
        if self._content_dir is None:
            raise RuntimeError(
                "CoIN-Bench content directory not found. "
                "Pass `content_dir` or provide a `data_path` with a sibling `content/` folder."
            )

        index: Dict[int, Dict[str, Any]] = {}
        for p in sorted(self._content_dir.glob("*.json.gz")):
            data = _load_json_gz(p)
            scene_base = p.name.split(".")[0]
            for ep in data.get("episodes", []):
                ep_id = int(ep["episode_id"])
                index[ep_id] = {"scene_base": scene_base, "episode": ep}
        self._episode_index = index

    def _load_scene_content(self, scene_base: str) -> Dict[str, Any]:
        if scene_base in self._scene_cache:
            return self._scene_cache[scene_base]
        if self._content_dir is None:
            raise RuntimeError("CoIN-Bench content directory not configured.")
        p = self._content_dir / f"{scene_base}.json.gz"
        data = _load_json_gz(p)
        self._scene_cache[scene_base] = data
        return data

    def get_episode_meta(self, episode_id: int) -> CoinBenchEpisodeMeta:
        self._build_episode_index()
        assert self._episode_index is not None
        entry = self._episode_index[episode_id]
        ep = entry["episode"]
        scene_base = entry["scene_base"]

        scene_content = self._load_scene_content(scene_base)
        scene_glb = Path(ep["scene_id"]).name
        inst = ep["object_instance_id"]
        goal_key = f"{scene_glb}_{inst}"
        goal0 = (scene_content.get("goals", {}).get(goal_key) or [None])[0]

        target_pos = goal0.get("position") if goal0 else None
        distractors = goal0.get("distractors") if goal0 else None
        distractors_list: List[List[float]] = [list(map(float, d)) for d in (distractors or [])]

        scene_path = None
        if self._scene_datasets_root is not None:
            scene_path = self._scene_datasets_root / str(ep["scene_id"])

        return CoinBenchEpisodeMeta(
            episode_id=int(ep["episode_id"]),
            scene_id=str(ep["scene_id"]),
            object_category=str(ep["object_category"]),
            object_instance_id=str(ep["object_instance_id"]),
            camera_spec=dict(ep.get("camera_spec") or {}),
            target_position=list(map(float, target_pos)) if target_pos is not None else None,
            distractor_positions=distractors_list,
            scene_path=scene_path,
        )

    def render_target_image(
        self,
        meta: CoinBenchEpisodeMeta,
        *,
        resolution: Tuple[int, int],
        cache: bool = True,
    ) -> np.ndarray:
        if cache and meta.episode_id in self._rendered_target_images:
            return self._rendered_target_images[meta.episode_id]
        if meta.scene_path is None:
            raise RuntimeError("Cannot render target image without `scene_dataset_config` (scene path missing).")
        if "position" not in meta.camera_spec or "rotation" not in meta.camera_spec:
            raise RuntimeError("camera_spec missing `position` and/or `rotation` for this episode.")

        import habitat_sim
        from habitat_sim.agent import SixDOFPose
        from habitat_sim.agent.agent import AgentState
        from habitat_sim.utils.common import quat_from_coeffs

        height, width = int(resolution[0]), int(resolution[1])
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = str(meta.scene_path)

        cam = habitat_sim.CameraSensorSpec()
        cam.uuid = "rgb"
        cam.sensor_type = habitat_sim.SensorType.COLOR
        cam.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        cam.resolution = [height, width]
        from_viewpoint = bool(meta.camera_spec.get("from_viewpoint", False))
        hfov = meta.camera_spec.get("hfov", None)
        if hfov is None:
            raise RuntimeError("camera_spec missing `hfov` for this episode.")
        elif float(hfov) > 0:
            cam.hfov = float(hfov)
        elif from_viewpoint:
            # CoIN-Bench outlier: qyAac8rV8Zk table_350 stores viewpoint poses with hfov=-1. (Only 10 episodes)
            # Override only for those viewpoint episodes to avoid affecting all others.
            cam.hfov = 45.0
            cam.orientation = [float(np.deg2rad(-6.0)), 0.0, 0.0]

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [cam]

        sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
        try:
            pos = np.asarray(meta.camera_spec["position"], dtype=np.float32)
            rot = quat_from_coeffs(meta.camera_spec["rotation"])  # [x,y,z,w]

            if from_viewpoint:
                agent = sim.get_agent(0)
                agent.set_state(AgentState(position=pos, rotation=rot), infer_sensor_states=True)
                obs = sim.get_sensor_observations()
            else:
                agent = sim.get_agent(0)
                st = agent.get_state()
                st.position = pos
                st.rotation = rot
                st.sensor_states = {"rgb": SixDOFPose(position=pos, rotation=rot)}
                agent.set_state(st, infer_sensor_states=False)
                obs = sim.get_sensor_observations()
            rgb = np.asarray(obs["rgb"][:, :, :3])
        finally:
            sim.close()

        if cache:
            self._rendered_target_images[meta.episode_id] = rgb
        return rgb
