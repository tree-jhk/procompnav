# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import json
import os
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
from torch import Tensor

from vlfm.mapping.frontier_map import FrontierMap
from vlfm.mapping.value_map import ValueMap
from vlfm.policy.base_objectnav_policy import BaseObjectNavPolicy
from vlfm.policy.utils.acyclic_enforcer import AcyclicEnforcer
from vlfm.utils.geometry_utils import closest_point_within_threshold
from vlfm.vlm.blip2itm import BLIP2ITMClient
from vlfm.vlm.detections import ObjectDetections

try:
    from habitat_baselines.common.tensor_dict import TensorDict
except Exception:
    pass

from colorama import Fore
from colorama import init as init_colorama

init_colorama(autoreset=True)

PROMPT_SEPARATOR = "|"


class BaseITMPolicy(BaseObjectNavPolicy):
    _target_object_color: Tuple[int, int, int] = (0, 255, 0)
    _selected__frontier_color: Tuple[int, int, int] = (0, 255, 255)
    _frontier_color: Tuple[int, int, int] = (0, 0, 255)
    _circle_marker_thickness: int = 2
    _circle_marker_radius: int = 5
    _last_value: float = float("-inf")
    _last_frontier: np.ndarray = np.zeros(2)

    @staticmethod
    def _vis_reduce_fn(i: np.ndarray) -> np.ndarray:
        return np.max(i, axis=-1)

    def __init__(
        self,
        text_prompt: str,
        use_max_confidence: bool = True,
        sync_explored_areas: bool = False,
        use_distance_value: bool = False,
        distance_threshold: float = 3.0,
        enable_loop_value: bool = False,
        loop_alpha: float = 0.02,
        loop_determine_threshold: float = 0.6,
        loop_blacklist_count_threshold: int = 10,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._itm = BLIP2ITMClient(port=int(os.environ.get("BLIP2ITM_PORT", "12182")))
        self._text_prompt = text_prompt
        self._value_map: ValueMap = ValueMap(
            value_channels=len(text_prompt.split(PROMPT_SEPARATOR)),
            use_max_confidence=use_max_confidence,
            obstacle_map=self._obstacle_map if sync_explored_areas else None,
            pixels_per_meter=30,
            size=1500,
        )
        self._acyclic_enforcer = AcyclicEnforcer()
        self._use_distance_value = use_distance_value
        self._distance_threshold = distance_threshold
        self._use_loop_value = enable_loop_value
        self._loop_alpha = float(loop_alpha)
        self._loop_mu = np.zeros(2, dtype=np.float32)
        self._loop_s = 0.0
        self._loop_seen_first_forward = False
        self._loop_determine_threshold = float(loop_determine_threshold)
        self._loop_blacklist_count_threshold = int(loop_blacklist_count_threshold)
        self._loop_blacklist_cells: set[Tuple[int, int]] = set()
        self._loop_count_key = None
        self._loop_count = 0
        self._loop_prev_term = 0.0
        self._loop_last_top_key = None
        self._loop_blacklist_triggered = False
        self._loop_blacklist_added_cells: List[Tuple[int, int]] = []
        self._loop_blacklist_lock = False
        self._last_sorted_frontiers: np.ndarray = np.zeros((0, 2))
        self._last_sorted_values: List[float] = []
        self._latest_frontier_score_log = "[FRONTIER_SCORE] mode=VLFM score=nan reason=not_computed [FRONTIER_RANK] none"
        print(
            f"[BaseITMPolicy] class={self.__class__.__name__}, "
            f"use_distance_value={self._use_distance_value}, "
            f"distance_threshold={self._distance_threshold}, "
            f"enable_loop_value={self._use_loop_value}, "
            f"loop_alpha={self._loop_alpha}, "
            f"loop_determine_threshold={self._loop_determine_threshold}, "
            f"loop_blacklist_count_threshold={self._loop_blacklist_count_threshold}"
        )

    def _reset(self) -> None:
        super()._reset()
        self._value_map.reset()
        self._acyclic_enforcer = AcyclicEnforcer()
        self._last_value = float("-inf")
        self._last_frontier = np.zeros(2)
        self._loop_mu = np.zeros(2, dtype=np.float32)
        self._loop_s = 0.0
        self._loop_seen_first_forward = False
        self._loop_blacklist_cells = set()
        self._loop_count_key = None
        self._loop_count = 0
        self._loop_prev_term = 0.0
        self._loop_last_top_key = None
        self._loop_blacklist_triggered = False
        self._loop_blacklist_added_cells = []
        self._loop_blacklist_lock = False

    def _explore(self, observations: Union[Dict[str, Tensor], "TensorDict"]) -> Tensor:
        if self._rotate_active():
            self._latest_frontier_score_log = (
                "[FRONTIER_SCORE] mode=VLFM score=nan reason=rotate_in_progress [FRONTIER_RANK] none"
            )
            return self._rotate_action()
        frontiers = self._observations_cache["frontier_sensor"]
        if np.array_equal(frontiers, np.zeros((1, 2))) or len(frontiers) == 0:
            # if getattr(self._object_map, "explore_only", False):
            self._latest_frontier_score_log = (
                "[FRONTIER_SCORE] mode=VLFM score=nan reason=no_frontiers [FRONTIER_RANK] none"
            )
            print("No frontiers found during exploration, executing random move.")
            return self._random_move()
            # print(Fore.RED + "[INFO] No frontiers found during exploration, calling STOP action.")
            # return self._stop_action
        self._reset_rotation_mode()
        best_frontier, best_value = self._get_best_frontier(observations, frontiers)
        if self._try_start_rotate():
            return self._rotate_action()
        pointnav_action = self._pointnav(best_frontier, stop=False, is_navigate_mode=False)

        return pointnav_action

    def _get_best_frontier(
        self,
        observations: Union[Dict[str, Tensor], "TensorDict"],
        frontiers: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Returns the best frontier and its value based on self._value_map.

        Args:
            observations (Union[Dict[str, Tensor], "TensorDict"]): The observations from
                the environment.
            frontiers (np.ndarray): The frontiers to choose from, array of 2D points.

        Returns:
            Tuple[np.ndarray, float]: The best frontier and its value.
        """
        # The points and values will be sorted in descending order
        sorted_pts, sorted_values = self._sort_frontiers_by_value(observations, frontiers)
        self._last_sorted_frontiers = sorted_pts
        self._last_sorted_values = [float(v) for v in sorted_values]
        frontier_ranking = [
            {
                "rank": int(i),
                "score": float(v),
                "x": float(p[0]),
                "y": float(p[1]),
            }
            for i, (p, v) in enumerate(zip(sorted_pts, sorted_values))
        ]
        frontier_ranking_str = " | ".join(
            [f"r{int(i)}:s={float(v):.4f},xy=({float(p[0]):.3f},{float(p[1]):.3f})" for i, (p, v) in enumerate(zip(sorted_pts, sorted_values))]
        )
        robot_xy = self._observations_cache["robot_xy"]
        best_frontier_idx = None
        top_two_values = tuple(sorted_values[:2])

        os.environ["DEBUG_INFO"] = ""
        # If there is a last point pursued, then we consider sticking to pursuing it
        # if it is still in the list of frontiers and its current value is not much
        # worse than self._last_value.
        if not np.array_equal(self._last_frontier, np.zeros(2)):
            curr_index = None

            for idx, p in enumerate(sorted_pts):
                if np.array_equal(p, self._last_frontier):
                    # Last point is still in the list of frontiers
                    curr_index = idx
                    break

            if curr_index is None:
                closest_index = closest_point_within_threshold(sorted_pts, self._last_frontier, threshold=0.5)

                if closest_index != -1:
                    # There is a point close to the last point pursued
                    curr_index = closest_index

            if curr_index is not None:
                curr_value = sorted_values[curr_index]
                if curr_value + 0.01 > self._last_value:
                    # The last point pursued is still in the list of frontiers and its
                    # value is not much worse than self._last_value
                    print("Sticking to last point.")
                    os.environ["DEBUG_INFO"] += "Sticking to last point. "
                    best_frontier_idx = curr_index

        # If there is no last point pursued, then just take the best point, given that
        # it is not cyclic.
        if best_frontier_idx is None:
            for idx, frontier in enumerate(sorted_pts):
                cyclic = self._acyclic_enforcer.check_cyclic(robot_xy, frontier, top_two_values)
                if cyclic:
                    print("Suppressed cyclic frontier.")
                    continue
                best_frontier_idx = idx
                break

        if best_frontier_idx is None:
            print("All frontiers are cyclic. Just choosing the closest one.")
            os.environ["DEBUG_INFO"] += "All frontiers are cyclic. "
            best_frontier_idx = max(
                range(len(frontiers)),
                key=lambda i: np.linalg.norm(frontiers[i] - robot_xy),
            )

        best_frontier = sorted_pts[best_frontier_idx]
        best_value = sorted_values[best_frontier_idx]
        self._acyclic_enforcer.add_state_action(robot_xy, best_frontier, top_two_values)
        self._last_value = best_value
        self._last_frontier = best_frontier
        mode, base, dist, loop, local, penalty = self._frontier_score_components(best_frontier, float(best_value))
        loop_over_threshold = int(loop >= self._loop_determine_threshold) if "+LOOP" in mode else 0
        loop_count_ready = int(self._loop_count >= self._loop_blacklist_count_threshold)
        frontier_score_log = (
            f"[FRONTIER_SCORE] mode={mode} score={float(best_value):.4f}"
            + (f" base={base:.4f}" if base is not None else "")
            + (f" dist={dist:.4f}" if dist is not None else "")
            + (f" loop={loop:.4f}" if loop is not None else "")
            + f" loop_thres={self._loop_determine_threshold:.4f}"
            + f" loop_over_thres={loop_over_threshold}"
            + f" loop_count={int(self._loop_count)}/{int(self._loop_blacklist_count_threshold)}"
            + f" loop_count_ready={loop_count_ready}"
            + f" loop_key={self._loop_last_top_key}"
            + f" loop_lock={int(self._loop_blacklist_lock)}"
            + f" blacklist_cells={int(len(self._loop_blacklist_cells))}"
            + (f" triggered=1 add={self._loop_blacklist_added_cells}" if self._loop_blacklist_triggered else "")
            + f" [FRONTIER_RANK] {frontier_ranking_str}"
        )
        self._latest_frontier_score_log = frontier_score_log
        os.environ["DEBUG_INFO"] += f" {frontier_score_log}"
        with open(self._object_map.current_episode_dir / "loop_score.jsonl", "a") as f:
            f.write(
                json.dumps(
                    {
                        "step": int(self._num_steps),
                        "mode": mode,
                        "loop": float(loop),
                        "loop_threshold": float(self._loop_determine_threshold),
                        "loop_over_threshold": int(loop_over_threshold),
                        "loop_count_ready": int(loop_count_ready),
                        "frontier_x": float(best_frontier[0]),
                        "frontier_y": float(best_frontier[1]),
                        "denom_loop": self._last_frontier_score_meta["denom_loop"],
                        "mu_x": self._last_frontier_score_meta["mu_x"],
                        "mu_y": self._last_frontier_score_meta["mu_y"],
                        "loop_count_key": None
                        if self._loop_last_top_key is None
                        else [int(self._loop_last_top_key[0]), int(self._loop_last_top_key[1])],
                        "loop_count": int(self._loop_count),
                        "loop_count_threshold": int(self._loop_blacklist_count_threshold),
                        "loop_prev_term": float(self._loop_prev_term),
                        "loop_blacklist_lock": int(self._loop_blacklist_lock),
                        "loop_blacklist_cells_count": int(len(self._loop_blacklist_cells)),
                        "loop_blacklist_cells": [[int(c[0]), int(c[1])] for c in sorted(self._loop_blacklist_cells)],
                        "loop_blacklist_triggered": bool(self._loop_blacklist_triggered),
                        "loop_blacklist_added_cells": [[int(c[0]), int(c[1])] for c in self._loop_blacklist_added_cells],
                        "frontier_ranking": frontier_ranking,
                    }
                )
                + "\n"
            )

        return best_frontier, best_value

    def _frontier_score_components(self, frontier: np.ndarray, score: float) -> Tuple[str, float, float, float, float, float]:
        prev_actions = getattr(self, "_prev_actions", None)
        use_prev_actions = (
            (not self._rotate_active())
            and isinstance(prev_actions, torch.Tensor)
            and prev_actions.dtype == torch.long
            and prev_actions.numel() > 0
        )
        apply_dist = use_prev_actions and self._use_distance_value
        apply_loop = use_prev_actions and self._use_loop_value and self._loop_seen_first_forward

        mode = "VLFM" + ("+DIST" if apply_dist else "") + ("+LOOP" if apply_loop else "")
        robot_xy = self._observations_cache["robot_xy"]
        dist = 0.0
        if apply_dist:
            d = float(np.linalg.norm(frontier[:2] - robot_xy))
            dist = float(np.exp(-d)) if d <= self._distance_threshold else 0.0
        loop = 0.0
        local = 0.0
        penalty = 0.0
        denom_loop = None
        denom_local = None
        mu_x = None
        mu_y = None
        if apply_loop:
            mu = self._loop_mu
            denom_loop = self._loop_s + 1e-6
            d2_loop = float(np.dot(robot_xy - mu, robot_xy - mu))
            loop = float(np.exp(-d2_loop / denom_loop))
            mu_x = float(mu[0])
            mu_y = float(mu[1])
        self._last_frontier_score_meta = {
            "denom_loop": None if denom_loop is None else float(denom_loop),
            "denom_local": None if denom_local is None else float(denom_local),
            "mu_x": mu_x,
            "mu_y": mu_y,
        }
        base = float(score) - dist
        return mode, base, dist, loop, local, penalty

    def _loop_cell_size(self) -> float:
        cfg = getattr(self._object_map, "pbp_config", {}) if hasattr(self, "_object_map") else {}
        return float(cfg.get("sufficient_exp_cell_size", 0.25))

    def _frontier_cell(self, frontier_xy: np.ndarray) -> Tuple[int, int]:
        cell_size = self._loop_cell_size()
        return tuple(np.floor(np.asarray(frontier_xy[:2], dtype=np.float32) / cell_size).astype(int).tolist())

    def _frontier_cells_rounded(self, frontier_xy: np.ndarray) -> List[Tuple[int, int]]:
        cx, cy = self._frontier_cell(frontier_xy)
        return [(cx, cy)]

    def _apply_loop_blacklist_to_value_map(self) -> None:
        if len(self._loop_blacklist_cells) == 0:
            return
        value_map = self._value_map._value_map
        ppm = float(self._value_map.pixels_per_meter)
        origin = self._value_map._episode_pixel_origin
        size = int(value_map.shape[0])
        cell_size = self._loop_cell_size()
        for cx, cy in self._loop_blacklist_cells:
            x0, x1 = float(cx) * cell_size, float(cx + 1) * cell_size
            y0, y1 = float(cy) * cell_size, float(cy + 1) * cell_size
            px0, px1 = int(-x0 * ppm) + int(origin[0]), int(-x1 * ppm) + int(origin[0])
            py0, py1 = int(-y0 * ppm) + int(origin[1]), int(-y1 * ppm) + int(origin[1])
            r0, r1 = size - px0, size - px1
            c0, c1 = py0, py1
            rmin, rmax = int(np.clip(min(r0, r1), 0, size - 1)), int(np.clip(max(r0, r1), 0, size - 1))
            cmin, cmax = int(np.clip(min(c0, c1), 0, size - 1)), int(np.clip(max(c0, c1), 0, size - 1))
            value_map[rmin : rmax + 1, cmin : cmax + 1, :] = -1e9

    def _postprocess_frontier_scores(
        self, sorted_frontiers: np.ndarray, sorted_values: List[float]
    ) -> Tuple[np.ndarray, List[float]]:
        if len(sorted_frontiers) == 0:
            return sorted_frontiers, sorted_values

        base = [float(v) for v in sorted_values]
        dist_terms = [0.0 for _ in base]
        loop_term = 0.0
        mode = "VLFM"
        self._loop_last_top_key = None
        self._loop_blacklist_triggered = False
        self._loop_blacklist_added_cells = []
        self._apply_loop_blacklist_to_value_map()

        prev_actions = getattr(self, "_prev_actions", None)
        use_prev_actions = (
            (not self._rotate_active())
            and isinstance(prev_actions, torch.Tensor)
            and prev_actions.dtype == torch.long
            and prev_actions.numel() > 0
        )
        if not use_prev_actions:
            self._loop_count_key = None
            self._loop_count = 0
            self._loop_prev_term = 0.0
            return sorted_frontiers, base

        robot_xy = self._observations_cache["robot_xy"]
        prev_id = int(prev_actions.reshape(-1)[0].item())

        # Loop-weighting behavior:
        # - rotate active: do not update (mu,s) and use base only
        # - prev action TURN_LEFT/RIGHT: do not update (mu,s), but apply loop boost
        # - prev action MOVE_FORWARD: update (mu,s), except skip the very first forward
        if prev_id == 1:  # MOVE_FORWARD
            if not self._loop_seen_first_forward:
                self._loop_seen_first_forward = True
            else:
                a = self._loop_alpha
                mu = (a * robot_xy + (1.0 - a) * self._loop_mu).astype(np.float32)
                self._loop_mu = mu
                diff = robot_xy - mu
                self._loop_s = float(a * float(np.dot(diff, diff)) + (1.0 - a) * self._loop_s)

        if self._use_distance_value:
            mode += "+DIST"
            for i, frontier in enumerate(sorted_frontiers):
                d = float(np.linalg.norm(frontier[:2] - robot_xy))
                dist_terms[i] = float(np.exp(-d)) if d <= self._distance_threshold else 0.0

        if self._use_loop_value and self._loop_seen_first_forward:
            mode += "+LOOP"
            denom_loop = self._loop_s + 1e-6
            d2_loop = float(np.dot(robot_xy - self._loop_mu, robot_xy - self._loop_mu))
            loop_term = float(np.exp(-d2_loop / denom_loop))

        if mode == "VLFM":
            return sorted_frontiers, base

        final = [b + d for (b, d) in zip(base, dist_terms)]
        for i, frontier in enumerate(sorted_frontiers):
            if any(cell in self._loop_blacklist_cells for cell in self._frontier_cells_rounded(frontier[:2])):
                final[i] = -1e9
        sorted_inds = np.argsort([-v for v in final])
        sorted_frontiers = np.array([sorted_frontiers[i] for i in sorted_inds])
        final = [final[i] for i in sorted_inds]
        if self._use_loop_value and self._loop_seen_first_forward and len(sorted_frontiers) > 0:
            top_frontier = sorted_frontiers[0]
            top_key = self._frontier_cell(top_frontier[:2])
            top_score = float(final[0])
            self._loop_last_top_key = top_key
            if loop_term >= self._loop_determine_threshold:
                if self._loop_blacklist_lock or top_score <= -1e8:
                    self._loop_count_key = None
                    self._loop_count = 0
                else:
                    if self._loop_count_key == top_key and loop_term >= self._loop_prev_term:
                        self._loop_count += 1
                    else:
                        self._loop_count_key = top_key
                        self._loop_count = 1
                    if self._loop_count >= self._loop_blacklist_count_threshold and self._loop_count_key is not None:
                        new_cells = [c for c in self._frontier_cells_rounded(top_frontier[:2]) if c not in self._loop_blacklist_cells]
                        self._loop_blacklist_cells.update(new_cells)
                        self._loop_blacklist_triggered = len(new_cells) > 0
                        self._loop_blacklist_added_cells = new_cells
                        self._apply_loop_blacklist_to_value_map()
                        for i, frontier in enumerate(sorted_frontiers):
                            if any(cell in self._loop_blacklist_cells for cell in self._frontier_cells_rounded(frontier[:2])):
                                final[i] = -1e9
                        sorted_inds = np.argsort([-v for v in final])
                        sorted_frontiers = np.array([sorted_frontiers[i] for i in sorted_inds])
                        final = [final[i] for i in sorted_inds]
                        self._loop_count_key = None
                        self._loop_count = 0
                        self._loop_blacklist_lock = True
            else:
                self._loop_count_key = None
                self._loop_count = 0
                self._loop_blacklist_lock = False
            self._loop_prev_term = loop_term
        else:
            self._loop_count_key = None
            self._loop_count = 0
            self._loop_prev_term = 0.0

        return sorted_frontiers, final

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        policy_info = super()._get_policy_info(detections)
        if self._loop_blacklist_cells:
            policy_info["loop_blacklist_cells"] = [[int(c[0]), int(c[1])] for c in sorted(self._loop_blacklist_cells)]
        if self._loop_blacklist_added_cells:
            policy_info["loop_blacklist_added_cells"] = [[int(c[0]), int(c[1])] for c in self._loop_blacklist_added_cells]

        if not self._visualize:
            return policy_info

        markers = []

        # Draw frontiers on to the cost map
        frontiers = self._observations_cache["frontier_sensor"]
        for frontier in frontiers:
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": self._frontier_color,
            }
            markers.append((frontier[:2], marker_kwargs))

        if not np.array_equal(self._last_goal, np.zeros(2)):
            # Draw the pointnav goal on to the cost map
            if any(np.array_equal(self._last_goal, frontier) for frontier in frontiers):
                color = self._selected__frontier_color
            else:
                color = self._target_object_color
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": color,
            }
            markers.append((self._last_goal, marker_kwargs))

        # Attach frontier values (top-K) for external logging/inspection
        if hasattr(self, "_last_sorted_frontiers") and len(self._last_sorted_values) > 0:
            top_k = min(5, len(self._last_sorted_values))
            frontier_values = []
            for idx in range(top_k):
                pt = self._last_sorted_frontiers[idx]
                val = float(self._last_sorted_values[idx])
                frontier_values.append(
                    {
                        "index": idx,
                        "x": float(pt[0]),
                        "y": float(pt[1]),
                        "value": val,
                    }
                )
            policy_info["frontier_values"] = frontier_values

        policy_info["value_map"] = cv2.cvtColor(
            self._value_map.visualize(markers, reduce_fn=self._vis_reduce_fn),
            cv2.COLOR_BGR2RGB,
        )

        return policy_info

    def _update_value_map(self) -> None:
        all_rgb = [i[0] for i in self._observations_cache["value_map_rgbd"]]
        cosines = [
            [
                self._itm.cosine(
                    rgb,
                    p.replace("target_object", self._target_object.replace("|", "/")),
                )
                for p in self._text_prompt.split(PROMPT_SEPARATOR)
            ]
            for rgb in all_rgb
        ]
        for cosine, (rgb, depth, tf, min_depth, max_depth, fov) in zip(
            cosines, self._observations_cache["value_map_rgbd"]
        ):
            self._value_map.update_map(np.array(cosine), depth, tf, min_depth, max_depth, fov)
        self._apply_loop_blacklist_to_value_map()

        self._value_map.update_agent_traj(
            self._observations_cache["robot_xy"],
            self._observations_cache["robot_heading"],
        )

    def _apply_distance_weighting(
        self, frontiers: np.ndarray, values: List[float]
    ) -> Tuple[np.ndarray, List[float]]:
        """Adjust semantic values with the distance-based exploration bonus."""
        if not self._use_distance_value or len(frontiers) == 0:
            return frontiers, values
        robot_xy = self._observations_cache.get("robot_xy")
        if robot_xy is None:
            return frontiers, values

        adjusted_values: List[float] = []
        for frontier, value in zip(frontiers, values):
            distance = float(np.linalg.norm(frontier[:2] - robot_xy))
            bonus = float(np.exp(-distance)) if distance <= self._distance_threshold else 0.0
            adjusted_values.append(float(value) + bonus)

        sorted_inds = np.argsort([-v for v in adjusted_values])
        sorted_frontiers = np.array([frontiers[i] for i in sorted_inds])
        sorted_values = [adjusted_values[i] for i in sorted_inds]
        return sorted_frontiers, sorted_values

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        raise NotImplementedError


class ITMPolicy(BaseITMPolicy):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._frontier_map: FrontierMap = FrontierMap()

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
        current_step: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        self._pre_step(observations, masks)
        if self._visualize:
            self._update_value_map()
        # return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)
        tmp = super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic,
                          current_step=current_step)
        return tmp

    def _reset(self) -> None:
        super()._reset()
        self._frontier_map.reset()

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        rgb = self._observations_cache["object_map_rgbd"][0][0]
        text = self._text_prompt.replace("target_object", self._target_object)
        self._frontier_map.update(frontiers, rgb, text)  # type: ignore
        sorted_frontiers, sorted_values = self._frontier_map.sort_waypoints()
        return self._postprocess_frontier_scores(sorted_frontiers, sorted_values)


class ITMPolicyV2(BaseITMPolicy):
    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
        current_step: int = 0,
    ) -> Any:
        self._pre_step(observations, masks)
        self._update_value_map()
        return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic,
                           current_step=current_step)

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        sorted_frontiers, sorted_values = self._value_map.sort_waypoints(frontiers, 0.5)
        return self._postprocess_frontier_scores(sorted_frontiers, sorted_values)


class ITMPolicyV3(ITMPolicyV2):
    def __init__(self, exploration_thresh: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._exploration_thresh = exploration_thresh

        def visualize_value_map(arr: np.ndarray) -> np.ndarray:
            # Get the values in the first channel
            first_channel = arr[:, :, 0]
            # Get the max values across the two channels
            max_values = np.max(arr, axis=2)
            # Create a boolean mask where the first channel is above the threshold
            mask = first_channel > exploration_thresh
            # Use the mask to select from the first channel or max values
            result = np.where(mask, first_channel, max_values)

            return result

        self._vis_reduce_fn = visualize_value_map  # type: ignore

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        sorted_frontiers, sorted_values = self._value_map.sort_waypoints(frontiers, 0.5, reduce_fn=self._reduce_values)

        return self._postprocess_frontier_scores(sorted_frontiers, sorted_values)

    def _reduce_values(self, values: List[Tuple[float, float]]) -> List[float]:
        """
        Reduce the values to a single value per frontier

        Args:
            values: A list of tuples of the form (target_value, exploration_value). If
                the highest target_value of all the value tuples is below the threshold,
                then we return the second element (exploration_value) of each tuple.
                Otherwise, we return the first element (target_value) of each tuple.

        Returns:
            A list of values, one per frontier.
        """
        target_values = [v[0] for v in values]
        max_target_value = max(target_values)

        if max_target_value < self._exploration_thresh:
            explore_values = [v[1] for v in values]
            return explore_values
        else:
            return [v[0] for v in values]
