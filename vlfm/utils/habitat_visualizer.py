# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from pathlib import Path
from frontier_exploration.utils.general_utils import xyz_to_habitat
from habitat.utils.common import flatten_dict
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.maps import MAP_TARGET_POINT_INDICATOR
from habitat.utils.visualizations.utils import overlay_text_to_image
from habitat_baselines.common.tensor_dict import TensorDict

from vlfm.utils.geometry_utils import transform_points
from vlfm.utils.img_utils import (
    reorient_rescale_map,
    resize_image,
    resize_images,
    rotate_image,
)
from vlfm.utils.visualization import add_text_to_image, pad_images

class HabitatVis:
    def __init__(self) -> None:
        self.rgb: List[np.ndarray] = []
        self.depth: List[np.ndarray] = []
        self.maps: List[np.ndarray] = []
        self.vis_maps: List[List[np.ndarray]] = []
        self.texts: List[List[str]] = []
        self.instance_imagegoal: List[np.ndarray] = []
        self.using_vis_maps = False
        self.using_annotated_rgb = False
        self.using_annotated_depth = False
        self._mem_debug_counter = 0
        self.instance_imagegoal_save_root: Optional[Path] = None
        self.save_instance_imagegoal_every_step = False
        self._saved_instance_imagegoal_steps: set[tuple[str, int]] = set()

    def set_instance_imagegoal_save_root(self, save_root: Optional[str]) -> None:
        self.instance_imagegoal_save_root = Path(save_root).expanduser() if save_root else None

    def reset(self) -> None:
        self.rgb = []
        self.depth = []
        self.maps = []
        self.vis_maps = []
        self.texts = []
        self.instance_imagegoal = []
        self.using_annotated_rgb = False
        self.using_annotated_depth = False
        self._mem_debug_counter = 0
        self._saved_instance_imagegoal_steps = set()

    @staticmethod
    def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
        arr = np.asarray(img)
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] != arr.shape[-1]:
            arr = np.transpose(arr[:3], (1, 2, 0))
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if arr.ndim == 3 and arr.shape[-1] > 3:
            arr = arr[..., :3]

        if arr.dtype != np.uint8:
            max_val = float(np.max(arr)) if arr.size else 0.0
            scale = 255.0 if max_val <= 1.0 else 1.0
            arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
        else:
            arr = arr.copy()
        return arr

    def _maybe_save_instance_imagegoal(
        self,
        instance_imagegoal: np.ndarray,
        info: Dict[str, Any],
        step_idx: int,
    ) -> None:
        episode_id = str(info.get("episode_id", "unknown"))
        step = int(info.get("step", step_idx))
        key = (episode_id, step if self.save_instance_imagegoal_every_step else 0)
        if key in self._saved_instance_imagegoal_steps:
            return

        root = self.instance_imagegoal_save_root
        info_root = info.get("data_backup_path")
        if root is None and info_root:
            root = Path(str(info_root)).expanduser()
        if root is None:
            return

        out_dir = root / episode_id / "instance_imagegoal"
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            return

        img_rgb = self._to_uint8_rgb(instance_imagegoal)
        fname = f"{step:04d}_instance_imagegoal.png" if self.save_instance_imagegoal_every_step else "instance_imagegoal.png"
        out_path = out_dir / fname
        try:
            cv2.imwrite(str(out_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            self._saved_instance_imagegoal_steps.add(key)
        except cv2.error:
            return

    def collect_data(
        self,
        observations: TensorDict,
        infos: List[Dict[str, Any]],
        policy_info: List[Dict[str, Any]],
    ) -> None:
        assert len(infos) == 1, "Only support one environment for now"

        if "annotated_depth" in policy_info[0]:
            depth = policy_info[0]["annotated_depth"]
            self.using_annotated_depth = True
        else:
            depth = (observations["depth"][0].cpu().numpy() * 255.0).astype(np.uint8)
            depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)
        depth = overlay_frame(depth, infos[0])
        self.depth.append(depth)

        if "annotated_rgb" in policy_info[0]:
            rgb = policy_info[0]["annotated_rgb"]
            self.using_annotated_rgb = True
        else:
            rgb = observations["rgb"][0].cpu().numpy()
        self.rgb.append(rgb)

        top_down_map = infos[0]["top_down_map"]
        original_map = top_down_map.get("map")
        if original_map is not None:
            top_down_map["map"] = original_map.copy()

        overlay_rotate_history_on_map(infos[0], policy_info[0])
        # Visualize target point cloud on the map
        color_point_cloud_on_map(infos, policy_info)

        map = maps.colorize_draw_agent_and_fit_to_height(infos[0]["top_down_map"], self.depth[0].shape[0])
        if original_map is not None:
            top_down_map["map"] = original_map
        self.maps.append(map)
        vis_map_imgs = [
            self._reorient_rescale_habitat_map(infos, policy_info[0][vkey])
            for vkey in ["obstacle_map", "value_map"]
            if vkey in policy_info[0]
        ]
        cell_top_down_map = None
        info_root = infos[0].get("data_backup_path")
        if info_root:
            cell_path = Path(str(info_root)).expanduser() / str(infos[0].get("episode_id", "unknown")) / "self_questioner" / "cell_top_down_map.png"
            if cell_path.exists():
                cell_top_down_map = cv2.imread(str(cell_path), cv2.IMREAD_COLOR)
                if cell_top_down_map is not None:
                    cell_top_down_map = cv2.cvtColor(cell_top_down_map, cv2.COLOR_BGR2RGB)
        if cell_top_down_map is not None:
            instance_imagegoal = cell_top_down_map
        elif "instance_imagegoal" in policy_info[0]:
            instance_imagegoal = np.asarray(policy_info[0]["instance_imagegoal"])
        else:
            instance_imagegoal = observations["instance_imagegoal"].squeeze().detach().cpu().numpy()
        vis_map_imgs.append(instance_imagegoal)
        if vis_map_imgs:
            self.using_vis_maps = True
            self.vis_maps.append(vis_map_imgs)
        text = [
            policy_info[0][text_key]
            for text_key in policy_info[0].get("render_below_images", [])
            if text_key in policy_info[0]
        ]
        self.texts.append(text)
        step_idx = len(self.instance_imagegoal)
        self._maybe_save_instance_imagegoal(instance_imagegoal, infos[0], step_idx)
        self.instance_imagegoal.append(instance_imagegoal)

    def flush_frames(self, failure_cause: str) -> List[np.ndarray]:
        """Flush all frames and return them"""
        # Because the annotated frames are actually one step delayed, pop the first one
        # and add a placeholder frame to the end (gets removed anyway)
        if self.using_annotated_rgb is not None:
            self.rgb.append(self.rgb.pop(0))
        if self.using_annotated_depth is not None:
            self.depth.append(self.depth.pop(0))
        if self.using_vis_maps:  # Cost maps are also one step delayed
            self.vis_maps.append(self.vis_maps.pop(0))

        frames = []
        num_frames = len(self.depth) - 1  # last frame is from next episode, remove it
        for i in range(num_frames):
            frame = self._create_frame(
                self.depth[i],
                self.rgb[i],
                self.maps[i],
                self.vis_maps[i],
                self.texts[i],
                self.instance_imagegoal[i],
            )
            failure_cause_text = "Failure cause: " + failure_cause
            frame = add_text_to_image(frame, failure_cause_text, top=True)
            frames.append(frame)

        if len(frames) > 0:
            frames = pad_images(frames, pad_from_top=True)

        frames = [resize_image(f, 480 * 2) for f in frames]

        self._log_buffer_usage(reason="flush_frames", num_frames=len(frames))
        self.reset()

        return frames

    def _log_buffer_usage(self, reason: str, num_frames: int) -> None:
        """
        Debug-only logging: approximate memory footprint of buffered frames.
        Keeps overhead tiny by sampling every call (flush).
        """
        self._mem_debug_counter += 1
        try:
            import psutil  # local import to avoid global dependency if missing
        except Exception:
            return
        if num_frames <= 0:
            return
        # Estimate numpy buffers
        buf_bytes = 0
        for arr_list in (self.rgb, self.depth, self.maps):
            for arr in arr_list:
                if hasattr(arr, "nbytes"):
                    buf_bytes += arr.nbytes
        for vis_list in self.vis_maps:
            for arr in vis_list:
                if hasattr(arr, "nbytes"):
                    buf_bytes += arr.nbytes
        proc = psutil.Process()
        rss = proc.memory_info().rss
        total_ram = psutil.virtual_memory().total
        rss_pct = (rss / total_ram) * 100 if total_ram else 0.0
        buf_pct = (buf_bytes / rss) * 100 if rss else 0.0
        print(
            f"[MEM][HabitatVis:{reason}] sample={self._mem_debug_counter} "
            f"frames={num_frames} rss={self._bytes_human(rss)} ({rss_pct:.2f}%) "
            f"buf={self._bytes_human(buf_bytes)} ({buf_pct:.2f}% of RSS) "
            f"rgb={len(self.rgb)} depth={len(self.depth)} maps={len(self.maps)} vis_maps={len(self.vis_maps)}"
        )

    @staticmethod
    def _bytes_human(num_bytes: float) -> str:
        units = ["B", "KB", "MB", "GB", "TB"]
        if num_bytes <= 0:
            return "0B"
        idx = min(int(np.log(num_bytes) / np.log(1024)), len(units) - 1)
        scaled = num_bytes / (1024 ** idx)
        return f"{scaled:.2f}{units[idx]}"

    @staticmethod
    def _reorient_rescale_habitat_map(infos: List[Dict[str, Any]], vis_map: np.ndarray) -> np.ndarray:
        # Rotate the cost map to match the agent's orientation at the start
        # of the episode
        start_yaw = infos[0]["start_yaw"]
        if start_yaw != 0.0:
            vis_map = rotate_image(vis_map, start_yaw, border_value=(255, 255, 255))

        # Rotate the image 90 degrees if the corresponding map is taller than it is wide
        habitat_map = infos[0]["top_down_map"]["map"]
        if habitat_map.shape[0] > habitat_map.shape[1]:
            vis_map = np.rot90(vis_map, 1)

        vis_map = reorient_rescale_map(vis_map)

        return vis_map

    @staticmethod
    def _create_frame(
        depth: np.ndarray,
        rgb: np.ndarray,
        map: np.ndarray,
        vis_map_imgs: List[np.ndarray],
        text: List[str],
        instance_imagegoal: np.ndarray,
    ) -> np.ndarray:
        """Create a frame using all the given images.

        First, the depth and rgb images are stacked vertically. Then, all the maps are
        combined as a separate images. Then these two images should be stitched together
        horizontally (depth-rgb on the left, maps on the right).

        The combined map image contains two rows of images and at least one column.
        First, the 'map' argument is at the top left, then the first element of the
        'vis_map_imgs' argument is at the bottom left. If there are more than one
        element in 'vis_map_imgs', then the second element is at the top right, the
        third element is at the bottom right, and so on.

        Args:
            depth: The depth image (H, W, 3).
            rgb: The rgb image (H, W, 3).
            map: The map image, a 3-channel rgb image, but can have different shape from
                depth and rgb.
            vis_map_imgs: A list of other map images. Each are 3-channel rgb images, but
                can have different sizes.
            text: A list of strings to be rendered above the images.
        Returns:
            np.ndarray: The combined frame image.
        """
        # Stack depth and rgb images vertically
        depth_rgb = np.vstack((depth, rgb))

        # Prepare the list of images to be combined
        map_imgs = [map] + vis_map_imgs
        if len(map_imgs) % 2 == 1:
            # If there are odd number of images, add a placeholder image
            map_imgs.append(np.ones_like(map_imgs[-1]) * 255)

        even_index_imgs = map_imgs[::2]
        odd_index_imgs = map_imgs[1::2]
        top_row = np.hstack(resize_images(even_index_imgs, match_dimension="height"))
        bottom_row = np.hstack(resize_images(odd_index_imgs, match_dimension="height"))

        frame = np.vstack(resize_images([top_row, bottom_row], match_dimension="width"))
        depth_rgb, frame = resize_images([depth_rgb, frame], match_dimension="height")
        frame = np.hstack((depth_rgb, frame))

        # Add text to the top of the frame
        for t in text[::-1]:
            frame = add_text_to_image(frame, t, top=True)

        return frame


def sim_xy_to_grid_xy(
    upper_bound: Tuple[int, int],
    lower_bound: Tuple[int, int],
    grid_resolution: Tuple[int, int],
    sim_xy: np.ndarray,
    remove_duplicates: bool = True,
) -> np.ndarray:
    """Converts simulation coordinates to grid coordinates.

    Args:
        upper_bound (Tuple[int, int]): The upper bound of the grid.
        lower_bound (Tuple[int, int]): The lower bound of the grid.
        grid_resolution (Tuple[int, int]): The resolution of the grid.
        sim_xy (np.ndarray): A numpy array of 2D simulation coordinates.
        remove_duplicates (bool): Whether to remove duplicate grid coordinates.

    Returns:
        np.ndarray: A numpy array of 2D grid coordinates.
    """
    grid_size = np.array(
        [
            abs(upper_bound[1] - lower_bound[1]) / grid_resolution[0],
            abs(upper_bound[0] - lower_bound[0]) / grid_resolution[1],
        ]
    )
    grid_xy = ((sim_xy - lower_bound[::-1]) / grid_size).astype(int)

    if remove_duplicates:
        grid_xy = np.unique(grid_xy, axis=0)

    return grid_xy


def color_point_cloud_on_map(infos: List[Dict[str, Any]], policy_info: List[Dict[str, Any]]) -> None:
    if len(policy_info[0]["target_point_cloud"]) == 0:
        return

    upper_bound = infos[0]["top_down_map"]["upper_bound"]
    lower_bound = infos[0]["top_down_map"]["lower_bound"]
    grid_resolution = infos[0]["top_down_map"]["grid_resolution"]
    tf_episodic_to_global = infos[0]["top_down_map"]["tf_episodic_to_global"]

    cloud_episodic_frame = policy_info[0]["target_point_cloud"][:, :3]
    cloud_global_frame_xyz = transform_points(tf_episodic_to_global, cloud_episodic_frame)
    cloud_global_frame_habitat = xyz_to_habitat(cloud_global_frame_xyz)
    cloud_global_frame_habitat_xy = cloud_global_frame_habitat[:, [2, 0]]

    grid_xy = sim_xy_to_grid_xy(
        upper_bound,
        lower_bound,
        grid_resolution,
        cloud_global_frame_habitat_xy,
        remove_duplicates=True,
    )

    new_map = infos[0]["top_down_map"]["map"].copy()
    new_map[grid_xy[:, 0], grid_xy[:, 1]] = MAP_TARGET_POINT_INDICATOR

    infos[0]["top_down_map"]["map"] = new_map


ROTATE_INDICATOR_VALUE = 250
maps.TOP_DOWN_MAP_COLORS[ROTATE_INDICATOR_VALUE] = np.array([255, 0, 0], dtype=np.uint8)


def overlay_rotate_history_on_map(info: Dict[str, Any], policy_info: Dict[str, Any]) -> None:
    top_down = info.get("top_down_map")
    if top_down is None:
        return
    rotate_locations = policy_info.get("rotate_locations", [])
    radius_m = policy_info.get("rotate_radius_m")
    if not rotate_locations or radius_m is None:
        return

    tf_episodic_to_global = top_down.get("tf_episodic_to_global")
    upper_bound = top_down.get("upper_bound")
    lower_bound = top_down.get("lower_bound")
    grid_resolution = top_down.get("grid_resolution")
    base_map = top_down.get("map")

    if (
        tf_episodic_to_global is None
        or upper_bound is None
        or lower_bound is None
        or grid_resolution is None
        or base_map is None
    ):
        return

    rotate_xy = np.asarray(rotate_locations, dtype=np.float32)
    if rotate_xy.size == 0 or rotate_xy.shape[-1] != 2:
        top_down.pop("rotate_circles", None)
        return

    rotate_xyz = np.concatenate(
        [rotate_xy, np.zeros((rotate_xy.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    global_xyz = transform_points(tf_episodic_to_global, rotate_xyz)
    habitat_xyz = xyz_to_habitat(global_xyz)
    habitat_xy = habitat_xyz[:, [2, 0]]

    grid_xy = sim_xy_to_grid_xy(
        upper_bound,
        lower_bound,
        grid_resolution,
        habitat_xy,
        remove_duplicates=False,
    )

    meters_per_cell = np.array(
        [
            abs(upper_bound[1] - lower_bound[1]) / grid_resolution[0],
            abs(upper_bound[0] - lower_bound[0]) / grid_resolution[1],
        ],
        dtype=np.float32,
    )
    mean_cell_size = float(np.mean(meters_per_cell))
    if mean_cell_size <= 0:
        return
    radius_px = max(1, int(round(radius_m / mean_cell_size)))

    height, width = base_map.shape[:2]
    for row, col in grid_xy:
        if 0 <= row < height and 0 <= col < width:
            cv2.circle(
                base_map,
                (int(col), int(row)),
                radius_px,
                color=int(ROTATE_INDICATOR_VALUE),
                thickness=1,
            )
            cv2.circle(
                base_map,
                (int(col), int(row)),
                2,
                color=int(ROTATE_INDICATOR_VALUE),
                thickness=-1,
            )


def overlay_frame(frame: np.ndarray, info: Dict[str, Any], additional: Optional[List[str]] = None) -> np.ndarray:
    """
    Renders text from the `info` dictionary to the `frame` image.
    """

    lines = []
    flattened_info = flatten_dict(info)
    for k, v in flattened_info.items():
        if isinstance(v, str):
            lines.append(f"{k}: {v}")
        else:
            try:
                lines.append(f"{k}: {v:.2f}")
            except TypeError:
                pass
    if additional is not None:
        lines.extend(additional)

    frame = overlay_text_to_image(frame, lines, font_size=0.25)

    return frame
