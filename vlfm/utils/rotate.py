from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch


class RotateController:
    def __init__(
        self,
        enabled: bool,
        rotate_radius: float,
        rotate_openness_threshold: float,
        num_angle_bin: int,
        save_panoramas: bool,
        panorama_output_dir: str,
    ) -> None:
        self.enabled = bool(enabled)
        self.rotate_radius = float(rotate_radius)
        self.rotate_openness_threshold = float(rotate_openness_threshold)
        self.num_angle_bin = int(num_angle_bin)
        self.save_panoramas = bool(save_panoramas)
        self.panorama_output_dir = panorama_output_dir
        self.openness_use_cuda = torch.cuda.is_available()
        self._openness_angle_bin_cache: Dict[int, int] = {}
        self.reset()

    def reset(self) -> None:
        self.rotate_used = 0
        self.in_progress = False
        self.steps_remaining = 0
        self.last_openness_score = -1.0
        self.rotate_locations: List[np.ndarray] = []
        self.panorama_frames: List[np.ndarray] = []
        self.panorama_mode: Optional[str] = None
        self.panorama_index = 0

    def active(self) -> bool:
        return self.in_progress and self.steps_remaining > 0

    def supported(self, obstacle_map: Optional[object]) -> bool:
        return self.enabled and obstacle_map is not None

    def record_initial_location(self, robot_xy: Optional[np.ndarray]) -> None:
        if robot_xy is not None and not self.rotate_locations:
            self.rotate_locations.append(robot_xy.copy())

    def record_location(self, robot_xy: np.ndarray) -> None:
        self.rotate_locations.append(robot_xy.copy())

    def record_panorama_frame(self, frame: Optional[np.ndarray], mode: str) -> None:
        if not self.save_panoramas or frame is None:
            return
        if self.panorama_mode != mode:
            self.panorama_mode = mode
            self.panorama_frames = []
        self.panorama_frames.append(frame.copy())

    def _concat_panoramic(self, images: List[np.ndarray]) -> Optional[np.ndarray]:
        if len(images) == 0:
            return None
        height, width = images[0].shape[0], images[0].shape[1]
        background_image = np.zeros((2 * height + 30, 3 * width + 40, 3), np.uint8)
        copy_images = np.array(images, dtype=np.uint8)
        for i in range(len(copy_images)):
            if i % 2 != 0:
                row = i // 6
                col = (i % 6) // 2
                copy_images[i] = cv2.putText(
                    copy_images[i],
                    f"Direction {i}",
                    (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (255, 0, 0),
                    6,
                    cv2.LINE_AA,
                )
                start_y = 10 * (row + 1) + row * height
                start_x = col * width + col * 10
                background_image[start_y : start_y + height, start_x : start_x + width, :] = copy_images[i]
        return background_image

    def finalize_panorama(self, mode: Optional[str], num_steps: int, output_dir: str) -> Optional[str]:
        if not self.save_panoramas:
            self.panorama_frames = []
            self.panorama_mode = None
            return None
        if not self.panorama_frames:
            self.panorama_mode = None
            return None
        if mode is not None and self.panorama_mode != mode:
            return None
        panorama = self._concat_panoramic(self.panorama_frames)
        self.panorama_frames = []
        current_mode = self.panorama_mode
        self.panorama_mode = None
        if panorama is None or current_mode is None:
            return None
        filename = f"{num_steps:04d}_{current_mode}_{self.panorama_index:04d}.png"
        self.panorama_index += 1
        path = f"{output_dir}/{filename}"
        panorama_bgr = cv2.cvtColor(panorama, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, panorama_bgr)
        return path

    def compute_openness(self, robot_xy: np.ndarray, obstacle_map: object) -> Tuple[float, str]:
        obstacle_mask = obstacle_map._navigable_map == 0
        obstacle_px = np.nonzero(obstacle_mask)
        if obstacle_px[0].size == 0:
            return 1.0, "GPU" if self.openness_use_cuda else "CPU"

        robot_px = obstacle_map._xy_to_px(robot_xy.reshape(1, 2))[0]
        dx = obstacle_px[1].astype(np.int64) - int(robot_px[0])
        dy = obstacle_px[0].astype(np.int64) - int(robot_px[1])
        valid = np.logical_or(dx != 0, dy != 0)
        if not np.any(valid):
            return 1.0, "GPU" if self.openness_use_cuda else "CPU"
        dx = dx[valid]
        dy = dy[valid]

        span = 2 * obstacle_map.size + 1
        keys = (dy + obstacle_map.size) * span + (dx + obstacle_map.size)
        uniq_keys, inv = np.unique(keys, return_inverse=True)
        uniq_dx = (uniq_keys % span) - obstacle_map.size
        uniq_dy = (uniq_keys // span) - obstacle_map.size
        uniq_bins = np.empty(len(uniq_keys), dtype=np.int64)
        miss_idx: List[int] = []
        for i, key in enumerate(uniq_keys):
            cached = self._openness_angle_bin_cache.get(int(key))
            if cached is None:
                miss_idx.append(i)
            else:
                uniq_bins[i] = cached

        backend = "GPU" if self.openness_use_cuda else "CPU"
        if miss_idx:
            miss_idx_arr = np.asarray(miss_idx, dtype=np.int64)
            miss_dx = uniq_dx[miss_idx_arr]
            miss_dy = uniq_dy[miss_idx_arr]
            if self.openness_use_cuda:
                dx_t = torch.from_numpy(miss_dx.astype(np.float32)).to(device="cuda")
                dy_t = torch.from_numpy(miss_dy.astype(np.float32)).to(device="cuda")
                angles = torch.atan2(dy_t, dx_t)
                bins = torch.remainder(
                    torch.floor((angles + np.pi) * (self.num_angle_bin / (2 * np.pi))).to(torch.long),
                    self.num_angle_bin,
                )
                miss_bins = bins.cpu().numpy().astype(np.int64)
            else:
                angles = np.arctan2(miss_dy.astype(np.float32), miss_dx.astype(np.float32))
                miss_bins = np.mod(
                    np.floor((angles + np.pi) * (self.num_angle_bin / (2 * np.pi))).astype(np.int64),
                    self.num_angle_bin,
                )
            uniq_bins[miss_idx_arr] = miss_bins
            for i, bin_idx in zip(miss_idx_arr, miss_bins):
                self._openness_angle_bin_cache[int(uniq_keys[i])] = int(bin_idx)

        blocked = np.zeros(self.num_angle_bin, dtype=bool)
        blocked[uniq_bins[inv]] = True
        return 1.0 - float(np.mean(blocked)), backend

    def try_start(
        self, robot_xy: Optional[np.ndarray], obstacle_map: Optional[object], rotation_turns: int
    ) -> Tuple[bool, str, float, str]:
        if not self.supported(obstacle_map) or self.in_progress or robot_xy is None:
            return False, "unsupported", -1.0, "GPU" if self.openness_use_cuda else "CPU"

        for prev_rotate in self.rotate_locations:
            if np.linalg.norm(robot_xy - prev_rotate) < self.rotate_radius:
                return False, "near_prev", -1.0, "GPU" if self.openness_use_cuda else "CPU"

        openness_score, backend = self.compute_openness(robot_xy, obstacle_map)
        self.last_openness_score = float(openness_score)
        if not (openness_score > self.rotate_openness_threshold):
            return False, "low_openness", openness_score, backend

        self.steps_remaining = max(int(rotation_turns), 1)
        self.in_progress = True
        self.rotate_used += 1
        self.record_location(robot_xy)
        return True, "started", openness_score, backend

    def step_action(
        self,
        rotate_fn: Callable[[], object],
        frame: Optional[np.ndarray],
        num_steps: int,
        output_dir: str,
    ) -> Tuple[object, Optional[str]]:
        if not self.in_progress:
            raise RuntimeError("Called step_action() while rotate is not active.")
        self.record_panorama_frame(frame, "rotate")
        action = rotate_fn()
        self.steps_remaining -= 1
        saved_path: Optional[str] = None
        if self.steps_remaining <= 0:
            self.in_progress = False
            self.steps_remaining = 0
            saved_path = self.finalize_panorama("rotate", num_steps, output_dir)
        return action, saved_path
