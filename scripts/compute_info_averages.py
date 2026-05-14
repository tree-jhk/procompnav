#!/usr/bin/env python3
"""
Compute per-key averages for numeric values stored in info.json files under evaluation folders.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Set, Tuple


_REACHED_PNG_RE = re.compile(r"^(?P<step>\d+)_reached_.*\.png$")
_MERGED_GROUP_RE = re.compile(r"^group_\d+_merged$")


def _split_episode_tokens(value: str) -> List[str]:
    tokens: List[str] = []
    for token in re.split(r"[,\s]+", value.strip()):
        if token:
            tokens.append(token)
    return tokens


def _load_episode_id_filter(
    episode_id_args: List[str],
    episode_id_files: List[str],
) -> Set[str] | None:
    def _parse_list_literal_or_tokens(raw: str) -> Set[str]:
        text = raw.strip()
        if not text:
            return set()

        parsed_obj = None
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed_obj = parser(text)
                break
            except Exception:
                continue

        if isinstance(parsed_obj, (list, tuple, set)):
            tokens: Set[str] = set()
            for item in parsed_obj:
                if item is None:
                    continue
                tokens.update(_split_episode_tokens(str(item)))
            return tokens

        return set(_split_episode_tokens(text))

    episode_ids: Set[str] = set()
    for raw in episode_id_args:
        episode_ids.update(_split_episode_tokens(raw))

    for source in episode_id_files:
        source = source.strip()
        if not source:
            continue
        source_path = Path(source).expanduser()
        if source_path.is_file():
            path = source_path.resolve()
            text = path.read_text(encoding="utf-8")
            stripped = text.strip()
            if stripped and stripped[0] in "[({":
                episode_ids.update(_parse_list_literal_or_tokens(stripped))
                continue

            for raw_line in text.splitlines():
                line = raw_line.split("#", 1)[0].strip()
                if not line:
                    continue
                episode_ids.update(_parse_list_literal_or_tokens(line))
            continue

        # Fallback: treat value itself as a list literal (e.g. "[1,2,3]") or CSV/whitespace tokens.
        if ("/" in source or "\\" in source) and not _parse_list_literal_or_tokens(source):
            raise FileNotFoundError(f"Episode ID file not found: {source}")
        episode_ids.update(_parse_list_literal_or_tokens(source))

    return episode_ids if episode_ids else None


def _collect_first_reached_steps(
    root_dir: Path,
    allowed_episode_ids: Set[str] | None = None,
) -> Dict[str, int]:
    """Return {episode_id: first_reached_step} from observation/*_reached_*.png filenames."""
    first_steps: Dict[str, int] = {}
    for episode_dir in sorted(root_dir.iterdir()):
        if not episode_dir.is_dir():
            continue
        if allowed_episode_ids is not None and episode_dir.name not in allowed_episode_ids:
            continue
        obs_dir = episode_dir / "observation"
        if not obs_dir.is_dir():
            continue
        min_step: int | None = None
        try:
            for image_path in obs_dir.iterdir():
                if not image_path.is_file():
                    continue
                match = _REACHED_PNG_RE.match(image_path.name)
                if not match:
                    continue
                step = int(match.group("step"))
                if min_step is None or step < min_step:
                    min_step = step
        except OSError as exc:
            print(f"Skipping observation scan for {episode_dir}: {exc}")
            continue
        if min_step is not None:
            first_steps[episode_dir.name] = min_step
    return first_steps


def _collect_min_pbp_selected_first_seen_steps(
    root_dir: Path, episode_ids: Iterable[str]
) -> Dict[str, int]:
    steps: Dict[str, int] = {}
    for episode_id in sorted(episode_ids):
        instances_path = root_dir / episode_id / "instances.jsonl"
        if not instances_path.is_file():
            continue
        min_step: int | None = None
        with instances_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                instance = json.loads(line)
                if instance.get("pbp_selected") is not True:
                    continue
                first_seen_step = instance.get("first_seen_step")
                if not isinstance(first_seen_step, (int, float)):
                    continue
                step = int(first_seen_step)
                if min_step is None or step < min_step:
                    min_step = step
        if min_step is not None:
            steps[episode_id] = min_step
    return steps


def _collect_true_episode_ids_by_key(
    root_dir: Path,
    key: str,
    allowed_episode_ids: Set[str] | None = None,
) -> Set[str]:
    """Return episode IDs where info.json[key] is truthy (bool True or numeric >= 1)."""
    episode_ids: Set[str] = set()
    eps = 1e-9
    for child in sorted(root_dir.iterdir()):
        if not child.is_dir():
            continue
        if allowed_episode_ids is not None and child.name not in allowed_episode_ids:
            continue
        info_path = child / "info.json"
        if not info_path.is_file():
            continue
        try:
            with info_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        value = data.get(key)
        if value is True:
            episode_ids.add(child.name)
        elif isinstance(value, (int, float)) and float(value) >= 1.0 - eps:
            episode_ids.add(child.name)
    return episode_ids


def _compute_binned_frequencies(
    steps: List[int],
    *,
    bin_size: int,
) -> tuple[list[int], list[int]]:
    if not steps:
        return ([], [])
    max_step = max(steps)
    max_bin = max_step // bin_size
    bin_starts = [i * bin_size for i in range(max_bin + 1)]
    freq = [0] * len(bin_starts)
    for step in steps:
        idx = min(step // bin_size, max_bin)
        freq[idx] += 1
    return (bin_starts, freq)


def _nice_tick_step(max_value: int, *, target_ticks: int = 6) -> int:
    if max_value <= 0:
        return 1
    raw = max_value / max(1, target_ticks)
    mag = 1
    while raw >= 10:
        raw /= 10
        mag *= 10
    if raw <= 1:
        base = 1
    elif raw <= 2:
        base = 2
    elif raw <= 5:
        base = 5
    else:
        base = 10
    return base * mag


def _save_first_reached_histogram(
    output_dir: Path,
    first_steps: List[int],
    *,
    bin_size: int = 50,
    total_episodes: int | None = None,
) -> Path | None:
    if not first_steps:
        return None

    bin_starts, freq = _compute_binned_frequencies(first_steps, bin_size=bin_size)

    out_path = output_dir / f"first_reached_step_hist_bin{bin_size}.png"
    tick_stride = max(1, len(bin_starts) // 20)

    try:
        from PIL import Image, ImageDraw, ImageFont

        margin_left, margin_right, margin_top, margin_bottom = 80, 25, 35, 65
        plot_h = 360
        max_width = 5000
        bar_w = max(8, min(35, (max_width - margin_left - margin_right) // max(1, len(bin_starts))))
        gap = max(2, bar_w // 8)
        img_w = margin_left + margin_right + bar_w * len(bin_starts)
        img_h = margin_top + margin_bottom + plot_h

        img = Image.new("RGB", (img_w, img_h), "white")
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        x0 = margin_left
        y0 = margin_top
        x1 = img_w - margin_right
        y1 = margin_top + plot_h
        draw.line((x0, y1, x1, y1), fill="black", width=2)
        draw.line((x0, y0, x0, y1), fill="black", width=2)

        max_freq = max(freq) if freq else 0
        tick_step = _nice_tick_step(max_freq, target_ticks=7)
        for y_val in range(0, max_freq + 1, tick_step):
            if max_freq <= 0:
                continue
            y = y1 - int((y_val / max_freq) * (plot_h - 10))
            draw.line((x0 - 5, y, x0, y), fill="black", width=1)
            draw.line((x0, y, x1, y), fill=(220, 220, 220), width=1)
            label = str(y_val)
            tw = draw.textlength(label, font=font)
            draw.text((x0 - 10 - tw, y - 6), label, fill="black", font=font)

        for i, count in enumerate(freq):
            if max_freq <= 0:
                bar_h = 0
            else:
                bar_h = int((count / max_freq) * (plot_h - 10))
            bx0 = margin_left + i * bar_w
            left = bx0 + gap // 2
            right = bx0 + bar_w - 1 - (gap - gap // 2)
            if right < left:
                left = bx0
                right = bx0 + bar_w - 1
            by1 = y1 - 1
            by0 = by1 - bar_h
            if bar_h > 0:
                try:
                    draw.rectangle(
                        (left, by0, right, by1),
                        fill=(52, 120, 246),
                        outline="black",
                        width=2,
                    )
                except TypeError:
                    draw.rectangle((left, by0, right, by1), fill=(52, 120, 246), outline="black")
                    draw.rectangle((left + 1, by0 + 1, right - 1, by1 - 1), outline="black")

        title = f"first_reached_step histogram (bin={bin_size}, reached={len(first_steps)}"
        if total_episodes is not None:
            title += f"/{total_episodes}"
        title += ")"
        draw.text((margin_left, 5), title, fill="black", font=font)
        draw.text((5, y0), "reached episodes", fill="black", font=font)
        xlabel = "first_reached_step / 10"
        xw = draw.textlength(xlabel, font=font)
        draw.text(((img_w - xw) / 2, y1 + 32), xlabel, fill="black", font=font)

        for i in range(0, len(bin_starts), tick_stride):
            bx = margin_left + i * bar_w
            draw.line((bx, y1, bx, y1 + 5), fill="black", width=1)
            label = str(bin_starts[i] // 10)
            tw = draw.textlength(label, font=font)
            draw.text((bx - tw / 2, y1 + 8), label, fill="black", font=font)

        img.save(out_path)
        return out_path
    except Exception as exc:
        print(f"  [WARN] PIL plot failed; trying matplotlib fallback: {exc}")

    try:
        import os

        os.environ.setdefault("KMP_DISABLE_SHM", "1")
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        fig_w = max(6.0, min(18.0, 0.35 * len(bin_starts)))
        plt.figure(figsize=(fig_w, 4.0))
        plt.bar(bin_starts, freq, width=bin_size * 0.9, align="edge")
        title = f"first_reached_step histogram (bin={bin_size}, reached={len(first_steps)}"
        if total_episodes is not None:
            title += f"/{total_episodes}"
        title += ")"
        plt.title(title)
        plt.xlabel(f"first_reached_step bin (size={bin_size})")
        plt.ylabel("reached episode count")
        plt.xticks(bin_starts[::tick_stride], rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return out_path
    except Exception as exc:
        print(f"  [WARN] matplotlib fallback failed; skipping plot: {exc}")
        return None


def _gather_numeric_values(
    info_path_iter: Iterable[Path],
) -> Tuple[
    Dict[str, float],
    Dict[str, int],
    Dict[str, int],
    Set[str],
    List[str],
    Dict[str, int],
    Dict[str, int],
    Dict[str, List[str]],
    Dict[str, float],
    List[str],
    List[str],
    List[str],
    Dict[str, float],
    Dict[str, float],
    Dict[str, int],
    Dict[str, float],
    Dict[str, float],
]:
    """Aggregate numeric values found in the provided info.json paths."""
    totals: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    true_counts: Dict[str, int] = defaultdict(int)
    binary_possible: Dict[str, bool] = {}
    EPS = 1e-9
    visited_episodes: List[str] = []
    detected_object_counts: Dict[str, int] = {}
    target_proximity_counts: Dict[str, int] = defaultdict(int)
    target_proximity_episode_ids: Dict[str, List[str]] = defaultdict(list)
    low_step_episodes: Dict[str, float] = {}
    success_episode_ids: List[str] = []
    distractor_success_episode_ids: List[str] = []
    possible_target_episode_ids: List[str] = []
    total_steps_by_episode: Dict[str, float] = {}
    num_detected_objects_by_episode: Dict[str, float] = {}
    num_merged_instances_by_episode: Dict[str, int] = {}
    spl_by_episode: Dict[str, float] = {}
    total_questions_by_episode: Dict[str, float] = {}

    for info_path in info_path_iter:
        try:
            with info_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Skipping {info_path}: {exc}")
            continue

        pbp_results = data.get("pbp_results")
        # Some eval runs store question depth under pbp_results[0]["pbp_depth"].
        if "pbp_depth" not in data:
            if isinstance(pbp_results, list) and pbp_results and isinstance(pbp_results[0], dict):
                pbp_depth = pbp_results[0].get("pbp_depth")
                if isinstance(pbp_depth, (int, float)):
                    data["pbp_depth"] = pbp_depth
        # re_explore: number of additional PBP executions after the first run.
        re_explore = 0
        if isinstance(pbp_results, list) and pbp_results:
            execution_indices: List[int] = []
            for pbp_result in pbp_results:
                if not isinstance(pbp_result, dict):
                    continue
                execution_index = pbp_result.get("execution_index")
                if isinstance(execution_index, (int, float)):
                    execution_indices.append(int(execution_index))
            if execution_indices:
                re_explore = max(0, max(execution_indices))
            else:
                re_explore = max(0, len(pbp_results) - 1)
        data["re_explore"] = int(re_explore)

        visited_flag = False
        success_flag = False
        distractor_success_flag = False
        episode_dir = info_path.parent
        episode_id = episode_dir.name

        for key, value in data.items():
            is_one = False

            if isinstance(value, bool):
                # Treat booleans as numbers (0 or 1) to preserve reached/stop rate semantics.
                numeric_value = float(value)
                binary_possible[key] = True
                if value:
                    is_one = True
            elif isinstance(value, (int, float)):
                numeric_value = float(value)
                if key not in binary_possible:
                    binary_possible[key] = True
                if abs(numeric_value) < EPS:
                    is_one = False
                elif abs(numeric_value - 1.0) < EPS:
                    is_one = True
                else:
                    binary_possible[key] = False
            else:
                continue

            totals[key] += numeric_value
            counts[key] += 1
            if is_one:
                true_counts[key] += 1
                if key == "visited_target":
                    visited_flag = True
                if key == "success":
                    success_flag = True
                if key == "distractor_success":
                    distractor_success_flag = True
            elif key == "visited_target" and numeric_value >= 1.0 - EPS:
                # Catch non-binary but truthy visited entries.
                visited_flag = True
            elif key == "success" and numeric_value >= 1.0 - EPS:
                success_flag = True
            elif key == "distractor_success" and numeric_value >= 1.0 - EPS:
                distractor_success_flag = True

        total_steps = data.get("total_steps")
        if isinstance(total_steps, (int, float)):
            total_steps_by_episode[episode_id] = float(total_steps)
            if total_steps < 400:
                low_step_episodes[episode_id] = float(total_steps)
        num_detected_objects = data.get("num_detected_objects")
        if isinstance(num_detected_objects, (int, float)):
            num_detected_objects_by_episode[episode_id] = float(num_detected_objects)
        spl = data.get("spl")
        if isinstance(spl, (int, float)):
            spl_by_episode[episode_id] = float(spl)
        total_questions = data.get("total_questions_to_human")
        if isinstance(total_questions, (int, float)):
            total_questions_by_episode[episode_id] = float(total_questions)

        proximity_state = data.get("target_proximity_state")
        if isinstance(proximity_state, str):
            target_proximity_counts[proximity_state] += 1
            target_proximity_episode_ids[proximity_state].append(episode_id)

        if visited_flag:
            visited_episodes.append(episode_id)
        if success_flag:
            success_episode_ids.append(episode_id)
        if distractor_success_flag:
            distractor_success_episode_ids.append(episode_id)

        detected_path = episode_dir / "detected_objects.jsonl"
        possible_target_instance_id: str | None = None
        if detected_path.is_file():
            try:
                with detected_path.open("r", encoding="utf-8") as det_f:
                    count = 0
                    for line in det_f:
                        count += 1
                        if possible_target_instance_id is not None:
                            continue
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            det = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        val = det.get("is_possible_target")
                        if val is True or val == 1 or val == 1.0:
                            inst = det.get("instance_id")
                            possible_target_instance_id = str(inst) if inst is not None else "None"
                    detected_object_counts[episode_id] = count
            except OSError as exc:
                print(f"Skipping detected objects for {episode_dir}: {exc}")
                detected_object_counts[episode_id] = 0
        else:
            detected_object_counts[episode_id] = 0

        if possible_target_instance_id is not None:
            possible_target_episode_ids.append(f"({episode_id},{possible_target_instance_id})")

        detected_objects_dir = episode_dir / "detected_objects"
        merged_count = 0
        if detected_objects_dir.is_dir():
            try:
                for entry in detected_objects_dir.iterdir():
                    if entry.is_dir() and _MERGED_GROUP_RE.match(entry.name):
                        merged_count += 1
            except OSError as exc:
                print(f"Skipping merged instance scan for {episode_dir}: {exc}")
        num_merged_instances_by_episode[episode_id] = merged_count

    binary_keys = {key for key, is_binary in binary_possible.items() if is_binary}
    return (
        totals,
        counts,
        true_counts,
        binary_keys,
        visited_episodes,
        detected_object_counts,
        target_proximity_counts,
        target_proximity_episode_ids,
        low_step_episodes,
        success_episode_ids,
        distractor_success_episode_ids,
        possible_target_episode_ids,
        total_steps_by_episode,
        num_detected_objects_by_episode,
        num_merged_instances_by_episode,
        spl_by_episode,
        total_questions_by_episode,
    )


def compute_directory_averages(
    root_dir: Path,
    allowed_episode_ids: Set[str] | None = None,
) -> Tuple[
    Dict[str, float],
    int,
    Dict[str, int],
    Dict[str, int],
    Set[str],
    List[str],
    Dict[str, int],
    Dict[str, int],
    Dict[str, List[str]],
    Dict[str, float],
    List[str],
    List[str],
    List[str],
    Dict[str, float],
    Dict[str, float],
    Dict[str, int],
    Dict[str, float],
    Dict[str, float],
]:
    """Compute averages for all info.json files nested one level below root_dir."""
    if not root_dir.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")
    if not root_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {root_dir}")

    info_files = []
    for child in sorted(root_dir.iterdir()):
        if not child.is_dir():
            continue
        if allowed_episode_ids is not None and child.name not in allowed_episode_ids:
            continue
        info_path = child / "info.json"
        if info_path.is_file():
            info_files.append(info_path)

    (
        totals,
        counts,
        true_counts,
        binary_keys,
        visited_episodes,
        detected_object_counts,
        target_proximity_counts,
        target_proximity_episode_ids,
        low_step_episodes,
        success_episode_ids,
        distractor_success_episode_ids,
        possible_target_episode_ids,
        total_steps_by_episode,
        num_detected_objects_by_episode,
        num_merged_instances_by_episode,
        spl_by_episode,
        total_questions_by_episode,
    ) = _gather_numeric_values(info_files)
    averages = {key: totals[key] / counts[key] for key in totals if counts[key] > 0}
    return (
        averages,
        len(info_files),
        counts,
        true_counts,
        binary_keys,
        visited_episodes,
        detected_object_counts,
        target_proximity_counts,
        target_proximity_episode_ids,
        low_step_episodes,
        success_episode_ids,
        distractor_success_episode_ids,
        possible_target_episode_ids,
        total_steps_by_episode,
        num_detected_objects_by_episode,
        num_merged_instances_by_episode,
        spl_by_episode,
        total_questions_by_episode,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute averages for numeric fields in info.json files under evaluation folders."
    )
    parser.add_argument(
        "directories",
        nargs="+",
        help="Directories containing numeric subdirectories with info.json files.",
    )
    parser.add_argument(
        "--sort-key",
        choices=["name", "value"],
        default="name",
        help="Sort averages by key name or by value (descending).",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Number of decimal places to display for averages.",
    )
    parser.add_argument(
        "--step-bin-size",
        type=int,
        default=100,
        help="Step bin size for non-cumulative step-bin summary table.",
    )
    parser.add_argument(
        "--episode-id",
        action="append",
        default=[],
        help=(
            "Episode ID filter. Repeat this option or pass comma/space-separated IDs. "
            "Only matching episodes are aggregated."
        ),
    )
    parser.add_argument(
        "--episode-id-file",
        type=str,
        action="append",
        default=[],
        help=(
            "Episode ID source. Accepts a file path (.txt/.json etc.) or a list literal "
            "string like \"[1,2,3]\"; can be repeated."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode_id_filter = _load_episode_id_filter(args.episode_id, args.episode_id_file)
    precision = max(args.precision, 0)
    step_bin_size = max(args.step_bin_size, 1)
    global_reached_count = 0
    global_non_reached_count = 0
    global_success_given_reached_count = 0
    global_reached_steps_sum = 0.0
    global_non_reached_steps_sum = 0.0
    global_reached_steps_values: List[float] = []
    global_non_reached_steps_values: List[float] = []
    global_reached_detected_values: List[float] = []
    global_non_reached_detected_values: List[float] = []
    global_success_detected_values: List[float] = []
    global_reached_pbp_first_seen_records: List[Tuple[int, bool]] = []
    global_reached_success_pbp_first_seen_steps: List[int] = []
    global_info_file_count = 0
    global_success_true = 0.0
    global_success_count = 0
    global_distractor_success_true = 0.0
    global_distractor_success_count = 0
    global_stop_called_true = 0.0
    global_stop_called_count = 0
    global_stop_tp_true = 0.0
    global_stop_fp_true = 0.0
    global_spl_sum = 0.0
    global_spl_count = 0
    global_total_questions_sum = 0.0
    global_interaction_questions_sum = 0.0
    global_interaction_episode_count = 0
    global_snq_exc_sum = 0.0
    global_snq_exc_count = 0
    global_snq_inc_sum = 0.0
    global_snq_inc_count = 0
    global_re_explore_sum = 0.0
    global_re_explore_count = 0
    global_response_len_total_tokens_sum = 0.0
    global_response_len_num_valid_responses_sum = 0.0
    global_elapsed_time_sum = 0.0
    global_elapsed_time_count = 0
    global_total_steps_sum = 0.0
    global_total_steps_count = 0
    global_detected_total_counts: Dict[int, int] = defaultdict(int)
    global_detected_success_counts: Dict[int, int] = defaultdict(int)
    global_reached_detected_total_counts: Dict[int, int] = defaultdict(int)
    global_reached_detected_success_counts: Dict[int, int] = defaultdict(int)
    global_detected_steps_sum: Dict[int, float] = defaultdict(float)
    global_detected_steps_count: Dict[int, int] = defaultdict(int)
    global_detected_spl_sum: Dict[int, float] = defaultdict(float)
    global_detected_spl_count: Dict[int, int] = defaultdict(int)
    global_reached_detected_steps_sum: Dict[int, float] = defaultdict(float)
    global_reached_detected_steps_count: Dict[int, int] = defaultdict(int)
    global_reached_detected_spl_sum: Dict[int, float] = defaultdict(float)
    global_reached_detected_spl_count: Dict[int, int] = defaultdict(int)
    global_merged_total_counts: Dict[int, int] = defaultdict(int)
    global_merged_success_counts: Dict[int, int] = defaultdict(int)
    global_reached_merged_total_counts: Dict[int, int] = defaultdict(int)
    global_reached_merged_success_counts: Dict[int, int] = defaultdict(int)
    global_merged_steps_sum: Dict[int, float] = defaultdict(float)
    global_merged_steps_count: Dict[int, int] = defaultdict(int)
    global_merged_spl_sum: Dict[int, float] = defaultdict(float)
    global_merged_spl_count: Dict[int, int] = defaultdict(int)
    global_reached_merged_steps_sum: Dict[int, float] = defaultdict(float)
    global_reached_merged_steps_count: Dict[int, int] = defaultdict(int)
    global_reached_merged_spl_sum: Dict[int, float] = defaultdict(float)
    global_reached_merged_spl_count: Dict[int, int] = defaultdict(int)
    global_collected_object_total_counts: Dict[int, int] = defaultdict(int)
    global_collected_object_tp_counts: Dict[int, int] = defaultdict(int)
    global_collected_object_fp_counts: Dict[int, int] = defaultdict(int)
    global_stop_collected_object_total_counts: Dict[int, int] = defaultdict(int)
    global_stop_collected_object_tp_counts: Dict[int, int] = defaultdict(int)
    global_stop_collected_object_fp_counts: Dict[int, int] = defaultdict(int)
    global_collected_object_steps_sum: Dict[int, float] = defaultdict(float)
    global_collected_object_steps_count: Dict[int, int] = defaultdict(int)
    global_collected_object_spl_sum: Dict[int, float] = defaultdict(float)
    global_collected_object_spl_count: Dict[int, int] = defaultdict(int)
    global_stop_collected_object_steps_sum: Dict[int, float] = defaultdict(float)
    global_stop_collected_object_steps_count: Dict[int, int] = defaultdict(int)
    global_stop_collected_object_spl_sum: Dict[int, float] = defaultdict(float)
    global_stop_collected_object_spl_count: Dict[int, int] = defaultdict(int)
    global_step_success_records: List[Tuple[float, bool]] = []
    global_detected_episode_records: List[Tuple[int, str, str, bool, float | None]] = []
    global_reached_detected_episode_records: List[Tuple[int, str, str, bool, float | None]] = []
    global_merged_episode_records: List[Tuple[int, str, str, bool, float | None]] = []
    global_reached_merged_episode_records: List[Tuple[int, str, str, bool, float | None]] = []
    stats_dir_count = 0

    if episode_id_filter is not None:
        print(f"[INFO] Episode filter enabled: {len(episode_id_filter)} IDs")

    for directory in args.directories:
        root = Path(directory).expanduser().resolve()
        try:
            (
                averages,
                file_count,
                counts,
                true_counts,
                binary_keys,
                visited_episodes,
                detected_object_counts,
                target_proximity_counts,
                target_proximity_episode_ids,
                low_step_episodes,
                success_episode_ids,
                distractor_success_episode_ids,
                possible_target_episode_ids,
                total_steps_by_episode,
                num_detected_objects_by_episode,
                num_merged_instances_by_episode,
                spl_by_episode,
                total_questions_by_episode,
            ) = compute_directory_averages(root, allowed_episode_ids=episode_id_filter)
        except Exception as exc:  # noqa: BLE001 - bubble up as user-facing error.
            print(f"[ERROR] {directory}: {exc}")
            continue

        if not file_count:
            print(f"[INFO] {root}: No info.json files found.")
            continue

        print(f"\nDirectory: {root} (info.json files: {file_count})")
        if not averages:
            print("  No numeric fields found to average.")
            continue
        fmt = f"{{:.{precision}f}}"

        def _format_total(value: float) -> str:
            if abs(value - round(value)) < 1e-9:
                return str(int(round(value)))
            return fmt.format(value)

        def _format_percent(value: float) -> str:
            return f"{value * 100.0:.2f}%"

        first_reached_by_episode = _collect_first_reached_steps(
            root,
            allowed_episode_ids=episode_id_filter,
        )
        first_reached_steps = list(first_reached_by_episode.values())
        if first_reached_steps:
            avg_first = sum(first_reached_steps) / len(first_reached_steps)
            bin_starts, freq = _compute_binned_frequencies(first_reached_steps, bin_size=50)
            if freq:
                max_freq = max(freq)
                mode_bins = [i for i, count in enumerate(freq) if count == max_freq]
                mode_parts = []
                for idx in mode_bins[:3]:
                    start = bin_starts[idx]
                    end = start + 50 - 1
                    mode_parts.append(f"{start}-{end}")
                mode_str = ", ".join(mode_parts) + (" ..." if len(mode_bins) > 3 else "")
                mode_info = f", mode_bin={mode_str} (count={max_freq})"
            else:
                mode_info = ""
            print(
                "  first_reached_step (reached_episodes="
                f"{len(first_reached_steps)}/{file_count}): min={min(first_reached_steps)}, "
                f"max={max(first_reached_steps)}, avg={avg_first:.1f}{mode_info}"
            )
            plot_path = _save_first_reached_histogram(
                root,
                first_reached_steps,
                bin_size=50,
                total_episodes=file_count,
            )
            if plot_path is not None:
                print(f"  saved first_reached_step histogram: {plot_path}")
        else:
            print("  first_reached_step: None")

        reached_episode_ids = set(first_reached_by_episode.keys())
        success_episode_set = set(success_episode_ids)
        distractor_success_episode_set = set(distractor_success_episode_ids)
        stop_called_episode_set = _collect_true_episode_ids_by_key(
            root,
            "stop_called",
            allowed_episode_ids=episode_id_filter,
        )
        reached_pbp_first_seen_by_episode = _collect_min_pbp_selected_first_seen_steps(
            root,
            reached_episode_ids,
        )
        success_reached_episode_ids = success_episode_set.intersection(reached_episode_ids)
        success_given_reached_count = len(success_reached_episode_ids)
        reached_success_pbp_first_seen_steps = [
            step
            for ep_id, step in reached_pbp_first_seen_by_episode.items()
            if ep_id in success_reached_episode_ids
        ]
        reached_count = len(reached_episode_ids)
        non_reached_count = max(file_count - reached_count, 0)
        reached_steps_values = [
            total_steps_by_episode[ep_id] for ep_id in reached_episode_ids if ep_id in total_steps_by_episode
        ]
        non_reached_steps_values = [
            steps for ep_id, steps in total_steps_by_episode.items() if ep_id not in reached_episode_ids
        ]
        reached_detected_values = [
            num_detected_objects_by_episode[ep_id]
            for ep_id in reached_episode_ids
            if ep_id in num_detected_objects_by_episode
        ]
        non_reached_detected_values = [
            value
            for ep_id, value in num_detected_objects_by_episode.items()
            if ep_id not in reached_episode_ids
        ]
        success_detected_values = [
            num_detected_objects_by_episode[ep_id]
            for ep_id in success_episode_set
            if ep_id in num_detected_objects_by_episode
        ]
        reached_steps_sum = sum(reached_steps_values)
        non_reached_steps_sum = sum(non_reached_steps_values)

        sr_given_reached = success_given_reached_count / reached_count if reached_count else 0.0
        avg_steps_given_reached = reached_steps_sum / reached_count if reached_count else 0.0
        avg_steps_given_non_reached = (
            non_reached_steps_sum / non_reached_count if non_reached_count else 0.0
        )
        print(
            "  Avg of SR given reached_episodes: "
            f"{success_given_reached_count}/{reached_count} = {_format_percent(sr_given_reached)}"
        )
        print(
            "  Avg of number of steps given reached_episodes: "
            f"{_format_total(reached_steps_sum)}/{reached_count} = "
            f"{fmt.format(avg_steps_given_reached)}"
        )
        print(
            "  Avg of number of steps given non reached_episodes: "
            f"{_format_total(non_reached_steps_sum)}/{non_reached_count} = "
            f"{fmt.format(avg_steps_given_non_reached)}"
        )
        if reached_steps_values:
            print(
                "  reached_episodes steps min/max/medium: "
                f"{_format_total(min(reached_steps_values))}/"
                f"{_format_total(max(reached_steps_values))}/"
                f"{_format_total(float(median(reached_steps_values)))}"
            )
        else:
            print("  reached_episodes steps min/max/medium: None")
        if non_reached_steps_values:
            print(
                "  non reached_episodes steps min/max/medium: "
                f"{_format_total(min(non_reached_steps_values))}/"
                f"{_format_total(max(non_reached_steps_values))}/"
                f"{_format_total(float(median(non_reached_steps_values)))}"
            )
        else:
            print("  non reached_episodes steps min/max/medium: None")
        if reached_detected_values:
            print(
                "  reached_episodes num_detected_objects min/max/medium/avg: "
                f"{_format_total(min(reached_detected_values))}/"
                f"{_format_total(max(reached_detected_values))}/"
                f"{_format_total(float(median(reached_detected_values)))}/"
                f"{_format_total(sum(reached_detected_values) / len(reached_detected_values))}"
            )
        else:
            print("  reached_episodes num_detected_objects min/max/medium/avg: None")
        if non_reached_detected_values:
            print(
                "  non reached_episodes num_detected_objects min/max/medium/avg: "
                f"{_format_total(min(non_reached_detected_values))}/"
                f"{_format_total(max(non_reached_detected_values))}/"
                f"{_format_total(float(median(non_reached_detected_values)))}/"
                f"{_format_total(sum(non_reached_detected_values) / len(non_reached_detected_values))}"
            )
        else:
            print("  non reached_episodes num_detected_objects min/max/medium/avg: None")
        if success_detected_values:
            print(
                "  success_episodes num_detected_objects min/max/avg: "
                f"{_format_total(min(success_detected_values))}/"
                f"{_format_total(max(success_detected_values))}/"
                f"{_format_total(sum(success_detected_values) / len(success_detected_values))}"
            )
        else:
            print("  success_episodes num_detected_objects min/max/avg: None")

        global_reached_count += reached_count
        global_non_reached_count += non_reached_count
        global_success_given_reached_count += success_given_reached_count
        global_reached_steps_sum += reached_steps_sum
        global_non_reached_steps_sum += non_reached_steps_sum
        global_reached_steps_values.extend(reached_steps_values)
        global_non_reached_steps_values.extend(non_reached_steps_values)
        global_reached_detected_values.extend(reached_detected_values)
        global_non_reached_detected_values.extend(non_reached_detected_values)
        global_success_detected_values.extend(success_detected_values)
        for ep_id, step in reached_pbp_first_seen_by_episode.items():
            global_reached_pbp_first_seen_records.append((step, ep_id in success_episode_set))
        global_reached_success_pbp_first_seen_steps.extend(reached_success_pbp_first_seen_steps)
        for ep_id, step in total_steps_by_episode.items():
            global_step_success_records.append((step, ep_id in success_episode_set))
        for ep_id, detected_value in num_detected_objects_by_episode.items():
            detected_key = int(round(detected_value))
            step_value = total_steps_by_episode.get(ep_id)
            spl_value = spl_by_episode.get(ep_id)
            success_flag = ep_id in success_episode_set
            global_detected_total_counts[detected_key] += 1
            if ep_id in reached_episode_ids:
                global_reached_detected_total_counts[detected_key] += 1
            if step_value is not None:
                global_detected_steps_sum[detected_key] += step_value
                global_detected_steps_count[detected_key] += 1
            if spl_value is not None:
                global_detected_spl_sum[detected_key] += spl_value
                global_detected_spl_count[detected_key] += 1
            global_detected_episode_records.append(
                (detected_key, root.name, ep_id, success_flag, step_value)
            )
            if ep_id in reached_episode_ids:
                if step_value is not None:
                    global_reached_detected_steps_sum[detected_key] += step_value
                    global_reached_detected_steps_count[detected_key] += 1
                if spl_value is not None:
                    global_reached_detected_spl_sum[detected_key] += spl_value
                    global_reached_detected_spl_count[detected_key] += 1
                global_reached_detected_episode_records.append(
                    (detected_key, root.name, ep_id, success_flag, step_value)
                )
            if success_flag:
                global_detected_success_counts[detected_key] += 1
                if ep_id in reached_episode_ids:
                    global_reached_detected_success_counts[detected_key] += 1
        for ep_id, merged_count in num_merged_instances_by_episode.items():
            merged_key = int(merged_count)
            step_value = total_steps_by_episode.get(ep_id)
            spl_value = spl_by_episode.get(ep_id)
            success_flag = ep_id in success_episode_set
            global_merged_total_counts[merged_key] += 1
            if ep_id in reached_episode_ids:
                global_reached_merged_total_counts[merged_key] += 1
            if step_value is not None:
                global_merged_steps_sum[merged_key] += step_value
                global_merged_steps_count[merged_key] += 1
            if spl_value is not None:
                global_merged_spl_sum[merged_key] += spl_value
                global_merged_spl_count[merged_key] += 1
            global_merged_episode_records.append(
                (merged_key, root.name, ep_id, success_flag, step_value)
            )
            if ep_id in reached_episode_ids:
                if step_value is not None:
                    global_reached_merged_steps_sum[merged_key] += step_value
                    global_reached_merged_steps_count[merged_key] += 1
                if spl_value is not None:
                    global_reached_merged_spl_sum[merged_key] += spl_value
                    global_reached_merged_spl_count[merged_key] += 1
                global_reached_merged_episode_records.append(
                    (merged_key, root.name, ep_id, success_flag, step_value)
                )
            if success_flag:
                global_merged_success_counts[merged_key] += 1
                if ep_id in reached_episode_ids:
                    global_reached_merged_success_counts[merged_key] += 1

        episode_ids_for_collected_objects = set(num_merged_instances_by_episode.keys()).union(
            detected_object_counts.keys()
        )
        for ep_id in episode_ids_for_collected_objects:
            # Rule:
            # - instances.jsonl exists => use detected_objects/group_* count
            # - instances.jsonl missing => use detected_objects.jsonl line count
            if (root / ep_id / "instances.jsonl").is_file():
                collected_object_count = int(num_merged_instances_by_episode.get(ep_id, 0))
            else:
                collected_object_count = int(detected_object_counts.get(ep_id, 0))

            global_collected_object_total_counts[collected_object_count] += 1
            if ep_id in success_episode_set:
                global_collected_object_tp_counts[collected_object_count] += 1
            if ep_id in distractor_success_episode_set:
                global_collected_object_fp_counts[collected_object_count] += 1
            step_value = total_steps_by_episode.get(ep_id)
            if step_value is not None:
                global_collected_object_steps_sum[collected_object_count] += step_value
                global_collected_object_steps_count[collected_object_count] += 1
            spl_value = spl_by_episode.get(ep_id)
            if spl_value is not None:
                global_collected_object_spl_sum[collected_object_count] += spl_value
                global_collected_object_spl_count[collected_object_count] += 1
            if ep_id in stop_called_episode_set:
                global_stop_collected_object_total_counts[collected_object_count] += 1
                if ep_id in success_episode_set:
                    global_stop_collected_object_tp_counts[collected_object_count] += 1
                if ep_id in distractor_success_episode_set:
                    global_stop_collected_object_fp_counts[collected_object_count] += 1
                stop_step_value = total_steps_by_episode.get(ep_id)
                if stop_step_value is not None:
                    global_stop_collected_object_steps_sum[collected_object_count] += stop_step_value
                    global_stop_collected_object_steps_count[collected_object_count] += 1
                stop_spl_value = spl_by_episode.get(ep_id)
                if stop_spl_value is not None:
                    global_stop_collected_object_spl_sum[collected_object_count] += stop_spl_value
                    global_stop_collected_object_spl_count[collected_object_count] += 1

        global_stop_tp_true += float(len(stop_called_episode_set.intersection(success_episode_set)))
        global_stop_fp_true += float(len(stop_called_episode_set.intersection(distractor_success_episode_set)))
        global_stop_called_true += float(len(stop_called_episode_set))
        global_stop_called_count += file_count

        global_info_file_count += file_count
        if "success" in counts:
            success_count = counts["success"]
            global_success_count += success_count
            if "success" in binary_keys:
                global_success_true += float(true_counts.get("success", 0))
            elif "success" in averages:
                global_success_true += float(averages["success"]) * float(success_count)
        if "distractor_success" in counts:
            distractor_success_count = counts["distractor_success"]
            global_distractor_success_count += distractor_success_count
            if "distractor_success" in binary_keys:
                global_distractor_success_true += float(true_counts.get("distractor_success", 0))
            elif "distractor_success" in averages:
                global_distractor_success_true += (
                    float(averages["distractor_success"]) * float(distractor_success_count)
                )
        if "spl" in counts and "spl" in averages:
            spl_count = counts["spl"]
            global_spl_count += spl_count
            global_spl_sum += float(averages["spl"]) * float(spl_count)
        if "total_questions_to_human" in counts and "total_questions_to_human" in averages:
            q_count = counts["total_questions_to_human"]
            global_total_questions_sum += float(averages["total_questions_to_human"]) * float(q_count)
        interaction_questions_sum = 0.0
        interaction_episode_count = 0
        for question_count in total_questions_by_episode.values():
            if question_count > 0.0:
                interaction_questions_sum += question_count
                interaction_episode_count += 1
        global_interaction_questions_sum += interaction_questions_sum
        global_interaction_episode_count += interaction_episode_count
        if "snq_excluding_nq0" in counts and "snq_excluding_nq0" in averages:
            snq_exc_count = counts["snq_excluding_nq0"]
            global_snq_exc_count += snq_exc_count
            global_snq_exc_sum += float(averages["snq_excluding_nq0"]) * float(snq_exc_count)
        if "snq_including_nq0" in counts and "snq_including_nq0" in averages:
            snq_inc_count = counts["snq_including_nq0"]
            global_snq_inc_count += snq_inc_count
            global_snq_inc_sum += float(averages["snq_including_nq0"]) * float(snq_inc_count)
        if "re_explore" in counts and "re_explore" in averages:
            re_explore_count = counts["re_explore"]
            global_re_explore_count += re_explore_count
            global_re_explore_sum += float(averages["re_explore"]) * float(re_explore_count)
        if "response_len_total_tokens" in counts and "response_len_total_tokens" in averages:
            rl_total_count = counts["response_len_total_tokens"]
            global_response_len_total_tokens_sum += float(averages["response_len_total_tokens"]) * float(rl_total_count)
        if (
            "response_len_num_valid_responses" in counts
            and "response_len_num_valid_responses" in averages
        ):
            rl_valid_count = counts["response_len_num_valid_responses"]
            global_response_len_num_valid_responses_sum += (
                float(averages["response_len_num_valid_responses"]) * float(rl_valid_count)
            )
        if "elapsed_time_sec" in counts and "elapsed_time_sec" in averages:
            et_count = counts["elapsed_time_sec"]
            global_elapsed_time_count += et_count
            global_elapsed_time_sum += float(averages["elapsed_time_sec"]) * float(et_count)
        if "total_steps" in counts and "total_steps" in averages:
            ts_count = counts["total_steps"]
            global_total_steps_count += ts_count
            global_total_steps_sum += float(averages["total_steps"]) * float(ts_count)
        stats_dir_count += 1

        if args.sort_key == "value":
            sorted_items = sorted(averages.items(), key=lambda item: item[1], reverse=True)
        else:
            sorted_items = sorted(averages.items(), key=lambda item: item[0])

        def _episode_sort_key(item: str) -> Tuple[int, str]:
            return (0, f"{int(item):010d}") if item.isdigit() else (1, item)

        possible_count = len(possible_target_episode_ids)
        possible_ratio = possible_count / file_count if file_count else 0.0
        print(f"is_possible_target: {possible_count}/{file_count} = {fmt.format(possible_ratio)}")
        pairs = sorted(
            possible_target_episode_ids,
            key=lambda item: _episode_sort_key(item.lstrip("(").split(",", 1)[0]),
        )
        pairs_str = " ".join(pairs) if pairs else "None"
        print(f"  is_possible_target episode_id,instance_id: {pairs_str}")
        if "total_steps" in averages:
            count = counts["total_steps"]
            print(f"  total_steps avg: {fmt.format(averages['total_steps'])} (count={count})")
        if "total_questions_to_human" in averages:
            count = counts["total_questions_to_human"]
            total_questions = float(averages["total_questions_to_human"]) * float(count)
            avg_per_episode = total_questions / float(file_count) if file_count else 0.0
            if abs(total_questions - round(total_questions)) < 1e-9:
                total_questions_str = str(int(round(total_questions)))
            else:
                total_questions_str = fmt.format(total_questions)
            print(
                "  total_questions_to_human avg: "
                f"{total_questions_str}/{file_count} = {fmt.format(avg_per_episode)}"
            )
        if "pbp_depth" in averages:
            count = counts["pbp_depth"]
            value = averages["pbp_depth"]
            total = value * count
            if abs(total - round(total)) < 1e-9:
                total_str = str(int(round(total)))
            else:
                total_str = fmt.format(total)
            print(f"Avg pbp_depth: {total_str}/{count} = {fmt.format(value)}")
        if "re_explore" in averages:
            count = counts["re_explore"]
            total_re_explore = float(averages["re_explore"]) * float(count)
            avg_per_episode = total_re_explore / float(file_count) if file_count else 0.0
            if abs(total_re_explore - round(total_re_explore)) < 1e-9:
                total_re_explore_str = str(int(round(total_re_explore)))
            else:
                total_re_explore_str = fmt.format(total_re_explore)
            print(
                "  re_explore count: "
                f"{total_re_explore_str}/{file_count} = {fmt.format(avg_per_episode)}"
            )
        for key, value in sorted_items:
            count = counts[key]
            if key in ("success", "distractor_success"):
                if key in binary_keys:
                    true_count = true_counts.get(key, 0)
                    print(f"{key}: {true_count}/{count} = {_format_percent(value)}")
                else:
                    print(f"{key}: {fmt.format(value)} (count={count})")
                if key == "success":
                    episode_ids = success_episode_ids
                elif key == "distractor_success":
                    episode_ids = distractor_success_episode_ids
                ids = sorted(episode_ids, key=_episode_sort_key)
                ids_str = " ".join(ids) if ids else "None"
                print(f"  {key} episode ids: {ids_str}")
            elif key == "spl":
                # Keep the same "x/y = avg" style as success for quick log parsing.
                total = value * count
                if abs(total - round(total)) < 1e-9:
                    total_str = str(int(round(total)))
                else:
                    total_str = fmt.format(total)
                label = "num_questions (pbp_depth)" if key == "pbp_depth" else key
                print(f"{label}: {total_str}/{count} = {_format_percent(value)}")

        total_proximity = sum(target_proximity_counts.values())
        if total_proximity:
            ratio_fmt = f"{{:.{precision}f}}"
            # print("  target_proximity_state counts:")
            proximity_order = ("reached", "adjacent", "nearby", "close")
            # for state in proximity_order:
            #     print(f"    {state}: {target_proximity_counts.get(state, 0)}")
            far_count = target_proximity_counts.get("far", 0)
            # if far_count:
            #     print(f"    far: {far_count}")
            unexpected_states = {
                state: count
                for state, count in target_proximity_counts.items()
                if state not in proximity_order and state != "far"
            }
            for state, count in sorted(unexpected_states.items()):
                print(f"    {state}: {count}")

            non_far = total_proximity - far_count
            ratio = non_far / total_proximity if total_proximity else 0.0
            
            reached = 0
            if "reached" in target_proximity_counts:
                reached = target_proximity_counts.get("reached", 0)
            elif "adjacent" in target_proximity_counts:
                reached = target_proximity_counts.get("adjacent", 0)
            adjacent = target_proximity_counts.get("adjacent", 0)
            nearby = target_proximity_counts.get("nearby", 0)
            close = target_proximity_counts.get("close", 0)
            def _print_ratio(label: str, numerator: int) -> None:
                ratio_value = numerator / total_proximity if total_proximity else 0.0
                print(
                    f"  {label}: {numerator}/{total_proximity} = "
                    f"{ratio_fmt.format(ratio_value)}"
                )

            _print_ratio("reached ratio", reached)
            _print_ratio("reached+adjacent ratio", reached + adjacent)
            _print_ratio("reached+adjacent+nearby ratio", reached + adjacent + nearby)
            _print_ratio(
                "reached+adjacent+nearby+close ratio",
                reached + adjacent + nearby + close,
            )

            def _print_episode_ids(label: str, state: str) -> None:
                episodes = target_proximity_episode_ids.get(state, [])
                if not episodes:
                    print(f"  {label}: None")
                    return
                ids = " ".join(sorted(episodes, key=_episode_sort_key))
                print(f"  {label}: {ids}")

            _print_episode_ids("reached episode ids", "reached")
            _print_episode_ids("adjacent episode ids", "adjacent")

        low_step_sorted = sorted(
            low_step_episodes.items(),
            key=lambda item: _episode_sort_key(item[0]),
        )

        # if low_step_sorted:
        #     print("  total_steps<400 episode ids:")
        #     for episode_id, steps in low_step_sorted:
        #         print(f"    {episode_id}: {steps:.0f}")
        # else:
        #     print("  total_steps<400 episode ids: None")

        # if visited_episodes:
        #     sorted_eps = ", ".join(sorted(visited_episodes, key=lambda item: int(item)))
        #     print(f"  visited_target episodes ({len(visited_episodes)}): {sorted_eps}")
        # else:
        #     print("  visited_target episodes (0): None")

        # if detected_object_counts:
        #     avg_detected = sum(detected_object_counts.values()) / len(detected_object_counts)
        #     print(
        #         f"detected_objects.jsonl counts (avg_detected_objects="
        #         f"{avg_detected:.{precision}f} over {len(detected_object_counts)} episodes):"
        #     )
        #     for episode_id in sorted(detected_object_counts.keys(), key=lambda item: int(item)):
        #         print(f"{episode_id}: {detected_object_counts[episode_id]}")

    if stats_dir_count:
        fmt = f"{{:.{precision}f}}"

        def _format_total(value: float) -> str:
            if abs(value - round(value)) < 1e-9:
                return str(int(round(value)))
            return fmt.format(value)

        global_sr_given_reached = (
            global_success_given_reached_count / global_reached_count if global_reached_count else 0.0
        )
        global_avg_steps_given_reached = (
            global_reached_steps_sum / global_reached_count if global_reached_count else 0.0
        )
        global_avg_steps_given_non_reached = (
            global_non_reached_steps_sum / global_non_reached_count
            if global_non_reached_count
            else 0.0
        )
        global_overall_sr = global_success_true / global_success_count if global_success_count else 0.0
        global_overall_dsr = (
            global_distractor_success_true / global_distractor_success_count
            if global_distractor_success_count
            else 0.0
        )
        global_stop_called_ratio = (
            global_stop_called_true / global_stop_called_count if global_stop_called_count else 0.0
        )
        global_stop_distractor_ratio = (
            global_stop_fp_true / global_stop_called_true if global_stop_called_true else 0.0
        )
        global_stop_success_ratio = (
            global_stop_tp_true / global_stop_called_true if global_stop_called_true else 0.0
        )
        fdr_denominator = global_success_true + global_distractor_success_true
        global_fdr = global_distractor_success_true / fdr_denominator if fdr_denominator else 0.0
        fdr_stop_denominator = global_stop_tp_true + global_stop_fp_true
        global_fdr_stop = global_stop_fp_true / fdr_stop_denominator if fdr_stop_denominator else 0.0
        global_overall_spl = global_spl_sum / global_spl_count if global_spl_count else 0.0
        global_avg_questions = (
            global_total_questions_sum / global_info_file_count if global_info_file_count else 0.0
        )
        global_avg_nq_new = (
            global_interaction_questions_sum / global_interaction_episode_count
            if global_interaction_episode_count
            else 0.0
        )
        global_avg_snq_exc = global_snq_exc_sum / global_snq_exc_count if global_snq_exc_count else 0.0
        global_avg_snq_inc = global_snq_inc_sum / global_snq_inc_count if global_snq_inc_count else 0.0
        global_avg_re_explore = (
            global_re_explore_sum / global_re_explore_count if global_re_explore_count else 0.0
        )
        global_rl_total = (
            global_response_len_total_tokens_sum / global_info_file_count if global_info_file_count else 0.0
        )
        global_rl_single = (
            global_response_len_total_tokens_sum / global_response_len_num_valid_responses_sum
            if global_response_len_num_valid_responses_sum
            else 0.0
        )
        global_avg_elapsed_time = (
            global_elapsed_time_sum / global_elapsed_time_count if global_elapsed_time_count else 0.0
        )
        global_avg_total_steps = (
            global_total_steps_sum / global_total_steps_count if global_total_steps_count else 0.0
        )
        print("\nAcross all directories:")
        reached_steps_stats = (
            f"{_format_total(min(global_reached_steps_values))}/"
            f"{_format_total(max(global_reached_steps_values))}/"
            f"{_format_total(float(median(global_reached_steps_values)))}"
            if global_reached_steps_values
            else "None"
        )
        non_reached_steps_stats = (
            f"{_format_total(min(global_non_reached_steps_values))}/"
            f"{_format_total(max(global_non_reached_steps_values))}/"
            f"{_format_total(float(median(global_non_reached_steps_values)))}"
            if global_non_reached_steps_values
            else "None"
        )
        reached_detected_stats = (
            f"{_format_total(min(global_reached_detected_values))}/"
            f"{_format_total(max(global_reached_detected_values))}/"
            f"{_format_total(float(median(global_reached_detected_values)))}/"
            f"{_format_total(sum(global_reached_detected_values) / len(global_reached_detected_values))}"
            if global_reached_detected_values
            else "None"
        )
        non_reached_detected_stats = (
            f"{_format_total(min(global_non_reached_detected_values))}/"
            f"{_format_total(max(global_non_reached_detected_values))}/"
            f"{_format_total(float(median(global_non_reached_detected_values)))}/"
            f"{_format_total(sum(global_non_reached_detected_values) / len(global_non_reached_detected_values))}"
            if global_non_reached_detected_values
            else "None"
        )
        success_detected_stats = (
            f"{_format_total(min(global_success_detected_values))}/"
            f"{_format_total(max(global_success_detected_values))}/"
            f"{_format_total(sum(global_success_detected_values) / len(global_success_detected_values))}"
            if global_success_detected_values
            else "None"
        )
        pbp_first_seen_count = len(global_reached_success_pbp_first_seen_steps)
        pbp_first_seen_sum = float(sum(global_reached_success_pbp_first_seen_steps))
        pbp_first_seen_avg = pbp_first_seen_sum / pbp_first_seen_count if pbp_first_seen_count else 0.0
        pbp_first_seen_avg_str = fmt.format(pbp_first_seen_avg) if pbp_first_seen_count else "None"
        pbp_first_seen_min_str = (
            _format_total(min(global_reached_success_pbp_first_seen_steps))
            if pbp_first_seen_count
            else "None"
        )
        pbp_first_seen_max_str = (
            _format_total(max(global_reached_success_pbp_first_seen_steps))
            if pbp_first_seen_count
            else "None"
        )
        rows = [
            (
                "Average SR",
                f"{_format_total(global_success_true)}/{global_success_count}",
                _format_percent(global_overall_sr),
            ),
            (
                "Average DSR",
                f"{_format_total(global_distractor_success_true)}/{global_distractor_success_count}",
                _format_percent(global_overall_dsr),
            ),
            (
                "stop_called ratio",
                f"{_format_total(global_stop_called_true)}/{global_stop_called_count}",
                _format_percent(global_stop_called_ratio),
            ),
            (
                "STOP->Distractor/STOP",
                f"{_format_total(global_stop_fp_true)}/{_format_total(global_stop_called_true)}",
                _format_percent(global_stop_distractor_ratio),
            ),
            (
                "STOP->Success/STOP",
                f"{_format_total(global_stop_tp_true)}/{_format_total(global_stop_called_true)}",
                _format_percent(global_stop_success_ratio),
            ),
            (
                "FDR",
                f"{_format_total(global_distractor_success_true)}/"
                f"({_format_total(global_success_true)}+{_format_total(global_distractor_success_true)})",
                _format_percent(global_fdr),
            ),
            (
                "FDR_stop",
                f"{_format_total(global_stop_fp_true)}/"
                f"({_format_total(global_stop_tp_true)}+{_format_total(global_stop_fp_true)})",
                _format_percent(global_fdr_stop),
            ),
            (
                "Average SPL",
                f"{_format_total(global_spl_sum)}/{global_spl_count}",
                _format_percent(global_overall_spl),
            ),
            (
                "Average NQ",
                f"{_format_total(global_total_questions_sum)}/{global_info_file_count}",
                fmt.format(global_avg_questions),
            ),
            (
                "NQ_new",
                f"{_format_total(global_interaction_questions_sum)}/{global_interaction_episode_count}",
                fmt.format(global_avg_nq_new) if global_interaction_episode_count else "None",
            ),
            (
                "SNQ_exc",
                f"{_format_total(global_snq_exc_sum)}/{global_snq_exc_count}",
                fmt.format(global_avg_snq_exc),
            ),
            (
                "SNQ_inc",
                f"{_format_total(global_snq_inc_sum)}/{global_snq_inc_count}",
                fmt.format(global_avg_snq_inc),
            ),
            (
                "re_explore",
                f"{_format_total(global_re_explore_sum)}/{global_re_explore_count}",
                fmt.format(global_avg_re_explore),
            ),
            ("RL_total", f"{_format_total(global_response_len_total_tokens_sum)}/{global_info_file_count}", fmt.format(global_rl_total)),
            (
                "RL_single",
                f"{_format_total(global_response_len_total_tokens_sum)}/"
                f"{_format_total(global_response_len_num_valid_responses_sum)}",
                fmt.format(global_rl_single),
            ),
            (
                "Average elapsed_time_sec",
                f"{_format_total(global_elapsed_time_sum)}/{global_elapsed_time_count}",
                fmt.format(global_avg_elapsed_time),
            ),
            (
                "Average total_steps",
                f"{_format_total(global_total_steps_sum)}/{global_total_steps_count}",
                fmt.format(global_avg_total_steps),
            ),
            ("sum(A)", "", str(global_reached_count)),
            ("sum(B-A)", "", str(global_non_reached_count)),
            (
                "Avg of SR given reached_episodes",
                f"{global_success_given_reached_count}/{global_reached_count}",
                _format_percent(global_sr_given_reached),
            ),
            (
                "Avg of number of steps given reached_episodes",
                f"{_format_total(global_reached_steps_sum)}/{global_reached_count}",
                fmt.format(global_avg_steps_given_reached),
            ),
            (
                "Avg of number of steps given non reached_episodes",
                f"{_format_total(global_non_reached_steps_sum)}/{global_non_reached_count}",
                fmt.format(global_avg_steps_given_non_reached),
            ),
            ("reached_episodes steps min/max/medium", "", reached_steps_stats),
            ("non reached_episodes steps min/max/medium", "", non_reached_steps_stats),
            ("reached_episodes num_detected_objects min/max/medium/avg", "", reached_detected_stats),
            ("non reached_episodes num_detected_objects min/max/medium/avg", "", non_reached_detected_stats),
            ("success_episodes num_detected_objects min/max/avg", "", success_detected_stats),
            (
                "Avg first_seen_step (pbp_selected=true, reached&success)",
                f"{_format_total(pbp_first_seen_sum)}/{pbp_first_seen_count}",
                pbp_first_seen_avg_str,
            ),
            ("Min first_seen_step (pbp_selected=true, reached&success)", "", pbp_first_seen_min_str),
            ("Max first_seen_step (pbp_selected=true, reached&success)", "", pbp_first_seen_max_str),
        ]
        metric_width = max(len("Metric"), max(len(name) for name, _, _ in rows))
        equation_width = max(len("Equation"), max(len(eq) for _, eq, _ in rows))
        value_width = max(len("Value"), max(len(value) for _, _, value in rows))
        border = (
            f"  +{'-' * (metric_width + 2)}+{'-' * (equation_width + 2)}+"
            f"{'-' * (value_width + 2)}+"
        )
        print(border)
        print(
            f"  | {'Metric'.ljust(metric_width)} | {'Equation'.ljust(equation_width)} | "
            f"{'Value'.ljust(value_width)} |"
        )
        print(border)
        for name, equation, value in rows:
            print(
                f"  | {name.ljust(metric_width)} | {equation.ljust(equation_width)} | "
                f"{value.ljust(value_width)} |"
            )
        print(border)

        percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        reached_sorted = sorted(global_reached_steps_values)
        non_reached_sorted = sorted(global_non_reached_steps_values)
        reached_success_pbp_first_seen_sorted = sorted(global_reached_success_pbp_first_seen_steps)

        def _lower_percentile_value(sorted_values: List[float], pct: int) -> str:
            if not sorted_values:
                return "None"
            n = len(sorted_values)
            idx = (pct * n + 99) // 100 - 1
            if idx < 0:
                idx = 0
            if idx >= n:
                idx = n - 1
            return _format_total(sorted_values[idx])

        def _lower_percentile_cutoff(sorted_values: List[int], pct: int) -> int | None:
            if not sorted_values:
                return None
            n = len(sorted_values)
            idx = (pct * n + 99) // 100 - 1
            if idx < 0:
                idx = 0
            if idx >= n:
                idx = n - 1
            return sorted_values[idx]

        p_rows = [
            (
                f"Bottom {pct}%",
                _lower_percentile_value(reached_sorted, pct),
                _lower_percentile_value(non_reached_sorted, pct),
                _lower_percentile_value(reached_success_pbp_first_seen_sorted, pct),
            )
            for pct in percentiles
        ]
        p_col0 = max(len("Percentile"), max(len(r[0]) for r in p_rows))
        p_col1 = max(len("Reached steps"), max(len(r[1]) for r in p_rows))
        p_col2 = max(len("Non-reached steps"), max(len(r[2]) for r in p_rows))
        p_col3 = max(len("R&S PBP first_seen_step"), max(len(r[3]) for r in p_rows))
        p_border = (
            f"  +{'-' * (p_col0 + 2)}+{'-' * (p_col1 + 2)}+"
            f"{'-' * (p_col2 + 2)}+{'-' * (p_col3 + 2)}+"
        )
        print("  Step percentile table (lower p% cutoff):")
        print(p_border)
        print(
            f"  | {'Percentile'.ljust(p_col0)} | {'Reached steps'.ljust(p_col1)} | "
            f"{'Non-reached steps'.ljust(p_col2)} | {'R&S PBP first_seen_step'.ljust(p_col3)} |"
        )
        print(p_border)
        for p_label, r_val, n_val, rs_pbp_val in p_rows:
            print(
                f"  | {p_label.ljust(p_col0)} | {r_val.ljust(p_col1)} | "
                f"{n_val.ljust(p_col2)} | {rs_pbp_val.ljust(p_col3)} |"
            )
        print(p_border)

        rs_sr_rows = []
        prev_cutoff: int | None = None
        for pct in percentiles:
            cutoff = _lower_percentile_cutoff(reached_success_pbp_first_seen_sorted, pct)
            if cutoff is None:
                rs_sr_rows.append((f"Bottom {pct}%", "None", "None", "None", "None"))
                prev_cutoff = cutoff
                continue

            cumulative_total = 0
            cumulative_success = 0
            interval_total = 0
            interval_success = 0
            rs_interval_steps: List[int] = []
            for step, success_flag in global_reached_pbp_first_seen_records:
                if step <= cutoff:
                    cumulative_total += 1
                    if success_flag:
                        cumulative_success += 1
                in_interval = step <= cutoff if prev_cutoff is None else prev_cutoff < step <= cutoff
                if in_interval:
                    interval_total += 1
                    if success_flag:
                        interval_success += 1
            for step in reached_success_pbp_first_seen_sorted:
                in_rs_interval = step <= cutoff if prev_cutoff is None else prev_cutoff < step <= cutoff
                if in_rs_interval:
                    rs_interval_steps.append(step)

            cumulative_sr = (
                f"{cumulative_success}/{cumulative_total} = "
                f"{_format_percent(cumulative_success / cumulative_total)}"
                if cumulative_total
                else "0/0 = None"
            )
            interval_sr = (
                f"{interval_success}/{interval_total} = "
                f"{_format_percent(interval_success / interval_total)}"
                if interval_total
                else "0/0 = None"
            )
            rs_interval_step_stats = (
                f"{_format_total(min(rs_interval_steps))}/"
                f"{fmt.format(sum(rs_interval_steps) / len(rs_interval_steps))}/"
                f"{_format_total(max(rs_interval_steps))}"
                if rs_interval_steps
                else "None"
            )
            rs_sr_rows.append(
                (f"Bottom {pct}%", _format_total(cutoff), cumulative_sr, interval_sr, rs_interval_step_stats)
            )
            prev_cutoff = cutoff

        rs_col0 = max(len("Percentile"), max(len(r[0]) for r in rs_sr_rows))
        rs_col1 = max(len("Cutoff step"), max(len(r[1]) for r in rs_sr_rows))
        rs_col2 = max(len("Cumulative SR"), max(len(r[2]) for r in rs_sr_rows))
        rs_col3 = max(len("Interval SR"), max(len(r[3]) for r in rs_sr_rows))
        rs_col4 = max(len("R&S interval step(min/avg/max)"), max(len(r[4]) for r in rs_sr_rows))
        rs_border = (
            f"  +{'-' * (rs_col0 + 2)}+{'-' * (rs_col1 + 2)}+"
            f"{'-' * (rs_col2 + 2)}+{'-' * (rs_col3 + 2)}+{'-' * (rs_col4 + 2)}+"
        )
        print("  R&S PBP first_seen_step SR table:")
        print(rs_border)
        print(
            f"  | {'Percentile'.ljust(rs_col0)} | {'Cutoff step'.ljust(rs_col1)} | "
            f"{'Cumulative SR'.ljust(rs_col2)} | {'Interval SR'.ljust(rs_col3)} | "
            f"{'R&S interval step(min/avg/max)'.ljust(rs_col4)} |"
        )
        print(rs_border)
        for p_label, cutoff_str, cumulative_sr, interval_sr, rs_interval_step_stats in rs_sr_rows:
            print(
                f"  | {p_label.ljust(rs_col0)} | {cutoff_str.ljust(rs_col1)} | "
                f"{cumulative_sr.ljust(rs_col2)} | {interval_sr.ljust(rs_col3)} | "
                f"{rs_interval_step_stats.ljust(rs_col4)} |"
            )
        print(rs_border)

        def _avg_str(sum_map: Dict[int, float], count_map: Dict[int, int], key: int) -> str:
            cnt = count_map.get(key, 0)
            if cnt <= 0:
                return "None"
            return fmt.format(sum_map[key] / cnt)

        def _avg_percent_str(sum_map: Dict[int, float], count_map: Dict[int, int], key: int) -> str:
            cnt = count_map.get(key, 0)
            if cnt <= 0:
                return "None"
            return _format_percent(sum_map[key] / cnt)

        def _print_top3_episodes(
            title: str,
            metric_name: str,
            records: List[Tuple[int, str, str, bool, float | None]],
        ) -> None:
            print(f"  {title}:")
            if not records:
                print("    None")
                return

            def _top_sort_key(item: Tuple[int, str, str, bool, float | None]) -> Tuple[int, int, str, str]:
                metric_value, dir_name, ep_id, _success, _steps = item
                ep_num = int(ep_id) if ep_id.isdigit() else 10**12
                return (-metric_value, ep_num, dir_name, ep_id)

            top_records = sorted(records, key=_top_sort_key)[:3]
            top_rows = []
            for metric_value, dir_name, ep_id, success, steps in top_records:
                episode_label = f"{dir_name}/{ep_id}"
                top_rows.append(
                    (
                        episode_label,
                        str(metric_value),
                        "Y" if success else "N",
                        _format_total(steps) if steps is not None else "None",
                    )
                )

            t_col0 = max(len("episode"), max(len(r[0]) for r in top_rows))
            t_col1 = max(len(metric_name), max(len(r[1]) for r in top_rows))
            t_col2 = max(len("success"), max(len(r[2]) for r in top_rows))
            t_col3 = max(len("steps"), max(len(r[3]) for r in top_rows))
            t_border = (
                f"  +{'-' * (t_col0 + 2)}+{'-' * (t_col1 + 2)}+"
                f"{'-' * (t_col2 + 2)}+{'-' * (t_col3 + 2)}+"
            )
            print(t_border)
            print(
                f"  | {'episode'.ljust(t_col0)} | {metric_name.ljust(t_col1)} | "
                f"{'success'.ljust(t_col2)} | {'steps'.ljust(t_col3)} |"
            )
            print(t_border)
            for episode_label, metric_value, success, steps in top_rows:
                print(
                    f"  | {episode_label.ljust(t_col0)} | {metric_value.ljust(t_col1)} | "
                    f"{success.ljust(t_col2)} | {steps.ljust(t_col3)} |"
                )
            print(t_border)

        def _print_success_by_bucket_table(
            title: str,
            bucket_name: str,
            total_name: str,
            keys: List[int],
            success_counts: Dict[int, int],
            total_counts: Dict[int, int],
            steps_sum: Dict[int, float],
            steps_count: Dict[int, int],
            spl_sum: Dict[int, float],
            spl_count: Dict[int, int],
        ) -> None:
            if not keys:
                print(f"  {title}: None")
                return

            rows_for_table = []
            for key in keys:
                success_cnt = success_counts.get(key, 0)
                total_cnt = total_counts[key]
                ratio = success_cnt / total_cnt if total_cnt else 0.0
                avg_steps = _avg_str(steps_sum, steps_count, key)
                avg_spl = _avg_percent_str(spl_sum, spl_count, key)
                rows_for_table.append(
                    (str(key), str(success_cnt), str(total_cnt), _format_percent(ratio), avg_steps, avg_spl)
                )

            col0 = max(len(bucket_name), max(len(r[0]) for r in rows_for_table))
            col1 = max(len("success episodes"), max(len(r[1]) for r in rows_for_table))
            col2 = max(len(total_name), max(len(r[2]) for r in rows_for_table))
            col3 = max(len("success ratio"), max(len(r[3]) for r in rows_for_table))
            col4 = max(len("avg steps"), max(len(r[4]) for r in rows_for_table))
            col5 = max(len("row SPL"), max(len(r[5]) for r in rows_for_table))
            table_border = (
                f"  +{'-' * (col0 + 2)}+{'-' * (col1 + 2)}+{'-' * (col2 + 2)}+"
                f"{'-' * (col3 + 2)}+{'-' * (col4 + 2)}+{'-' * (col5 + 2)}+"
            )
            print(f"  {title}:")
            print(table_border)
            print(
                f"  | {bucket_name.ljust(col0)} | {'success episodes'.ljust(col1)} | "
                f"{total_name.ljust(col2)} | {'success ratio'.ljust(col3)} | "
                f"{'avg steps'.ljust(col4)} | {'row SPL'.ljust(col5)} |"
            )
            print(table_border)
            for value, success_cnt, total_cnt, ratio_str, avg_steps, avg_spl in rows_for_table:
                print(
                    f"  | {value.ljust(col0)} | {success_cnt.ljust(col1)} | "
                    f"{total_cnt.ljust(col2)} | {ratio_str.ljust(col3)} | "
                    f"{avg_steps.ljust(col4)} | {avg_spl.ljust(col5)} |"
                )
            print(table_border)

        def _print_fdr_by_bucket_table(
            title: str,
            bucket_name: str,
            keys: List[int],
            tp_counts: Dict[int, int],
            fp_counts: Dict[int, int],
            total_counts: Dict[int, int],
        ) -> None:
            if not keys:
                print(f"  {title}: None")
                return

            rows_for_table = []
            for key in keys:
                tp_cnt = tp_counts.get(key, 0)
                fp_cnt = fp_counts.get(key, 0)
                tp_fp_cnt = tp_cnt + fp_cnt
                fdr_str = _format_percent(fp_cnt / tp_fp_cnt) if tp_fp_cnt else "None"
                rows_for_table.append(
                    (str(key), str(tp_cnt), str(fp_cnt), str(tp_fp_cnt), fdr_str, str(total_counts.get(key, 0)))
                )

            col0 = max(len(bucket_name), max(len(r[0]) for r in rows_for_table))
            col1 = max(len("TP(success)"), max(len(r[1]) for r in rows_for_table))
            col2 = max(len("FP(distractor)"), max(len(r[2]) for r in rows_for_table))
            col3 = max(len("TP+FP"), max(len(r[3]) for r in rows_for_table))
            col4 = max(len("FDR"), max(len(r[4]) for r in rows_for_table))
            col5 = max(len("total episodes"), max(len(r[5]) for r in rows_for_table))
            table_border = (
                f"  +{'-' * (col0 + 2)}+{'-' * (col1 + 2)}+{'-' * (col2 + 2)}+"
                f"{'-' * (col3 + 2)}+{'-' * (col4 + 2)}+{'-' * (col5 + 2)}+"
            )
            print(f"  {title}:")
            print(table_border)
            print(
                f"  | {bucket_name.ljust(col0)} | {'TP(success)'.ljust(col1)} | "
                f"{'FP(distractor)'.ljust(col2)} | {'TP+FP'.ljust(col3)} | "
                f"{'FDR'.ljust(col4)} | {'total episodes'.ljust(col5)} |"
            )
            print(table_border)
            for value, tp_cnt, fp_cnt, tp_fp_cnt, fdr_str, total_cnt in rows_for_table:
                print(
                    f"  | {value.ljust(col0)} | {tp_cnt.ljust(col1)} | "
                    f"{fp_cnt.ljust(col2)} | {tp_fp_cnt.ljust(col3)} | "
                    f"{fdr_str.ljust(col4)} | {total_cnt.ljust(col5)} |"
                )
            print(table_border)

        def _print_stop_summary_by_bucket_table(
            title: str,
            bucket_name: str,
            keys: List[int],
            success_counts: Dict[int, int],
            total_counts: Dict[int, int],
            steps_sum: Dict[int, float],
            steps_count: Dict[int, int],
            spl_sum: Dict[int, float],
            spl_count: Dict[int, int],
        ) -> None:
            if not keys:
                print(f"  {title}: None")
                return

            rows_for_table = []
            for key in keys:
                success_cnt = success_counts.get(key, 0)
                total_cnt = total_counts.get(key, 0)
                sr_str = _format_percent(success_cnt / total_cnt) if total_cnt else "None"
                avg_steps = _avg_str(steps_sum, steps_count, key)
                avg_spl = _avg_percent_str(spl_sum, spl_count, key)
                rows_for_table.append((str(key), str(success_cnt), str(total_cnt), sr_str, avg_spl, avg_steps))

            col0 = max(len(bucket_name), max(len(r[0]) for r in rows_for_table))
            col1 = max(len("success episodes"), max(len(r[1]) for r in rows_for_table))
            col2 = max(len("stop_called episodes"), max(len(r[2]) for r in rows_for_table))
            col3 = max(len("SR"), max(len(r[3]) for r in rows_for_table))
            col4 = max(len("avg SPL"), max(len(r[4]) for r in rows_for_table))
            col5 = max(len("avg steps"), max(len(r[5]) for r in rows_for_table))
            table_border = (
                f"  +{'-' * (col0 + 2)}+{'-' * (col1 + 2)}+{'-' * (col2 + 2)}+"
                f"{'-' * (col3 + 2)}+{'-' * (col4 + 2)}+{'-' * (col5 + 2)}+"
            )
            print(f"  {title}:")
            print(table_border)
            print(
                f"  | {bucket_name.ljust(col0)} | {'success episodes'.ljust(col1)} | "
                f"{'stop_called episodes'.ljust(col2)} | {'SR'.ljust(col3)} | "
                f"{'avg SPL'.ljust(col4)} | {'avg steps'.ljust(col5)} |"
            )
            print(table_border)
            for value, success_cnt, total_cnt, sr_str, avg_spl, avg_steps in rows_for_table:
                print(
                    f"  | {value.ljust(col0)} | {success_cnt.ljust(col1)} | "
                    f"{total_cnt.ljust(col2)} | {sr_str.ljust(col3)} | "
                    f"{avg_spl.ljust(col4)} | {avg_steps.ljust(col5)} |"
                )
            print(table_border)

        def _print_sr_spl_steps_by_bucket_table(
            title: str,
            bucket_name: str,
            total_name: str,
            keys: List[int],
            success_counts: Dict[int, int],
            total_counts: Dict[int, int],
            steps_sum: Dict[int, float],
            steps_count: Dict[int, int],
            spl_sum: Dict[int, float],
            spl_count: Dict[int, int],
        ) -> None:
            if not keys:
                print(f"  {title}: None")
                return

            rows_for_table = []
            for key in keys:
                success_cnt = success_counts.get(key, 0)
                total_cnt = total_counts.get(key, 0)
                sr_str = _format_percent(success_cnt / total_cnt) if total_cnt else "None"
                avg_spl = _avg_percent_str(spl_sum, spl_count, key)
                avg_steps = _avg_str(steps_sum, steps_count, key)
                rows_for_table.append((str(key), str(success_cnt), str(total_cnt), sr_str, avg_spl, avg_steps))

            col0 = max(len(bucket_name), max(len(r[0]) for r in rows_for_table))
            col1 = max(len("success episodes"), max(len(r[1]) for r in rows_for_table))
            col2 = max(len(total_name), max(len(r[2]) for r in rows_for_table))
            col3 = max(len("SR"), max(len(r[3]) for r in rows_for_table))
            col4 = max(len("avg SPL"), max(len(r[4]) for r in rows_for_table))
            col5 = max(len("avg steps"), max(len(r[5]) for r in rows_for_table))
            table_border = (
                f"  +{'-' * (col0 + 2)}+{'-' * (col1 + 2)}+{'-' * (col2 + 2)}+"
                f"{'-' * (col3 + 2)}+{'-' * (col4 + 2)}+{'-' * (col5 + 2)}+"
            )
            print(f"  {title}:")
            print(table_border)
            print(
                f"  | {bucket_name.ljust(col0)} | {'success episodes'.ljust(col1)} | "
                f"{total_name.ljust(col2)} | {'SR'.ljust(col3)} | "
                f"{'avg SPL'.ljust(col4)} | {'avg steps'.ljust(col5)} |"
            )
            print(table_border)
            for value, success_cnt, total_cnt, sr_str, avg_spl, avg_steps in rows_for_table:
                print(
                    f"  | {value.ljust(col0)} | {success_cnt.ljust(col1)} | "
                    f"{total_cnt.ljust(col2)} | {sr_str.ljust(col3)} | "
                    f"{avg_spl.ljust(col4)} | {avg_steps.ljust(col5)} |"
                )
            print(table_border)

        def _print_cumulative_step_bin_table(
            title: str,
            records: List[Tuple[float, bool]],
            total_episode_count: int,
            *,
            bin_size: int = 50,
        ) -> None:
            if total_episode_count <= 0:
                print(f"  {title}: None")
                return
            if not records:
                print(f"  {title}: None (no total_steps records)")
                return

            max_step = max(step for step, _ in records)
            max_upper = int(math.ceil(max_step / bin_size) * bin_size)
            max_upper = max(bin_size, max_upper)
            rows_for_table = []

            prev_upper = 0
            for upper in range(bin_size, max_upper + 1, bin_size):
                cumulative_episode_count = sum(1 for step, _ in records if step <= float(upper))
                cumulative_success_count = sum(
                    1 for step, success_flag in records if success_flag and step <= float(upper)
                )
                ep_ratio = (
                    f"{cumulative_episode_count}/{total_episode_count} = "
                    f"{_format_percent(cumulative_episode_count / total_episode_count)}"
                )
                success_ratio = (
                    f"{cumulative_success_count}/{total_episode_count} = "
                    f"{_format_percent(cumulative_success_count / total_episode_count)}"
                )
                rows_for_table.append((f"{prev_upper}-{upper}", ep_ratio, success_ratio))
                prev_upper = upper

            col0 = max(len("step bin"), max(len(r[0]) for r in rows_for_table))
            col1 = max(len("cum episodes / total"), max(len(r[1]) for r in rows_for_table))
            col2 = max(len("cum success / total"), max(len(r[2]) for r in rows_for_table))
            table_border = (
                f"  +{'-' * (col0 + 2)}+{'-' * (col1 + 2)}+{'-' * (col2 + 2)}+"
            )
            print(f"  {title}:")
            print(table_border)
            print(
                f"  | {'step bin'.ljust(col0)} | {'cum episodes / total'.ljust(col1)} | "
                f"{'cum success / total'.ljust(col2)} |"
            )
            print(table_border)
            for bin_label, ep_ratio, success_ratio in rows_for_table:
                print(
                    f"  | {bin_label.ljust(col0)} | {ep_ratio.ljust(col1)} | "
                    f"{success_ratio.ljust(col2)} |"
                )
            print(table_border)

        def _print_step_bin_table(
            title: str,
            records: List[Tuple[float, bool]],
            *,
            bin_size: int = 100,
        ) -> None:
            if not records:
                print(f"  {title}: None (no total_steps records)")
                return

            total_episode_count = len(records)
            episode_counts: Dict[int, int] = defaultdict(int)
            success_counts: Dict[int, int] = defaultdict(int)
            for step, success_flag in records:
                step_value = max(0, int(step))
                bin_start = (step_value // bin_size) * bin_size
                episode_counts[bin_start] += 1
                if success_flag:
                    success_counts[bin_start] += 1

            if not episode_counts:
                print(f"  {title}: None (no valid step bins)")
                return

            max_bin_start = max(episode_counts.keys())
            rows_for_table = []
            cumulative_episode_count = 0
            cumulative_success_count = 0
            for bin_start in range(0, max_bin_start + 1, bin_size):
                bin_end = bin_start + bin_size - 1
                episode_count = episode_counts.get(bin_start, 0)
                success_count = success_counts.get(bin_start, 0)
                cumulative_episode_count += episode_count
                cumulative_success_count += success_count
                ratio_str = (
                    f"{success_count}/{episode_count} = "
                    f"{_format_percent(success_count / episode_count)}"
                    if episode_count
                    else "0/0 = None"
                )
                cumulative_ratio_str = (
                    f"{cumulative_success_count}/{total_episode_count} = "
                    f"{_format_percent(cumulative_success_count / total_episode_count)}"
                    if total_episode_count
                    else "0/0 = None"
                )
                rows_for_table.append(
                    (
                        f"{bin_start}-{bin_end}",
                        str(episode_count),
                        str(success_count),
                        ratio_str,
                        cumulative_ratio_str,
                    )
                )

            col0 = max(len("step bin"), max(len(r[0]) for r in rows_for_table))
            col1 = max(len("number of episodes"), max(len(r[1]) for r in rows_for_table))
            col2 = max(len("number of success episode"), max(len(r[2]) for r in rows_for_table))
            col3 = max(
                len("number of success episode / number of episodes"),
                max(len(r[3]) for r in rows_for_table),
            )
            col4 = max(
                len("cumulative success rate v2"),
                max(len(r[4]) for r in rows_for_table),
            )
            table_border = (
                f"  +{'-' * (col0 + 2)}+{'-' * (col1 + 2)}+{'-' * (col2 + 2)}+"
                f"{'-' * (col3 + 2)}+{'-' * (col4 + 2)}+"
            )
            print(f"  {title}:")
            print(table_border)
            print(
                f"  | {'step bin'.ljust(col0)} | {'number of episodes'.ljust(col1)} | "
                f"{'number of success episode'.ljust(col2)} | "
                f"{'number of success episode / number of episodes'.ljust(col3)} | "
                f"{'cumulative success rate v2'.ljust(col4)} |"
            )
            print(table_border)
            for bin_label, episode_count, success_count, ratio_str, cumulative_ratio_str in rows_for_table:
                print(
                    f"  | {bin_label.ljust(col0)} | {episode_count.ljust(col1)} | "
                    f"{success_count.ljust(col2)} | {ratio_str.ljust(col3)} | "
                    f"{cumulative_ratio_str.ljust(col4)} |"
                )
            print(table_border)

        detected_keys = sorted(global_detected_total_counts.keys())
        _print_success_by_bucket_table(
            "Success by num_detected_objects",
            "num_detected_objects",
            "total episodes",
            detected_keys,
            global_detected_success_counts,
            global_detected_total_counts,
            global_detected_steps_sum,
            global_detected_steps_count,
            global_detected_spl_sum,
            global_detected_spl_count,
        )
        _print_top3_episodes(
            "Top-3 episodes by num_detected_objects",
            "num_detected_objects",
            global_detected_episode_records,
        )

        reached_detected_keys = sorted(global_reached_detected_total_counts.keys())
        _print_success_by_bucket_table(
            "reached_episodes - Success by num_detected_objects",
            "num_detected_objects",
            "reached episodes",
            reached_detected_keys,
            global_reached_detected_success_counts,
            global_reached_detected_total_counts,
            global_reached_detected_steps_sum,
            global_reached_detected_steps_count,
            global_reached_detected_spl_sum,
            global_reached_detected_spl_count,
        )
        _print_top3_episodes(
            "Top-3 reached episodes by num_detected_objects",
            "num_detected_objects",
            global_reached_detected_episode_records,
        )

        merged_keys = sorted(global_merged_total_counts.keys())
        _print_success_by_bucket_table(
            "Success by num_merged_instances",
            "num_merged_instances",
            "total episodes",
            merged_keys,
            global_merged_success_counts,
            global_merged_total_counts,
            global_merged_steps_sum,
            global_merged_steps_count,
            global_merged_spl_sum,
            global_merged_spl_count,
        )
        _print_top3_episodes(
            "Top-3 episodes by num_merged_instances",
            "num_merged_instances",
            global_merged_episode_records,
        )

        reached_merged_keys = sorted(global_reached_merged_total_counts.keys())
        _print_success_by_bucket_table(
            "reached_episodes - Success by num_merged_instances",
            "num_merged_instances",
            "reached episodes",
            reached_merged_keys,
            global_reached_merged_success_counts,
            global_reached_merged_total_counts,
            global_reached_merged_steps_sum,
            global_reached_merged_steps_count,
            global_reached_merged_spl_sum,
            global_reached_merged_spl_count,
        )
        _print_top3_episodes(
            "Top-3 reached episodes by num_merged_instances",
            "num_merged_instances",
            global_reached_merged_episode_records,
        )

        collected_object_keys = sorted(global_collected_object_total_counts.keys())
        _print_fdr_by_bucket_table(
            "FDR by collected_object_count",
            "collected_object_count",
            collected_object_keys,
            global_collected_object_tp_counts,
            global_collected_object_fp_counts,
            global_collected_object_total_counts,
        )
        _print_sr_spl_steps_by_bucket_table(
            "SR/SPL/avg steps by collected_object_count",
            "collected_object_count",
            "total episodes",
            collected_object_keys,
            global_collected_object_tp_counts,
            global_collected_object_total_counts,
            global_collected_object_steps_sum,
            global_collected_object_steps_count,
            global_collected_object_spl_sum,
            global_collected_object_spl_count,
        )
        stop_collected_object_keys = sorted(global_stop_collected_object_total_counts.keys())
        _print_fdr_by_bucket_table(
            "FDR_stop by collected_object_count",
            "collected_object_count",
            stop_collected_object_keys,
            global_stop_collected_object_tp_counts,
            global_stop_collected_object_fp_counts,
            global_stop_collected_object_total_counts,
        )
        _print_stop_summary_by_bucket_table(
            "stop_called=True by collected_object_count (SR/SPL/avg steps)",
            "collected_object_count",
            stop_collected_object_keys,
            global_stop_collected_object_tp_counts,
            global_stop_collected_object_total_counts,
            global_stop_collected_object_steps_sum,
            global_stop_collected_object_steps_count,
            global_stop_collected_object_spl_sum,
            global_stop_collected_object_spl_count,
        )
        _print_step_bin_table(
            f"Step-bin summary (non-cumulative, bin={step_bin_size})",
            global_step_success_records,
            bin_size=step_bin_size,
        )


if __name__ == "__main__":
    main()
