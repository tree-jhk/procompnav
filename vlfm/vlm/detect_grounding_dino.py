#!/usr/bin/env python3

# python vlfm/vlm/detect_grounding_dino.py --target "chair" --box-threshold 0.35 --text-threshold 0.25

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlfm.vlm.grounding_dino import GroundingDINO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def make_caption(target: str) -> str:
    target = target.strip()
    if not target:
        raise ValueError("--target must not be empty")
    return f"{target} ."


def slugify_target(target: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", target.strip().lower()).strip("_")
    return slug or "target"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect objects with GroundingDINO.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=SCRIPT_DIR / "data",
        help="Directory with input images.",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help='Target object text to detect (e.g., "tv monitor", "chair").',
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save annotated images and results.json.",
    )
    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


def list_images(input_dir: Path) -> List[Path]:
    return sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def load_rgb_image(path: Path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_rgb_image(path: Path, image_rgb) -> None:
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(path), bgr)
    if not ok:
        raise ValueError(f"Failed to write image: {path}")


def main() -> None:
    args = parse_args()
    caption = make_caption(args.target)

    input_dir = args.input_dir.resolve()
    if args.output_dir is None:
        output_dir = (SCRIPT_DIR / "data" / f"{slugify_target(args.target)}_results").resolve()
    else:
        output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input dir not found: {input_dir}")

    image_paths = list_images(input_dir)
    if not image_paths:
        print(f"No images found in: {input_dir}")
        return

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        device = "cpu"

    config_path = PROJECT_ROOT / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
    weights_path = PROJECT_ROOT / "data" / "groundingdino_swint_ogc.pth"
    detector = GroundingDINO(
        config_path=str(config_path),
        weights_path=str(weights_path),
        caption=caption,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=torch.device(device),
    )

    all_results: Dict[str, List[dict]] = {}
    total_dets = 0

    for image_path in image_paths:
        image_rgb = load_rgb_image(image_path)
        detections = detector.predict(image_rgb, caption=caption)

        det_items: List[dict] = []
        for box, logit, phrase in zip(detections.boxes, detections.logits, detections.phrases):
            det_items.append(
                {
                    "phrase": str(phrase),
                    "score": float(logit.item()),
                    "box_xyxy": [float(v) for v in box.tolist()],
                }
            )

        all_results[image_path.name] = det_items
        total_dets += len(det_items)

        print(f"{image_path.name}: {len(det_items)} detections")
        annotated = detections.annotated_frame if detections.annotated_frame is not None else image_rgb
        out_img = output_dir / f"{image_path.stem}_annotated{image_path.suffix}"
        save_rgb_image(out_img, annotated)

    results_path = output_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(image_paths)} images")
    print(f"Total detections: {total_dets}")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
