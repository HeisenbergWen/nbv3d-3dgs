#!/usr/bin/env python3
"""Compute two-seed rendering disagreement heatmaps for 3DGS outputs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}


@dataclass
class RoiCandidate:
    image: str
    rank: int
    score: float
    bbox_xywh: list[int]
    patch_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two 3DGS render folders from different random seeds and "
            "generate RGB disagreement maps, heatmaps, overlays, and top-k "
            "patch ROI candidates."
        )
    )
    parser.add_argument(
        "--seed-a-renders",
        type=Path,
        required=True,
        help="Render directory for the first seed, e.g. output/...seed0/train/ours_30000/renders.",
    )
    parser.add_argument(
        "--seed-b-renders",
        type=Path,
        required=True,
        help="Render directory for the second seed, e.g. output/...seed1/train/ours_30000/renders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where disagreement outputs will be written.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=64,
        help="Patch size for ROI scoring. Default: 64.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top patches saved per image. Default: 5.",
    )
    parser.add_argument(
        "--heatmap-percentile",
        type=float,
        default=99.0,
        help="Percentile used to normalize heatmap visualization. Default: 99.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.45,
        help="Heatmap opacity in overlay images. Default: 0.45.",
    )
    parser.add_argument(
        "--resize-b-to-a",
        action="store_true",
        help="Resize seed B images to seed A size if dimensions differ.",
    )
    return parser.parse_args()


def list_images(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Render directory does not exist: {folder}")
    images = {
        path.name: path
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix in IMAGE_SUFFIXES
    }
    if not images:
        raise RuntimeError(f"No image files found in: {folder}")
    return images


def load_rgb(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def to_uint8(array: np.ndarray) -> np.ndarray:
    return np.clip(array * 255.0, 0, 255).astype(np.uint8)


def normalize_for_display(values: np.ndarray, percentile: float) -> np.ndarray:
    vmax = float(np.percentile(values, percentile))
    if vmax <= 1e-8:
        vmax = float(values.max())
    if vmax <= 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip(values / vmax, 0.0, 1.0).astype(np.float32)


def apply_colormap(values: np.ndarray) -> np.ndarray:
    """Apply a small blue-cyan-yellow-red colormap to values in [0, 1]."""
    stops = np.array(
        [
            [0.00, 0.00, 0.08],
            [0.00, 0.25, 0.75],
            [0.00, 0.85, 0.95],
            [1.00, 0.90, 0.10],
            [1.00, 0.05, 0.00],
        ],
        dtype=np.float32,
    )
    x = np.clip(values, 0.0, 1.0)
    scaled = x * (len(stops) - 1)
    lower = np.floor(scaled).astype(np.int32)
    upper = np.clip(lower + 1, 0, len(stops) - 1)
    weight = (scaled - lower)[..., None]
    return stops[lower] * (1.0 - weight) + stops[upper] * weight


def patch_candidates(diff: np.ndarray, patch_size: int, top_k: int, image_name: str) -> list[RoiCandidate]:
    height, width = diff.shape
    candidates: list[RoiCandidate] = []

    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            patch = diff[y : min(y + patch_size, height), x : min(x + patch_size, width)]
            score = float(patch.mean())
            candidates.append(
                RoiCandidate(
                    image=image_name,
                    rank=0,
                    score=score,
                    bbox_xywh=[x, y, int(patch.shape[1]), int(patch.shape[0])],
                    patch_size=patch_size,
                )
            )

    candidates.sort(key=lambda item: item.score, reverse=True)
    top = candidates[:top_k]
    for rank, candidate in enumerate(top, start=1):
        candidate.rank = rank
    return top


def draw_roi_boxes(image: np.ndarray, rois: list[RoiCandidate]) -> Image.Image:
    output = Image.fromarray(to_uint8(image))
    draw = ImageDraw.Draw(output)
    for roi in rois:
        x, y, w, h = roi.bbox_xywh
        color = (255, 32, 32) if roi.rank == 1 else (255, 220, 0)
        draw.rectangle([x, y, x + w - 1, y + h - 1], outline=color, width=3)
        draw.text((x + 4, y + 4), f"{roi.rank}:{roi.score:.4f}", fill=color)
    return output


def save_image(path: Path, array: np.ndarray) -> None:
    Image.fromarray(to_uint8(array)).save(path)


def compare_pair(
    image_name: str,
    path_a: Path,
    path_b: Path,
    output_dir: Path,
    patch_size: int,
    top_k: int,
    heatmap_percentile: float,
    overlay_alpha: float,
    resize_b_to_a: bool,
) -> tuple[list[RoiCandidate], dict[str, float]]:
    image_a = load_rgb(path_a)
    image_b = load_rgb(path_b)

    if image_a.shape != image_b.shape:
        if not resize_b_to_a:
            raise ValueError(
                f"Image size mismatch for {image_name}: "
                f"{path_a} has {image_a.shape}, {path_b} has {image_b.shape}. "
                "Use --resize-b-to-a if this mismatch is expected."
            )
        resized = Image.open(path_b).convert("RGB").resize(
            (image_a.shape[1], image_a.shape[0]), Image.Resampling.BILINEAR
        )
        image_b = np.asarray(resized, dtype=np.float32) / 255.0

    abs_rgb = np.abs(image_a - image_b)
    diff = abs_rgb.mean(axis=2)
    display = normalize_for_display(diff, heatmap_percentile)
    heatmap = apply_colormap(display)
    overlay = np.clip(image_a * (1.0 - overlay_alpha) + heatmap * overlay_alpha, 0.0, 1.0)

    stem = Path(image_name).stem
    save_image(output_dir / "rgb_diff" / f"{stem}_diff.png", abs_rgb)
    save_image(output_dir / "heatmap" / f"{stem}_heatmap.png", heatmap)
    save_image(output_dir / "overlay" / f"{stem}_overlay.png", overlay)

    rois = patch_candidates(diff, patch_size, top_k, image_name)
    draw_roi_boxes(overlay, rois).save(output_dir / "roi_visualization" / f"{stem}_roi.png")

    summary = {
        "mean_disagreement": float(diff.mean()),
        "max_disagreement": float(diff.max()),
        "p95_disagreement": float(np.percentile(diff, 95)),
        "p99_disagreement": float(np.percentile(diff, 99)),
    }
    return rois, summary


def main() -> None:
    args = parse_args()
    seed_a = list_images(args.seed_a_renders)
    seed_b = list_images(args.seed_b_renders)
    common_names = sorted(set(seed_a) & set(seed_b))
    missing_a = sorted(set(seed_b) - set(seed_a))
    missing_b = sorted(set(seed_a) - set(seed_b))

    if not common_names:
        raise RuntimeError("No matching image names found between the two render directories.")

    output_dir = args.output_dir.resolve()
    for child in ["rgb_diff", "heatmap", "overlay", "roi_visualization"]:
        (output_dir / child).mkdir(parents=True, exist_ok=True)

    all_rois: list[RoiCandidate] = []
    image_summaries: dict[str, dict[str, float]] = {}

    print(f"Seed A renders: {args.seed_a_renders.resolve()}")
    print(f"Seed B renders: {args.seed_b_renders.resolve()}")
    print(f"Output dir: {output_dir}")
    print(f"Matched images: {len(common_names)}")
    if missing_a:
        print(f"Images only in seed B: {len(missing_a)}")
    if missing_b:
        print(f"Images only in seed A: {len(missing_b)}")

    for image_name in common_names:
        rois, summary = compare_pair(
            image_name=image_name,
            path_a=seed_a[image_name],
            path_b=seed_b[image_name],
            output_dir=output_dir,
            patch_size=args.patch_size,
            top_k=args.top_k,
            heatmap_percentile=args.heatmap_percentile,
            overlay_alpha=args.overlay_alpha,
            resize_b_to_a=args.resize_b_to_a,
        )
        all_rois.extend(rois)
        image_summaries[image_name] = summary

    all_rois.sort(key=lambda item: item.score, reverse=True)
    payload = {
        "seed_a_renders": str(args.seed_a_renders.resolve()),
        "seed_b_renders": str(args.seed_b_renders.resolve()),
        "patch_size": args.patch_size,
        "top_k_per_image": args.top_k,
        "matched_images": len(common_names),
        "missing_from_seed_a": missing_a,
        "missing_from_seed_b": missing_b,
        "image_summaries": image_summaries,
        "roi_candidates": [asdict(candidate) for candidate in all_rois],
    }

    with (output_dir / "roi_candidates.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Wrote ROI candidates: {output_dir / 'roi_candidates.json'}")
    print("Done.")


if __name__ == "__main__":
    main()
