#!/usr/bin/env python3
"""Create base/candidate/test splits for active-view 3DGS experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


@dataclass(frozen=True)
class SplitImage:
    index: int
    name: str
    path: str
    role: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a full public dataset scene into base, candidate, and test "
            "views for defect-guided active-view 3DGS experiments."
        )
    )
    parser.add_argument("--src-scene", type=Path, required=True)
    parser.add_argument("--src-image-dir", default="input")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-name", default=None)
    parser.add_argument("--base-count", type=int, default=6)
    parser.add_argument("--test-stride", type=int, default=8)
    parser.add_argument("--manifest-name", default="split_manifest.json")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def list_images(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    images = sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix in IMAGE_SUFFIXES)
    if not images:
        raise RuntimeError(f"No images found in: {image_dir}")
    return images


def uniform_indices(indices: list[int], count: int) -> list[int]:
    if count <= 0:
        raise ValueError("count must be positive")
    if count > len(indices):
        raise ValueError(f"Requested {count} indices, but only {len(indices)} are available")
    if count == 1:
        return [indices[len(indices) // 2]]
    selected_positions = [round(i * (len(indices) - 1) / (count - 1)) for i in range(count)]
    return [indices[pos] for pos in selected_positions]


def build_split(images: list[Path], base_count: int, test_stride: int) -> list[SplitImage]:
    if test_stride <= 1:
        raise ValueError("--test-stride must be greater than 1")
    all_indices = list(range(len(images)))
    test_indices = set(all_indices[::test_stride])
    train_pool = [idx for idx in all_indices if idx not in test_indices]
    base_indices = set(uniform_indices(train_pool, base_count))
    candidate_indices = set(train_pool) - base_indices

    split: list[SplitImage] = []
    for idx, image in enumerate(images):
        if idx in test_indices:
            role = "test"
        elif idx in base_indices:
            role = "base"
        elif idx in candidate_indices:
            role = "candidate"
        else:
            raise AssertionError(f"Unassigned image index: {idx}")
        split.append(SplitImage(index=idx, name=image.name, path=str(image.resolve()), role=role))
    return split


def main() -> None:
    args = parse_args()
    src_scene = args.src_scene.resolve()
    src_image_dir = (src_scene / args.src_image_dir).resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / args.manifest_name

    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"Manifest already exists: {manifest_path}. Use --overwrite to replace it.")

    images = list_images(src_image_dir)
    split = build_split(images, args.base_count, args.test_stride)
    scene_name = args.scene_name or src_scene.name

    counts = {
        "base": sum(item.role == "base" for item in split),
        "candidate": sum(item.role == "candidate" for item in split),
        "test": sum(item.role == "test" for item in split),
    }
    if counts["base"] != args.base_count:
        raise RuntimeError(f"Expected {args.base_count} base views, got {counts['base']}")
    if not counts["candidate"]:
        raise RuntimeError("Candidate pool is empty")
    if not counts["test"]:
        raise RuntimeError("Test split is empty")

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "scene_name": scene_name,
        "source_scene": str(src_scene),
        "source_image_dir": str(src_image_dir),
        "base_count": args.base_count,
        "test_stride": args.test_stride,
        "counts": counts,
        "images": [asdict(item) for item in split],
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote {manifest_path}")
    print(f"Counts: {counts}")


if __name__ == "__main__":
    main()
