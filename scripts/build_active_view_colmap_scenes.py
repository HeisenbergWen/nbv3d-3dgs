#!/usr/bin/env python3
"""Build active-view scenes that preserve a shared full-scene COLMAP frame."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
SPARSE_TEXT_FILES = ("cameras.txt", "images.txt", "points3D.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one 3DGS training scene per active-view run by filtering a "
            "full-scene COLMAP model. This keeps train and test cameras in the "
            "same coordinate frame, enabling holdout test-view rendering."
        )
    )
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--selected-views", type=Path, required=True)
    parser.add_argument(
        "--source-scene",
        type=Path,
        required=True,
        help="Full 3DGS/COLMAP scene root containing source images and sparse model.",
    )
    parser.add_argument("--source-image-dir", default="input")
    parser.add_argument("--source-sparse-dir", default="sparse/0")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--image-dir-name", default="images")
    parser.add_argument("--copy-mode", choices=("copy", "symlink"), default="copy")
    parser.add_argument("--colmap-bin", default="colmap")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_id(method: str, k: int, seed: int | None) -> str:
    if method == "base_6":
        return "base_6"
    suffix = f"_seed{seed}" if seed is not None else ""
    return f"{method}_k{k}{suffix}"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def link_or_copy(src: Path, dst: Path, copy_mode: str) -> None:
    if copy_mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def sparse_has_text(path: Path) -> bool:
    return all((path / name).is_file() for name in SPARSE_TEXT_FILES)


def sparse_has_binary(path: Path) -> bool:
    return all((path / name).is_file() for name in ("cameras.bin", "images.bin", "points3D.bin"))


def prepare_sparse_text(source_sparse: Path, output_root: Path, colmap_bin: str) -> Path:
    if sparse_has_text(source_sparse):
        return source_sparse
    if not sparse_has_binary(source_sparse):
        raise FileNotFoundError(
            f"Expected COLMAP text or binary sparse model in {source_sparse}. "
            "Need cameras/images/points3D as .txt or .bin files."
        )

    text_dir = output_root / "_source_sparse_text"
    if sparse_has_text(text_dir):
        return text_dir
    if text_dir.exists():
        shutil.rmtree(text_dir)
    text_dir.mkdir(parents=True)
    subprocess.run(
        [
            colmap_bin,
            "model_converter",
            "--input_path",
            str(source_sparse),
            "--output_path",
            str(text_dir),
            "--output_type",
            "TXT",
        ],
        check=True,
    )
    return text_dir


def colmap_image_name(image_record_line: str) -> str:
    parts = image_record_line.strip().split()
    if len(parts) < 10:
        raise ValueError(f"Invalid COLMAP image record: {image_record_line!r}")
    return Path(parts[9]).name


def filter_images_txt(src: Path, dst: Path, allowed_names: set[str]) -> int:
    lines = src.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    kept = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.lstrip().startswith("#"):
            out.append(line)
            i += 1
            continue
        if i + 1 >= len(lines):
            raise ValueError(f"COLMAP images.txt has an image record without points2D line: {line!r}")
        points_line = lines[i + 1]
        image_name = colmap_image_name(line)
        if image_name in allowed_names:
            out.extend([line, points_line])
            kept += 1
        i += 2

    missing = sorted(allowed_names - collect_image_names(src))
    if missing:
        preview = ", ".join(missing[:10])
        suffix = " ..." if len(missing) > 10 else ""
        raise RuntimeError(f"Missing {len(missing)} selected images from source COLMAP model: {preview}{suffix}")

    dst.write_text("\n".join(out) + "\n", encoding="utf-8")
    return kept


def collect_image_names(images_txt: Path) -> set[str]:
    names: set[str] = set()
    lines = images_txt.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.lstrip().startswith("#"):
            i += 1
            continue
        names.add(colmap_image_name(line))
        i += 2
    return names


def copy_sparse_subset(source_text_sparse: Path, dst_sparse: Path, allowed_names: set[str]) -> int:
    dst_sparse.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source_text_sparse / "cameras.txt", dst_sparse / "cameras.txt")
    shutil.copy2(source_text_sparse / "points3D.txt", dst_sparse / "points3D.txt")
    return filter_images_txt(source_text_sparse / "images.txt", dst_sparse / "images.txt", allowed_names)


def validate_source_images(source_images: Path, image_names: set[str]) -> None:
    missing = sorted(name for name in image_names if not (source_images / name).is_file())
    if missing:
        preview = ", ".join(missing[:10])
        suffix = " ..." if len(missing) > 10 else ""
        raise FileNotFoundError(f"Missing {len(missing)} source images in {source_images}: {preview}{suffix}")


def main() -> None:
    args = parse_args()
    split = load_json(args.split_manifest)
    selections = load_json(args.selected_views)
    source_scene = args.source_scene.resolve()
    source_images = (source_scene / args.source_image_dir).resolve()
    source_sparse = (source_scene / args.source_sparse_dir).resolve()
    output_root = args.output_root.resolve()

    if not source_images.is_dir():
        raise FileNotFoundError(f"Source image directory does not exist: {source_images}")
    if not source_sparse.is_dir():
        raise FileNotFoundError(f"Source sparse directory does not exist: {source_sparse}")

    output_root.mkdir(parents=True, exist_ok=True)
    source_text_sparse = prepare_sparse_text(source_sparse, output_root, args.colmap_bin)

    base_views = [item for item in split["images"] if item["role"] == "base"]
    test_views = [item for item in split["images"] if item["role"] == "test"]
    candidate_names = {item["name"] for item in split["images"] if item["role"] == "candidate"}
    base_names = {item["name"] for item in base_views}
    test_names = {item["name"] for item in test_views}
    all_split_names = {item["name"] for item in split["images"]}
    validate_source_images(source_images, all_split_names)

    runs_manifest: dict = {
        "split_manifest": str(args.split_manifest.resolve()),
        "selected_views": str(args.selected_views.resolve()),
        "source_scene": str(source_scene),
        "source_image_dir": args.source_image_dir,
        "source_sparse_dir": str(source_sparse),
        "source_sparse_text_dir": str(source_text_sparse),
        "output_root": str(output_root),
        "image_dir_name": args.image_dir_name,
        "copy_mode": args.copy_mode,
        "colmap_coordinate_frame": "shared_source_scene",
        "runs": [],
    }

    for run in selections["runs"]:
        method = run["method"]
        k = int(run["k"])
        seed = run.get("seed")
        selected = run["selected"]
        selected_names = {item["name"] for item in selected}
        if selected_names & test_names:
            raise RuntimeError(f"Run {method} k={k} seed={seed} selected test views: {selected_names & test_names}")
        if not selected_names <= candidate_names:
            raise RuntimeError(f"Run {method} k={k} seed={seed} selected non-candidate views")

        train_items = sorted(base_views + selected, key=lambda item: int(item["index"]))
        train_names = {item["name"] for item in train_items}
        if len(train_names) != len(train_items):
            raise RuntimeError(f"Duplicate training image in run {method} k={k} seed={seed}")
        if not base_names <= train_names:
            raise RuntimeError(f"Run {method} k={k} seed={seed} is missing base views")
        if train_names & test_names:
            raise RuntimeError(f"Run {method} k={k} seed={seed} includes test views")

        rid = run_id(method, k, seed)
        scene_dir = output_root / rid
        image_dir = scene_dir / args.image_dir_name
        sparse_dir = scene_dir / "sparse" / "0"
        if scene_dir.exists():
            if not args.overwrite:
                raise FileExistsError(f"Scene already exists: {scene_dir}. Use --overwrite to replace it.")
            shutil.rmtree(scene_dir)
        image_dir.mkdir(parents=True, exist_ok=False)

        selected_txt = scene_dir / "selected_train_views.txt"
        with selected_txt.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("# role\tindex\tfilename\n")
            for item in train_items:
                src = source_images / item["name"]
                link_or_copy(src, image_dir / item["name"], args.copy_mode)
                role = "base" if item["name"] in base_names else "selected_candidate"
                handle.write(f"{role}\t{item['index']}\t{item['name']}\n")

        kept_cameras = copy_sparse_subset(source_text_sparse, sparse_dir, train_names)
        if kept_cameras != len(train_items):
            raise RuntimeError(f"Expected {len(train_items)} COLMAP image records for {rid}, got {kept_cameras}")

        run_manifest = {
            "run_id": rid,
            "method": method,
            "k": k,
            "seed": seed,
            "scene_dir": str(scene_dir),
            "image_dir": str(image_dir),
            "sparse_dir": str(sparse_dir),
            "source_scene": str(source_scene),
            "source_image_dir": args.source_image_dir,
            "source_sparse_dir": str(source_sparse),
            "colmap_coordinate_frame": "shared_source_scene",
            "train_views": train_items,
            "test_views": test_views,
            "selected_train_views": str(selected_txt),
            "train_command_constraints": {
                "iterations": 30000,
                "must_not_use_start_checkpoint": True,
                "must_not_run_convert": True,
            },
        }
        (scene_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
        runs_manifest["runs"].append(run_manifest)
        print(f"Created {rid}: {len(train_items)} train images in shared COLMAP frame")

    runs_manifest_path = output_root / "active_view_runs_manifest.json"
    runs_manifest_path.write_text(json.dumps(runs_manifest, indent=2), encoding="utf-8")
    print(f"Wrote {runs_manifest_path}")


if __name__ == "__main__":
    main()
