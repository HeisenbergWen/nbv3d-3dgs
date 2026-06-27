#!/usr/bin/env python3
"""Run closed-loop next-best-view selection for LLFF 3DGS experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from build_active_view_colmap_scenes import (
    copy_sparse_subset,
    link_or_copy,
    prepare_sparse_text,
    validate_source_images,
)
from colmap_text_utils import (
    ColmapModel,
    ImageRecord,
    angle_between,
    camera_center,
    load_colmap_text_model,
    project_point,
    vector_norm,
    vector_sub,
    view_direction,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFECT_SCORE_POLICIES = {"nbv3d", "nearest3d"}
POSE_SCORE_POLICIES = {"nbv3d", "nearest3d", "farthest", "maxcoverage", "maxparallax"}
SCORE_VARIANTS = {
    "full": (0.45, 0.25, 0.20, 0.10),
    "wo_p": (0.45, 0.00, 0.20, 0.10),
    "wo_c": (0.45, 0.25, 0.00, 0.10),
    "wo_r": (0.45, 0.25, 0.20, 0.00),
    "v_only": (1.00, 0.00, 0.00, 0.00),
    "p_only": (0.00, 1.00, 0.00, 0.00),
    "c_only": (0.00, 0.00, 1.00, 0.00),
    "wo_v": (0.00, 0.25, 0.20, 0.10),
    "v_p": (0.45, 0.25, 0.00, 0.00),
    "v_p_c": (0.45, 0.25, 0.20, 0.00),
}


@dataclass(frozen=True)
class ScoreBreakdown:
    index: int
    name: str
    score: float
    defect_visibility: float
    parallax_gain: float
    coverage_gain: float
    redundancy_penalty: float
    nearest_defect_distance: float
    nearest_train_distance: float


@dataclass(frozen=True)
class DefectPoint:
    point3d_id: int
    xyz: tuple[float, float, float]
    weight: float
    hit_count: int
    source_images: tuple[str, ...]


@dataclass(frozen=True)
class PoseScoreBreakdown:
    index: int
    name: str
    score: float
    defect_visibility: float
    parallax_gain: float
    coverage_gain: float
    redundancy_penalty: float
    visible_defect_points: int
    weighted_visible_defect: float
    nearest_train_distance: float
    camera_center: tuple[float, float, float]
    view_direction: tuple[float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Closed-loop NBV runner for LLFF 3DGS. Each round adds real candidate "
            "views, computes disagreement when required by the selection policy, "
            "and evaluates holdout test views. By default, each new train output "
            "starts from scratch; --resume-training is only for interrupted "
            "outputs with existing checkpoints."
        )
    )
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--source-scene", type=Path, required=True)
    parser.add_argument("--source-image-dir", default="images")
    parser.add_argument("--source-sparse-dir", default="sparse/0")
    parser.add_argument("--gs-dir", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--shared-init-root",
        type=Path,
        default=None,
        help=(
            "Directory for shared Round-0 scene/model artifacts. When set, "
            "round 0 uses <root>/<scene>/base<N>/scene and seed<seed>/ "
            "outputs across all methods."
        ),
    )
    parser.add_argument("--scene-name", default="llff_horns")
    parser.add_argument("--base-count", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--views-per-round", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument(
        "--optimize-seed-schedule",
        action="store_true",
        help=(
            "Use two seeds only in pose-aware rounds that select a next view. "
            "Final pose-aware rounds and all random rounds use only --eval-seed."
        ),
    )
    parser.add_argument(
        "--evaluation-roi-candidates",
        type=Path,
        default=None,
        help=(
            "Use a shared holdout ROI candidates file for final evaluation. "
            "Required for optimized random runs, which do not train a second seed."
        ),
    )
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument(
        "--pass-llffhold-to-render",
        action="store_true",
        help=(
            "Pass --llffhold to render.py. Keep disabled for official 3DGS "
            "render.py versions that do not expose this argument."
        ),
    )
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--top-k-roi", type=int, default=5)
    parser.add_argument(
        "--policy",
        choices=("nbv3d", "nearest3d", "farthest", "maxcoverage", "maxparallax", "nbv", "nearest", "random"),
        default="nbv3d",
        help="Closed-loop selection policy. Default: nbv3d.",
    )
    parser.add_argument(
        "--score-variant",
        choices=tuple(SCORE_VARIANTS),
        default="full",
        help=(
            "Score ablation used by nbv3d-style pose scoring. Minimal publishable "
            "matrix uses full, wo_p, wo_c, wo_r, and v_only."
        ),
    )
    parser.add_argument(
        "--score-weights",
        nargs=4,
        type=float,
        metavar=("V", "P", "C", "R"),
        default=None,
        help="Override --score-variant with explicit V P C R weights. R is subtracted.",
    )
    parser.add_argument(
        "--method-label",
        default=None,
        help="Stable label used in run IDs and reports. Defaults to policy or policy_scorevariant.",
    )
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--min-defect-points", type=int, default=20)
    parser.add_argument(
        "--allow-low-defect-points",
        action="store_true",
        help=(
            "Compatibility flag. Nonzero low defect point counts now warn and "
            "continue by default; zero mapped points still fail."
        ),
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--copy-mode", choices=("copy", "symlink"), default="copy")
    parser.add_argument("--colmap-bin", default="colmap")
    parser.add_argument(
        "--resume-training",
        action="store_true",
        help="Resume incomplete train outputs through batch_run_active_view_3dgs.py.",
    )
    parser.add_argument(
        "--checkpoint-iterations",
        nargs="+",
        type=int,
        default=[],
        help="Iteration numbers passed through to train.py --checkpoint_iterations.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Save train.py checkpoints every N iterations through the batch runner.",
    )
    parser.add_argument(
        "--prune-models-after-round",
        action="store_true",
        help=(
            "After all renders and disagreement products for a round are complete, "
            "delete model point clouds/checkpoints except the final eval-seed model."
        ),
    )
    parser.add_argument("--resize-renders", action="store_true")
    parser.add_argument("--compute-lpips", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_command(command: list[str], cwd: Path | None = None) -> None:
    print(" ".join(command))
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def normalize(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    min_value = min(values.values())
    max_value = max(values.values())
    if abs(max_value - min_value) < 1e-12:
        return {key: 1.0 for key in values}
    return {key: (value - min_value) / (max_value - min_value) for key, value in values.items()}


def method_label(args: argparse.Namespace) -> str:
    if args.method_label:
        label = args.method_label
    elif args.score_variant != "full" and args.policy == "nbv3d":
        label = f"{args.policy}_{args.score_variant}"
    else:
        label = args.policy
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in label)
    if not cleaned:
        raise ValueError("--method-label cannot be empty")
    return cleaned


def score_weights(args: argparse.Namespace) -> tuple[float, float, float, float]:
    if args.score_weights is not None:
        return tuple(float(item) for item in args.score_weights)  # type: ignore[return-value]
    return SCORE_VARIANTS[args.score_variant]


def weighted_nearest_distance(index: int, roi_sources: list[tuple[int, float]]) -> float:
    total_weight = sum(max(score, 1e-8) for _, score in roi_sources)
    if total_weight <= 0:
        return 0.0
    return sum(abs(index - source_index) * max(score, 1e-8) for source_index, score in roi_sources) / total_weight


def view_by_render_alias(train_views: list[dict]) -> dict[str, dict]:
    aliases: dict[str, dict] = {}
    for pos, view in enumerate(sorted(train_views, key=lambda item: int(item["index"]))):
        stem = f"{pos:05d}"
        for key in (view["name"], Path(view["name"]).stem, stem, f"{stem}.png", f"{stem}.jpg", f"{stem}.jpeg"):
            aliases[key] = view
    return aliases


def roi_sources_from_file(path: Path | None, train_views: list[dict]) -> list[tuple[int, float]]:
    if path is None or not path.is_file():
        return [(int(view["index"]), 1.0) for view in train_views]
    payload = load_json(path)
    aliases = view_by_render_alias(train_views)
    sources: list[tuple[int, float]] = []
    for candidate in payload.get("roi_candidates", []):
        image = str(candidate.get("image", ""))
        view = aliases.get(image) or aliases.get(Path(image).stem)
        if view is not None:
            sources.append((int(view["index"]), float(candidate.get("score", 1.0))))
    if not sources:
        return [(int(view["index"]), 1.0) for view in train_views]
    return sources


def score_candidates(
    candidates: list[dict],
    train_views: list[dict],
    selected_nbvs: list[dict],
    roi_sources: list[tuple[int, float]],
) -> list[ScoreBreakdown]:
    candidate_indices = [int(view["index"]) for view in candidates]
    max_gap = max(candidate_indices) - min(candidate_indices) if len(candidate_indices) > 1 else 1
    train_indices = [int(view["index"]) for view in train_views]
    selected_indices = [int(view["index"]) for view in selected_nbvs]

    defect_distances = {int(view["index"]): weighted_nearest_distance(int(view["index"]), roi_sources) for view in candidates}
    train_distances = {
        int(view["index"]): min(abs(int(view["index"]) - train_index) for train_index in train_indices)
        for view in candidates
    }
    coverage_distances = {
        int(view["index"]): (
            min(abs(int(view["index"]) - selected_index) for selected_index in selected_indices)
            if selected_indices
            else train_distances[int(view["index"])]
        )
        for view in candidates
    }

    parallax_gain = normalize(train_distances)
    coverage_gain = normalize(coverage_distances)
    max_defect_distance = max(max(defect_distances.values()), float(max_gap), 1.0)
    scored: list[ScoreBreakdown] = []
    for view in candidates:
        index = int(view["index"])
        defect_visibility = 1.0 - min(defect_distances[index] / max_defect_distance, 1.0)
        redundancy_penalty = 1.0 - parallax_gain[index]
        score = (
            0.50 * defect_visibility
            + 0.25 * parallax_gain[index]
            + 0.15 * coverage_gain[index]
            - 0.10 * redundancy_penalty
        )
        scored.append(
            ScoreBreakdown(
                index=index,
                name=str(view["name"]),
                score=float(score),
                defect_visibility=float(defect_visibility),
                parallax_gain=float(parallax_gain[index]),
                coverage_gain=float(coverage_gain[index]),
                redundancy_penalty=float(redundancy_penalty),
                nearest_defect_distance=float(defect_distances[index]),
                nearest_train_distance=float(train_distances[index]),
            )
        )
    scored.sort(key=lambda item: (-item.score, item.index))
    return scored


def select_next_views(
    policy: str,
    remaining_candidates: list[dict],
    train_views: list[dict],
    selected_nbvs: list[dict],
    roi_sources: list[tuple[int, float]],
    views_per_round: int,
    rng: random.Random,
) -> tuple[list[dict], list[ScoreBreakdown]]:
    scored = score_candidates(remaining_candidates, train_views, selected_nbvs, roi_sources)
    by_name = {view["name"]: view for view in remaining_candidates}
    if policy == "random":
        chosen = sorted(rng.sample(remaining_candidates, views_per_round), key=lambda item: int(item["index"]))
    elif policy == "nearest":
        nearest = sorted(scored, key=lambda item: (item.nearest_defect_distance, item.index))
        chosen = [by_name[item.name] for item in nearest[:views_per_round]]
    else:
        chosen = [by_name[item.name] for item in scored[:views_per_round]]
    return chosen, scored


def image_aliases(image_name: str, render_pos: int | None = None) -> list[str]:
    keys = [image_name, Path(image_name).stem]
    if render_pos is not None:
        stem = f"{render_pos:05d}"
        keys.extend([stem, f"{stem}.png", f"{stem}.jpg", f"{stem}.jpeg"])
    return list(dict.fromkeys(keys))


def build_train_view_aliases(train_views: list[dict]) -> dict[str, dict]:
    aliases: dict[str, dict] = {}
    for pos, view in enumerate(sorted(train_views, key=lambda item: int(item["index"]))):
        for key in image_aliases(str(view["name"]), pos):
            aliases[key] = view
    return aliases


def defect_points_payload(defect_points: list[DefectPoint], source_roi: Path | None, min_defect_points: int) -> dict:
    return {
        "source_roi_candidates": str(source_roi) if source_roi else None,
        "defect_point_count": len(defect_points),
        "min_defect_points": min_defect_points,
        "below_min_defect_points": len(defect_points) < min_defect_points,
        "total_weight": sum(point.weight for point in defect_points),
        "defect_points": [
            {
                "point3d_id": point.point3d_id,
                "xyz": list(point.xyz),
                "weight": point.weight,
                "hit_count": point.hit_count,
                "source_images": list(point.source_images),
            }
            for point in defect_points
        ],
    }


def build_defect_points3d(
    roi_candidates: Path | None,
    train_views: list[dict],
    model: ColmapModel,
    output_path: Path,
    min_defect_points: int,
    allow_low_defect_points: bool,
    dry_run: bool,
) -> list[DefectPoint]:
    train_aliases = build_train_view_aliases(train_views)
    aggregate: dict[int, dict] = {}

    if roi_candidates is None or not roi_candidates.is_file():
        if not dry_run:
            raise FileNotFoundError(f"Missing ROI candidates for 3D defect field: {roi_candidates}")
        for view in train_views:
            image = model.images.get(str(view["name"]))
            if image is None:
                continue
            for obs in image.observations:
                if obs.point3d_id >= 0 and obs.point3d_id in model.points3d:
                    item = aggregate.setdefault(
                        obs.point3d_id,
                        {"weight": 0.0, "hit_count": 0, "source_images": set()},
                    )
                    item["weight"] += 1.0
                    item["hit_count"] += 1
                    item["source_images"].add(image.name)
    else:
        payload = load_json(roi_candidates)
        for roi in payload.get("roi_candidates", []):
            raw_image = str(roi.get("image", ""))
            source_view = train_aliases.get(raw_image) or train_aliases.get(Path(raw_image).stem)
            if source_view is None:
                continue
            image = model.images.get(str(source_view["name"]))
            if image is None:
                continue
            x0, y0, width, height = [float(value) for value in roi["bbox_xywh"]]
            x1 = x0 + width
            y1 = y0 + height
            weight = float(roi.get("score", 1.0))
            for obs in image.observations:
                if obs.point3d_id < 0 or obs.point3d_id not in model.points3d:
                    continue
                if x0 <= obs.x < x1 and y0 <= obs.y < y1:
                    item = aggregate.setdefault(
                        obs.point3d_id,
                        {"weight": 0.0, "hit_count": 0, "source_images": set()},
                    )
                    item["weight"] += max(weight, 1e-8)
                    item["hit_count"] += 1
                    item["source_images"].add(image.name)

    defect_points = [
        DefectPoint(
            point3d_id=point_id,
            xyz=model.points3d[point_id].xyz,
            weight=float(item["weight"]),
            hit_count=int(item["hit_count"]),
            source_images=tuple(sorted(item["source_images"])),
        )
        for point_id, item in aggregate.items()
        if point_id in model.points3d
    ]
    defect_points.sort(key=lambda item: (-item.weight, item.point3d_id))
    write_json(output_path, defect_points_payload(defect_points, roi_candidates, min_defect_points))
    if not defect_points and not dry_run:
        raise RuntimeError(
            "Mapped 0 3D defect points from disagreement ROIs. Increase --top-k-roi/"
            "patch coverage or inspect COLMAP 2D-3D observations."
        )
    if len(defect_points) < min_defect_points and not dry_run:
        message = (
            f"Only mapped {len(defect_points)} 3D defect points, below "
            f"--min-defect-points {min_defect_points}. Increase --top-k-roi/"
            "patch coverage or inspect COLMAP 2D-3D observations. Continuing "
            "with a low-confidence 3D defect field."
        )
        print(f"[warn] {message}")
    return defect_points


def image_record_for_view(model: ColmapModel, view: dict) -> ImageRecord:
    image = model.images.get(str(view["name"]))
    if image is None:
        raise KeyError(f"Image {view['name']} is missing from source COLMAP images.txt")
    return image


def visible_defect_ids(
    image: ImageRecord,
    model: ColmapModel,
    defect_points: list[DefectPoint],
) -> set[int]:
    camera = model.cameras[image.camera_id]
    visible: set[int] = set()
    for point in defect_points:
        if project_point(camera, image, point.xyz) is not None:
            visible.add(point.point3d_id)
    return visible


def score_candidates_3d(
    candidates: list[dict],
    train_views: list[dict],
    defect_points: list[DefectPoint],
    model: ColmapModel,
    weights: tuple[float, float, float, float],
) -> list[PoseScoreBreakdown]:
    if not defect_points:
        return []
    train_images = [image_record_for_view(model, view) for view in train_views]
    train_centers = [camera_center(image) for image in train_images]
    train_dirs = [view_direction(image) for image in train_images]
    train_visible_by_point: dict[int, int] = {point.point3d_id: 0 for point in defect_points}
    for image in train_images:
        for point_id in visible_defect_ids(image, model, defect_points):
            train_visible_by_point[point_id] += 1

    total_weight = max(sum(point.weight for point in defect_points), 1e-8)
    candidate_distances: dict[str, float] = {}
    candidate_dirs: dict[str, tuple[float, float, float]] = {}
    candidate_centers: dict[str, tuple[float, float, float]] = {}
    for view in candidates:
        image = image_record_for_view(model, view)
        center = camera_center(image)
        direction = view_direction(image)
        candidate_centers[str(view["name"])] = center
        candidate_dirs[str(view["name"])] = direction
        candidate_distances[str(view["name"])] = min(vector_norm(vector_sub(center, train_center)) for train_center in train_centers)
    max_candidate_distance = max(max(candidate_distances.values()), 1e-8)

    scored: list[PoseScoreBreakdown] = []
    for view in candidates:
        image = image_record_for_view(model, view)
        camera = model.cameras[image.camera_id]
        center = candidate_centers[str(view["name"])]
        direction = candidate_dirs[str(view["name"])]
        weighted_visible = 0.0
        weighted_parallax = 0.0
        weighted_coverage = 0.0
        visible_count = 0

        for point in defect_points:
            if project_point(camera, image, point.xyz) is None:
                continue
            visible_count += 1
            weighted_visible += point.weight
            candidate_ray = vector_sub(center, point.xyz)
            best_angle = 0.0
            for train_center in train_centers:
                train_ray = vector_sub(train_center, point.xyz)
                best_angle = max(best_angle, angle_between(candidate_ray, train_ray))
            weighted_parallax += point.weight * min(best_angle / (math.pi * 0.5), 1.0)
            weighted_coverage += point.weight / (1.0 + float(train_visible_by_point.get(point.point3d_id, 0)))

        defect_visibility = weighted_visible / total_weight
        parallax_gain = weighted_parallax / max(weighted_visible, 1e-8)
        coverage_gain = weighted_coverage / total_weight
        position_redundancy = 1.0 - min(candidate_distances[str(view["name"])] / max_candidate_distance, 1.0)
        direction_redundancy = 1.0 - min(max(angle_between(direction, train_dir) for train_dir in train_dirs) / math.pi, 1.0)
        redundancy_penalty = 0.5 * position_redundancy + 0.5 * direction_redundancy
        v_weight, p_weight, c_weight, r_weight = weights
        score = (
            v_weight * defect_visibility
            + p_weight * parallax_gain
            + c_weight * coverage_gain
            - r_weight * redundancy_penalty
        )
        scored.append(
            PoseScoreBreakdown(
                index=int(view["index"]),
                name=str(view["name"]),
                score=float(score),
                defect_visibility=float(defect_visibility),
                parallax_gain=float(parallax_gain),
                coverage_gain=float(coverage_gain),
                redundancy_penalty=float(redundancy_penalty),
                visible_defect_points=visible_count,
                weighted_visible_defect=float(weighted_visible),
                nearest_train_distance=float(candidate_distances[str(view["name"])]),
                camera_center=center,
                view_direction=direction,
            )
        )
    scored.sort(key=lambda item: (-item.score, item.index))
    return scored


def select_next_views_3d(
    policy: str,
    remaining_candidates: list[dict],
    scored: list[PoseScoreBreakdown],
    views_per_round: int,
    rng: random.Random,
) -> list[dict]:
    by_name = {view["name"]: view for view in remaining_candidates}
    if policy == "random":
        return sorted(rng.sample(remaining_candidates, views_per_round), key=lambda item: int(item["index"]))
    if policy == "nearest3d":
        ordered = sorted(scored, key=lambda item: (-item.defect_visibility, -item.visible_defect_points, item.index))
        return [by_name[item.name] for item in ordered[:views_per_round]]
    if policy == "farthest":
        ordered = sorted(scored, key=lambda item: (-item.nearest_train_distance, item.index))
        return [by_name[item.name] for item in ordered[:views_per_round]]
    if policy == "maxcoverage":
        ordered = sorted(scored, key=lambda item: (-item.visible_defect_points, -item.weighted_visible_defect, item.index))
        return [by_name[item.name] for item in ordered[:views_per_round]]
    if policy == "maxparallax":
        ordered = sorted(scored, key=lambda item: (-item.parallax_gain, item.index))
        return [by_name[item.name] for item in ordered[:views_per_round]]
    return [by_name[item.name] for item in scored[:views_per_round]]


def all_colmap_points_as_uniform_defects(model: ColmapModel) -> list[DefectPoint]:
    points = [
        DefectPoint(
            point3d_id=point.point3d_id,
            xyz=point.xyz,
            weight=1.0,
            hit_count=1,
            source_images=(),
        )
        for point in model.points3d.values()
    ]
    points.sort(key=lambda item: item.point3d_id)
    return points


def write_candidate_scores_csv(
    path: Path,
    scored: list[PoseScoreBreakdown],
    selected_views: list[dict],
) -> None:
    selected_names = {str(item["name"]) for item in selected_views}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "candidate_view_id",
                "candidate_name",
                "S_total",
                "V_visibility",
                "P_parallax",
                "C_coverage",
                "R_redundancy",
                "nearest_train_distance",
                "visible_defect_points",
                "weighted_visible_defect",
                "rank",
                "selected_or_not",
            ],
        )
        writer.writeheader()
        for rank, item in enumerate(scored, start=1):
            writer.writerow(
                {
                    "candidate_view_id": item.index,
                    "candidate_name": item.name,
                    "S_total": item.score,
                    "V_visibility": item.defect_visibility,
                    "P_parallax": item.parallax_gain,
                    "C_coverage": item.coverage_gain,
                    "R_redundancy": item.redundancy_penalty,
                    "nearest_train_distance": item.nearest_train_distance,
                    "visible_defect_points": item.visible_defect_points,
                    "weighted_visible_defect": item.weighted_visible_defect,
                    "rank": rank,
                    "selected_or_not": item.name in selected_names,
                }
            )


def validate_split(split: dict, base_count: int, rounds: int, views_per_round: int) -> tuple[list[dict], list[dict], list[dict]]:
    images = split["images"]
    base_views = sorted([item for item in images if item["role"] == "base"], key=lambda item: int(item["index"]))
    candidates = sorted([item for item in images if item["role"] == "candidate"], key=lambda item: int(item["index"]))
    test_views = sorted([item for item in images if item["role"] == "test"], key=lambda item: int(item["index"]))
    names_by_role = {
        "base": {item["name"] for item in base_views},
        "candidate": {item["name"] for item in candidates},
        "test": {item["name"] for item in test_views},
    }
    if len(base_views) != base_count:
        raise RuntimeError(f"Expected {base_count} base views, got {len(base_views)}")
    if not names_by_role["base"].isdisjoint(names_by_role["candidate"]):
        raise RuntimeError("Base and candidate views overlap")
    if not names_by_role["base"].isdisjoint(names_by_role["test"]):
        raise RuntimeError("Base and test views overlap")
    if not names_by_role["candidate"].isdisjoint(names_by_role["test"]):
        raise RuntimeError("Candidate and test views overlap")
    required_candidates = rounds * views_per_round
    if len(candidates) < required_candidates:
        raise RuntimeError(f"Need at least {required_candidates} candidate views, got {len(candidates)}")
    if not test_views:
        raise RuntimeError("Test split is empty")
    return base_views, candidates, test_views


def run_id(label: str, round_index: int, train_seed: int) -> str:
    return f"closed_loop_{label}_round{round_index:02d}_seed{train_seed}"


def shared_init_base_dir(args: argparse.Namespace) -> Path | None:
    if args.shared_init_root is None:
        return None
    return args.shared_init_root.resolve() / args.scene_name / f"base{args.base_count}"


def shared_init_scene_dir(args: argparse.Namespace) -> Path | None:
    base_dir = shared_init_base_dir(args)
    return None if base_dir is None else base_dir / "scene"


def shared_init_output_dir(args: argparse.Namespace, seed: int) -> Path | None:
    base_dir = shared_init_base_dir(args)
    return None if base_dir is None else base_dir / f"seed{seed}"


def run_output_dir(output_root: Path, run: dict) -> Path:
    return Path(run.get("output_dir", output_root / str(run["run_id"]))).resolve()


def run_output_dirs(output_root: Path, runs: list[dict]) -> dict[str, Path]:
    return {str(run["run_id"]): run_output_dir(output_root, run) for run in runs}


def train_seeds_for_round(
    policy: str,
    round_index: int,
    rounds: int,
    train_seeds: list[int],
    eval_seed: int,
    optimize: bool,
) -> list[int]:
    if not optimize:
        return list(train_seeds)
    needs_pose_disagreement = policy in DEFECT_SCORE_POLICIES and round_index < rounds
    return list(train_seeds) if needs_pose_disagreement else [eval_seed]


def build_round_scene(
    scene_dir: Path,
    source_images: Path,
    source_text_sparse: Path,
    train_views: list[dict],
    test_views: list[dict],
    selected_nbvs: list[dict],
    args: argparse.Namespace,
) -> None:
    train_names = {item["name"] for item in train_views}
    if scene_dir.exists():
        if not args.overwrite:
            manifest_path = scene_dir / "run_manifest.json"
            if not manifest_path.is_file():
                raise FileExistsError(f"Scene exists without run_manifest.json: {scene_dir}")
            existing = load_json(manifest_path)
            existing_names = {item["name"] for item in existing.get("train_views", [])}
            if existing_names != train_names:
                raise FileExistsError(
                    f"Scene exists with different train views: {scene_dir}. "
                    "Use --overwrite or a fresh --work-root."
                )
            return
        shutil.rmtree(scene_dir)

    image_dir = scene_dir / "images"
    sparse_dir = scene_dir / "sparse" / "0"
    image_dir.mkdir(parents=True, exist_ok=False)
    selected_names = {item["name"] for item in selected_nbvs}
    for item in sorted(train_views, key=lambda view: int(view["index"])):
        link_or_copy(source_images / item["name"], image_dir / item["name"], args.copy_mode)
    kept = copy_sparse_subset(source_text_sparse, sparse_dir, train_names)
    if kept != len(train_views):
        raise RuntimeError(f"Expected {len(train_views)} COLMAP image records in {scene_dir}, got {kept}")

    selected_txt = scene_dir / "selected_train_views.txt"
    with selected_txt.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# role\tindex\tfilename\n")
        for item in sorted(train_views, key=lambda view: int(view["index"])):
            role = "selected_nbv" if item["name"] in selected_names else "base"
            handle.write(f"{role}\t{item['index']}\t{item['name']}\n")

    run_manifest = {
        "scene_dir": str(scene_dir),
        "train_views": train_views,
        "test_views": test_views,
        "selected_nbvs": selected_nbvs,
        "selected_train_views": str(selected_txt),
        "colmap_coordinate_frame": "shared_source_scene",
        "train_command_constraints": {
            "iterations": args.iterations,
            "must_not_use_start_checkpoint": not args.resume_training,
            "resume_training_allowed": args.resume_training,
            "must_not_run_convert": True,
        },
    }
    write_json(scene_dir / "run_manifest.json", run_manifest)


def round_runs_manifest(
    path: Path,
    policy: str,
    label: str,
    round_index: int,
    scene_dir: Path,
    train_views: list[dict],
    test_views: list[dict],
    selected_nbvs: list[dict],
    round_train_seeds: list[int],
    args: argparse.Namespace,
) -> dict:
    runs = []
    for seed in round_train_seeds:
        rid = run_id(label, round_index, seed)
        run = {
            "run_id": rid,
            "method": f"closed_loop_{label}",
            "policy": policy,
            "k": len(selected_nbvs),
            "seed": seed,
            "scene_dir": str(scene_dir),
            "train_views": train_views,
            "test_views": test_views,
            "selected_nbvs": selected_nbvs,
            "source_scene": str(args.source_scene.resolve()),
            "source_image_dir": args.source_image_dir,
            "source_sparse_dir": args.source_sparse_dir,
            "colmap_coordinate_frame": "shared_source_scene",
        }
        if round_index == 0:
            output_dir = shared_init_output_dir(args, seed)
            if output_dir is not None:
                run["output_dir"] = str(output_dir)
                run["shared_round0"] = True
        runs.append(run)
    payload = {
        "scene_name": args.scene_name,
        "policy": policy,
        "round_index": round_index,
        "round_train_seeds": round_train_seeds,
        "split_manifest": str(args.split_manifest.resolve()),
        "source_scene": str(args.source_scene.resolve()),
        "output_root": str(args.output_root.resolve()),
        "runs": runs,
    }
    write_json(path, payload)
    return payload


def eval_runs_manifest(path: Path, round_payloads: list[dict], args: argparse.Namespace) -> dict:
    runs = []
    for payload in round_payloads:
        for run in payload["runs"]:
            if int(run["seed"]) == args.eval_seed:
                runs.append(run)
    output = {
        "scene_name": args.scene_name,
        "policy": args.policy,
        "eval_seed": args.eval_seed,
        "split_manifest": str(args.split_manifest.resolve()),
        "source_scene": str(args.source_scene.resolve()),
        "output_root": str(args.output_root.resolve()),
        "runs": runs,
    }
    write_json(path, output)
    return output


def run_batch(
    runs_manifest: Path,
    output_root: Path,
    args: argparse.Namespace,
    run_ids: list[str],
    train_seed: int | None = None,
    train: bool = False,
    render: bool = False,
    render_extra: str | None = None,
    render_check_subdir: str | None = None,
    force_render: bool = False,
) -> None:
    command = [
        args.python_bin,
        str(SCRIPT_DIR / "batch_run_active_view_3dgs.py"),
        "--runs-manifest",
        str(runs_manifest),
        "--gs-dir",
        str(args.gs_dir),
        "--output-root",
        str(output_root),
        "--iterations",
        str(args.iterations),
        "--run-ids",
        *run_ids,
        "--no-convert",
    ]
    if train_seed is not None:
        command += ["--train-seed", str(train_seed)]
    if args.resume_training:
        command.append("--resume-training")
    if args.checkpoint_iterations:
        command += ["--checkpoint-iterations", *[str(item) for item in args.checkpoint_iterations]]
    if args.checkpoint_interval is not None:
        command += ["--checkpoint-interval", str(args.checkpoint_interval)]
    if not train:
        command.append("--no-train")
    if not render:
        command.append("--no-render")
    if force_render:
        command.append("--force-render")
    if render_extra:
        command.append(f"--render-extra={render_extra}")
    if render_check_subdir:
        command += ["--render-check-subdir", render_check_subdir]
    if args.dry_run:
        command.append("--dry-run")
    run_command(command)


def compute_disagreement(seed_a: Path, seed_b: Path, output_dir: Path, args: argparse.Namespace) -> Path | None:
    if args.dry_run:
        return None
    command = [
        args.python_bin,
        str(SCRIPT_DIR / "compute_disagreement_heatmap.py"),
        "--seed-a-renders",
        str(seed_a),
        "--seed-b-renders",
        str(seed_b),
        "--output-dir",
        str(output_dir),
        "--patch-size",
        str(args.patch_size),
        "--top-k",
        str(args.top_k_roi),
    ]
    run_command(command)
    return output_dir / "roi_candidates.json"


def run_final_evaluation(
    runs_manifest: Path,
    roi_candidates: Path | None,
    output_dir: Path,
    args: argparse.Namespace,
    label: str,
) -> dict | None:
    if args.dry_run:
        return None
    command = [
        args.python_bin,
        str(SCRIPT_DIR / "evaluate_active_view_results.py"),
        "--runs-manifest",
        str(runs_manifest),
        "--render-root",
        str(args.output_root),
        "--render-subdir",
        f"test/ours_{args.iterations}/renders",
        "--roi-source-run-id",
        run_id(label, 0, args.eval_seed),
        "--output-dir",
        str(output_dir),
    ]
    if roi_candidates is not None:
        command += ["--roi-candidates", str(roi_candidates), "--roi-match-key", "original"]
    if args.resize_renders:
        command.append("--resize-renders")
    if args.compute_lpips:
        command.append("--compute-lpips")
    run_command(command)

    summary_path = output_dir / "active_view_eval_summary.json"
    report_path = output_dir / "active_view_eval_report.md"
    closed_summary = output_dir / "closed_loop_eval_summary.json"
    closed_report = output_dir / "closed_loop_eval_report.md"
    shutil.copy2(summary_path, closed_summary)
    shutil.copy2(report_path, closed_report)
    return load_json(closed_summary)


def prune_round_models(
    output_root: Path,
    run_ids: list[str],
    keep_run_id: str | None,
) -> list[dict]:
    pruned = []
    for rid in run_ids:
        if rid == keep_run_id:
            continue
        output_dir = output_root / rid
        removed = []
        point_cloud_dir = output_dir / "point_cloud"
        if point_cloud_dir.is_dir():
            shutil.rmtree(point_cloud_dir)
            removed.append(str(point_cloud_dir))
        for checkpoint in output_dir.glob("chkpnt*.pth"):
            checkpoint.unlink()
            removed.append(str(checkpoint))
        record = {
            "run_id": rid,
            "kept_run_id": keep_run_id,
            "removed": removed,
        }
        write_json(output_dir / "model_pruned.json", record)
        pruned.append(record)
        print(f"[prune] {rid}: removed {len(removed)} model artifacts")
    return pruned


def main() -> None:
    args = parse_args()
    label = method_label(args)
    weights = score_weights(args)
    if args.views_per_round != 1:
        raise ValueError("v1 closed-loop NBV requires --views-per-round 1")
    if args.eval_seed not in set(args.train_seeds):
        raise ValueError("--eval-seed must be one of --train-seeds")
    if args.optimize_seed_schedule and args.policy not in POSE_SCORE_POLICIES | {"random"}:
        raise ValueError("--optimize-seed-schedule supports pose-scored policies and random")
    if args.policy in DEFECT_SCORE_POLICIES and args.rounds > 0 and len(args.train_seeds) < 2:
        raise ValueError("Pose-aware view selection requires at least two --train-seeds")
    if args.optimize_seed_schedule and args.policy not in DEFECT_SCORE_POLICIES and args.evaluation_roi_candidates is None:
        raise ValueError("Optimized non-defect-policy runs require --evaluation-roi-candidates")

    split = load_json(args.split_manifest)
    base_views, candidates, test_views = validate_split(split, args.base_count, args.rounds, args.views_per_round)
    source_scene = args.source_scene.resolve()
    source_images = (source_scene / args.source_image_dir).resolve()
    source_sparse = (source_scene / args.source_sparse_dir).resolve()
    if not source_images.is_dir():
        raise FileNotFoundError(f"Source image directory does not exist: {source_images}")
    if not source_sparse.is_dir():
        raise FileNotFoundError(f"Source sparse directory does not exist: {source_sparse}")
    validate_source_images(source_images, {item["name"] for item in split["images"]})

    work_root = args.work_root.resolve()
    output_root = args.output_root.resolve()
    scenes_root = work_root / "scenes"
    manifests_root = work_root / "manifests"
    disagreement_root = work_root / "disagreement"
    eval_dir = work_root / "eval"
    work_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    source_text_sparse = prepare_sparse_text(source_sparse, work_root, args.colmap_bin)
    colmap_model = load_colmap_text_model(source_text_sparse)
    pose_score_policy = args.policy in POSE_SCORE_POLICIES
    defect_score_policy = args.policy in DEFECT_SCORE_POLICIES

    rng = random.Random(args.selection_seed)
    selected_nbvs: list[dict] = []
    remaining = list(candidates)
    round_payloads: list[dict] = []
    round_records: list[dict] = []
    previous_roi_candidates: Path | None = None
    holdout_roi_candidates = args.evaluation_roi_candidates.resolve() if args.evaluation_roi_candidates else None
    if holdout_roi_candidates is not None and not holdout_roi_candidates.is_file() and not args.dry_run:
        raise FileNotFoundError(f"Shared evaluation ROI candidates do not exist: {holdout_roi_candidates}")

    for round_index in range(args.rounds + 1):
        needs_next_view = round_index < args.rounds
        needs_pose_disagreement = defect_score_policy and needs_next_view
        round_train_seeds = train_seeds_for_round(
            args.policy,
            round_index,
            args.rounds,
            args.train_seeds,
            args.eval_seed,
            args.optimize_seed_schedule,
        )
        train_views = sorted(base_views + selected_nbvs, key=lambda item: int(item["index"]))
        scene_dir = (
            shared_init_scene_dir(args)
            if round_index == 0 and shared_init_scene_dir(args) is not None
            else scenes_root / f"round_{round_index:02d}"
        )
        assert scene_dir is not None
        build_round_scene(scene_dir, source_images, source_text_sparse, train_views, test_views, selected_nbvs, args)
        round_manifest_path = manifests_root / f"round_{round_index:02d}_runs_manifest.json"
        payload = round_runs_manifest(
            round_manifest_path,
            args.policy,
            label,
            round_index,
            scene_dir,
            train_views,
            test_views,
            selected_nbvs,
            round_train_seeds,
            args,
        )
        round_payloads.append(payload)

        seed_run_ids = [run_id(label, round_index, seed) for seed in round_train_seeds]
        output_dirs_by_run = run_output_dirs(output_root, payload["runs"])
        for seed in round_train_seeds:
            run_batch(
                round_manifest_path,
                output_root,
                args,
                [run_id(label, round_index, seed)],
                train_seed=seed,
                train=True,
                render=False,
            )
        if needs_pose_disagreement or not args.optimize_seed_schedule:
            for seed in round_train_seeds:
                run_batch(
                    round_manifest_path,
                    output_root,
                    args,
                    [run_id(label, round_index, seed)],
                    train=False,
                    render=True,
                )

        render_source_scene = shlex.quote(str(source_scene).replace("\\", "/"))
        render_parts = [
            f"-s {render_source_scene}",
            f"--images {args.source_image_dir}",
            "--eval",
        ]
        if args.pass_llffhold_to_render:
            render_parts.append(f"--llffhold {args.llffhold}")
        render_parts.append("--skip_train")
        render_extra = " ".join(render_parts)
        holdout_run_ids = [run_id(label, round_index, args.eval_seed)]
        if round_index == 0 and holdout_roi_candidates is None:
            holdout_run_ids = seed_run_ids[:2]
        run_batch(
            round_manifest_path,
            output_root,
            args,
            holdout_run_ids,
            train=False,
            render=True,
            render_extra=render_extra,
            render_check_subdir=f"test/ours_{args.iterations}/renders",
            force_render=not args.prune_models_after_round,
        )

        if round_index == 0 and holdout_roi_candidates is None:
            if len(seed_run_ids) < 2:
                raise RuntimeError("Generating holdout ROI candidates requires two round-0 seeds")
            holdout_roi_candidates = compute_disagreement(
                output_dirs_by_run[seed_run_ids[0]] / f"test/ours_{args.iterations}/renders",
                output_dirs_by_run[seed_run_ids[1]] / f"test/ours_{args.iterations}/renders",
                disagreement_root / "holdout_base",
                args,
            )

        train_roi_candidates = None
        if needs_pose_disagreement or (defect_score_policy and not args.optimize_seed_schedule):
            if len(seed_run_ids) < 2:
                raise RuntimeError("Training disagreement requires two seeds")
            train_roi_candidates = compute_disagreement(
                output_dirs_by_run[seed_run_ids[0]] / f"train/ours_{args.iterations}/renders",
                output_dirs_by_run[seed_run_ids[1]] / f"train/ours_{args.iterations}/renders",
                disagreement_root / f"round_{round_index:02d}_train",
                args,
            )
        previous_roi_candidates = train_roi_candidates

        selected_next: list[dict] = []
        selected_next_details: list[dict] = []
        score_payload: list[dict] = []
        defect_points3d_path: Path | None = None
        defect_points3d_count = 0
        selected_nbvs_before_choice = list(selected_nbvs)
        if round_index < args.rounds:
            if pose_score_policy:
                if defect_score_policy:
                    defect_points3d_path = disagreement_root / f"round_{round_index:02d}_train" / "defect_points3D.json"
                    defect_points = build_defect_points3d(
                        previous_roi_candidates,
                        train_views,
                        colmap_model,
                        defect_points3d_path,
                        args.min_defect_points,
                        args.allow_low_defect_points,
                        args.dry_run,
                    )
                else:
                    defect_points3d_path = disagreement_root / f"round_{round_index:02d}_train" / "coverage_points3D.json"
                    defect_points = all_colmap_points_as_uniform_defects(colmap_model)
                    if not args.dry_run:
                        write_json(defect_points3d_path, defect_points_payload(defect_points, None, 0))
                defect_points3d_count = len(defect_points)
                scored3d = score_candidates_3d(remaining, train_views, defect_points, colmap_model, weights)
                if not scored3d:
                    raise RuntimeError("No 3D candidate scores were produced")
                selected_next = select_next_views_3d(
                    args.policy,
                    remaining,
                    scored3d,
                    args.views_per_round,
                    rng,
                )
                score_payload = [asdict(item) for item in scored3d]
                score_csv_path = disagreement_root / f"candidate_scores_round{round_index:02d}.csv"
                if not args.dry_run:
                    write_candidate_scores_csv(score_csv_path, scored3d, selected_next)
                score_by_name = {item.name: item for item in scored3d}
                for view in selected_next:
                    score_item = score_by_name[str(view["name"])]
                    detail = dict(view)
                    detail.update(
                        {
                            "camera_center": list(score_item.camera_center),
                            "view_direction": list(score_item.view_direction),
                            "visible_defect_points": score_item.visible_defect_points,
                            "score_breakdown": asdict(score_item),
                        }
                    )
                    selected_next_details.append(detail)
            else:
                if args.policy == "random":
                    selected_next = sorted(
                        rng.sample(remaining, args.views_per_round),
                        key=lambda item: int(item["index"]),
                    )
                else:
                    roi_sources = roi_sources_from_file(previous_roi_candidates, train_views)
                    selected_next, scored = select_next_views(
                        args.policy,
                        remaining,
                        train_views,
                        selected_nbvs,
                        roi_sources,
                        args.views_per_round,
                        rng,
                    )
                    score_payload = [asdict(item) for item in scored]
                selected_next_details = list(selected_next)
            selected_names = {item["name"] for item in selected_next}
            selected_nbvs.extend(selected_next)
            remaining = [item for item in remaining if item["name"] not in selected_names]

        pruned_model_outputs = []
        if args.prune_models_after_round and not args.dry_run and not (args.shared_init_root is not None and round_index == 0):
            keep_run_id = None
            if round_index == args.rounds:
                keep_run_id = run_id(label, round_index, args.eval_seed)
            pruned_model_outputs = prune_round_models(output_root, seed_run_ids, keep_run_id)

        round_records.append(
            {
                "round_id": f"round_{round_index:02d}",
                "round_index": round_index,
                "policy": args.policy,
                "method_label": label,
                "score_variant": args.score_variant,
                "score_weights": {
                    "V": weights[0],
                    "P": weights[1],
                    "C": weights[2],
                    "R": weights[3],
                },
                "train_views": train_views,
                "selected_nbvs_in_training": selected_nbvs_before_choice,
                "selected_next": selected_next_details,
                "selected_nbvs_after_choice": list(selected_nbvs),
                "remaining_candidates": remaining,
                "runs_manifest": str(round_manifest_path),
                "scene_dir": str(scene_dir),
                "seed_outputs": {
                    str(seed): str(output_dirs_by_run[run_id(label, round_index, seed)]) for seed in round_train_seeds
                },
                "round_train_seeds": round_train_seeds,
                "train_disagreement": str(train_roi_candidates) if train_roi_candidates else None,
                "defect_points3d_path": str(defect_points3d_path) if defect_points3d_path else None,
                "defect_points3d_count": defect_points3d_count,
                "holdout_roi_candidates": str(holdout_roi_candidates) if holdout_roi_candidates else None,
                "candidate_scores_csv": (
                    str(disagreement_root / f"candidate_scores_round{round_index:02d}.csv")
                    if pose_score_policy and round_index < args.rounds
                    else None
                ),
                "candidate_pose_scores": score_payload if pose_score_policy else [],
                "nbv_scores": score_payload if not pose_score_policy else [],
                "colmap_camera_model_warnings": list(colmap_model.warnings),
                "pruned_model_outputs": pruned_model_outputs,
            }
        )

    eval_manifest_path = manifests_root / "closed_loop_eval_runs_manifest.json"
    eval_runs_manifest(eval_manifest_path, round_payloads, args)
    eval_summary = run_final_evaluation(eval_manifest_path, holdout_roi_candidates, eval_dir, args, label)

    metrics_by_run = {}
    if eval_summary is not None:
        metrics_by_run = {item["run_id"]: item for item in eval_summary.get("runs", [])}
        for record in round_records:
            rid = run_id(label, int(record["round_index"]), args.eval_seed)
            record["metrics"] = metrics_by_run.get(rid)

    manifest = {
        "scene_name": args.scene_name,
        "policy": args.policy,
        "method_label": label,
        "score_variant": args.score_variant,
        "score_weights": {
            "V": weights[0],
            "P": weights[1],
            "C": weights[2],
            "R": weights[3],
        },
        "selection_seed": args.selection_seed,
        "split_manifest": str(args.split_manifest.resolve()),
        "source_scene": str(source_scene),
        "source_image_dir": args.source_image_dir,
        "source_sparse_dir": args.source_sparse_dir,
        "work_root": str(work_root),
        "output_root": str(output_root),
        "shared_init_root": str(args.shared_init_root.resolve()) if args.shared_init_root else None,
        "shared_round0_enabled": args.shared_init_root is not None,
        "base_count": args.base_count,
        "rounds": args.rounds,
        "views_per_round": args.views_per_round,
        "iterations": args.iterations,
        "patch_size": args.patch_size,
        "top_k_roi": args.top_k_roi,
        "train_seeds": args.train_seeds,
        "optimize_seed_schedule": args.optimize_seed_schedule,
        "evaluation_roi_candidates": str(holdout_roi_candidates) if holdout_roi_candidates else None,
        "eval_seed": args.eval_seed,
        "llffhold": args.llffhold,
        "pass_llffhold_to_render": args.pass_llffhold_to_render,
        "min_defect_points": args.min_defect_points,
        "allow_low_defect_points": args.allow_low_defect_points,
        "prune_models_after_round": args.prune_models_after_round,
        "resize_renders": args.resize_renders,
        "compute_lpips": args.compute_lpips,
        "dry_run": args.dry_run,
        "counts": {
            "base": len(base_views),
            "candidate": len(candidates),
            "test": len(test_views),
        },
        "rounds_detail": round_records,
        "eval_runs_manifest": str(eval_manifest_path),
        "eval_summary": str(eval_dir / "closed_loop_eval_summary.json") if not args.dry_run else None,
        "eval_report": str(eval_dir / "closed_loop_eval_report.md") if not args.dry_run else None,
    }
    manifest_path = work_root / "closed_loop_manifest.json"
    write_json(manifest_path, manifest)
    write_json(
        work_root / "selected_views.json",
        {
            "scene_name": args.scene_name,
            "policy": args.policy,
            "method_label": label,
            "selection_seed": args.selection_seed,
            "base_views": base_views,
            "selected_views": selected_nbvs,
            "rounds": [
                {
                    "round_index": record["round_index"],
                    "selected_next": record["selected_next"],
                    "candidate_scores_csv": record.get("candidate_scores_csv"),
                }
                for record in round_records
                if record.get("selected_next")
            ],
        },
    )
    print(f"Wrote {manifest_path}")
    if not args.dry_run:
        print(f"Wrote {eval_dir / 'closed_loop_eval_summary.json'}")
        print(f"Wrote {eval_dir / 'closed_loop_eval_report.md'}")


if __name__ == "__main__":
    main()
