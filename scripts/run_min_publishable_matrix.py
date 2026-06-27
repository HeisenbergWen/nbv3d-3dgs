#!/usr/bin/env python3
"""Run or print config-driven LLFF active-view experiment matrices."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "experiment_matrix" / "min_publishable_llff.json"


@dataclass(frozen=True)
class MatrixRun:
    scene: str
    label: str
    policy: str
    score_variant: str
    training_seed: int
    selection_seed: int | None
    work_root: str
    output_root: str
    command: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Expand a configured active-view matrix, including fixed splits, "
            "shared Round 0 directories, main baselines, and optional V/P/C/R "
            "ablations."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("split", "main", "ablations", "all"), default="all")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--colmap-bin", default="colmap")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--matrix-manifest",
        type=Path,
        default=None,
        help="Default: <av_root>/matrix_manifests/min_publishable_matrix.json",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def as_path(value: object) -> Path:
    return Path(str(value)).expanduser()


def command_text(command: list[str]) -> str:
    return " ".join(command)


def run_command(command: list[str], cwd: Path, dry_run: bool) -> None:
    print(command_text(command))
    if dry_run:
        return
    subprocess.run(command, cwd=str(cwd), check=True)


def split_manifest(config: dict, scene: dict) -> Path:
    return as_path(config["av_root"]) / "splits" / str(scene["name"]) / "split_manifest.json"


def method_work_root(config: dict, scene_name: str, label: str, training_seed: int) -> Path:
    return as_path(config["av_root"]) / "min_publishable" / scene_name / label / f"trainseed{training_seed}"


def method_output_root(config: dict, scene_name: str, label: str, training_seed: int) -> Path:
    return as_path(config["out_root"]) / "min_publishable" / scene_name / label / f"trainseed{training_seed}"


def shared_init_root(config: dict) -> Path:
    return as_path(config.get("shared_init_root", as_path(config["av_root"]) / "shared_init"))


def split_command(config: dict, scene: dict, args: argparse.Namespace) -> list[str]:
    command = [
        args.python_bin,
        str(SCRIPT_DIR / "create_active_view_split.py"),
        "--src-scene",
        str(as_path(scene["full_scene"])),
        "--src-image-dir",
        str(scene.get("source_image_dir", config.get("source_image_dir", "input"))),
        "--output-dir",
        str(split_manifest(config, scene).parent),
        "--scene-name",
        str(scene["name"]),
        "--base-count",
        str(config.get("base_count", 6)),
        "--test-stride",
        str(config.get("test_stride", 8)),
    ]
    if args.overwrite:
        command.append("--overwrite")
    return command


def aux_seed(config: dict, training_seed: int) -> int:
    return int(training_seed) + int(config.get("disagreement_seed_offset", 1000))


def closed_loop_command(
    config: dict,
    scene: dict,
    label: str,
    policy: str,
    score_variant: str,
    training_seed: int,
    selection_seed: int | None,
    args: argparse.Namespace,
    evaluation_roi_candidates: Path | None,
) -> list[str]:
    defect_policy = policy in {"nbv3d", "nearest3d"}
    train_seeds = [training_seed, aux_seed(config, training_seed)] if defect_policy else [training_seed]
    command = [
        args.python_bin,
        str(SCRIPT_DIR / "run_closed_loop_nbv_llff.py"),
        "--split-manifest",
        str(split_manifest(config, scene)),
        "--source-scene",
        str(as_path(scene["full_scene"])),
        "--source-image-dir",
        str(scene.get("source_image_dir", config.get("source_image_dir", "input"))),
        "--source-sparse-dir",
        str(scene.get("source_sparse_dir", config.get("source_sparse_dir", "sparse/0"))),
        "--gs-dir",
        str(as_path(config["gs_dir"])),
        "--work-root",
        str(method_work_root(config, str(scene["name"]), label, training_seed)),
        "--output-root",
        str(method_output_root(config, str(scene["name"]), label, training_seed)),
        "--shared-init-root",
        str(shared_init_root(config)),
        "--scene-name",
        str(scene["name"]),
        "--base-count",
        str(config.get("base_count", 6)),
        "--rounds",
        str(config.get("rounds", 3)),
        "--views-per-round",
        str(config.get("views_per_round", 1)),
        "--policy",
        policy,
        "--score-variant",
        score_variant,
        "--method-label",
        label,
        "--iterations",
        str(config.get("iterations", 30000)),
        "--train-seeds",
        *[str(seed) for seed in train_seeds],
        "--eval-seed",
        str(training_seed),
        "--selection-seed",
        str(selection_seed if selection_seed is not None else config.get("selection_seed", 0)),
        "--llffhold",
        str(config.get("llffhold", 8)),
        "--patch-size",
        str(config.get("patch_size", 64)),
        "--top-k-roi",
        str(config.get("top_k_roi", 10)),
        "--min-defect-points",
        str(config.get("min_defect_points", 20)),
        "--colmap-bin",
        args.colmap_bin,
        "--optimize-seed-schedule",
    ]
    if evaluation_roi_candidates is not None:
        command += ["--evaluation-roi-candidates", str(evaluation_roi_candidates)]
    if bool(config.get("allow_low_defect_points", False)):
        command.append("--allow-low-defect-points")
    if bool(config.get("resize_renders", True)):
        command.append("--resize-renders")
    if bool(config.get("compute_lpips", False)):
        command.append("--compute-lpips")
    if bool(config.get("prune_models_after_round", True)):
        command.append("--prune-models-after-round")
    if args.overwrite:
        command.append("--overwrite")
    if args.dry_run:
        command.append("--dry-run")
    return command


def main_methods(config: dict) -> list[dict]:
    configured = config.get("main_methods")
    if configured is not None:
        methods = []
        for item in configured:
            method = {
                "label": str(item["label"]),
                "policy": str(item["policy"]),
                "score_variant": str(item.get("score_variant", "full")),
                "selection_seeds": item.get("selection_seeds", [None]),
            }
            methods.append(method)
        return methods

    methods = [
        {"label": "nbv3d_full", "policy": "nbv3d", "score_variant": "full", "selection_seeds": [None]},
        {"label": "nearest3d_v_only", "policy": "nearest3d", "score_variant": "v_only", "selection_seeds": [None]},
        {"label": "farthest", "policy": "farthest", "score_variant": "full", "selection_seeds": [None]},
        {"label": "maxcoverage", "policy": "maxcoverage", "score_variant": "full", "selection_seeds": [None]},
    ]
    if bool(config.get("include_maxparallax", False)):
        methods.append(
            {"label": "maxparallax", "policy": "maxparallax", "score_variant": "full", "selection_seeds": [None]}
        )
    random_seeds = config.get("random_selection_seeds", [0, 1, 2, 3, 4])
    methods.extend(random_methods(random_seeds))
    return methods


def random_methods(random_seeds: list[int]) -> list[dict]:
    return [
        {
            "label": f"random_seed{seed}",
            "policy": "random",
            "score_variant": "full",
            "selection_seeds": [int(seed)],
        }
        for seed in random_seeds
    ]


def ablation_methods() -> list[dict]:
    return [
        {"label": "ablation_full", "policy": "nbv3d", "score_variant": "full", "selection_seeds": [None]},
        {"label": "ablation_wo_p", "policy": "nbv3d", "score_variant": "wo_p", "selection_seeds": [None]},
        {"label": "ablation_wo_c", "policy": "nbv3d", "score_variant": "wo_c", "selection_seeds": [None]},
        {"label": "ablation_wo_r", "policy": "nbv3d", "score_variant": "wo_r", "selection_seeds": [None]},
        {"label": "ablation_v_only", "policy": "nbv3d", "score_variant": "v_only", "selection_seeds": [None]},
    ]


def configured_ablation_methods(config: dict) -> list[dict]:
    variants = config.get("ablation_variants")
    if variants is None:
        return ablation_methods()
    output = []
    for variant in variants:
        raw = str(variant)
        label = "ablation_full" if raw == "full" else f"ablation_{raw}"
        output.append({"label": label, "policy": "nbv3d", "score_variant": raw, "selection_seeds": [None]})
    return output


def scene_filter(config: dict, stage: str) -> set[str] | None:
    key = f"{stage}_scene_names"
    names = config.get(key)
    if names is None and stage == "main":
        names = config.get("scene_names")
    if names is None:
        return None
    return {str(name) for name in names}


def stage_scenes(config: dict, stage: str) -> list[dict]:
    allowed = scene_filter(config, stage)
    scenes = list(config.get("scenes", []))
    if allowed is None:
        return scenes
    return [scene for scene in scenes if str(scene["name"]) in allowed]


def planned_runs(config: dict, args: argparse.Namespace) -> list[MatrixRun]:
    include_main = args.stage in {"main", "all"}
    include_ablations = args.stage in {"ablations", "all"}
    shared_full_label = "nbv3d_full" if include_main else "ablation_full"
    groups: list[tuple[list[dict], list[dict]]] = []
    if include_main:
        groups.append((main_methods(config), stage_scenes(config, "main")))
    if include_ablations:
        groups.append((configured_ablation_methods(config), stage_scenes(config, "ablation")))

    runs: list[MatrixRun] = []
    training_seeds = [int(seed) for seed in config.get("training_seeds", [0, 1, 2])]
    for methods, scenes in groups:
        for scene in scenes:
            scene_name = str(scene["name"])
            for training_seed in training_seeds:
                shared_roi = (
                    method_work_root(config, scene_name, shared_full_label, training_seed)
                    / "disagreement"
                    / "holdout_base"
                    / "roi_candidates.json"
                )
                for method in methods:
                    for selection_seed in method["selection_seeds"]:
                        label = str(method["label"])
                        evaluation_roi = None if label == shared_full_label else shared_roi
                        command = closed_loop_command(
                            config,
                            scene,
                            label,
                            str(method["policy"]),
                            str(method["score_variant"]),
                            training_seed,
                            selection_seed,
                            args,
                            evaluation_roi,
                        )
                        runs.append(
                            MatrixRun(
                                scene=scene_name,
                                label=label,
                                policy=str(method["policy"]),
                                score_variant=str(method["score_variant"]),
                                training_seed=training_seed,
                                selection_seed=selection_seed,
                                work_root=str(method_work_root(config, scene_name, label, training_seed)),
                                output_root=str(method_output_root(config, scene_name, label, training_seed)),
                                command=command,
                            )
                        )
    return runs


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    project_dir = as_path(config.get("project_dir", PROJECT_DIR))
    matrix_manifest = args.matrix_manifest or (
        as_path(config["av_root"]) / "matrix_manifests" / "min_publishable_matrix.json"
    )

    if args.stage in {"split", "all"}:
        for scene in config.get("scenes", []):
            run_command(split_command(config, scene, args), project_dir, args.dry_run)

    runs = planned_runs(config, args) if args.stage != "split" else []
    for run in runs:
        run_command(run.command, project_dir, args.dry_run)

    write_json(
        matrix_manifest,
        {
            "config": str(args.config.resolve()),
            "stage": args.stage,
            "dry_run": args.dry_run,
            "runs": [asdict(run) for run in runs],
        },
    )
    print(f"Wrote {matrix_manifest}")


if __name__ == "__main__":
    main()
