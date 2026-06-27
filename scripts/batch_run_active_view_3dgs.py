#!/usr/bin/env python3
"""Batch convert, train, and render active-view 3DGS scenes."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}
CHECKPOINT_PREFIX = "chkpnt"
CHECKPOINT_SUFFIX = ".pth"


@dataclass
class StageResult:
    run_id: str
    stage: str
    status: str
    command: list[str]
    log_path: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read active_view_runs_manifest.json and run official 3DGS "
            "convert.py, train.py, and render.py for every selected scene."
        )
    )
    parser.add_argument("--runs-manifest", type=Path, required=True)
    parser.add_argument("--gs-dir", type=Path, required=True, help="Official gaussian-splatting directory.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python-bin", default="python")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument(
        "--train-seed",
        type=int,
        default=None,
        help="Optional training seed. Only pass this if train.py supports --seed.",
    )
    parser.add_argument("--run-ids", nargs="+", default=None, help="Only run these run_id values.")
    parser.add_argument("--skip-run-ids", nargs="+", default=[], help="Skip these run_id values.")
    parser.add_argument("--only-methods", nargs="+", default=None)
    parser.add_argument("--no-convert", action="store_true")
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--force-convert", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-render", action="store_true")
    parser.add_argument(
        "--resume-training",
        action="store_true",
        help=(
            "Resume an incomplete existing train output from its latest "
            "chkpnt*.pth. Fresh runs with no output directory still start normally."
        ),
    )
    parser.add_argument(
        "--checkpoint-iterations",
        nargs="+",
        type=int,
        default=[],
        help="Optional iteration numbers passed to train.py --checkpoint_iterations.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Save train.py checkpoints every N iterations up to --iterations.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--log-root", type=Path, default=None)
    parser.add_argument("--convert-extra", default="", help="Extra args passed to convert.py, shell-style string.")
    parser.add_argument("--train-extra", default="", help="Extra args passed to train.py, shell-style string.")
    parser.add_argument("--render-extra", default="", help="Extra args passed to render.py, shell-style string.")
    parser.add_argument(
        "--render-check-subdir",
        default=None,
        help="Render completion subdir under each output. Default: train/ours_<iterations>/renders.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def has_files(path: Path, suffixes: set[str] | None = None) -> bool:
    if not path.is_dir():
        return False
    for child in path.iterdir():
        if child.is_file() and (suffixes is None or child.suffix in suffixes):
            return True
    return False


def convert_complete(scene_dir: Path) -> bool:
    sparse0 = scene_dir / "sparse" / "0"
    sparse = scene_dir / "sparse"
    return has_files(sparse0) or has_files(sparse)


def train_complete(output_dir: Path, iterations: int) -> bool:
    point_cloud = output_dir / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
    marker = output_dir / f".train_complete_iteration_{iterations}"
    return point_cloud.is_file() or marker.is_file()


def write_train_complete_marker(output_dir: Path, iterations: int) -> None:
    marker = output_dir / f".train_complete_iteration_{iterations}"
    marker.write_text(f"completed iteration {iterations}\n", encoding="utf-8")


def output_has_training_state(output_dir: Path) -> bool:
    if not output_dir.is_dir():
        return False
    return any(output_dir.iterdir())


def checkpoint_iteration(path: Path) -> int | None:
    name = path.name
    if not name.startswith(CHECKPOINT_PREFIX) or not name.endswith(CHECKPOINT_SUFFIX):
        return None
    raw_iteration = name[len(CHECKPOINT_PREFIX) : -len(CHECKPOINT_SUFFIX)]
    if not raw_iteration.isdigit():
        return None
    return int(raw_iteration)


def latest_checkpoint(output_dir: Path, max_iteration: int) -> Path | None:
    checkpoints: list[tuple[int, Path]] = []
    if not output_dir.is_dir():
        return None
    for item in output_dir.glob(f"{CHECKPOINT_PREFIX}*{CHECKPOINT_SUFFIX}"):
        iteration = checkpoint_iteration(item)
        if iteration is not None and iteration < max_iteration:
            checkpoints.append((iteration, item))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda item: item[0])[1]


def checkpoint_schedule(args: argparse.Namespace) -> list[int]:
    scheduled = {item for item in args.checkpoint_iterations if 0 < item <= args.iterations}
    if args.checkpoint_interval is not None:
        if args.checkpoint_interval <= 0:
            raise ValueError("--checkpoint-interval must be positive")
        scheduled.update(range(args.checkpoint_interval, args.iterations + 1, args.checkpoint_interval))
    return sorted(scheduled)


def render_complete(output_dir: Path, render_check_subdir: str) -> bool:
    return has_files(output_dir / render_check_subdir, IMAGE_SUFFIXES)


def command_to_text(command: list[str]) -> str:
    return " ".join(shlex.quote(item) for item in command)


def run_command(command: list[str], cwd: Path, log_path: Path, dry_run: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command_text = command_to_text(command)
    print(command_text)
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        log.write(f"$ {command_text}\n\n")
        if dry_run:
            log.write("[dry-run] command not executed\n")
            return 0
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return process.wait()


def next_log_path(log_root: Path, stem: str) -> Path:
    path = log_root / f"{stem}.log"
    if not path.exists():
        return path
    index = 1
    while True:
        candidate = log_root / f"{stem}.{index}.log"
        if not candidate.exists():
            return candidate
        index += 1


def selected_runs(runs: list[dict], args: argparse.Namespace) -> list[dict]:
    run_ids = set(args.run_ids) if args.run_ids else None
    skip_run_ids = set(args.skip_run_ids)
    methods = set(args.only_methods) if args.only_methods else None
    selected = []
    for run in runs:
        run_id = run["run_id"]
        method = run["method"]
        if run_ids is not None and run_id not in run_ids:
            continue
        if run_id in skip_run_ids:
            continue
        if methods is not None and method not in methods:
            continue
        selected.append(run)
    return selected


def stage_status(
    run_id: str,
    stage: str,
    command: list[str],
    log_path: Path | None,
    status: str,
) -> StageResult:
    return StageResult(
        run_id=run_id,
        stage=stage,
        status=status,
        command=command,
        log_path=str(log_path) if log_path else None,
    )


def main() -> None:
    args = parse_args()
    runs_manifest = load_json(args.runs_manifest)
    gs_dir = args.gs_dir.resolve()
    output_root = args.output_root.resolve()
    log_root = (args.log_root or (output_root / "_batch_logs")).resolve()
    render_check_subdir = args.render_check_subdir or f"train/ours_{args.iterations}/renders"

    if not gs_dir.is_dir():
        raise FileNotFoundError(f"3DGS directory does not exist: {gs_dir}")
    for script_name in ("convert.py", "train.py", "render.py"):
        if not (gs_dir / script_name).is_file():
            raise FileNotFoundError(f"Missing {script_name} in {gs_dir}")

    runs = selected_runs(runs_manifest["runs"], args)
    if not runs:
        raise RuntimeError("No runs selected")
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    convert_extra = shlex.split(args.convert_extra)
    train_extra = shlex.split(args.train_extra)
    render_extra = shlex.split(args.render_extra)
    checkpoint_iterations = checkpoint_schedule(args)
    results: list[StageResult] = []

    for run in runs:
        run_id = run["run_id"]
        scene_dir = Path(run["scene_dir"]).resolve()
        output_dir = Path(run.get("output_dir", output_root / run_id)).resolve()
        print(f"\n=== {run_id} ===")
        if not scene_dir.is_dir():
            raise FileNotFoundError(f"Scene directory does not exist for {run_id}: {scene_dir}")

        if not args.no_convert:
            command = [args.python_bin, "convert.py", "-s", str(scene_dir)] + convert_extra
            log_path = next_log_path(log_root, f"{run_id}__convert")
            if convert_complete(scene_dir) and not args.force_convert:
                print(f"[skip] convert complete for {run_id}")
                results.append(stage_status(run_id, "convert", command, None, "skipped"))
            else:
                code = run_command(command, gs_dir, log_path, args.dry_run)
                status = "ok" if code == 0 else f"failed:{code}"
                results.append(stage_status(run_id, "convert", command, log_path, status))
                if code != 0 and not args.continue_on_error:
                    break

        if not args.no_train:
            command = [
                args.python_bin,
                "train.py",
                "-s",
                str(scene_dir),
                "-m",
                str(output_dir),
                "--iterations",
                str(args.iterations),
                "--save_iterations",
                str(args.iterations),
                "--test_iterations",
                str(args.iterations),
            ]
            if args.train_seed is not None:
                command += ["--seed", str(args.train_seed)]
            if checkpoint_iterations:
                command += ["--checkpoint_iterations", *[str(item) for item in checkpoint_iterations]]
            log_path = next_log_path(log_root, f"{run_id}__train")
            if train_complete(output_dir, args.iterations) and not args.force_train:
                print(f"[skip] train complete for {run_id}")
                results.append(stage_status(run_id, "train", command, None, "skipped"))
            else:
                start_checkpoint = None
                if args.resume_training and output_has_training_state(output_dir) and not args.force_train:
                    start_checkpoint = latest_checkpoint(output_dir, args.iterations)
                    if start_checkpoint is None:
                        raise FileNotFoundError(
                            f"{run_id} has an incomplete existing output directory but no resumable "
                            f"{CHECKPOINT_PREFIX}*{CHECKPOINT_SUFFIX} before iteration {args.iterations}: {output_dir}"
                        )
                    command += ["--start_checkpoint", str(start_checkpoint)]
                command += train_extra
                if start_checkpoint is not None:
                    print(f"[resume] train {run_id} from {start_checkpoint}")
                code = run_command(command, gs_dir, log_path, args.dry_run)
                if code == 0 and not args.dry_run:
                    point_cloud = output_dir / "point_cloud" / f"iteration_{args.iterations}" / "point_cloud.ply"
                    if not point_cloud.is_file():
                        raise RuntimeError(
                            f"train.py exited successfully but final point cloud is missing for {run_id}: {point_cloud}"
                        )
                    write_train_complete_marker(output_dir, args.iterations)
                status = "ok" if code == 0 else f"failed:{code}"
                results.append(stage_status(run_id, "train", command, log_path, status))
                if code != 0 and not args.continue_on_error:
                    break

        if not args.no_render:
            command = [args.python_bin, "render.py", "-m", str(output_dir)] + render_extra
            log_path = next_log_path(log_root, f"{run_id}__render")
            if render_complete(output_dir, render_check_subdir) and not args.force_render:
                print(f"[skip] render complete for {run_id}")
                results.append(stage_status(run_id, "render", command, None, "skipped"))
            else:
                code = run_command(command, gs_dir, log_path, args.dry_run)
                status = "ok" if code == 0 else f"failed:{code}"
                results.append(stage_status(run_id, "render", command, log_path, status))
                if code != 0 and not args.continue_on_error:
                    break

    summary = {
        "runs_manifest": str(args.runs_manifest.resolve()),
        "gs_dir": str(gs_dir),
        "output_root": str(output_root),
        "iterations": args.iterations,
        "train_seed": args.train_seed,
        "resume_training": args.resume_training,
        "checkpoint_iterations": checkpoint_iterations,
        "dry_run": args.dry_run,
        "render_check_subdir": render_check_subdir,
        "results": [asdict(item) for item in results],
    }
    summary_path = log_root / "batch_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {summary_path}")

    failed = [item for item in results if item.status.startswith("failed")]
    if failed:
        print("Failed stages:")
        for item in failed:
            print(f"  {item.run_id} {item.stage}: {item.status}")
        sys.exit(1)


if __name__ == "__main__":
    main()
