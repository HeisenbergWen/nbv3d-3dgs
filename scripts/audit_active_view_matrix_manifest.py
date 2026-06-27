#!/usr/bin/env python3
"""Audit expanded active-view matrix manifests for fairness controls."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a run_min_publishable_matrix.py manifest.")
    parser.add_argument("--matrix-manifest", type=Path, required=True)
    parser.add_argument("--expected-runs", type=int, default=None)
    parser.add_argument("--require-shared-init", action="store_true")
    parser.add_argument("--ablation-scenes", nargs="*", default=None)
    parser.add_argument("--random-selection-seeds", nargs="*", type=int, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def command_has(command: object, flag: str) -> bool:
    return isinstance(command, list) and flag in command


def main() -> int:
    args = parse_args()
    payload = load_json(args.matrix_manifest)
    runs = payload.get("runs", [])
    if not isinstance(runs, list):
        raise RuntimeError("Manifest field 'runs' must be a list")

    errors: list[str] = []
    if args.expected_runs is not None and len(runs) != args.expected_runs:
        errors.append(f"Expected {args.expected_runs} runs, got {len(runs)}")

    identities = [
        (
            run.get("scene"),
            run.get("label"),
            run.get("training_seed"),
            run.get("selection_seed"),
        )
        for run in runs
    ]
    duplicates = [identity for identity, count in Counter(identities).items() if count > 1]
    if duplicates:
        errors.append(f"Found duplicate run identities: {duplicates[:5]}")

    if args.require_shared_init:
        missing = [run for run in runs if not command_has(run.get("command"), "--shared-init-root")]
        if missing:
            errors.append(f"{len(missing)} runs are missing --shared-init-root")

    if args.ablation_scenes is not None:
        allowed = set(args.ablation_scenes)
        actual = {str(run.get("scene")) for run in runs if str(run.get("label", "")).startswith("ablation_")}
        outside = sorted(actual - allowed)
        if outside:
            errors.append(f"Ablation scenes outside allowed set {sorted(allowed)}: {outside}")

    if args.random_selection_seeds is not None:
        allowed_seeds = set(args.random_selection_seeds)
        actual_seeds = {
            int(run["selection_seed"])
            for run in runs
            if str(run.get("label", "")).startswith("random_seed") and run.get("selection_seed") is not None
        }
        outside = sorted(actual_seeds - allowed_seeds)
        missing = sorted(allowed_seeds - actual_seeds)
        if outside:
            errors.append(f"Random selection seeds outside allowed set {sorted(allowed_seeds)}: {outside}")
        if missing:
            errors.append(f"Random selection seeds not present: {missing}")

    print(f"manifest: {args.matrix_manifest}")
    print(f"runs: {len(runs)}")
    print(f"by_scene: {dict(sorted(Counter(str(run.get('scene')) for run in runs).items()))}")
    print(f"by_label: {dict(sorted(Counter(str(run.get('label')) for run in runs).items()))}")

    if errors:
        print("AUDIT FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    print("AUDIT PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
