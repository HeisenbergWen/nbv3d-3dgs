#!/usr/bin/env python3
"""Summarize active-view matrix runs into CSV result tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


METRICS = (
    "global_l1",
    "global_psnr",
    "global_ssim",
    "global_lpips",
    "roi_l1",
    "roi_psnr",
    "roi_ssim",
    "roi_lpips",
    "non_roi_l1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate closed-loop eval summaries from run_min_publishable_matrix.py."
    )
    parser.add_argument("--matrix-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--strict", action="store_true", help="Fail if any configured run is missing.")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0 if values else float("nan")
    avg = mean(values)
    return math.sqrt(sum((item - avg) ** 2 for item in values) / (len(values) - 1))


def ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0 if values else float("nan")
    return 1.96 * std(values) / math.sqrt(len(values))


def metric_by_round(eval_summary: dict, eval_seed: int) -> dict[int, dict]:
    by_round = {}
    for item in eval_summary.get("runs", []):
        if int(item.get("seed", -1)) != int(eval_seed):
            continue
        run_id = str(item.get("run_id", ""))
        if "_round" not in run_id:
            continue
        raw_round = run_id.split("_round", 1)[1].split("_", 1)[0]
        if raw_round.isdigit():
            by_round[int(raw_round)] = item
    return by_round


def auc(rows: list[dict], metric: str) -> float:
    values = [finite(row.get(metric)) for row in rows]
    valid = [value for value in values if value is not None]
    if not valid:
        return float("nan")
    if len(valid) == 1:
        return valid[0]
    area = 0.0
    for left, right in zip(valid[:-1], valid[1:]):
        area += 0.5 * (left + right)
    return area / (len(valid) - 1)


def load_run(entry: dict, strict: bool) -> dict | None:
    work_root = Path(entry["work_root"])
    eval_path = work_root / "eval" / "closed_loop_eval_summary.json"
    manifest_path = work_root / "closed_loop_manifest.json"
    if not eval_path.is_file() or not manifest_path.is_file():
        if strict:
            missing = eval_path if not eval_path.is_file() else manifest_path
            raise FileNotFoundError(f"Missing result file: {missing}")
        return None

    manifest = load_json(manifest_path)
    eval_summary = load_json(eval_path)
    eval_seed = int(manifest.get("eval_seed", entry.get("training_seed", 0)))
    by_round = metric_by_round(eval_summary, eval_seed)
    if not by_round:
        if strict:
            raise RuntimeError(f"No evaluated rounds in {eval_path}")
        return None
    rounds = [by_round[index] for index in sorted(by_round)]
    final = rounds[-1]
    row = {
        "scene": entry["scene"],
        "label": entry["label"],
        "policy": entry["policy"],
        "score_variant": entry["score_variant"],
        "training_seed": entry["training_seed"],
        "selection_seed": "" if entry.get("selection_seed") is None else entry["selection_seed"],
        "rounds": len(rounds) - 1,
        "work_root": entry["work_root"],
        "eval_summary": str(eval_path),
    }
    for metric in METRICS:
        row[f"final_{metric}"] = final.get(metric)
        row[f"auc_{metric}"] = auc(rounds, metric)
        first = finite(rounds[0].get(metric))
        last = finite(final.get(metric))
        row[f"delta_{metric}"] = last - first if first is not None and last is not None else float("nan")
    return row


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(str(row["scene"]), str(row["label"]))].append(row)

    output = []
    for (scene, label), items in sorted(groups.items()):
        record = {"scene": scene, "label": label, "n": len(items)}
        for metric in METRICS:
            for prefix in ("final", "auc", "delta"):
                values = [finite(item.get(f"{prefix}_{metric}")) for item in items]
                valid = [value for value in values if value is not None]
                record[f"{prefix}_{metric}_mean"] = mean(valid)
                record[f"{prefix}_{metric}_std"] = std(valid)
                record[f"{prefix}_{metric}_ci95"] = ci95(valid)
        output.append(record)
    return output


def markdown(agg_rows: list[dict]) -> str:
    lines = [
        "# Minimal Publishable Matrix Summary",
        "",
        "| Scene | Method | n | Final Global PSNR | Final ROI PSNR | Final SSIM | Final LPIPS | Final ROI L1 | AUC ROI PSNR |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in agg_rows:
        lines.append(
            f"| {row['scene']} | {row['label']} | {row['n']} | "
            f"{row['final_global_psnr_mean']:.3f} +/- {row['final_global_psnr_std']:.3f} | "
            f"{row['final_roi_psnr_mean']:.3f} +/- {row['final_roi_psnr_std']:.3f} | "
            f"{row['final_global_ssim_mean']:.4f} +/- {row['final_global_ssim_std']:.4f} | "
            f"{row['final_global_lpips_mean']:.4f} +/- {row['final_global_lpips_std']:.4f} | "
            f"{row['final_roi_l1_mean']:.6f} +/- {row['final_roi_l1_std']:.6f} | "
            f"{row['auc_roi_psnr_mean']:.3f} +/- {row['auc_roi_psnr_std']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    manifest = load_json(args.matrix_manifest)
    rows = []
    for entry in manifest.get("runs", []):
        row = load_run(entry, args.strict)
        if row is not None:
            rows.append(row)

    if not rows:
        raise RuntimeError("No completed runs found")

    row_fields = list(rows[0].keys())
    agg_rows = aggregate(rows)
    agg_fields = list(agg_rows[0].keys())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "min_publishable_run_results.csv", rows, row_fields)
    write_csv(args.output_dir / "min_publishable_aggregate_results.csv", agg_rows, agg_fields)
    (args.output_dir / "min_publishable_summary.md").write_text(markdown(agg_rows), encoding="utf-8")
    print(f"Wrote {args.output_dir / 'min_publishable_run_results.csv'}")
    print(f"Wrote {args.output_dir / 'min_publishable_aggregate_results.csv'}")
    print(f"Wrote {args.output_dir / 'min_publishable_summary.md'}")


if __name__ == "__main__":
    main()
