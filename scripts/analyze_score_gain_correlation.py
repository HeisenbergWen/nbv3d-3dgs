#!/usr/bin/env python3
"""Analyze correlation between candidate view scores and measured quality gains."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join candidate_scores_round*.csv with exhaustive/proxy candidate "
            "metrics and compute Pearson, Spearman, Kendall, top-k hit, and regret."
        )
    )
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--metric", default="roi_psnr")
    parser.add_argument("--baseline", type=float, default=None)
    parser.add_argument("--score-column", default="S_total")
    parser.add_argument("--candidate-key", default="candidate_view_id")
    parser.add_argument("--output-stats", type=Path, default=Path("results/tables/score_gain_correlation.csv"))
    parser.add_argument("--output-joined", type=Path, default=None)
    parser.add_argument("--output-plot", type=Path, default=None)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    output = [0.0] * len(values)
    idx = 0
    while idx < len(indexed):
        end = idx + 1
        while end < len(indexed) and indexed[end][1] == indexed[idx][1]:
            end += 1
        avg_rank = 0.5 * (idx + 1 + end)
        for pos in range(idx, end):
            output[indexed[pos][0]] = avg_rank
        idx = end
    return output


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x <= 1e-12 or den_y <= 1e-12:
        return float("nan")
    return num / (den_x * den_y)


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(ranks(xs), ranks(ys))


def kendall_tau(xs: list[float], ys: list[float]) -> float:
    concordant = 0
    discordant = 0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            product = dx * dy
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else float("nan")


def join_rows(args: argparse.Namespace) -> list[dict]:
    scores = read_csv(args.scores)
    metrics = read_csv(args.metrics)
    metrics_by_id = {str(row[args.candidate_key]): row for row in metrics if args.candidate_key in row}
    joined = []
    for score_row in scores:
        key = str(score_row.get(args.candidate_key, ""))
        metric_row = metrics_by_id.get(key)
        if metric_row is None:
            continue
        score = finite(score_row.get(args.score_column))
        metric_value = finite(metric_row.get(args.metric))
        if score is None or metric_value is None:
            continue
        gain = metric_value - args.baseline if args.baseline is not None else metric_value
        row = dict(score_row)
        row.update({f"measured_{args.metric}": metric_value, f"delta_{args.metric}": gain})
        joined.append(row)
    return joined


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot(joined: list[dict], args: argparse.Namespace) -> None:
    if args.output_plot is None:
        return
    import matplotlib.pyplot as plt  # type: ignore

    xs = [float(row[args.score_column]) for row in joined]
    ys = [float(row[f"delta_{args.metric}"]) for row in joined]
    args.output_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.0, 4.0))
    plt.scatter(xs, ys, s=24, alpha=0.8)
    plt.xlabel(args.score_column)
    plt.ylabel(f"Delta {args.metric}" if args.baseline is not None else args.metric)
    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    joined = join_rows(args)
    if not joined:
        raise RuntimeError("No joined candidates. Check --candidate-key and metric column names.")

    xs = [float(row[args.score_column]) for row in joined]
    ys = [float(row[f"delta_{args.metric}"]) for row in joined]
    score_order = sorted(range(len(joined)), key=lambda idx: xs[idx], reverse=True)
    gain_order = sorted(range(len(joined)), key=lambda idx: ys[idx], reverse=True)
    selected_indices = [
        idx
        for idx, row in enumerate(joined)
        if str(row.get("selected_or_not", "")).lower() in {"true", "1", "yes"}
    ]
    selected_idx = selected_indices[0] if selected_indices else score_order[0]
    best_gain_idx = gain_order[0]
    top3_score = set(score_order[:3])
    stats = {
        "scores": str(args.scores.resolve()),
        "metrics": str(args.metrics.resolve()),
        "metric": args.metric,
        "score_column": args.score_column,
        "candidate_count": len(joined),
        "pearson": pearson(xs, ys),
        "spearman": spearman(xs, ys),
        "kendall": kendall_tau(xs, ys),
        "top1_accuracy": 1.0 if selected_idx == best_gain_idx else 0.0,
        "top3_hit_rate": 1.0 if best_gain_idx in top3_score else 0.0,
        "selection_regret": ys[best_gain_idx] - ys[selected_idx],
        "selected_candidate": joined[selected_idx].get(args.candidate_key),
        "best_gain_candidate": joined[best_gain_idx].get(args.candidate_key),
    }

    write_csv(args.output_stats, [stats], list(stats.keys()))
    if args.output_joined is not None:
        write_csv(args.output_joined, joined, list(joined[0].keys()))
    plot(joined, args)
    print(f"Wrote {args.output_stats}")
    if args.output_joined is not None:
        print(f"Wrote {args.output_joined}")
    if args.output_plot is not None:
        print(f"Wrote {args.output_plot}")


if __name__ == "__main__":
    main()
