#!/usr/bin/env python3
"""Aggregate active-view 3DGS render metrics across experiment runs."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}


@dataclass
class ImageMetrics:
    image: str
    original: str
    global_l1: float
    global_psnr: float
    global_ssim: float
    global_lpips: float
    roi_l1: float
    roi_psnr: float
    roi_ssim: float
    roi_lpips: float
    non_roi_l1: float


@dataclass
class RunMetrics:
    run_id: str
    method: str
    k: int
    seed: int | None
    render_dir: str
    global_l1: float
    global_psnr: float
    global_ssim: float
    global_lpips: float
    roi_l1: float
    roi_psnr: float
    roi_ssim: float
    roi_lpips: float
    non_roi_l1: float
    evaluated_images: int
    roi_images: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate active-view experiment renders. Global metrics are computed "
            "for all matched renders. ROI metrics are computed only for images "
            "that have matching ROI candidates."
        )
    )
    parser.add_argument("--runs-manifest", type=Path, required=True)
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument(
        "--render-subdir",
        default="test/ours_30000/renders",
        help="Subdirectory under each run_id in render-root. Default: test/ours_30000/renders.",
    )
    parser.add_argument(
        "--view-split",
        choices=("auto", "train", "test"),
        default="auto",
        help=(
            "Manifest view split used to match renders to originals. "
            "Default: auto, which uses train_views for train/... render-subdir "
            "and test_views otherwise."
        ),
    )
    parser.add_argument("--roi-candidates", type=Path, default=None)
    parser.add_argument(
        "--roi-match-key",
        choices=("auto", "original", "render"),
        default="original",
        help=(
            "How ROI candidates are matched to evaluated images. "
            "Default: original, which matches stable original image names. "
            "Use render only for ROI files generated for the exact same render directory."
        ),
    )
    parser.add_argument(
        "--roi-source-run-id",
        default="base_6",
        help=(
            "Run id whose render numbering was used when ROI candidates were generated. "
            "Used to remap ROI entries such as 00000.png back to stable original names. "
            "Default: base_6."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resize-renders", action="store_true")
    parser.add_argument(
        "--compute-lpips",
        action="store_true",
        help="Compute LPIPS if torch and lpips are installed; otherwise LPIPS is NaN.",
    )
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def load_rgb_resized(path: Path, size_wh: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize(size_wh, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Render directory does not exist: {folder}")
    images = sorted(path for path in folder.iterdir() if path.is_file() and path.suffix in IMAGE_SUFFIXES)
    if not images:
        raise RuntimeError(f"No renders found in: {folder}")
    return images


def mse(render: np.ndarray, original: np.ndarray, mask: np.ndarray | None = None) -> float:
    error = (render - original) ** 2
    if mask is not None:
        if not mask.any():
            return float("nan")
        error = error[mask]
    return float(error.mean())


def psnr(render: np.ndarray, original: np.ndarray, mask: np.ndarray | None = None) -> float:
    value = mse(render, original, mask)
    if not np.isfinite(value):
        return float("nan")
    if value <= 1e-12:
        return float("inf")
    return float(-10.0 * math.log10(value))


def l1(render: np.ndarray, original: np.ndarray, mask: np.ndarray | None = None) -> float:
    error = np.abs(render - original).mean(axis=2)
    if mask is not None:
        if not mask.any():
            return float("nan")
        error = error[mask]
    return float(error.mean())


def ssim_metric(render: np.ndarray, original: np.ndarray, mask: np.ndarray | None = None) -> float:
    try:
        from skimage.metrics import structural_similarity  # type: ignore
    except ImportError:
        return float("nan")
    if mask is not None:
        if not mask.any():
            return float("nan")
        ys, xs = np.where(mask)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        render = render[y0:y1, x0:x1]
        original = original[y0:y1, x0:x1]
    height, width = render.shape[:2]
    if height < 7 or width < 7:
        return float("nan")
    return float(structural_similarity(original, render, channel_axis=2, data_range=1.0))


class LpipsComputer:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.model = None
        self.torch = None
        if not enabled:
            return
        try:
            import lpips  # type: ignore
            import torch  # type: ignore
        except ImportError:
            return
        self.torch = torch
        self.model = lpips.LPIPS(net="alex")
        self.model.eval()

    def __call__(self, render: np.ndarray, original: np.ndarray, mask: np.ndarray | None = None) -> float:
        if self.model is None or self.torch is None:
            return float("nan")
        if mask is not None:
            if not mask.any():
                return float("nan")
            ys, xs = np.where(mask)
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            render = render[y0:y1, x0:x1]
            original = original[y0:y1, x0:x1]
        render_tensor = self.torch.from_numpy(render).permute(2, 0, 1).unsqueeze(0).float() * 2.0 - 1.0
        original_tensor = self.torch.from_numpy(original).permute(2, 0, 1).unsqueeze(0).float() * 2.0 - 1.0
        with self.torch.no_grad():
            return float(self.model(render_tensor, original_tensor).item())


def summarize(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def view_name_keys(view: dict) -> list[str]:
    keys = [str(view["name"]), Path(str(view["name"])).stem]
    if "path" in view:
        path = Path(str(view["path"]))
        keys.extend([path.name, path.stem])
    return list(dict.fromkeys(keys))


def render_name_keys(path: Path) -> list[str]:
    return [path.name, path.stem]


def build_source_render_aliases(source_views: list[dict]) -> dict[str, list[str]]:
    aliases: dict[str, list[str]] = {}
    sorted_views = sorted(source_views, key=lambda item: int(item["index"]))
    for pos, view in enumerate(sorted_views):
        original_keys = view_name_keys(view)
        render_stem = f"{pos:05d}"
        render_keys = [render_stem, f"{render_stem}.png", f"{render_stem}.jpg", f"{render_stem}.jpeg"]
        for key in render_keys:
            aliases[key] = original_keys
    return aliases


def add_roi_candidate(by_image: dict[str, list[dict]], key: str, candidate: dict) -> None:
    by_image.setdefault(key, []).append(candidate)


def load_roi_candidates(path: Path | None, source_views: list[dict] | None = None) -> dict[str, list[dict]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_image: dict[str, list[dict]] = {}
    source_aliases = build_source_render_aliases(source_views) if source_views else {}
    for candidate in payload.get("roi_candidates", []):
        item = dict(candidate)
        item["bbox_xywh"] = [int(round(value)) for value in item["bbox_xywh"]]
        image = str(item["image"])
        raw_keys = [image, Path(image).stem]
        for key in raw_keys:
            add_roi_candidate(by_image, key, item)
            for alias in source_aliases.get(key, []):
                add_roi_candidate(by_image, alias, item)
    return by_image


def make_roi_mask(height: int, width: int, candidates: list[dict]) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for candidate in candidates:
        x, y, w, h = candidate["bbox_xywh"]
        x0 = max(0, min(width, x))
        y0 = max(0, min(height, y))
        x1 = max(0, min(width, x + w))
        y1 = max(0, min(height, y + h))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    return mask


def resolve_view_key(render_subdir: str, view_split: str) -> str:
    if view_split != "auto":
        return f"{view_split}_views"
    normalized = render_subdir.replace("\\", "/").strip("/")
    first_part = normalized.split("/", 1)[0]
    return "train_views" if first_part == "train" else "test_views"


def match_originals(render_paths: list[Path], views: list[dict], view_key: str) -> list[tuple[Path, dict]]:
    views_by_name = {view["name"]: view for view in views}
    views_by_stem = {Path(view["name"]).stem: view for view in views}
    sorted_views = sorted(views, key=lambda item: int(item["index"]))
    pairs: list[tuple[Path, dict]] = []
    for pos, render_path in enumerate(render_paths):
        original = views_by_name.get(render_path.name) or views_by_stem.get(render_path.stem)
        if original is None:
            if pos >= len(sorted_views):
                raise IndexError(f"Cannot map render {render_path.name} to {view_key}")
            original = sorted_views[pos]
        pairs.append((render_path, original))
    return pairs


def resolve_roi_candidates(
    roi_by_image: dict[str, list[dict]],
    render_path: Path,
    original_view: dict,
    roi_match_key: str,
) -> list[dict] | None:
    original_keys = view_name_keys(original_view)
    render_keys = render_name_keys(render_path)
    if roi_match_key == "original":
        lookup_keys = original_keys
    elif roi_match_key == "render":
        lookup_keys = render_keys
    else:
        lookup_keys = original_keys + render_keys
    for key in lookup_keys:
        candidates = roi_by_image.get(key)
        if candidates:
            return candidates
    return None


def evaluate_run(
    run: dict,
    render_root: Path,
    render_subdir: str,
    view_key: str,
    roi_by_image: dict[str, list[dict]],
    roi_match_key: str,
    resize: bool,
    lpips_computer: LpipsComputer,
) -> tuple[RunMetrics, list[ImageMetrics]]:
    run_id = run["run_id"]
    run_output_dir = Path(run.get("output_dir", render_root / run_id))
    render_dir = run_output_dir / render_subdir
    render_paths = list_images(render_dir)
    if view_key not in run:
        raise KeyError(f"Run {run_id} does not contain {view_key}")
    pairs = match_originals(render_paths, run[view_key], view_key)
    per_image: list[ImageMetrics] = []

    for render_path, original_view in pairs:
        original_path = Path(original_view["path"])
        original = load_rgb(original_path)
        render = load_rgb(render_path)
        size_wh = (original.shape[1], original.shape[0])
        if render.shape[:2] != original.shape[:2]:
            if not resize:
                raise ValueError(
                    f"Size mismatch for {render_path}: got {render.shape[:2]}, expected {original.shape[:2]}. "
                    "Use --resize-renders to resize."
                )
            render = load_rgb_resized(render_path, size_wh)

        roi_candidates = resolve_roi_candidates(roi_by_image, render_path, original_view, roi_match_key)
        roi_mask = make_roi_mask(original.shape[0], original.shape[1], roi_candidates) if roi_candidates else None
        non_roi_mask = ~roi_mask if roi_mask is not None else None
        per_image.append(
            ImageMetrics(
                image=render_path.name,
                original=original_view["name"],
                global_l1=l1(render, original),
                global_psnr=psnr(render, original),
                global_ssim=ssim_metric(render, original),
                global_lpips=lpips_computer(render, original),
                roi_l1=l1(render, original, roi_mask) if roi_mask is not None else float("nan"),
                roi_psnr=psnr(render, original, roi_mask) if roi_mask is not None else float("nan"),
                roi_ssim=ssim_metric(render, original, roi_mask) if roi_mask is not None else float("nan"),
                roi_lpips=lpips_computer(render, original, roi_mask) if roi_mask is not None else float("nan"),
                non_roi_l1=l1(render, original, non_roi_mask) if non_roi_mask is not None else float("nan"),
            )
        )

    run_metrics = RunMetrics(
        run_id=run_id,
        method=run["method"],
        k=int(run["k"]),
        seed=run.get("seed"),
        render_dir=str(render_dir),
        global_l1=summarize([item.global_l1 for item in per_image]),
        global_psnr=summarize([item.global_psnr for item in per_image]),
        global_ssim=summarize([item.global_ssim for item in per_image]),
        global_lpips=summarize([item.global_lpips for item in per_image]),
        roi_l1=summarize([item.roi_l1 for item in per_image]),
        roi_psnr=summarize([item.roi_psnr for item in per_image]),
        roi_ssim=summarize([item.roi_ssim for item in per_image]),
        roi_lpips=summarize([item.roi_lpips for item in per_image]),
        non_roi_l1=summarize([item.non_roi_l1 for item in per_image]),
        evaluated_images=len(per_image),
        roi_images=sum(bool(np.isfinite(item.roi_l1)) for item in per_image),
    )
    return run_metrics, per_image


def main() -> None:
    args = parse_args()
    runs_manifest = json.loads(args.runs_manifest.read_text(encoding="utf-8"))
    view_key = resolve_view_key(args.render_subdir, args.view_split)
    view_split_label = view_key[:-len("_views")]
    source_run = next(
        (run for run in runs_manifest["runs"] if run["run_id"] == args.roi_source_run_id),
        None,
    )
    source_views = source_run.get(view_key) if source_run and view_key in source_run else None
    roi_by_image = load_roi_candidates(args.roi_candidates, source_views)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lpips_computer = LpipsComputer(args.compute_lpips)

    run_metrics: list[RunMetrics] = []
    per_image_payload: dict[str, list[dict]] = {}
    for run in runs_manifest["runs"]:
        metrics, per_image = evaluate_run(
            run,
            args.render_root,
            args.render_subdir,
            view_key,
            roi_by_image,
            args.roi_match_key,
            args.resize_renders,
            lpips_computer,
        )
        run_metrics.append(metrics)
        per_image_payload[metrics.run_id] = [asdict(item) for item in per_image]

    summary = {
        "runs_manifest": str(args.runs_manifest.resolve()),
        "render_root": str(args.render_root.resolve()),
        "render_subdir": args.render_subdir,
        "view_split": view_split_label,
        "roi_candidates": str(args.roi_candidates.resolve()) if args.roi_candidates else None,
        "roi_match_key": args.roi_match_key,
        "roi_source_run_id": args.roi_source_run_id if source_views else None,
        "runs": [asdict(item) for item in run_metrics],
        "per_image": per_image_payload,
    }
    (output_dir / "active_view_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Active View Evaluation",
        "",
        f"View split: `{view_split_label}`",
        "",
        "| Run | Method | k | Seed | Global L1 down | Global PSNR up | Global SSIM up | Global LPIPS down | ROI L1 down | ROI PSNR up | ROI SSIM up | ROI LPIPS down | Non-ROI L1 down | Images | ROI Images |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in sorted(run_metrics, key=lambda metric: (metric.method, metric.k, metric.seed if metric.seed is not None else -1)):
        seed = "" if item.seed is None else str(item.seed)
        lines.append(
            f"| {item.run_id} | {item.method} | {item.k} | {seed} | "
            f"{item.global_l1:.6f} | {item.global_psnr:.3f} | "
            f"{item.global_ssim:.4f} | {item.global_lpips:.4f} | "
            f"{item.roi_l1:.6f} | {item.roi_psnr:.3f} | "
            f"{item.roi_ssim:.4f} | {item.roi_lpips:.4f} | {item.non_roi_l1:.6f} | "
            f"{item.evaluated_images} | {item.roi_images} |"
        )
    (output_dir / "active_view_eval_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_dir / 'active_view_eval_summary.json'}")
    print(f"Wrote {output_dir / 'active_view_eval_report.md'}")


if __name__ == "__main__":
    main()
