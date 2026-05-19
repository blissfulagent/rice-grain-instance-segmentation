"""Lean rice-grain instance segmentation pipeline.

The script builds a stylized instance map from a dark-background rice image:
1. make a clean foreground mask,
2. place one seed near each grain center,
3. estimate each grain's local direction,
4. render each grain as a colored ellipse on a black background.

Run:
    python rice_segmentation.py --input inputs/InputImage.jpg --expected inputs/ExpectedOutput.jpeg
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import peak_local_max


@dataclass(frozen=True)
class Config:
    input_path: str = "inputs/InputImage.jpg"
    expected_path: Optional[str] = "inputs/ExpectedOutput.jpeg"
    output_dir: str = "outputs"

    foreground_threshold: int = 60
    blur_kernel: int = 5
    morph_kernel: int = 3
    close_iterations: int = 1

    min_seed_distance: int = 12
    seed_threshold_abs: float = 2.0
    orientation_radius: int = 40
    ellipse_scale: float = 2.0

    clip_to_foreground: bool = False
    clip_dilation: int = 15
    random_seed: int = 42


def read_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def ensure_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


# Stage 1: grayscale and smoothing

def preprocess(image_bgr: np.ndarray, cfg: Config) -> Dict[str, np.ndarray]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur_size = ensure_odd(cfg.blur_kernel)
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    return {"gray": gray, "blurred": blurred}


# Stage 2: foreground extraction

def make_foreground_mask(blurred: np.ndarray, cfg: Config) -> np.ndarray:
    mask = np.uint8(blurred > cfg.foreground_threshold) * 255
    kernel_size = ensure_odd(cfg.morph_kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=cfg.close_iterations)
    return mask


# Stage 3: seed placement

def find_grain_seeds(mask: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    distance_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    seed_points = peak_local_max(
        distance_map,
        min_distance=cfg.min_seed_distance,
        threshold_abs=cfg.seed_threshold_abs,
        labels=(mask > 0),
        exclude_border=False,
    )
    return seed_points, distance_map


def draw_seed_overlay(image_bgr: np.ndarray, seed_points: np.ndarray) -> np.ndarray:
    overlay = image_bgr.copy()
    for row, col in seed_points:
        cv2.circle(overlay, (int(col), int(row)), 3, (0, 0, 255), -1)
    return overlay


# Stage 4: local orientation estimation

def estimate_local_pose(
    row: int,
    col: int,
    mask: np.ndarray,
    gray: np.ndarray,
    cfg: Config,
) -> Tuple[float, float, float]:
    height, width = gray.shape
    radius = cfg.orientation_radius

    y0, y1 = max(0, row - radius), min(height, row + radius + 1)
    x0, x1 = max(0, col - radius), min(width, col + radius + 1)

    yy, xx = np.mgrid[y0:y1, x0:x1]
    local_mask = (mask[y0:y1, x0:x1] > 0).astype(np.float32)
    local_gray = gray[y0:y1, x0:x1].astype(np.float32)

    sigma = max(radius / 2.0, 1.0)
    gaussian_focus = np.exp(-((yy - row) ** 2 + (xx - col) ** 2) / (2 * sigma * sigma))
    brightness_weight = np.clip(local_gray - 40.0, 0.0, None) / 100.0
    weights = local_mask * gaussian_focus * (1.0 + brightness_weight)

    total = float(weights.sum())
    if total < 1e-6:
        return 0.0, 70.0, 24.0

    cx = float((weights * xx).sum() / total)
    cy = float((weights * yy).sum() / total)
    dx = xx - cx
    dy = yy - cy

    cov_xx = float((weights * dx * dx).sum() / total)
    cov_yy = float((weights * dy * dy).sum() / total)
    cov_xy = float((weights * dx * dy).sum() / total)

    covariance = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    angle = float(np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])))
    major_axis = 4.2 * float(np.sqrt(max(eigenvalues[0], 1.0)))
    minor_axis = 3.2 * float(np.sqrt(max(eigenvalues[1], 1.0)))

    major_axis = float(np.clip(major_axis, 40.0, 95.0)) * cfg.ellipse_scale
    minor_axis = float(np.clip(minor_axis, 14.0, 34.0)) * cfg.ellipse_scale

    if major_axis / max(minor_axis, 1.0) < 1.8:
        major_axis = minor_axis * 2.4

    return angle, major_axis, minor_axis


# Stage 5: instance rendering

def render_instances(
    image_shape: Tuple[int, int],
    mask: np.ndarray,
    gray: np.ndarray,
    seed_points: np.ndarray,
    cfg: Config,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    height, width = image_shape
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    rng = np.random.default_rng(cfg.random_seed)

    clip_mask = None
    if cfg.clip_to_foreground:
        dilate_size = ensure_odd(cfg.clip_dilation)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
        clip_mask = cv2.dilate(mask, kernel, iterations=1)

    instance_rows: List[Dict[str, float]] = []
    for instance_id, (row, col) in enumerate(seed_points, start=1):
        angle, major_axis, minor_axis = estimate_local_pose(int(row), int(col), mask, gray, cfg)
        color = rng.integers(45, 256, size=3, dtype=np.uint8).tolist()

        instance_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(
            instance_mask,
            ((float(col), float(row)), (major_axis, minor_axis), angle),
            color=255,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

        if clip_mask is not None:
            instance_mask = cv2.bitwise_and(instance_mask, instance_mask, mask=clip_mask)

        canvas[instance_mask > 0] = color
        instance_rows.append(
            {
                "id": float(instance_id),
                "x": float(col),
                "y": float(row),
                "major_axis": float(major_axis),
                "minor_axis": float(minor_axis),
                "angle_deg": float(angle),
            }
        )

    return canvas, instance_rows


# Stage 6: saving and comparison

def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def normalize_for_view(image: np.ndarray) -> np.ndarray:
    if image.max() == 0:
        return image.copy()
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def foreground_metrics(output_bgr: np.ndarray, expected_bgr: Optional[np.ndarray]) -> Dict[str, Optional[float]]:
    if expected_bgr is None:
        return {"foreground_iou": None, "dice_score": None}

    output_mask = np.any(output_bgr > 0, axis=2)
    expected_mask = np.any(expected_bgr > 10, axis=2)

    intersection = np.logical_and(output_mask, expected_mask).sum()
    union = np.logical_or(output_mask, expected_mask).sum()
    total = output_mask.sum() + expected_mask.sum()

    iou = float(intersection / union) if union else 0.0
    dice = float((2 * intersection) / total) if total else 0.0
    return {"foreground_iou": iou, "dice_score": dice}


def save_instance_csv(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "x", "y", "major_axis", "minor_axis", "angle_deg"]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_comparison(
    save_path: Path,
    original_bgr: np.ndarray,
    mask: np.ndarray,
    seed_overlay_bgr: np.ndarray,
    output_bgr: np.ndarray,
    expected_bgr: Optional[np.ndarray],
) -> None:
    panels = [
        ("Input", cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)),
        ("Foreground mask", mask, "gray"),
        ("Seed centers", cv2.cvtColor(seed_overlay_bgr, cv2.COLOR_BGR2RGB)),
        ("Generated output", cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)),
    ]

    if expected_bgr is not None:
        panels.append(("Expected output", cv2.cvtColor(expected_bgr, cv2.COLOR_BGR2RGB)))

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    if len(panels) == 1:
        axes = [axes]

    for axis, panel in zip(axes, panels):
        title = panel[0]
        image = panel[1]
        cmap = panel[2] if len(panel) == 3 else None
        axis.imshow(image, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(cfg: Config, show: bool = False) -> Dict[str, object]:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = read_image(cfg.input_path)
    expected = read_image(cfg.expected_path) if cfg.expected_path else None

    prep = preprocess(image, cfg)
    mask = make_foreground_mask(prep["blurred"], cfg)
    seeds, distance_map = find_grain_seeds(mask, cfg)
    seed_overlay = draw_seed_overlay(image, seeds)
    output, rows = render_instances(image.shape[:2], mask, prep["gray"], seeds, cfg)
    metrics = foreground_metrics(output, expected)

    save_image(output_dir / "01_gray.png", prep["gray"])
    save_image(output_dir / "02_blurred.png", prep["blurred"])
    save_image(output_dir / "03_foreground_mask.png", mask)
    save_image(output_dir / "04_distance_map.png", normalize_for_view(distance_map))
    save_image(output_dir / "05_seed_overlay.png", seed_overlay)
    save_image(output_dir / "06_final_segmented_output.png", output)
    save_instance_csv(output_dir / "seed_instances.csv", rows)
    save_comparison(output_dir / "comparison.png", image, mask, seed_overlay, output, expected)

    with (output_dir / "run_summary.txt").open("w") as file:
        file.write(f"Detected seed instances: {len(seeds)}\n")
        file.write(f"Foreground threshold: {cfg.foreground_threshold}\n")
        file.write(f"Minimum seed distance: {cfg.min_seed_distance}\n")
        file.write(f"Ellipse scale: {cfg.ellipse_scale}\n")
        file.write(f"Clip to foreground: {cfg.clip_to_foreground}\n")
        file.write(f"Foreground IoU: {metrics['foreground_iou']}\n")
        file.write(f"Dice score: {metrics['dice_score']}\n")

    if show:
        plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        plt.title("Final segmented output")
        plt.axis("off")
        plt.show()

    print(f"Saved outputs to: {output_dir}")
    print(f"Detected seed instances: {len(seeds)}")
    if metrics["foreground_iou"] is not None:
        print(f"Foreground IoU: {metrics['foreground_iou']:.4f}")
        print(f"Dice score: {metrics['dice_score']:.4f}")

    return {
        "output": output,
        "mask": mask,
        "seeds": seeds,
        "metrics": metrics,
        "output_dir": output_dir,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lean rice-grain instance segmentation pipeline")
    parser.add_argument("--input", default=Config.input_path, help="Path to the raw rice image")
    parser.add_argument("--expected", default=Config.expected_path, help="Path to the reference output image")
    parser.add_argument("--no-expected", action="store_true", help="Run without a reference image")
    parser.add_argument("--output-dir", default=Config.output_dir, help="Directory for generated files")
    parser.add_argument("--threshold", type=int, default=Config.foreground_threshold, help="Foreground threshold")
    parser.add_argument("--min-seed-distance", type=int, default=Config.min_seed_distance, help="Minimum distance between seeds")
    parser.add_argument("--ellipse-scale", type=float, default=Config.ellipse_scale, help="Scale factor for rendered ellipses")
    parser.add_argument("--clip-to-foreground", action="store_true", help="Clip rendered ellipses to a dilated foreground mask")
    parser.add_argument("--show", action="store_true", help="Display the final output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        input_path=args.input,
        expected_path=None if args.no_expected else args.expected,
        output_dir=args.output_dir,
        foreground_threshold=args.threshold,
        min_seed_distance=args.min_seed_distance,
        ellipse_scale=args.ellipse_scale,
        clip_to_foreground=args.clip_to_foreground,
    )
    run(cfg, show=args.show)


if __name__ == "__main__":
    main()
