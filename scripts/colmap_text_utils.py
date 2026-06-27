#!/usr/bin/env python3
"""Small COLMAP text-model helpers for pose-aware NBV scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]

    @property
    def fx(self) -> float:
        if self.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
            return self.params[0]
        if self.model == "PINHOLE":
            return self.params[0]
        return self.params[0]

    @property
    def fy(self) -> float:
        if self.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
            return self.params[0]
        if self.model == "PINHOLE":
            return self.params[1]
        return self.params[0]

    @property
    def cx(self) -> float:
        if self.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
            return self.params[1]
        if self.model == "PINHOLE":
            return self.params[2]
        return self.width * 0.5

    @property
    def cy(self) -> float:
        if self.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
            return self.params[2]
        if self.model == "PINHOLE":
            return self.params[3]
        return self.height * 0.5


@dataclass(frozen=True)
class Observation:
    x: float
    y: float
    point3d_id: int


@dataclass(frozen=True)
class ImageRecord:
    image_id: int
    qvec: tuple[float, float, float, float]
    tvec: tuple[float, float, float]
    camera_id: int
    name: str
    observations: tuple[Observation, ...]


@dataclass(frozen=True)
class Point3D:
    point3d_id: int
    xyz: tuple[float, float, float]
    rgb: tuple[int, int, int]
    error: float


@dataclass(frozen=True)
class ColmapModel:
    cameras: dict[int, Camera]
    images: dict[str, ImageRecord]
    points3d: dict[int, Point3D]
    warnings: tuple[str, ...]


def parse_cameras(path: Path) -> tuple[dict[int, Camera], list[str]]:
    cameras: dict[int, Camera] = {}
    warnings: list[str] = []
    supported = {"PINHOLE", "SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        camera = Camera(
            camera_id=int(parts[0]),
            model=parts[1],
            width=int(parts[2]),
            height=int(parts[3]),
            params=tuple(float(item) for item in parts[4:]),
        )
        cameras[camera.camera_id] = camera
        if camera.model not in supported:
            warnings.append(f"Unsupported camera model {camera.model}; using first parameters as a pinhole approximation")
        elif camera.model in {"SIMPLE_RADIAL", "RADIAL"}:
            warnings.append(f"Camera model {camera.model} distortion is ignored for NBV projection")
    return cameras, warnings


def parse_images(path: Path) -> dict[str, ImageRecord]:
    images: dict[str, ImageRecord] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line or line.startswith("#"):
            index += 1
            continue
        if index + 1 >= len(lines):
            raise ValueError(f"COLMAP image record without points2D line: {line!r}")
        parts = line.split()
        image_id = int(parts[0])
        qvec = tuple(float(item) for item in parts[1:5])
        tvec = tuple(float(item) for item in parts[5:8])
        camera_id = int(parts[8])
        name = Path(parts[9]).name
        points_parts = lines[index + 1].strip().split()
        observations: list[Observation] = []
        if len(points_parts) % 3 != 0:
            raise ValueError(f"Invalid COLMAP points2D line for {name}")
        for obs_idx in range(0, len(points_parts), 3):
            observations.append(
                Observation(
                    x=float(points_parts[obs_idx]),
                    y=float(points_parts[obs_idx + 1]),
                    point3d_id=int(points_parts[obs_idx + 2]),
                )
            )
        images[name] = ImageRecord(
            image_id=image_id,
            qvec=qvec,  # type: ignore[arg-type]
            tvec=tvec,  # type: ignore[arg-type]
            camera_id=camera_id,
            name=name,
            observations=tuple(observations),
        )
        index += 2
    return images


def parse_points3d(path: Path) -> dict[int, Point3D]:
    points: dict[int, Point3D] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        point_id = int(parts[0])
        points[point_id] = Point3D(
            point3d_id=point_id,
            xyz=(float(parts[1]), float(parts[2]), float(parts[3])),
            rgb=(int(parts[4]), int(parts[5]), int(parts[6])),
            error=float(parts[7]),
        )
    return points


def load_colmap_text_model(sparse_dir: Path) -> ColmapModel:
    cameras, warnings = parse_cameras(sparse_dir / "cameras.txt")
    return ColmapModel(
        cameras=cameras,
        images=parse_images(sparse_dir / "images.txt"),
        points3d=parse_points3d(sparse_dir / "points3D.txt"),
        warnings=tuple(warnings),
    )


def qvec_to_rotmat(qvec: tuple[float, float, float, float]) -> tuple[tuple[float, float, float], ...]:
    qw, qx, qy, qz = qvec
    return (
        (
            1.0 - 2.0 * qy * qy - 2.0 * qz * qz,
            2.0 * qx * qy - 2.0 * qw * qz,
            2.0 * qz * qx + 2.0 * qw * qy,
        ),
        (
            2.0 * qx * qy + 2.0 * qw * qz,
            1.0 - 2.0 * qz * qz - 2.0 * qx * qx,
            2.0 * qy * qz - 2.0 * qw * qx,
        ),
        (
            2.0 * qz * qx - 2.0 * qw * qy,
            2.0 * qy * qz + 2.0 * qw * qx,
            1.0 - 2.0 * qx * qx - 2.0 * qy * qy,
        ),
    )


def mat_vec_mul(matrix: tuple[tuple[float, float, float], ...], vector: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(sum(matrix[row][col] * vector[col] for col in range(3)) for row in range(3))  # type: ignore[return-value]


def camera_center(image: ImageRecord) -> tuple[float, float, float]:
    rotation = qvec_to_rotmat(image.qvec)
    tx, ty, tz = image.tvec
    return tuple(
        -sum(rotation[row][col] * (tx, ty, tz)[row] for row in range(3))
        for col in range(3)
    )  # type: ignore[return-value]


def view_direction(image: ImageRecord) -> tuple[float, float, float]:
    rotation = qvec_to_rotmat(image.qvec)
    direction = tuple(rotation[2][col] for col in range(3))
    norm = math.sqrt(sum(item * item for item in direction))
    if norm <= 1e-12:
        return (0.0, 0.0, 1.0)
    return tuple(item / norm for item in direction)  # type: ignore[return-value]


def world_to_camera(image: ImageRecord, xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    rotated = mat_vec_mul(qvec_to_rotmat(image.qvec), xyz)
    return (
        rotated[0] + image.tvec[0],
        rotated[1] + image.tvec[1],
        rotated[2] + image.tvec[2],
    )


def project_point(
    camera: Camera,
    image: ImageRecord,
    xyz: tuple[float, float, float],
    depth_epsilon: float = 1e-6,
) -> tuple[float, float, float] | None:
    x_cam, y_cam, z_cam = world_to_camera(image, xyz)
    if z_cam <= depth_epsilon:
        return None
    x = camera.fx * (x_cam / z_cam) + camera.cx
    y = camera.fy * (y_cam / z_cam) + camera.cy
    if x < 0.0 or x >= camera.width or y < 0.0 or y >= camera.height:
        return None
    return x, y, z_cam


def vector_sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vector_norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(sum(item * item for item in vector))


def angle_between(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    norm_a = vector_norm(a)
    norm_b = vector_norm(b)
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    dot = sum(a[idx] * b[idx] for idx in range(3)) / (norm_a * norm_b)
    dot = max(-1.0, min(1.0, dot))
    return math.acos(dot)
