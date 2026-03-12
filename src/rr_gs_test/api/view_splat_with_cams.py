import math
from dataclasses import dataclass
from pathlib import Path

from jaxtyping import Float32
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from numpy import ndarray
from PIL import Image
from rr_gs_test.gaussians3d import Gaussians3D
from serde import serde
from serde.json import from_json
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.ops.conventions import CC, convert_pose
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole
import cv2


@serde
class BlenderFrame:
    file_path: str
    rotation: float
    transform_matrix: Float32[ndarray, "4 4"]


@serde
class BlenderTransform:
    camera_angle_x: float
    frames: list[BlenderFrame]


@dataclass
class BlenderSequenceData:
    pinhole_params: list[PinholeParameters]
    image_paths: list[Path]


@dataclass
class ViewSplatConfig:
    """Configuration for single-image metric depth estimation."""

    rr_config: RerunTyroConfig
    """Rerun logging configuration."""
    example_dir: Path = Path("data/chair/")
    """Path to the input example directory containing the 3D splat data."""


def splat_blueprint(entity_path: str, gaussians: Gaussians3D) -> rrb.Blueprint:
    bounds_min = gaussians.centers.min(axis=0)
    bounds_max = gaussians.centers.max(axis=0)
    center = 0.5 * (bounds_min + bounds_max)
    extent = bounds_max - bounds_min
    distance: float = max(float(np.linalg.norm(extent)), 1.0) * 1.5

    return rrb.Blueprint(
        rrb.Spatial3DView(
            origin="/",
            name="Scene",
            overrides={entity_path: rrb.Visualizer("GaussianSplats3D")},
            eye_controls=rrb.EyeControls3D(
                position=center + np.array([distance, distance * 0.5, distance], dtype=np.float32),
                look_target=center,
                eye_up=(0.0, 1.0, 0.0),
            ),
        )
    )


def _resolve_blender_image_path(dataset_root: Path, file_path: str) -> Path:
    relative_path = Path(file_path.removeprefix("./"))
    if relative_path.suffix == "":
        relative_path: Path = relative_path.with_suffix(".png")
    return dataset_root / relative_path


def blender_to_simplecv(camera_data: BlenderTransform, dataset_root: Path) -> BlenderSequenceData:
    if len(camera_data.frames) == 0:
        raise ValueError("Blender camera data must contain at least one frame.")

    image_paths: list[Path] = [
        _resolve_blender_image_path(dataset_root, frame.file_path) for frame in camera_data.frames
    ]

    with Image.open(image_paths[0]) as first_image:
        width, height = first_image.size

    focal_length: float = 0.5 * width / math.tan(camera_data.camera_angle_x / 2.0)
    intrinsics: Intrinsics = Intrinsics.from_focal_principal_point(
        camera_conventions="RDF",
        fl_x=focal_length,
        fl_y=focal_length,
        cx=width / 2.0,
        cy=height / 2.0,
        width=width,
        height=height,
    )

    pinhole_params: list[PinholeParameters] = []
    for frame, image_path in zip(camera_data.frames, image_paths, strict=True):
        world_T_cam_gl = np.asarray(frame.transform_matrix, dtype=np.float32)
        world_T_cam_cv = np.asarray(convert_pose(world_T_cam_gl, CC.GL, CC.CV), dtype=np.float32)
        extrinsics = Extrinsics(
            world_R_cam=world_T_cam_cv[:3, :3],
            world_t_cam=world_T_cam_cv[:3, 3],
        )
        pinhole_params.append(
            PinholeParameters(
                name=image_path.stem,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
            )
        )

    return BlenderSequenceData(
        pinhole_params=pinhole_params,
        image_paths=image_paths,
    )


def main(cfg: ViewSplatConfig) -> None:
    ply_path: Path = cfg.example_dir / "chair.ply"
    camera_data_path: Path = cfg.example_dir / "transforms_train.json"
    camera_data: BlenderTransform = from_json(BlenderTransform, camera_data_path.read_text())
    blender_sequence: BlenderSequenceData = blender_to_simplecv(camera_data, cfg.example_dir)
    print(f"Camera angle x: {camera_data.camera_angle_x} radians")
    print(f"Loaded {len(blender_sequence.pinhole_params)} camera/image pairs from {cfg.example_dir}")

    gaussians: Gaussians3D = Gaussians3D.from_ply(ply_path)
    parent_log_path = Path("/world/splat")
    cameras_log_path = Path("/world/cameras")
    # required or else the viewer will not show the entity
    rr.send_blueprint(splat_blueprint("", gaussians))
    # rr.log("/", rr.ViewCoordinates.RFU, static=True)
    rr.log(str(parent_log_path), gaussians, static=True)
    for index, (camera, image_path) in enumerate(
        zip(blender_sequence.pinhole_params, blender_sequence.image_paths, strict=True)
    ):
        cam_log_path: Path = cameras_log_path / f"cam_{index:03d}"
        image_log_path: Path = cam_log_path / "pinhole" / "image"
        log_pinhole(
            camera,
            cam_log_path,
            image_plane_distance=0.25,
        )
        image = Image.open(image_path)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2RGB)
        rr.log(str(image_log_path), rr.Image(image_cv).compress(jpeg_quality=75), static=True)
