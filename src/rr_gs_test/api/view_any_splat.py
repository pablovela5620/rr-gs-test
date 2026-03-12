from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from PIL import Image
from rr_gs_test.gaussians3d import Gaussians3D
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole
import cv2


@dataclass
class ViewSplatConfig:
    """Configuration for single-image metric depth estimation."""

    rr_config: RerunTyroConfig
    """Rerun logging configuration."""
    ply_path: Path = Path("data/chair/chair.ply")
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


def main(cfg: ViewSplatConfig) -> None:
    ply_path: Path = cfg.ply_path

    gaussians: Gaussians3D = Gaussians3D.from_ply(ply_path)
    parent_log_path = Path("/world/splat")
    # required or else the viewer will not show the entity
    rr.send_blueprint(splat_blueprint("", gaussians))
    # rr.log("/", rr.ViewCoordinates.RFU, static=True)
    rr.log(str(parent_log_path), gaussians, static=True)
