from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from rr_gs_test.gaussians3d import Gaussians3D
from simplecv.rerun_log_utils import get_safe_application_id


@dataclass
class RerunTyroConfig:
    application_id: str = field(default_factory=get_safe_application_id)
    """Name of the application."""
    recording_id: str | UUID | None = None
    """Recording ID."""
    connect: bool = False
    """Whether to connect to an existing rerun instance or not."""
    save: Path | None = None
    """Path to save the rerun data without visualizing it."""
    serve: bool = False
    """Serve the rerun data."""
    headless: bool = False
    """Run rerun in headless mode."""
    executable_name: str = "rerun"
    """Executable name passed to ``rerun.spawn`` when launching the viewer."""
    executable_path: str | None = None
    """Optional absolute or relative path to the Rerun executable."""
    memory_limit: str = "75%"
    """Viewer memory limit passed through to ``rerun.spawn``."""
    server_memory_limit: str = "4GiB"
    """gRPC proxy memory limit passed through to ``rerun.spawn``."""

    def __post_init__(self) -> None:
        rr.init(
            application_id=self.application_id,
            recording_id=self.recording_id,
            default_enabled=True,
            strict=True,
        )
        if self.serve:
            rr.serve_grpc()
            rr.serve_web_viewer(open_browser=not self.headless)
        elif self.connect:
            rr.connect_grpc()
        elif self.save is not None:
            rr.save(self.save)
        elif not self.headless:
            rr.spawn(
                executable_name=self.executable_name,
                executable_path=self.executable_path,
                memory_limit=self.memory_limit,
                server_memory_limit=self.server_memory_limit,
            )


@dataclass
class ViewSplatConfig:
    """Configuration for single-image metric depth estimation."""

    rr_config: RerunTyroConfig
    """Rerun logging configuration."""
    ply_path: Path = Path("data/room/splat-bedroom.ply")
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
    print(f"Loading splat from {ply_path}")

    gaussians: Gaussians3D = Gaussians3D.from_ply(ply_path)
    parent_log_path = Path("/world/splat")
    # required or else the viewer will not show the entity
    rr.send_blueprint(splat_blueprint("", gaussians))
    # rr.log("/", rr.ViewCoordinates.RFU, static=True)
    rr.log(str(parent_log_path), gaussians, static=True)
