"""Log the example chair Gaussian splat by spawning the packaged viewer."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from .gaussians3d import Gaussians3D

APP_ID = "rr-gs-test"
VIEWER_PORT = 9876
VIEWER_URL = "rerun+http://127.0.0.1:9876/proxy"
VIEWER_EXECUTABLE_NAME = os.environ.get("RR_GS_VIEWER_EXECUTABLE_NAME", "rerun-gs-viewer")
VIEWER_EXECUTABLE_PATH = os.environ.get("RR_GS_VIEWER_EXECUTABLE_PATH")
VIEW_ROOT = "/"
DEFAULT_ENTITY_PATH = "world/splats"
DEFAULT_PLY = (
    Path(__file__).resolve().parents[3] / "rerun-simple-gs" / "examples" / "chair.ply"
)


def args_from_argv() -> tuple[Path, str]:
    args = sys.argv[1:]
    if len(args) > 2:
        raise SystemExit(
            "usage: pixi run log-chair [scene.ply] [entity/path]"
        )
    ply_path = Path(args[0]) if args else DEFAULT_PLY
    entity_path = args[1] if len(args) == 2 else DEFAULT_ENTITY_PATH
    return ply_path, entity_path


def splat_blueprint(entity_path: str, gaussians: Gaussians3D) -> rrb.Blueprint:
    bounds_min = gaussians.centers.min(axis=0)
    bounds_max = gaussians.centers.max(axis=0)
    center = 0.5 * (bounds_min + bounds_max)
    extent = bounds_max - bounds_min
    distance = max(float(np.linalg.norm(extent)), 1.0) * 1.5

    return rrb.Blueprint(
        rrb.Spatial3DView(
            origin=VIEW_ROOT,
            name="Scene",
            overrides={entity_path: rrb.Visualizer("GaussianSplats3D")},
            eye_controls=rrb.EyeControls3D(
                position=center + np.array([distance, distance * 0.5, distance], dtype=np.float32),
                look_target=center,
                eye_up=(0.0, 1.0, 0.0),
            ),
        )
    )


def connect_or_spawn_viewer() -> None:
    spawn_kwargs = {
        "port": VIEWER_PORT,
        "connect": True,
        "executable_name": VIEWER_EXECUTABLE_NAME,
    }
    if VIEWER_EXECUTABLE_PATH:
        spawn_kwargs["executable_path"] = VIEWER_EXECUTABLE_PATH

    rr.spawn(**spawn_kwargs)


def main() -> None:
    ply_path, entity_path = args_from_argv()
    if not ply_path.exists():
        raise SystemExit(f"PLY not found: {ply_path}")

    gaussians = Gaussians3D.from_ply(ply_path)

    rr.init(APP_ID, spawn=False)
    connect_or_spawn_viewer()
    rr.send_blueprint(splat_blueprint(entity_path, gaussians))
    rr.log(entity_path, rr.Clear(recursive=True), static=True)
    rr.log(entity_path, gaussians, static=True)
    rr.disconnect()

    print(f"Logged {ply_path} to {VIEWER_URL} as {entity_path}")


if __name__ == "__main__":
    main()
