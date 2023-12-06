import argparse
from pathlib import Path
from typing import List

from scantools import (
    logger,
    run_meshing,
    run_navvis_to_capture,
    run_rendering,
    to_meshlab_visualization,
)
from scantools.capture import Capture


def run(capture_path: Path, session_ids: List[str], navvis_dir: Path):
    if capture_path.exists():
        capture = Capture.load(capture_path)
    else:
        capture = Capture(sessions={}, path=capture_path)

    tiles_format = "none"
    mesh_id = "mesh"
    # meshing_method = "advancing_front"
    meshing_method = "poisson"

    for session in session_ids:
        if session not in capture.sessions:
            logger.info("Exporting NavVis session %s.", session)
            run_navvis_to_capture.run(
                navvis_dir / session,
                capture,
                tiles_format,
                session,
            )

        if (
            not capture.sessions[session].proc
            or mesh_id not in capture.sessions[session].proc.meshes
        ):
            logger.info("Meshing session %s.", session)
            run_meshing.run(
                capture,
                session,
                "point_cloud_final",
                mesh_id,
                method=meshing_method,
            )

        if not capture.sessions[session].depths:
            logger.info("Rendering session %s.", session)
            run_rendering.run(capture, session, mesh_id=mesh_id + "_simplified")

        to_meshlab_visualization.run(
            capture,
            session,
            f"trajectory_{session}",
            export_mesh=True,
            export_poses=True,
            mesh_id=mesh_id,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture_path", type=Path, required=True)
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--sessions", nargs="*", type=str, default=[])
    args = parser.parse_args()

    run(args.capture_path, args.sessions, args.input_path)
