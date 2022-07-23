import argparse
from typing import List, Dict, Optional
from pathlib import Path

from . import logger
from .capture import Capture, Session
from .utils.colmap import Camera, Image, write_model


def add_session_to_colmap(session: Session, colmap_cameras: Dict, colmap_images: Dict,
                          image_prefix: str, keys: Optional[List] = None):
    # Prepare COLMAP cameras.
    camera_id_capture2colmap = {}

    for camera_id, sensor in session.cameras.items():
        colmap_camera_id = len(colmap_cameras) + 1
        colmap_cameras[colmap_camera_id] = Camera(colmap_camera_id, **sensor.asdict)
        camera_id_capture2colmap[camera_id] = colmap_camera_id

    T_cams2s = session.trajectories
    if session.proc is not None:
        T_s2w = session.proc.alignment_global.get_abs_pose('pose_graph_optimized')
    else:
        T_s2w = None
    if T_s2w is not None:
        T_cams2w = T_s2w * T_cams2s
    else:
        # No alignment to global world found.
        # Using the current's session coordinate frame as world frame.
        T_cams2w = T_cams2s

    # Prepare COLMAP images.
    if keys is None:
        keys = sorted(session.images.key_pairs())
    for ts, camera_id in keys:
        T_w2cam = session.get_pose(ts, camera_id, T_cams2w).inverse()
        colmap_image_id = len(colmap_images) + 1
        colmap_camera_id = camera_id_capture2colmap[camera_id]
        colmap_images[colmap_image_id] = Image(
            colmap_image_id, T_w2cam.qvec, T_w2cam.t, colmap_camera_id,
            (image_prefix / session.images[ts, camera_id]).as_posix(), [], [])


def run(capture: Capture, session_ids: List[str], output_path: Path, ext: str = '.bin'):
    output_path.mkdir(exist_ok=True, parents=True)

    colmap_cameras = {}
    colmap_images = {}
    colmap_points = {}
    for session_id in session_ids:
        prefix = capture.data_path(session_id).relative_to(capture.sessions_path())
        add_session_to_colmap(capture.sessions[session_id], colmap_cameras, colmap_images, prefix)

    # Write to disk.
    logger.info('Writing COLMAP empty %s reconstruction to %s.', ext, output_path.resolve())
    write_model(colmap_cameras, colmap_images, colmap_points, str(output_path), ext=ext)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--capture_path', type=Path, required=True)
    parser.add_argument('--session_ids', type=str, nargs='+', required=True)
    parser.add_argument('--output_path', type=Path, required=True)
    args = parser.parse_args().__dict__
    args['capture'] = Capture.load(args.pop('capture_path'))

    run(**args)
