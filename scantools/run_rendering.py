import argparse
from pathlib import Path
from tqdm import tqdm
import PIL.Image
import numpy as np

from . import logger
from .capture import Capture, RecordsDepth
from .proc.rendering import Renderer
from .utils.io import read_mesh, write_depth


def run(capture: Capture, session_id: str, mesh_id: str = 'mesh',
        render_images: bool = False):

    session = capture.sessions[session_id]
    assert session.proc is not None
    assert session.proc.meshes is not None
    assert mesh_id in session.proc.meshes
    assert session.images is not None

    mesh_path = capture.proc_path(session_id) / session.proc.meshes[mesh_id]
    mesh = read_mesh(mesh_path)
    renderer = Renderer(mesh)

    depths = session.depths
    if depths is None:
        depths = session.depths = RecordsDepth()
    output_dir = capture.data_path(session_id)
    prefix = 'render'
    for ts, camera_id in tqdm(session.images.key_pairs()):
        if session.rigs is None:
            pose_cam2w = session.trajectories[ts, camera_id]
        else:
            rig_id = next(iter(session.trajectories[ts]))
            rig_from_cam = session.rigs[rig_id, camera_id]
            world_from_rig = session.trajectories[ts, rig_id]
            pose_cam2w = world_from_rig * rig_from_cam

        camera = session.sensors[camera_id]
        rgb, depth_map = renderer.render_from_capture(pose_cam2w, camera)

        image_path = session.images[ts, camera_id]
        depth_path = Path(prefix, image_path).with_suffix('.depth.png')
        depths[ts, camera_id] = depth_path.as_posix()

        output_path = output_dir / depth_path
        output_path.parent.mkdir(exist_ok=True, parents=True)
        write_depth(output_path, depth_map)

        if render_images:
            im = PIL.Image.fromarray((rgb * 255).astype(np.uint8))
            im.save(output_dir / prefix / image_path)

    logger.info('Wrote the depth renderings to %s.', output_dir)
    # Hacky but safer than rewriting the whole session data
    depths.save(capture.session_path(session_id) / session.filename('depths'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     argument_default=argparse.SUPPRESS)
    parser.add_argument('--capture_path', type=Path, required=True)
    parser.add_argument('--session_id', type=str, required=True)
    parser.add_argument('--mesh_id', type=str)
    parser.add_argument('--render_images', action='store_true')
    args = parser.parse_args().__dict__
    args['capture'] = Capture.load(args.pop('capture_path'))

    run(**args)
