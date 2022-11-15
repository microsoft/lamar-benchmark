import argparse
from pathlib import Path
import os
from tqdm import tqdm

from . import logger
from .capture import Capture
from .viz.meshlab import MeshlabProject


def run(capture: Capture, session_id: str, output_name: str,
        export_mesh: bool = False, export_poses: bool = False,
        mesh_id: str = 'mesh', overwrite: bool = True):
    if not (export_mesh or export_poses):
        raise ValueError('Specify poses or mesh or both.')

    session = capture.sessions[session_id]
    output_path = capture.viz_path() / f'{output_name}.mlp'
    if not overwrite and output_path.exists():
        logger.info('Will append to MLP file %s.', output_path.resolve())
        input_path = output_path
    else:
        input_path = None
    mlp = MeshlabProject(path=input_path)

    if export_mesh:
        assert session.proc is not None
        assert session.proc.meshes is not None
        assert mesh_id in session.proc.meshes
        mesh_path = capture.proc_path(session_id) / session.proc.meshes[mesh_id]
        label = f'{session_id}/{mesh_id}'
        mlp.add_mesh(label, os.path.relpath(mesh_path, output_path.parent))

    if export_poses:
        for ts, camera_id in tqdm(session.images.key_pairs()):
            camera = session.sensors[camera_id]
            pose = session.get_pose(ts, camera_id)
            label = f'{ts}/{camera_id}'
            mlp.add_camera(label, camera, pose)

    if input_path is None:
        logger.info('Writing MLP file to %s.', output_path.resolve())
    output_path.parent.mkdir(exist_ok=True, parents=True)
    mlp.write(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--capture_path', type=Path, required=True)
    parser.add_argument('--session_id', type=str, required=True)
    parser.add_argument('--output_name', type=str, required=True)
    parser.add_argument('--export_mesh', action='store_true')
    parser.add_argument('--export_poses', action='store_true')
    parser.add_argument('--mesh_id', type=str, default='mesh')
    parser.add_argument('--no-overwrite', dest='overwrite', action='store_false')
    parser.set_defaults(overwrite=True)
    args = parser.parse_args().__dict__
    args['capture'] = Capture.load(args.pop('capture_path'))

    run(**args)
