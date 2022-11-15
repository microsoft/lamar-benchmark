import argparse
from pathlib import Path
import inspect

from . import logger
from .capture import Capture, Proc
from .proc.meshing import mesh_from_pointcloud, simplify_mesh
from .utils.io import read_pointcloud, write_mesh


def run(capture: Capture, session_id: str,
        pointcloud_id: str = 'point_cloud_final', mesh_id: str = 'mesh',
        add_simplified: bool = True, **kwargs):

    session = capture.sessions[session_id]
    if session.proc:
        assert mesh_id not in session.proc.meshes
    else:
        session.proc = Proc()

    pcd_filename = session.pointclouds[0, pointcloud_id]
    pcd_path = capture.data_path(session_id) / pcd_filename
    pcd = read_pointcloud(pcd_path)

    # We run mesh simplification mostly for visualization purposes.
    # For large scenes, we cannot run the simplification on the entire mesh at once.
    # Instead, we run it inside the reconstruction, which allows for block-simplification
    # after the advancing front surface reconstruction.
    if add_simplified:
        kwargs = {'simplify_factor': 5, 'simplify_error': 1e-8, **kwargs}
    logger.info('Running meshing with args: %s', kwargs)
    mesh, simplified, _ = mesh_from_pointcloud(pcd, **kwargs)

    mesh_path = Path(Proc.meshes_dirname, f'{mesh_id}.ply')
    output_path = capture.proc_path(session_id) / mesh_path
    output_path.parent.mkdir(exist_ok=True, parents=True)
    write_mesh(output_path, mesh)
    session.proc.meshes[mesh_id] = mesh_path
    # We don't need to save the Proc object as it automatically discovers the mesh files

    if add_simplified:
        simpl_id = mesh_id + '_simplified'
        simpl_path = Path(Proc.meshes_dirname, f'{simpl_id}.ply')
        write_mesh(capture.proc_path(session_id) / simpl_path, simplified)
        session.proc.meshes[simpl_id] = simpl_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     argument_default=argparse.SUPPRESS)
    parser.add_argument('--capture_path', type=Path, required=True)
    parser.add_argument('--session_id', type=str, required=True)
    parser.add_argument('--pointcloud_id', type=str)
    parser.add_argument('--mesh_id', type=str)
    fn_params = inspect.signature(mesh_from_pointcloud).parameters
    fn_params = {n: p.default for n, p in fn_params.items() if p.default != inspect.Parameter.empty}
    parser.add_argument('--params', nargs='*',
                        help=f'Optional function parameters in {fn_params}. Example: psr_depth=13')
    args = parser.parse_args().__dict__
    args['capture'] = Capture.load(args.pop('capture_path'))

    # Parse the provided parameters
    for param in args.pop('params'):
        name, val = param.split('=')
        assert name in fn_params.keys()
        args[name] = eval(val)  # pylint: disable=eval-used

    run(**args)
