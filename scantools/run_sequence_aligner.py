import argparse
from pathlib import Path
import pickle
import os
import numpy as np
from tqdm import tqdm

from . import logger
from .capture import Capture, Proc, Pose
from .proc.alignment import Paths, save_stats, image_matching as imatch
from .proc.alignment.sequence import (
    SequenceAlignmentConf as Conf,
    align_trajectories_with_voting,
    optimize_sequence_pose_graph_gnc,
    optimize_sequence_bundle_gnc)
from .proc.alignment.localization import BatchLocalizer
from .utils.misc import add_bool_arg

from .proc.rendering import Renderer
from .utils.io import read_mesh, write_image
from .viz.alignment import plot_rendering_diff
from .viz.meshlab import MeshlabProject


conf_hololens = dict(
    localizer=dict(
        pnp_error_multiplier=1.0,  # rig
        keyframing=dict(
            max_rotation=np.inf,  # disable the rotation check
        ),
    ),
    pgo=dict(
        rel_noise_tracking=0.01,
    ),
    ba=dict(
        rel_noise_tracking=0.01,
    ),
)

conf_ios = dict(
    localizer=dict(
        pnp_error_multiplier=3.0,  # single image
    ),
    pgo=dict(
        rel_noise_tracking=0.05,
    ),
    ba=dict(
        rel_noise_tracking=0.05,
    ),
)


def run(capture: Capture, ref_id: str, query_id: str, conf: Conf,
        overwrite: bool = False, vis_mesh_id: str = 'mesh',
        visualize: bool = True, visualize_diff: bool = True, visualize_inliers: bool = False):
    session_q = capture.sessions[query_id]

    # Save the paths to the working directories
    paths = Paths(capture.registration_path(), conf.matching, query_id, ref_id)
    paths.outputs.mkdir(exist_ok=True, parents=True)

    if visualize:
        # Meshlab project with all trajectories
        mlp_path = paths.outputs / 'alignment_pnp+pgo+ba.mlp'
        mlp = MeshlabProject()
        proc = capture.sessions[ref_id].proc
        mesh_path = capture.proc_path(ref_id) / proc.meshes[vis_mesh_id]
        T_mesh2global = proc.alignment_global.get_abs_pose('pose_graph_optimized')
        if T_mesh2global is None:
            T_mesh2global = Pose.from_4x4mat(np.eye(4))
        mlp.add_mesh(f'{ref_id}/mesh', os.path.relpath(mesh_path, mlp_path.parent),
                     T=T_mesh2global.to_4x4mat())

    logger.info('Estimating camera poses via localization.')
    loc = BatchLocalizer(
        capture, conf.matching, conf.localizer, paths, visualize=visualize_inliers)
    poses_loc = loc.run(overwrite)
    matches_2d3d = loc.matcher.matches_2d3d
    if visualize:
        mlp.add_trajectory('loc', poses_loc, session_q, 'red')
        mlp.write(mlp_path)
    if len(poses_loc.key_pairs()) == 0:
        logger.warning('Could not localize any image!')
        return False

    logger.info('Running pose initialization via voting.')
    poses_tracking = session_q.trajectories
    poses_init, stats = align_trajectories_with_voting(poses_tracking, poses_loc, conf.init)
    save_stats(paths.stats('init'), stats)
    if poses_init is None:
        logger.warning('Initial pose voting failed!')
        return False
    poses_init.save(paths.outputs / 'trajectory_pgo_init.txt')

    logger.info('Running pose graph optimization.')
    poses_pgo, stats = optimize_sequence_pose_graph_gnc(
        poses_tracking, poses_loc, poses_init, conf.pgo)
    poses_pgo.save(paths.outputs / 'trajectory_pgo.txt')
    save_stats(paths.stats('pgo'), stats)
    # TODO: save the optimization robust weights for visualization
    if visualize:
        mlp.add_trajectory('init', poses_init, session_q, 'lime')
        mlp.add_trajectory('pgo', poses_pgo, session_q, 'yellow')
        mlp.write(mlp_path)

    # Second localization guided by the PGO poses
    if conf.reloc.do_rematching:
        conf_rematch = conf.matching.update(dict(
            Rt_thresh=conf.reloc.Rt_thresh,
            pairs_file=conf.reloc.pairs_file.format(conf.matching.num_pairs)))
        conf_relocalize = conf.localizer.update(dict(
            dump_name=conf.reloc.dump_name, min_num_inliers=conf.reloc.min_num_inliers))

        reloc = BatchLocalizer(
            capture, conf_rematch, conf_relocalize,
            Paths(capture.registration_path(), conf_rematch, query_id, ref_id),
            visualize=visualize_inliers)
        poses_loc2 = reloc.run(
            overwrite, from_overlap=True, query_poses=poses_pgo)
        poses_pgo, stats = optimize_sequence_pose_graph_gnc(
            poses_tracking, poses_loc2, poses_pgo, conf.pgo)
        poses_pgo.save(paths.outputs / 'trajectory_pgo2.txt')
        save_stats(paths.stats('pgo2'), stats)
        matches_2d3d = reloc.matcher.matches_2d3d
        if visualize:
            mlp.add_trajectory('loc2', poses_loc2, session_q, 'magenta')
            mlp.add_trajectory('pgo2', poses_pgo, session_q, 'orange')
            mlp.write(mlp_path)

    # Include 2D-3D point constraints into the optimization
    logger.info('Running bundle adjustment.')
    with open(capture.session_path(ref_id) / 'tracks.pkl', 'rb') as f:
        tracks_ref = pickle.load(f)
    poses_opt, stats = optimize_sequence_bundle_gnc(
        poses_tracking, poses_pgo, session_q,
        matches_2d3d, tracks_ref, conf.ba,
        multipliers=[1])
    poses_opt.save(paths.outputs / 'trajectory_ba.txt')
    save_stats(paths.stats('ba'), stats)
    if visualize:
        mlp.add_trajectory('opt', poses_opt, session_q, 'blue')
        mlp.write(mlp_path)

    if session_q.proc is None:
        session_q.proc = Proc()
    session_q.proc.alignment_trajectories = poses_opt
    session_q.proc.save(capture.proc_path(query_id))

    if visualize_diff:
        logger.info('Generating visualization diffs by rendering.')
        mesh = read_mesh(
            capture.proc_path(ref_id) / capture.sessions[ref_id].proc.meshes[vis_mesh_id])
        renderer = Renderer(mesh)
        viz_keys = sorted(poses_opt.key_pairs())
        viz_keys = imatch.subsample_list(viz_keys, min(len(viz_keys), 100))
        for ts, cam_id in tqdm(viz_keys):
            T_cam2w = poses_opt[ts, cam_id]
            camera = session_q.sensors.get(cam_id)
            if camera is None:  # it's a rig! - pick any first camera that exists at timestamp
                subcam_id, _ = next(iter(session_q.images[ts].items()))
                T_cam2rig = session_q.rigs[cam_id][subcam_id]
                camera = session_q.sensors[subcam_id]
                T_cam2w = T_cam2w * T_cam2rig
                impath = session_q.images[ts, subcam_id]
            else:
                impath = session_q.images[ts, cam_id]
            diff = plot_rendering_diff(
                renderer, capture.data_path(query_id) / impath, camera, T_cam2w)
            viz_path = capture.viz_path() / 'sequence_alignment' / query_id / impath
            viz_path.parent.mkdir(exist_ok=True, parents=True)
            write_image(viz_path, diff)

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     argument_default=argparse.SUPPRESS)
    parser.add_argument('--capture_path', type=Path, required=True)
    parser.add_argument('--query_id', type=str, required=True)
    parser.add_argument('--ref_id', type=str, required=True)
    parser.add_argument('--matching_confs', nargs=3, type=str, required=True,
                        help='Configuration names for retrieval, local features, and matching')
    parser.add_argument('--num_loc_queries', type=lambda x: int(x) if x.isdigit() else float(x))
    add_bool_arg(parser, 'visualize', default=True)
    args = parser.parse_args().__dict__
    args['capture'] = Capture.load(args.pop('capture_path'))
    args['conf'] = Conf(matching=imatch.MatchingConf(*args.pop('matching_confs')))

    run(**args)
