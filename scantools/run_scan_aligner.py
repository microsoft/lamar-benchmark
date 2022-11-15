import argparse
from pathlib import Path
from typing import Dict, Tuple
import os
import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import open3d as o3d

from . import logger
from .capture import Capture, Pose
from .proc.alignment import Paths, image_matching as imatch
from .proc.alignment.scan import (
    lift_points_to_3d, ransac_alignment, icp_alignment, residual_errors_3d,
    ScanAlignmentConf as Conf)
from .utils.io import read_pointcloud, write_pointcloud
from .utils.misc import add_bool_arg
from .viz.meshlab import MeshlabProject


def lift_matches_to_3d(impath2key_q: Dict[str, Tuple[int, str]],
                       impath2key_r: Dict[str, Tuple[int, str]],
                       q_id: str, r_id: str, capture: Capture,
                       conf: Conf, paths: Paths
                       ) -> Tuple[np.ndarray, np.ndarray]:

    retrieval = imatch.parse_retrieval(paths.pairs)
    _, path_keypoints_q = paths.features(q_id)
    _, path_keypoints_r = paths.features(r_id)

    all_p3d = []
    pair2idxs = {}
    num_all_matches = 0
    for q in tqdm(impath2key_q):
        refs = retrieval[q]
        matches = imatch.get_matches(paths.matches, zip([q]*len(refs), refs))

        # for each query, select the ref with largest number of matches
        num_matches = [len(m) for m in matches]
        idx = np.argmax(num_matches)
        max_matches = num_matches[idx]
        if max_matches < conf.min_num_matches_lift:
            continue
        ref = refs[idx]
        matches = matches[idx]

        kp_q = imatch.get_keypoints(path_keypoints_q, [q])[0][0][matches[:, 0]]
        kp_r = imatch.get_keypoints(path_keypoints_r, [ref])[0][0][matches[:, 1]]

        p3d_q, valid_q = lift_points_to_3d(
            kp_q, impath2key_q[q], capture.sessions[q_id], capture.data_path(q_id))
        p3d_r, valid_r = lift_points_to_3d(
            kp_r, impath2key_r[ref], capture.sessions[r_id], capture.data_path(r_id))

        valid = valid_q & valid_r
        matches = np.stack([p3d_q[valid], p3d_r[valid]], 1)
        pair2idxs[(q, ref)] = (num_all_matches, len(matches), np.where(valid)[0])
        all_p3d.append(matches)
        num_all_matches += len(matches)

    p3d_q, p3d_r = np.swapaxes(np.concatenate(all_p3d, 0), 0, 1)
    return p3d_q, p3d_r, pair2idxs


def visualize_alignment_errors(pcd_q: o3d.geometry.PointCloud,
                               pcd_r: o3d.geometry.PointCloud,
                               T_ransac: np.ndarray, T_icp: np.ndarray,
                               conf: Conf, paths: Paths):
    max_err = conf.viz_error_thresh
    logger.info('Computing alignment statistics.')
    (errs_ransac, errs_icp), (mask_ransac, mask_icp), (_, pcd_q_aligned) = residual_errors_3d(
        pcd_q, pcd_r, [T_icp, T_ransac], max_err)
    logger.info('Alignment RANSAC -> ICP: %.2f/%.2f/%.2f -> %.2f/%.2f/%.2f mean/med/q3 errors [cm]',
                *[100*x for e in (errs_ransac[mask_ransac], errs_icp[mask_icp])
                  for x in (e.mean(), np.median(e), np.percentile(e, 75))])

    hist_args = dict(bins=5000, alpha=0.5)
    plt.hist(errs_ransac[mask_ransac], label='RANSAC', **hist_args)
    plt.hist(errs_icp[mask_icp], label='ICP', **hist_args)
    plt.xlim([1e-3, max_err])
    plt.xscale('log')
    plt.legend()
    plt.savefig(paths.outputs / conf.plot_residuals_file)
    plt.close()

    errs_icp /= max_err
    colors = np.asarray(pcd_q_aligned.colors)
    chunk = 10000000
    for i in range(math.ceil(len(colors)/chunk)):
        sl = slice(i*chunk, (i+1)*chunk)
        colors[sl] = matplotlib.cm.bwr(errs_icp[sl])[:, :3]  # can cause OOM
    del errs_ransac, mask_ransac, errs_icp, mask_icp
    pcd_q_aligned.normals = o3d.utility.Vector3dVector()
    write_pointcloud(paths.outputs / conf.viz_residuals_file, pcd_q_aligned)


def run(capture: Capture, ref_id: str, query_id: str, conf: Conf,
        pcd_id: str = 'point_cloud_final', visualize: bool = True):

    session_r = capture.sessions[ref_id]
    session_q = capture.sessions[query_id]
    image_root = capture.sessions_path()  # base dir of hloc image naming

    # Save the paths to the working directories
    paths = Paths(capture.registration_path(), conf.matching, query_id, ref_id)
    paths.outputs.mkdir(exist_ok=True, parents=True)

    # Select some or all images within the query and reference sessions
    impath2key_q = imatch.list_images_for_matching(
        session_q, capture.data_path(query_id).relative_to(image_root), conf.matching,
        keyframing=conf.keyframing)
    impath2key_r = imatch.list_images_for_matching(
        session_r, capture.data_path(ref_id).relative_to(image_root), conf.matching)

    # Feature extraction, retrieval, and matching
    imatch.pairwise_matching(
        impath2key_q.keys(), impath2key_r.keys(), image_root,
        paths.session(query_id), paths.session(ref_id), paths.outputs, conf.matching)

    # Turn the 2D-2D matches into 3D-3D using the rendered depth
    logger.info('Computing 3D-3D matches.')
    p3d_q, p3d_r, pair2idxs = lift_matches_to_3d(
        impath2key_q, impath2key_r, query_id, ref_id, capture, conf, paths)

    logger.info('Running RANSAC scan alignment.')
    T_ransac, inliers, inlier_ratio = ransac_alignment(p3d_q, p3d_r, conf.ransac)
    pair2inliers = {p: np.stack([inliers[s:s+e], v], -1) for p, (s, e, v) in pair2idxs.items()}
    imatch.write_inliers(paths.outputs / conf.ransac.inlier_file, pair2inliers)

    pcd_q = read_pointcloud(capture.data_path(query_id) / session_q.pointclouds[0, pcd_id])
    pcd_r = read_pointcloud(capture.data_path(ref_id) / session_r.pointclouds[0, pcd_id])
    logger.info('Running ICP refinement.')
    T_icp, icp_info = icp_alignment(pcd_q, pcd_r, T_ransac, conf.icp)

    session_q.proc.alignment_global['ransac', ref_id] = (Pose.from_4x4mat(T_ransac),
                                                         [inlier_ratio])
    session_q.proc.alignment_global['icp', ref_id] = (Pose.from_4x4mat(T_icp),
                                                      icp_info.reshape(-1).tolist())
    session_q.proc.save(capture.proc_path(query_id))

    if visualize:
        mlp_path = paths.outputs / 'alignment_icp.mlp'
        mlp = MeshlabProject()
        mesh_path = capture.proc_path(ref_id) / session_r.proc.meshes['mesh']
        mlp.add_mesh(f'{ref_id}/mesh', os.path.relpath(mesh_path, mlp_path.parent))
        mesh_path = capture.proc_path(query_id) / session_q.proc.meshes['mesh']
        mlp.add_mesh(f'{query_id}/mesh', os.path.relpath(mesh_path, mlp_path.parent), T=T_icp)
        mlp.write(mlp_path)
        visualize_alignment_errors(pcd_q, pcd_r, T_ransac, T_icp, conf, paths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     argument_default=argparse.SUPPRESS)
    parser.add_argument('--capture_path', type=Path, required=True)
    parser.add_argument('--query_id', type=str, required=True)
    parser.add_argument('--ref_id', type=str, required=True)
    parser.add_argument('--matching_confs', nargs=3, type=str, required=True,
                        help='Confguration names for retrieval, local features, and matching')
    add_bool_arg(parser, 'visualize', default=True)
    args = parser.parse_args().__dict__
    args['capture'] = Capture.load(args.pop('capture_path'))
    args['conf'] = Conf(matching=imatch.MatchingConf(*args.pop('matching_confs')))

    run(**args)
