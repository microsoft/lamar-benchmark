from pathlib import Path
from typing import List, Tuple, Callable
from collections import defaultdict
import numpy as np

import pycolmap

from scantools.capture import Camera, Pose, Trajectories
from scantools.proc.alignment.image_matching import get_keypoints, get_matches

KeyType = Tuple[int, str]


def recover_matches_2d3d(query: str, ref_key_names: List[Tuple[KeyType, str]],
                         mapping, query_features: Path, match_file: Path):
    (p2d,), (noise,) = get_keypoints(query_features, [query])
    p2d_to_p3d = defaultdict(list)
    num_matches = 0

    if len(ref_key_names) == 0:
        ref_keys = ref_names = []
    else:
        ref_keys, ref_names = zip(*ref_key_names)
    all_matches = get_matches(match_file, zip([query]*len(ref_names), ref_names))

    ret = {
        'kp_q': [np.empty((0, 2))],
        'p3d': [np.empty((0, 3))],
        'indices': [np.empty((0,), int)],
        'node_ids_ref': [np.empty((0, 2), object)]
    }
    for idx, (ref_key, matches) in enumerate(zip(ref_keys, all_matches)):
        if len(matches) == 0:
            continue
        valid, p3ds, p3d_ids = mapping.get_points3D(ref_key, matches[:, 1])
        matches = matches[valid]
        num_matches += len(matches)

        p2d_q = []
        p3d = []
        indices = []
        node_ids_ref = []
        for (i, j), p3d_id, xyz in zip(matches, p3d_ids, p3ds):
            # avoid duplicate observations
            if p3d_id != -1 and p3d_id in p2d_to_p3d[i]:
                continue
            p2d_to_p3d[i].append(p3d_id)
            p2d_q.append(p2d[i])
            p3d.append(xyz)
            indices.append(idx)
            node_ids_ref.append((ref_key, j))
        if len(p2d_q) == 0:
            continue

        ret['kp_q'].append(np.array(p2d_q))
        ret['p3d'].append(np.array(p3d))
        ret['indices'].append(np.array(indices))
        ret['node_ids_ref'].append(np.array(node_ids_ref, dtype=object))
    ret = {k: np.concatenate(v, 0) for k, v in ret.items()}

    return {**ret, 'keypoint_noise': noise}


def estimate_camera_pose(query: str, camera: Camera,
                         ref_key_names: List[Tuple[KeyType, str]],
                         recover_matches: Callable,
                         pnp_error_multiplier: float,
                         return_covariance: bool) -> Pose:
    matches_2d3d = recover_matches(query, ref_key_names)
    keypoint_noise = matches_2d3d['keypoint_noise']

    ret = pycolmap.absolute_pose_estimation(
        matches_2d3d['kp_q'], matches_2d3d['p3d'],
        camera.asdict, pnp_error_multiplier * keypoint_noise,
        return_covariance=return_covariance)

    if ret['success']:
        if return_covariance:
            ret['covariance'] *= keypoint_noise ** 2
            # the covariance returned by pycolmap is on the left side,
            # which is the right side of the inverse.
            pose = Pose(*Pose(ret['qvec'], ret['tvec']).inv.qt, ret['covariance'])
        else:
            pose = Pose(ret['qvec'], ret['tvec']).inv
    else:
        pose = None

    ret = {**ret, 'matches_2d3d_list': [matches_2d3d]}
    return pose, ret


def estimate_camera_pose_rig(queries: List[str], cameras: List[Camera], T_cams2rig: List[Pose],
                             refs_key_names: List[List[Tuple[KeyType, str]]],
                             recover_matches: Callable,
                             pnp_error_multiplier: float, return_covariance: bool) -> Pose:
    matches_2d3d_list = []
    keypoint_noises = []
    for query, ref_key_names in zip(queries, refs_key_names):
        matches_2d3d = recover_matches(query, ref_key_names)
        matches_2d3d_list.append(matches_2d3d)
        keypoint_noises.append(matches_2d3d['keypoint_noise'])

    p2d_m_list = [m['kp_q'] for m in matches_2d3d_list]
    p3d_m_list = [m['p3d'] for m in matches_2d3d_list]
    camera_dicts = [camera.asdict for camera in cameras]
    rel_poses = [T.inverse() for T in T_cams2rig]
    qvecs = [p.qvec for p in rel_poses]
    tvecs = [p.t for p in rel_poses]
    keypoint_noise = np.mean(keypoint_noises)

    ret = pycolmap.rig_absolute_pose_estimation(
        p2d_m_list, p3d_m_list, camera_dicts, qvecs,
        tvecs, pnp_error_multiplier * keypoint_noise,
        return_covariance=return_covariance)

    if ret['success']:
        if return_covariance:
            ret['covariance'] *= keypoint_noise ** 2
            # the covariance returned by pycolmap is on the left side,
            # which is the right side of the inverse.
            pose = Pose(*Pose(ret['qvec'], ret['tvec']).inv.qt, ret['covariance'])
        else:
            pose = Pose(ret['qvec'], ret['tvec']).inv
    else:
        pose = None

    ret = {**ret, 'matches_2d3d_list': matches_2d3d_list}
    return pose, ret


def compute_pose_errors(query_keys: List, T_c2w: Trajectories, T_c2w_gt: Trajectories):
    err_r, err_t = [], []
    for key in query_keys:
        if key in T_c2w:
            dr, dt = (T_c2w[key].inverse() * T_c2w_gt[key]).magnitude()
        else:
            dr = np.inf
            dt = np.inf
        err_r.append(dr)
        err_t.append(dt)
    err_r = np.stack(err_r)
    err_t = np.stack(err_t)
    return err_r, err_t
