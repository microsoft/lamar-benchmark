from typing import Optional, List, Dict, Tuple
import numpy as np
from tqdm.contrib.concurrent import thread_map

from ..capture import Capture, Session, Trajectories, KeyType


def plane_check(plane_points, check_points, eps=1e-6):
    # Input:
    #   plane_points is N x 3 x 3 in format (O, X, Y)
    #   check_points is M x P x 3
    # Output:
    #   all_on_positive_side is N x M with True if all check points are
    #   on the positive side of the plane
    # Compute plane normals.
    normals = np.cross(plane_points[:, 1] - plane_points[:, 0],
                       plane_points[:, 2] - plane_points[:, 0])
    normals /= np.linalg.norm(normals, axis=-1)[:, np.newaxis]  # N x 3
    # Translate check points.
    check_points = check_points[np.newaxis, :, :, :] - plane_points[:, 0][:, np.newaxis, np.newaxis]
    # Compute dot products with all check points.
    dot_products = np.einsum('nd,nmpd->nmp', normals, check_points)
    # Positive check.
    all_on_positive_side = np.all(dot_products > eps, axis=-1)  # N x M
    return all_on_positive_side


def _pyramid_non_intersection_partial_check(pyramid_points1, pyramid_points2):
    # This is a helper function.
    # Input:
    #   pyramid_points1 is N x 5 x 3 in format (O, top_left, bottom_left, bottom_right, top_right)
    #   pyramid_points2 is M x 5 x 3 in same format as above
    # Output:
    #   does_not_intersect is N x M with True if the associated pyramids do not intersect
    # Need to make sure that plane normals point towards outside.
    # Plane 1 - O, top_left, bottom_left
    all_on_positive_side1 = plane_check(
        np.stack([pyramid_points1[:, 0], pyramid_points1[:, 1], pyramid_points1[:, 2]], axis=1),
        pyramid_points2
    )
    # Plane 2 - O, bottom_left, bottom_right
    all_on_positive_side2 = plane_check(
        np.stack([pyramid_points1[:, 0], pyramid_points1[:, 2], pyramid_points1[:, 3]], axis=1),
        pyramid_points2
    )
    # Plane 3 - O, bottom_right, top_right
    all_on_positive_side3 = plane_check(
        np.stack([pyramid_points1[:, 0], pyramid_points1[:, 3], pyramid_points1[:, 4]], axis=1),
        pyramid_points2
    )
    # Plane 4 - O, top_right, top_left
    all_on_positive_side4 = plane_check(
        np.stack([pyramid_points1[:, 0], pyramid_points1[:, 4], pyramid_points1[:, 1]], axis=1),
        pyramid_points2
    )
    # Plane 5 - top_left, top_right, bottom_left
    all_on_positive_side5 = plane_check(
        np.stack([pyramid_points1[:, 1], pyramid_points1[:, 4], pyramid_points1[:, 2]], axis=1),
        pyramid_points2
    )
    # If at least one plane passes the plane check then the pyramids don't intersect.
    does_not_intersect = (all_on_positive_side1 | all_on_positive_side2 | all_on_positive_side3
                          | all_on_positive_side4 | all_on_positive_side5)
    return does_not_intersect


def pyramid_intersection_check(pyramid_points1, pyramid_points2, batch_size=5_000, num_threads=8):
    # Input:
    #   pyramid_points1 is N x 5 x 3 in format (O, top_left, bottom_left, bottom_right, top_right)
    #   pyramid_points2 is M x 5 x 3 in same format as above
    # Output:
    #   intersects is N x M with True if the associated pyramids intersect
    N = pyramid_points1.shape[0]
    M = pyramid_points2.shape[0]
    params = []
    for batch_idx_N in range(int(np.ceil(N / batch_size))):
        start_idx_N = batch_idx_N * batch_size
        end_idx_N = min(start_idx_N + batch_size, N)
        for batch_idx_M in range(int(np.ceil(M / batch_size))):
            start_idx_M = batch_idx_M * batch_size
            end_idx_M = min(start_idx_M + batch_size, M)
            params.append([start_idx_N, end_idx_N, start_idx_M, end_idx_M])
        assert end_idx_M == M
    assert end_idx_N == N

    intersects = np.zeros((N, M), dtype=bool)
    def _worker_fn(param):
        start_idx_N, end_idx_N, start_idx_M, end_idx_M = param
        does_not_intersect1 = _pyramid_non_intersection_partial_check(
            pyramid_points1[start_idx_N : end_idx_N],
            pyramid_points2[start_idx_M : end_idx_M])
        does_not_intersect2 = _pyramid_non_intersection_partial_check(
            pyramid_points2[start_idx_M : end_idx_M],
            pyramid_points1[start_idx_N : end_idx_N])
        intersects[start_idx_N : end_idx_N, start_idx_M : end_idx_M] = ~(does_not_intersect1 | does_not_intersect2.T)
    if len(params) < 4:
        for p in params:
            _worker_fn(p)
    else:
        thread_map(_worker_fn, params, max_workers=num_threads)
    return intersects


def pyramid_from_camera(rot_mat: np.ndarray, tvec: np.ndarray, width: int, height: int,
                        fx: float, fy: float, cx: float, cy: float, max_depth: float):
    # rot_mat, tvec are cam-to-world
    O = np.array([0, 0, 0]).astype(np.float32)
    top_left = np.array([(0 - cx) / fx, (0 - cy) / fy, 1.0]) * max_depth
    bottom_left = np.array([(0 - cx) / fx, (height - cy) / fy, 1.0]) * max_depth
    bottom_right = np.array([(width - cx) / fx, (height - cy) / fy, 1.0]) * max_depth
    top_right = np.array([(width - cx) / fx, (0 - cy) / fy, 1.0]) * max_depth
    vertices = [O, top_left, bottom_left, bottom_right, top_right]
    return np.vstack(list(map(lambda x: rot_mat @ x + tvec, vertices)))


def pyramids_from_trajectory(keys: List, session: Session, poses: Optional[Trajectories] = None,
                             max_depth=20.) -> np.ndarray:
    pyramids = []
    for ts, camera_id in keys:
        pose = session.get_pose(ts, camera_id, poses)
        camera = session.sensors[camera_id]
        pyramid = pyramid_from_camera(
            pose.R, pose.t, camera.width, camera.height,
            *camera.projection_params, max_depth=max_depth)
        pyramids.append(pyramid)
    return np.asarray(pyramids, np.float32)


def frustum_intersections(keys_q: List, session_q: Session, T_q: Optional[Trajectories] = None,
                          keys_r: Optional[List] = None, session_r: Optional[Session] = None,
                          poses_r: Optional[Trajectories] = None, max_depth: float = 20.
                          ) -> np.ndarray:
    frustums_q = pyramids_from_trajectory(keys_q, session_q, poses=T_q, max_depth=max_depth)
    if keys_r is None:
        frustums_r = frustums_q
    else:
        frustums_r = pyramids_from_trajectory(keys_r, session_r, poses=poses_r, max_depth=max_depth)
    return pyramid_intersection_check(frustums_q, frustums_r)


def frustum_intersection_multisessions(capture: Capture,
                                       keys: List[Tuple[str, KeyType]],
                                       session2trajectory: Optional[Dict[str, Trajectories]] = None,
                                       max_depth: float = 20.0,
                                       **kwargs) -> np.ndarray:
    frustums = []
    for sid, (ts, cam_id) in keys:
        session = capture.sessions[sid]
        if session2trajectory is None:
            traj = session.proc.alignment_trajectories or session.trajectories
        else:
            traj = session2trajectory[sid]
        pose = session.get_pose(ts, cam_id, traj)
        camera = session.sensors[cam_id]
        pyramid = pyramid_from_camera(
            pose.R, pose.t, camera.width, camera.height, *camera.projection_params, max_depth)
        frustums.append(pyramid.astype(np.float32))
    frustums = np.stack(frustums)
    return pyramid_intersection_check(frustums, frustums, **kwargs)
