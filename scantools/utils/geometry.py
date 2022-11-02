from typing import Tuple, Optional
import numpy as np
from scipy.interpolate import interpn

from ..capture import Camera, Pose


def to_homogeneous(p: np.ndarray) -> np.ndarray:
    return np.pad(p, ((0, 0),)*(p.ndim-1) + ((0, 1),), constant_values=1)


def from_homogeneous(points):
    return points[..., :-1] / points[..., -1:]


def project(p3d: np.ndarray, camera: Camera, pose: Optional[Pose] = None,
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if pose is not None:
        p3d = pose.transform_points(p3d)
    z = p3d[..., -1]
    p2d = camera.world2image(from_homogeneous(p3d))
    valid = (z > 0) & camera.in_image(p2d)
    return p2d, z, valid


def backproject(p2d: np.ndarray, z: np.ndarray, camera: Camera,
                pose: Optional[Pose] = None) -> np.ndarray:
    assert p2d.shape[-1] == 2
    p2d = camera.image2world(p2d)
    p3d = to_homogeneous(p2d) * z[..., None]
    if pose is not None:
        p3d = pose.transform_points(p3d)
    return p3d


def sample_depth(p2d: np.ndarray, depth: np.ndarray,
                 fast: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    assert len(p2d.shape) == 2 and p2d.shape[1] == 2
    assert len(depth.shape) == 2
    h, w = depth.shape
    p2d = p2d - 0.5  # from COLMAP coordinates
    mask = (p2d >= 0) & (p2d <= (np.array([w, h]) - 1))
    p2d = np.where(mask, p2d, 0)
    p2d = p2d[:, ::-1]  # (x, y) to (y, x)
    if fast:  # nearest
        z = depth[tuple(np.round(p2d).astype(int).T)]
        valid = z > 0
    else:  # bilinear
        h, w = depth.shape
        grid = (np.arange(h), np.arange(w))
        z = interpn(grid, depth, p2d, method='linear')
        valid = (depth <= 0).astype(float)
        valid = interpn(grid, valid, p2d, method='linear') == 0
        z[~valid] = 0.
    valid &= mask.all(1)
    return z, valid


def sample_depth_grid(depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert len(depth.shape) == 2
    h, w = depth.shape
    p2d = np.mgrid[:h, :w].reshape(2, -1)[::-1].T
    z = depth.reshape(-1)
    valid = z > 0
    return p2d, z, valid


def vector_to_cross_product_matrix(v: np.ndarray) -> np.ndarray:
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def compute_epipolar_error(T_q2w_r: Pose, T_q2w_t: Pose, camera_r: Camera,
                           camera_t: Camera, p2d_r: np.ndarray, p2d_t: np.ndarray) -> np.ndarray:
    T_r2t = T_q2w_t.inverse() * T_q2w_r
    E = vector_to_cross_product_matrix(T_r2t.t) @ T_r2t.r.as_matrix()
    F = (
        np.linalg.inv(camera_t.K).T @ E @
        np.linalg.inv(camera_r.K))
    l2d_r2t = (F @ to_homogeneous(p2d_r).T).T
    l2d_t2r = (F.T @ to_homogeneous(p2d_t).T).T
    errors = 0.5 * (
        np.abs(np.sum(to_homogeneous(p2d_t) * l2d_r2t, axis=1)) /
        np.linalg.norm(l2d_r2t[:, : 2], axis=1) +
        np.abs(np.sum(to_homogeneous(p2d_r) * l2d_t2r, axis=1)) /
        np.linalg.norm(l2d_t2r[:, : 2], axis=1)
    )
    return errors
