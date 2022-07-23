from typing import Tuple, Optional
import numpy as np

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
