import logging
from typing import Optional, List, Tuple
import numpy as np
from tqdm import tqdm

from .rendering import Renderer, compute_rays
from ..capture import Session, Trajectories, Pose, Camera
from ..utils.geometry import project
from ..utils.frustum import frustum_intersections

logger = logging.getLogger(__name__)


def vector_cos(a, b):
    """Cosine similarity ... x N x M between ... x N x D and ... x M x D arrays
    """
    dot = (a[..., None, :] @ b[..., None])[..., 0, 0]
    return dot / (np.linalg.norm(a, axis=-1)*np.linalg.norm(b, axis=-1))


class OverlapTracer:
    def __init__(self, renderer: Renderer, stride: int = 1, num_rays: Optional[float] = None):
        self.renderer = renderer
        self.stride = stride
        self.num_rays = num_rays

    def get_stride(self, cam: Camera) -> int:
        '''Sometimes it is more convenient to define the number of rays per image edge
           rather than the stride.
        '''
        if self.num_rays is None:
            return self.stride
        return max(max(cam.width, cam.height) // self.num_rays, 1)

    def compute_overlap_pair(self, T_1: Pose, T_2: Pose, cam_1: Camera, cam_2: Camera
                             ) -> np.ndarray:
        rays1 = compute_rays(T_1, cam_1, self.get_stride(cam_1))
        intersections = self.renderer.compute_intersections(rays1)
        ov_strided = self.compute_overlap_from_rays(rays1, *intersections, T_2, cam_1, cam_2)
        return ov_strided

    def upsample_overlap(self, ov_strided: np.ndarray, cam: Camera) -> np.ndarray:
        h, w = ov_strided.shape
        stride = self.get_stride(cam)
        ov = np.zeros((cam.height, cam.width))
        ov[:h*stride, :w*stride] = ov_strided.repeat(stride, 0).repeat(stride, 1)
        return ov

    def compute_overlap_from_rays(self, rays1: Tuple, p3d_1: np.ndarray, valid_1: np.ndarray,
                                  T_2: Pose, cam_1: Camera, cam_2: Camera) -> Optional[np.ndarray]:
        if np.count_nonzero(valid_1) == 0:
            return None

        _, _, valid_1_2 = project(p3d_1, cam_2, pose=T_2.inverse())
        if np.count_nonzero(valid_1_2) == 0:
            return None

        rays2_dir = p3d_1[valid_1_2] - T_2.t
        rays2 = (np.repeat(T_2.t.astype(np.float32)[None], len(rays2_dir), 0), rays2_dir)
        p3d_2, valid_2 = self.renderer.compute_intersections(rays2)
        error = p3d_2 - p3d_1[valid_1_2][valid_2]
        visible = np.einsum('nd,nd->n', error, error) < 1e-4  # 1 cm
        if np.count_nonzero(visible) == 0:
            return None

        idxs = np.where(valid_1)[0][valid_1_2][valid_2][visible]
        n1 = rays1[1][idxs]
        n2 = rays2[1][np.where(valid_2)[0][visible]]
        weight = (vector_cos(n1, n2)+1)/2

        stride = self.get_stride(cam_1)
        w, h = cam_1.width // stride, cam_1.height // stride
        ov_strided = np.zeros((h, w))
        ov_strided[(idxs//w, idxs%w)] = weight
        return ov_strided

    def trajectory_overlap(self, keys_q: List, session_q: Session,
                           poses_q: Optional[Trajectories] = None,
                           keys_r: Optional[List] = None, session_r: Optional[Session] = None,
                           poses_r: Optional[Trajectories] = None,
                           mask: Optional[Trajectories] = None) -> np.ndarray:

        intersects = frustum_intersections(keys_q, session_q, poses_q, keys_r, session_r, poses_r)
        if mask is not None:
            intersects &= mask
        discard = ~intersects

        is_self = keys_r is None
        if is_self:
            keys_r, session_r, poses_r = keys_q, session_q, poses_q

        overlap_matrix = np.full((len(keys_q), len(keys_r)), -1, float)
        overlap_matrix[discard] = 0

        # cache the image poses as they might be compositions of rig poses
        T_rs = {k: session_r.get_pose(*k, poses_r) for k in keys_r}

        for i, (ts_q, id_q) in enumerate(tqdm(keys_q)):
            T_q = session_q.get_pose(ts_q, id_q, poses_q)
            cam_q = session_q.cameras[id_q]
            rays_q = compute_rays(T_q, cam_q, stride=self.get_stride(cam_q))
            intersections = self.renderer.compute_intersections(rays_q)

            for j, (ts_r, id_r) in enumerate(keys_r):
                if discard[i, j] or (ts_q, id_q) == (ts_r, id_r):
                    continue
                T_r = T_rs[ts_r, id_r]
                cam_r = session_r.cameras[id_r]
                ov = self.compute_overlap_from_rays(rays_q, *intersections, T_r, cam_q, cam_r)
                overlap_matrix[i, j] = 0.0 if ov is None else ov.mean()

        return overlap_matrix

    def release_renderer(self):
        self.renderer.release()
