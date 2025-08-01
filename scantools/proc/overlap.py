import logging
from typing import Optional, List, Tuple
import itertools
import numpy as np
from tqdm import tqdm

from .rendering import Renderer, compute_rays
from ..capture import Capture, Session, Trajectories, Pose, Camera
from ..utils.geometry import project
from ..utils.io import read_mesh
from ..utils.frustum import frustum_intersections

logger = logging.getLogger(__name__)
TRACERS = {}


def vector_cos(a, b):
    """Cosine similarity ... x N x M between ... x N x D and ... x M x D arrays
    """
    dot = (a[..., None, :] @ b[..., None])[..., 0, 0]
    return dot / (np.linalg.norm(a, axis=-1)*np.linalg.norm(b, axis=-1))


def overlay(overlap: np.ndarray, image: np.ndarray):
    ov = image.copy()
    mask = overlap > 0
    overlap = overlap[mask][..., None]/2
    ov[mask] = ((1-overlap)*image[mask] + overlap*np.array([0, 255, 0])).astype(int)
    return ov


class OverlapTracer:
    def __init__(self, renderer: Renderer, stride: int = 1, num_rays: Optional[int] = None):
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

        overlap_matrix = np.full((len(keys_q), len(keys_r)), -1, dtype=np.float16)
        overlap_matrix[discard] = 0

        # cache the image poses as they might be compositions of rig poses
        T_rs = {k: session_r.get_pose(*k, poses_r) for k in keys_r}

        for i, (ts_q, id_q) in enumerate(tqdm(keys_q)):
            T_q = session_q.get_pose(ts_q, id_q, poses_q)
            cam_q = session_q.sensors[id_q]
            rays_q = compute_rays(T_q, cam_q, stride=self.get_stride(cam_q))
            intersections = self.renderer.compute_intersections(rays_q)

            for j, (ts_r, id_r) in enumerate(keys_r):
                if discard[i, j] or (ts_q, id_q) == (ts_r, id_r):
                    continue
                T_r = T_rs[ts_r, id_r]
                cam_r = session_r.sensors[id_r]
                ov = self.compute_overlap_from_rays(rays_q, *intersections, T_r, cam_q, cam_r)
                overlap_matrix[i, j] = 0.0 if ov is None else ov.mean()

        return overlap_matrix


def compute_overlaps_for_scans(capture: Capture, session_ids: List[str], combined_id: str,
                               keys: Optional[List] = None, mesh_id: str = 'mesh',
                               num_rays: int = 60) -> List[List[np.ndarray]]:
    num = len(session_ids)
    overlaps = [[None for _ in range(num)] for _ in range(num)]
    sessions = [capture.sessions[i] for i in session_ids]
    if keys is None:
        keys = [sorted(session.images.key_pairs()) for session in sessions]

    mesh = read_mesh(capture.proc_path(combined_id)
                     / capture.sessions[combined_id].proc.meshes[mesh_id])
    tracer = OverlapTracer(Renderer(mesh), num_rays=num_rays)

    for i, j in itertools.product(range(num), repeat=2):
        logger.info('Computing overlaps for sessions (%s, %s).', session_ids[i], session_ids[j])
        T_q = sessions[i].trajectories
        T_q2w = sessions[i].proc.alignment_global.get_abs_pose('pose_graph_optimized')
        if T_q2w is not None:
            T_q = T_q2w * T_q
        if i == j:
            T_r = T_q
        else:
            T_r = sessions[j].trajectories
            T_r2w = sessions[j].proc.alignment_global.get_abs_pose('pose_graph_optimized')
            if T_r2w is not None:
                T_r = T_r2w * T_r
        ov = tracer.trajectory_overlap(keys[i], sessions[i], T_q, keys[j], sessions[j], T_r)
        overlaps[i][j] = ov

    return overlaps


def compute_overlaps_for_sequence(capture: Capture, id_q: str, id_ref: str,
                                  keys_q: Optional[List] = None, keys_ref: Optional[List] = None,
                                  T_q: Optional[Trajectories] = None,
                                  num_rays: int = 60,
                                  do_caching: bool = True) -> Tuple[List[np.ndarray]]:
    session_q = capture.sessions[id_q]
    if keys_q is None:
        keys_q = sorted(session_q.images.key_pairs())
    session_ref = capture.sessions[id_ref]
    if keys_ref is None:
        keys_ref = sorted(capture.sessions[id_ref].images.key_pairs())

    logger.info('Computing overlaps for sessions (%s, %s).', id_q, id_ref)

    # Use either the provided query poses or the globally-aligned trajectory
    if T_q is None:
        T_q = session_q.proc.alignment_trajectories
        if T_q is None:
            T_q = session_q.trajectories

    # Sub-mesh selection based on reference image sequence.
    if session_ref.proc.subsessions:
        sub_mesh_id_list = [f'mesh_simplified_{sub_id.replace("/", "_")}'
                            for sub_id in session_ref.proc.subsessions]
        sub_mesh_id_list = [m if m in session_ref.proc.meshes else m.replace('_simplified', '')
                            for m in sub_mesh_id_list]
        valid_image_indices_list = [
            np.where([cam_id.startswith(sub_id) for _, cam_id in keys_ref])[0]
            for sub_id in session_ref.proc.subsessions]
        selected_keys_ref_list = [
            [(ts, cam_id) for ts, cam_id in keys_ref if cam_id.startswith(sub_id)]
            for sub_id in session_ref.proc.subsessions]
    else:
        # No subsessions so we should use a single mesh.
        sub_mesh_id_list = ['mesh_simplified']
        if sub_mesh_id_list[0] not in session_ref.proc.meshes:
            sub_mesh_id_list = ['mesh']
        valid_image_indices_list = [np.array(list(range(len(keys_ref))))]
        selected_keys_ref_list = [keys_ref]

    ov_q2r = np.zeros([len(keys_q), len(keys_ref)], dtype=np.float16)
    ov_r2q = np.zeros([len(keys_ref), len(keys_q)], dtype=np.float16)
    for sub_info in zip(sub_mesh_id_list, valid_image_indices_list, selected_keys_ref_list):
        sub_mesh_id, valid_image_indices, selected_keys_ref = sub_info
        sub_mesh_path = capture.proc_path(id_ref) / session_ref.proc.meshes[sub_mesh_id]
        if do_caching and sub_mesh_path in TRACERS:
            tracer = TRACERS[sub_mesh_path]
        else:
            tracer = OverlapTracer(Renderer(read_mesh(sub_mesh_path)), num_rays=num_rays)
            if do_caching:
                TRACERS[sub_mesh_path] = tracer

        # All reference meshes are already aligned in a common coordinate system.
        ov_q2r_ = tracer.trajectory_overlap(
            keys_q, session_q, T_q, selected_keys_ref, session_ref)
        ov_r2q_ = tracer.trajectory_overlap(
            selected_keys_ref, session_ref, None, keys_q, session_q, T_q,
            mask=(ov_q2r_.T > 0.01))

        # Update global overlap matrix.
        ov_q2r[:, valid_image_indices] = ov_q2r_
        ov_r2q[valid_image_indices, :] = ov_r2q_

    return ov_q2r, ov_r2q


def pairs_from_overlap(overlaps, num_pairs) -> List[List[int]]:
    pairs = []
    for ov_i in overlaps:
        idx = np.argpartition(-ov_i, num_pairs)[:num_pairs]  # not sorted
        idx = idx[np.argsort(-ov_i[idx])]
        idx = idx[ov_i[idx] > 0]
        pairs.append(idx.tolist())
    return pairs


def pairs_for_sequence(capture: Capture, id_q: str, id_ref: str, num_pairs: int,
                       keys_q: Optional[List] = None, keys_refs: Optional[List] = None,
                       T_q: Optional[Trajectories] = None) -> List[List[int]]:

    overlaps_q2r, overlaps_r2q = compute_overlaps_for_sequence(
        capture, id_q, id_ref, keys_q, keys_refs, T_q)
    overlaps = (overlaps_q2r + overlaps_r2q.T) / 2
    return pairs_from_overlap(overlaps, num_pairs)
