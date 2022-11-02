import logging
from typing import Tuple, Optional, List
from pathlib import Path
import dataclasses
import numpy as np
import open3d as o3d

from .image_matching import MatchingConf, KeyFramingConf
from ...capture import Session
from ...utils.geometry import sample_depth, backproject
from ...utils.io import read_depth
from ...utils.configuration import BaseConf

o3dreg = o3d.pipelines.registration
PointCloud = o3d.geometry.PointCloud
logger = logging.getLogger(__name__)


class RANSACConf(BaseConf):
    distance_thresh: float = 5e-2  # inlier threshold [meters]
    max_iteration: int = 1000000
    inlier_file = None  # where to store the indices of inlier matches

    def __post_init__(self):
        self.inlier_file = f'ransac_inliers_{self.distance_thresh*100:.1f}cm.h5'

class ICPConf(BaseConf):
    distance_thresh: float = 5e-2  # max correspondence distance [meters]
    robust_thresh: float = 5e-2  # quadratic range of cost function [meters]
    max_iteration: int = 50

class ScanAlignmentConf(BaseConf):
    matching: MatchingConf

    ransac: RANSACConf = dataclasses.field(default_factory=RANSACConf)
    icp: ICPConf = dataclasses.field(default_factory=ICPConf)
    keyframing: KeyFramingConf = None  # subsample the query images
    min_num_matches_lift: int = 10  # discard pairs with too few matches

    viz_error_thresh: float = 5e-2  # maximum error of the residuals [meters]
    viz_residuals_file = None  # name of the PLY file of residuals
    plot_residuals_file = None  # name of the histogram file of residuals

    def __post_init__(self):
        self.viz_residuals_file = f'align_residuals_{self.viz_error_thresh*100:.1f}cm.ply'
        self.plot_residuals_file = f'align_residuals_{self.viz_error_thresh*100:.1f}cm.png'


def lift_points_to_3d(p2d: np.ndarray, key: Tuple[int, str], session: Session, data_path: Path,
                      align: bool = False, depth_prefix: Optional[str] = 'render',
                      ) -> Tuple[np.ndarray, np.ndarray]:
    pose = session.trajectories[key]
    ts, sensor_id = key
    camera = session.sensors[sensor_id]
    if depth_prefix is not None:
        sensor_id = depth_prefix + '/' + sensor_id
    depth = read_depth(data_path / session.depths[ts, sensor_id])

    if align:
        T_session2w = session.proc.alignment_global.get_abs_pose('pose_graph_optimized')
        if T_session2w is not None:
            pose = T_session2w * pose

    z, valid = sample_depth(p2d, depth, fast=False)
    p3d = backproject(p2d, z, camera, pose)
    return p3d, valid


def ransac_alignment(p3d_q: np.ndarray, p3d_r: np.ndarray, conf: RANSACConf) -> np.ndarray:
    if p3d_q.shape != p3d_r.shape:
        raise ValueError('Lists of query and reference points must be matching.')
    op3d_q = PointCloud(o3d.utility.Vector3dVector(p3d_q))
    op3d_r = PointCloud(o3d.utility.Vector3dVector(p3d_r))
    o3d_matches = o3d.utility.Vector2iVector(np.arange(len(p3d_q))[:, None].repeat(2, 1))

    # sources (q) to target (r)
    check = [o3dreg.CorrespondenceCheckerBasedOnDistance(conf.distance_thresh)]
    crit = o3dreg.RANSACConvergenceCriteria(
        max_iteration=conf.max_iteration, confidence=0.99999999)
    result = o3dreg.registration_ransac_based_on_correspondence(
        op3d_q, op3d_r, o3d_matches, conf.distance_thresh,
        o3dreg.TransformationEstimationPointToPoint(False),
        checkers=check, criteria=crit,
    )
    n_matches = len(p3d_q)
    n_inliers = len(result.correspondence_set)
    inlier_ratio = n_inliers/n_matches
    logger.info('RANSAC alignment: %d/%d inliers (ratio=%.2f%%)',
                n_inliers, n_matches, inlier_ratio*100)

    inliers = np.full(len(p3d_q), False, bool)
    inliers[np.asarray(result.correspondence_set)[:, 0]] = True
    return result.transformation.copy(), inliers, inlier_ratio


def icp_alignment(pcd_q: PointCloud, pcd_r: PointCloud, T_init: np.ndarray,
                  conf: ICPConf) -> np.ndarray:
    # sources (q) to target (r)
    crit = o3dreg.ICPConvergenceCriteria(max_iteration=conf.max_iteration)
    loss = o3dreg.HuberLoss(k=conf.robust_thresh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        result_icp = o3dreg.registration_icp(
            pcd_q, pcd_r, conf.distance_thresh, T_init, criteria=crit,
            estimation_method=o3dreg.TransformationEstimationPointToPlane(loss))

    dt = np.linalg.norm(result_icp.transformation[:3, 3] - T_init[:3, 3])
    logger.info('ICP refinement: update by %.2f cm.', dt*100)

    information = o3dreg.get_information_matrix_from_point_clouds(
        pcd_q, pcd_r, conf.distance_thresh, result_icp.transformation)

    return result_icp.transformation.copy(), information


def residual_errors_3d(pcd_q: PointCloud, pcd_r: PointCloud, Ts_q2r: List[np.ndarray], max_error: float
                       ) -> Tuple[np.ndarray, np.ndarray, PointCloud]:
    device = o3d.core.Device('cpu:0')
    dtype = o3d.core.Dtype.Float32
    pts_r = o3d.core.Tensor(np.asarray(pcd_r.points), dtype=dtype, device=device)
    nns = o3d.core.nns.NearestNeighborSearch(pts_r)
    nns.hybrid_index()

    errs = []
    masks = []
    for T in Ts_q2r:
        pcd_q_aligned = PointCloud(pcd_q)  # copy
        pcd_q_aligned.transform(T)
        pts_q = o3d.core.Tensor(np.asarray(pcd_q_aligned.points), dtype=dtype, device=device)
        indices, squared_distances, *_ = nns.hybrid_search(pts_q, max_error**2, 1)
        distances = squared_distances.sqrt()[:, 0].numpy()
        mask = indices[:, 0].numpy() == -1
        distances[mask] = max_error
        errs.append(distances)
        masks.append(~mask)
        del pcd_q_aligned, indices, squared_distances
    pcds_q_aligned = []
    for T in Ts_q2r:
        pcd_q_aligned = PointCloud(pcd_q)  # copy
        pcd_q_aligned.transform(T)
        pcds_q_aligned.append(pcd_q_aligned)
    return errs, masks, pcds_q_aligned
