import numpy as np
import pyceres


def add_pose_graph_factors_to_sequence(problem, loss, poses_relative, poses_opt, relative_noise):
    residuals = []
    keys = sorted(poses_opt)
    for (ts1, cam_id1), (ts2, cam_id2) in zip(keys[:-1], keys[1:]):
        qt1 = poses_opt[ts1, cam_id1]
        qt2 = poses_opt[ts2, cam_id2]
        T_1tow_track = poses_relative[ts1, cam_id1]
        T_2tow_track = poses_relative[ts2, cam_id2]
        T_2to1_track = T_1tow_track.inverse() * T_2tow_track
        sigma_pose = np.abs(np.r_[T_2to1_track.r.as_rotvec(), T_2to1_track.t]) * relative_noise
        sigma_pose = sigma_pose.clip(min=0.001)
        cov_tracking = np.diag(sigma_pose)**2
        factor = pyceres.factors.PoseGraphRelativeCost(*T_2to1_track.inverse().qt, cov_tracking)
        residuals.append(problem.add_residual_block(factor, loss, [*qt1, *qt2]))
        problem.set_parameterization(qt1[0], pyceres.QuaternionParameterization())
        problem.set_parameterization(qt2[0], pyceres.QuaternionParameterization())
    return residuals
