import logging
from typing import Tuple
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import pyceres

from .refinement import add_pose_graph_factors_to_sequence
from ...capture import Pose, Trajectories
from ...utils.configuration import BaseConf

logger = logging.getLogger(__name__)


class InitializerConf(BaseConf):
    distance_thresh: float = 2.  # inlier threshold in translation [meters]
    angle_thresh: float = 20.  # inlier threshold in rotation [degrees]
    min_num_inliers: int = 4

class PGOConf(BaseConf):
    rel_noise_tracking: float = 0.01  # noise of the tracking
    cost_tracking: Tuple = ('Null',)
    cost_loc: Tuple = ('Arctan', 100.)
    num_threads: int = -1

class BAConf(BaseConf):
    rel_noise_tracking: float = 0.01  # noise of the tracking
    cost_tracking: Tuple = ('Null',)
    noise_point3d: float = 0.02 # estimate of mesh accuracy [m]
    noise_bundle_multiplier: float = 1  # noise of the keypoint detection
    cost_bundle: Tuple = ('Huber', 2.5)
    reproj_filter_lambda: float = 5
    num_threads: int = -1


def align_trajectories_with_voting(traj_query, traj_ref, conf: InitializerConf):
    keys = sorted(set(traj_query.key_pairs()) & set(traj_ref.key_pairs()))
    if len(keys) == 0:
        return False, None
    T_query = [traj_query[k] for k in keys]
    T_ref = [traj_ref[k] for k in keys]

    nums = []
    best_data = (np.inf, None, None, None, None)
    best_num = 0
    for idx in tqdm(range(len(keys)), disable=(logger.level > logging.INFO)):
        T_q2r = T_ref[idx] * T_query[idx].inverse()
        diffs = [T_ref[i].inverse() * T_q2r * T_query[i] for i in range(len(keys))]
        dt = np.stack([np.linalg.norm(diff.t) for diff in diffs])
        dR = np.stack([diff.r.magnitude() for diff in diffs])

        inliers = (dt < conf.distance_thresh) & (dR < np.deg2rad(conf.angle_thresh))
        num = np.count_nonzero(inliers)
        err_t = np.median(dt[inliers])
        err_rot = np.rad2deg(np.median(dR[inliers]))
        score = err_t + err_rot
        nums.append(num)

        if num > best_num or (num == best_num and score < best_data[0]):
            best_num = num
            best_data = (score, err_t, err_rot, inliers, T_q2r)

    _, err_t, err_rot, inliers, T_q2r = best_data
    logger.info('Voting trajectory alignment: dR=%.2fdeg dt=%.3fm inliers=%d/%d (%.1f%%)',
                err_rot, err_t, inliers.sum(), len(inliers), 100*inliers.mean())
    success = len(inliers) >= conf.min_num_inliers
    return success, T_q2r


def get_loss(name: str, *params) -> pyceres.LossFunction:
    return pyceres.LossFunction({
        "name": name.lower().replace('null', 'trivial'),
        "params": params,
    })


def print_pose_stats(prefix: str, T_1tow: Trajectories, T_2tow: Trajectories):
    keys = set(T_1tow.key_pairs()) & set(T_2tow.key_pairs())
    dr, dt = tuple(np.array([(T_1tow[k].inverse() * T_2tow[k]).magnitude() for k in keys]).T)
    dt *= 100  # m to cm
    logger.info('%s: mean/med/q1/q9 dR=%.1f/%.1f/%.1f/%.1f deg, dt=%.1f/%.1f/%.1f/%.1f cm',
                prefix,
                dr.mean(), np.median(dr), np.percentile(dr, 10), np.percentile(dr, 90),
                dt.mean(), np.median(dt), np.percentile(dt, 10), np.percentile(dt, 90))


def optimize_sequence_pose_graph_gnc(poses_tracking, poses_loc, poses_init, conf: PGOConf):
    '''Apply Graduated Non-Convexity (GNC) to the robust cost of the localization prior.
       Loose implementation of the paper:
       "Graduated Non-Convexity for Robust Spatial Perception: From Non-Minimal Solvers
       to Global Outlier Rejection", Yang, Antonante, Tzoumas, Carlone, ICRA/RAL, 2020.
       https://arxiv.org/pdf/1909.08605.pdf
    '''
    cost_loc, mu_target = conf.cost_loc
    multiplier = 1.4
    mu = mu_target * 10 * multiplier
    poses_opt = poses_init
    while mu > mu_target:
        mu = max(mu_target, mu / multiplier)
        poses_opt = optimize_sequence_pose_graph(
            poses_tracking, poses_loc, poses_opt, conf.update(dict(cost_loc=(cost_loc, mu))))
    return poses_opt


def optimize_pgo_with_init(poses_tracking: Trajectories, poses_loc: Trajectories,
                           conf_init: InitializerConf, conf_pgo: PGOConf):
    # Initial alignment (assuming that the tracking is rigid)
    logger.info('Running pose initialization via voting.')
    success, T_q2r = align_trajectories_with_voting(poses_tracking, poses_loc, conf_init)
    if not success:
        return None, None

    poses_init = Trajectories()
    for k in poses_tracking.key_pairs():
        poses_init[k] = T_q2r * poses_tracking[k]

    # Pose graph between the localization prior and the sequential relative tracking
    logger.info('Running pose graph optimization.')
    poses_pgo = optimize_sequence_pose_graph_gnc(poses_tracking, poses_loc, poses_init, conf_pgo)
    return poses_init, poses_pgo


def optimize_sequence_pose_graph(poses_tracking, poses_loc, poses_init, conf: PGOConf):
    loss_loc = get_loss(*conf.cost_loc)
    loss_tracking = get_loss(*conf.cost_tracking)

    # pyceres will optimize these variables in-place
    qt_opt = {k: deepcopy(poses_init[k].inverse().qt) for k in poses_init.key_pairs()}

    problem = pyceres.Problem()
    for k in poses_loc.key_pairs():
        T_w_i = poses_loc[k]
        # PoseGraphAbsoluteCost expects the measurement as T_i_w but the covariance in frame i
        factor = pyceres.factors.PoseGraphAbsoluteCost(*T_w_i.inv.qt, T_w_i.covar)
        problem.add_residual_block(factor, loss_loc, [*qt_opt[k]])
    add_pose_graph_factors_to_sequence(
        problem, loss_tracking, poses_tracking, qt_opt, conf.rel_noise_tracking)

    for qt in qt_opt.values():
        problem.set_parameterization(qt[0], pyceres.QuaternionParameterization())

    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
    options.max_num_iterations = 100
    options.minimizer_progress_to_stdout = False
    options.num_threads = conf.num_threads
    summary = pyceres.SolverSummary()
    pyceres.solve(options, problem, summary)
    logger.info('%s', summary.BriefReport())

    poses_opt = Trajectories()
    for k, qt in qt_opt.items():
        poses_opt[k] = Pose(*qt).inverse()
    print_pose_stats('Pose graph optimization', poses_init, poses_opt)
    return poses_opt


def print_reprojection_stats(prefix, reproj_errors, std):
    logger.info('%s reproj. stats: mean/med/q1/q9 %.1f/%.1f/%.1f/%.1f px.',
                prefix,
                np.mean(reproj_errors),
                np.median(reproj_errors),
                np.percentile(reproj_errors, 10),
                np.percentile(reproj_errors, 90))
    logger.info('%s reproj. stats: 1std/2std/3std/4std/5std %.2f/%.2f/%.2f/%.2f/%.2f%%.',
                prefix,
                np.mean(reproj_errors <= std) * 100,
                np.mean(reproj_errors <= 2 * std) * 100,
                np.mean(reproj_errors <= 3 * std) * 100,
                np.mean(reproj_errors <= 4 * std) * 100,
                np.mean(reproj_errors <= 5 * std) * 100)


def print_tracking_stats(prefix, poses_tracking, poses):
    keys = list(sorted(poses_tracking.key_pairs()))
    rel_dr = []
    rel_dt = []
    for key1, key2 in zip(keys[: -1], keys[1 :]):
        pose_2to1_tracking = poses_tracking[key1].inverse() * poses_tracking[key2]
        norm_r_tracking, norm_t_tracking = pose_2to1_tracking.magnitude()
        pose_2to1 = poses[key1].inverse() * poses[key2]
        dr, dt = (pose_2to1_tracking.inverse() * pose_2to1).magnitude()
        rel_dr.append(dr / norm_r_tracking)
        rel_dt.append(dt / norm_t_tracking)
    rel_dr = np.array(rel_dr) * 100
    rel_dt = np.array(rel_dt) * 100
    logger.info('%s tracking stats: mean/med/q1/q9 rel_dR=%.1f/%.1f/%.1f/%.1f%%, '
                'rel_dt=%.1f/%.1f/%.1f/%.1f%%', prefix,
                np.mean(rel_dr), np.median(rel_dr),
                np.percentile(rel_dr, 10), np.percentile(rel_dr, 90),
                np.mean(rel_dt), np.median(rel_dt),
                np.percentile(rel_dt, 10), np.percentile(rel_dt, 90))
    logger.info('%s tracking stats: 1%%/2%%/3%%/4%%/5%% %.2f/%.2f/%.2f/%.2f/%.2f%%.',
                prefix,
                np.mean((rel_dr <= 1) & (rel_dt <= 1)) * 100,
                np.mean((rel_dr <= 2) & (rel_dt <= 2)) * 100,
                np.mean((rel_dr <= 3) & (rel_dt <= 3)) * 100,
                np.mean((rel_dr <= 4) & (rel_dt <= 4)) * 100,
                np.mean((rel_dr <= 5) & (rel_dt <= 5)) * 100)


def optimize_sequence_bundle_gnc(poses_tracking, poses_init, session_q,
                                 matches_2d3d, tracks_ref, conf: BAConf,
                                 multipliers=(4.0, 2.0, 1.0)):
    robust_loss = conf.cost_bundle[0]
    robust_threshold = conf.cost_bundle[1]
    poses_opt = poses_init
    for multiplier in multipliers:
        current_conf = conf.update(dict(cost_bundle=(robust_loss, multiplier * robust_threshold)))
        logger.info('Running BA with %s.', str(current_conf))
        poses_opt = optimize_sequence_bundle(
            poses_tracking, poses_opt, session_q, matches_2d3d, tracks_ref, current_conf)
    return poses_opt


def optimize_sequence_bundle(poses_tracking, poses_init, session_q,
                             matches_2d3d, tracks_ref, conf: BAConf,
                             use_reference_tracks=True):
    loss_bundle = get_loss(*conf.cost_bundle)
    loss_tracking = get_loss(*conf.cost_tracking)
    if conf.noise_point3d is None:
        noise_point3d = None
    else:
        # Convert from uniform noise over cube to Gaussian distribution with same std.
        noise_point3d = np.eye(3) * conf.noise_point3d / np.sqrt(3)


    keys = sorted(poses_init.key_pairs())
    data_bundle = []  # list all the cameras with their transforms
    for ts, id_ in keys:
        if id_ in session_q.sensors:
            data_bundle.append((ts, id_, id_, None))
        else:  # it's a rig!
            for cam_id, T_cam2rig in session_q.rigs[id_].items():
                data_bundle.append((ts, id_, cam_id, T_cam2rig))

    # parameters to be optimized in-place by pyceres
    qt_opt = {k: deepcopy(poses_init[k].inverse().qt) for k in keys}
    camera_params = {k: np.array(cam.params) for k, cam in session_q.cameras.items()}
    p3d_opt = {}
    qt_init = deepcopy(qt_opt)
    problem = pyceres.Problem()

    n_obs = 0
    n_points_tracks_ref = 0
    n_filtered = 0
    n_filtered_tracks_ref = 0
    avg_noise_bundle = []
    for ts, rig_id, cam_id, T_cam2rig in tqdm(data_bundle, disable=(logger.level > logging.INFO)):
        matches = matches_2d3d.get((ts, cam_id))
        if matches is None:
            continue
        p2ds = matches['kp_q']
        p3ds = matches['p3d']
        node_ids_ref = matches['node_ids_ref']
        stddev_keypoint = matches['keypoint_noise'] * conf.noise_bundle_multiplier
        avg_noise_bundle.append(stddev_keypoint)
        camera = session_q.sensors[cam_id]
        T_w2cam = (poses_init[ts, rig_id] * (T_cam2rig or Pose())).inverse()
        for p2d, p3d, node_id_ref in zip(p2ds, p3ds, node_ids_ref):
            n_obs += 1
            node_id_ref = tuple(node_id_ref)
            is_track = False
            if use_reference_tracks and node_id_ref in tracks_ref['node_to_root_mapping']:
                node_id_ref = tracks_ref['node_to_root_mapping'][node_id_ref]
                p3d = tracks_ref['track_p3d'][node_id_ref]
                is_track = True

            if T_cam2rig is None:
                factor = pyceres.factors.BundleAdjustmentCost(
                    camera.model.model_id, p2d, stddev_keypoint)
            else:
                factor = pyceres.factors.BundleAdjustmentRigCost(
                    camera.model.model_id, p2d, *T_cam2rig.inverse().qt, stddev_keypoint)
            residual = factor.evaluate(*qt_opt[ts, rig_id], p3d, camera_params[cam_id])[0]
            behind = T_w2cam.transform_points(p3d)[-1] <= 0
            if behind or residual is None or np.linalg.norm(residual) > conf.reproj_filter_lambda:
                n_filtered += 1
                n_filtered_tracks_ref += is_track
                continue

            if node_id_ref not in p3d_opt:
                n_points_tracks_ref += is_track
            p3d_ = p3d_opt.setdefault(node_id_ref, p3d.copy())
            problem.add_residual_block(
                factor, loss_bundle, [*qt_opt[ts, rig_id], p3d_, camera_params[cam_id]])
            problem.set_parameter_block_constant(camera_params[cam_id])

            # add a prior on the 3D points or keep them fixed
            if noise_point3d is None:
                problem.set_parameter_block_constant(p3d_)
            else:
                factor = pyceres.factors.NormalPrior(p3d, covariance=noise_point3d**2)
                problem.add_residual_block(factor, pyceres.TrivialLoss(), [p3d_])

    logger.info('Bundle adjustment with %d (%d) points and %d observations.',
                len(p3d_opt), n_points_tracks_ref, n_obs)
    logger.info('%d (%d) filtered observations.', n_filtered, n_filtered_tracks_ref)
    p3d_init = deepcopy(p3d_opt)

    add_pose_graph_factors_to_sequence(
        problem, loss_tracking, poses_tracking, qt_opt, conf.rel_noise_tracking)

    for q, _ in qt_opt.values():
        problem.set_parameterization(q, pyceres.QuaternionParameterization())

    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
    options.max_num_iterations = 100
    options.minimizer_progress_to_stdout = False
    options.num_threads = conf.num_threads
    summary = pyceres.SolverSummary()
    pyceres.solve(options, problem, summary)
    logger.info('%s', summary.BriefReport())

    if len(p3d_init) > 0:
        diff = np.linalg.norm(np.array([p3d_init[k] for k in p3d_opt])
                              - np.array([p3d_opt[k] for k in p3d_opt]), axis=1)
        logger.info('Movement of 3D points: mean/med/q1/q9/max %e/%e/%e/%e/%e m.',
                    np.mean(diff), np.median(diff),
                    np.percentile(diff, 10), np.percentile(diff, 90),
                    np.max(diff))

    poses_opt = Trajectories()
    for k, qt in qt_opt.items():
        poses_opt[k] = Pose(*qt).inverse()
    print_pose_stats('Bundle adjustment', poses_init, poses_opt)

    # Tracking stats.
    print_tracking_stats('Pre-BA', poses_tracking, poses_init)
    print_tracking_stats('Post-BA', poses_tracking, poses_opt)

    # Reprojection statistics.
    logger.info('Computing reprojection statistics.')
    initial_reprojection_errors = []
    final_reprojection_errors = []
    for ts, rig_id, cam_id, T_cam2rig in tqdm(data_bundle, disable=(logger.level > logging.INFO)):
        matches = matches_2d3d.get((ts, cam_id))
        if matches is None:
            continue
        p2ds = matches['kp_q']
        p3ds = matches['p3d']
        node_ids_ref = matches['node_ids_ref']
        stddev_keypoint = matches['keypoint_noise'] * conf.noise_bundle_multiplier
        camera = session_q.sensors[cam_id]
        T_w2cam = (poses_init[ts, rig_id] * (T_cam2rig or Pose())).inverse()
        for p2d, p3d, node_id_ref in zip(p2ds, p3ds, node_ids_ref):
            node_id_ref = tuple(node_id_ref)
            if use_reference_tracks and node_id_ref in tracks_ref['node_to_root_mapping']:
                node_id_ref = tracks_ref['node_to_root_mapping'][node_id_ref]
                p3d = tracks_ref['track_p3d'][node_id_ref]

            factor = pyceres.factors.BundleAdjustmentRigCost(
                camera.model.model_id, p2d, *(T_cam2rig or Pose()).inverse().qt, stddev_keypoint)
            residual = factor.evaluate(*qt_init[ts, rig_id], p3d, np.array(camera.params))[0]
            behind = T_w2cam.transform_points(p3d)[-1] <= 0
            if behind or residual is None or np.linalg.norm(residual) > conf.reproj_filter_lambda:
                continue
            initial_reprojection_errors.append(np.linalg.norm(residual)*stddev_keypoint)

            residual = factor.evaluate(
                *qt_opt[ts, rig_id], p3d_opt[node_id_ref], np.array(camera.params))[0]
            if residual is None or np.linalg.norm(residual) > conf.reproj_filter_lambda:
                continue
            final_reprojection_errors.append(np.linalg.norm(residual)*stddev_keypoint)
    initial_reprojection_errors = np.array(initial_reprojection_errors)
    final_reprojection_errors = np.array(final_reprojection_errors)
    if initial_reprojection_errors.shape[0] > 0:
        print_reprojection_stats('Pre-BA', initial_reprojection_errors, np.mean(avg_noise_bundle))
        print_reprojection_stats('Post-BA', final_reprojection_errors, np.mean(avg_noise_bundle))
    return poses_opt
