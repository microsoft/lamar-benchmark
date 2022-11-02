from typing import List, Tuple, Optional, Dict
from copy import deepcopy
import logging
import dataclasses
from tqdm import tqdm
import numpy as np
import pycolmap
import pyceres

from ...utils.configuration import BaseConf
from ...capture import Pose, Trajectories, Session, KeyType
from .image_matching import MatchingConf, KeyFramingConf
from .localization import RelocConf

logger = logging.getLogger(__name__)
QTDictType = Dict[KeyType, Tuple[np.ndarray]]


class RefinementConf(BaseConf):
    matching: MatchingConf
    num_pairs_seq: int = 20
    Rt_thresh_seq: Tuple[float, float] = (120.0, 20.0)

    keyframings: Dict[Session.Device, KeyFramingConf] = dataclasses.field(default_factory=lambda: {
        Session.Device.HOLOLENS: None,
        Session.Device.PHONE: None,
    })
    keyframing: KeyFramingConf = dataclasses.field(default_factory=KeyFramingConf)

    max_reprojection_error: float = 2.0  # multiplier of the detection noise
    num_refinement_iterations: int = 2
    num_solver_iterations: int = 50

    rel_noise_tracking: Dict[Session.Device, float] = dataclasses.field(default_factory=lambda: {
        Session.Device.HOLOLENS: 0.01,
        Session.Device.PHONE: 0.05,
    })

    matching_seq_ref: MatchingConf = None
    matching_seq_seq: MatchingConf = None

    def __post_init__(self):
        self.matching_seq_ref = self.matching.update(dict(
            pairs_file=RelocConf.pairs_file.format(self.matching.num_pairs)))
        self.matching_seq_seq = self.matching.update(dict(
            pairs_file='pairs_spatial.txt',
            num_pairs=self.num_pairs_seq,
            Rt_thresh=self.Rt_thresh_seq,
        ))
        for k in self.keyframings.keys():
            if self.keyframings[k] is None:
                self.keyframings[k] = self.keyframing


class Triangulator:
    def __init__(self,
                 graph: pycolmap.CorrespondenceGraph,
                 reconstruction: pycolmap.Reconstruction,
                 max_reproj_error: float,
                 max_angle_error: float = 2.0,
                 min_tri_angle: float = 1.5):
        self.graph = graph
        self.reconstruction = reconstruction

        self.triangulator = pycolmap.IncrementalTriangulator(graph, reconstruction)
        self.options = pycolmap.IncrementalTriangulatorOptions()
        self.options.create_max_angle_error = max_angle_error
        self.options.continue_max_angle_error = max_angle_error
        self.options.merge_max_reproj_error = max_reproj_error
        self.options.complete_max_reproj_error = max_reproj_error

        self.max_reproj_error = max_reproj_error
        self.min_tri_angle = min_tri_angle

    def triangulate_all_images(self, completion_freq=0.02):
        images = [
            (image.image_id, self.graph.num_correspondences_for_image(image.image_id))
            for image in self.reconstruction.images.values()]
        images = sorted(images, key=lambda x: -x[1])
        if isinstance(completion_freq, float):
            completion_freq = int(completion_freq * len(images))
        for idx, (image_id, _) in enumerate(tqdm(images)):
            self.triangulator.triangulate_image(self.options, image_id)
            if (idx + 1) % completion_freq == 0:
                self.complete_and_merge_tracks()
        self.recursive_complete_and_merge_tracks()

    def complete_and_merge_tracks(self):
        num_completed_obs = self.triangulator.complete_all_tracks(self.options)
        num_merged_obs = self.triangulator.merge_all_tracks(self.options)
        return num_completed_obs + num_merged_obs

    def recursive_complete_and_merge_tracks(self):
        while self.complete_and_merge_tracks():
            pass

    def filter_points(self):
        return self.reconstruction.filter_all_points3D(self.max_reproj_error, self.min_tri_angle)


def add_pose_graph_factors_to_sequence(problem, loss, poses_relative, poses_opt, relative_noise,
                                       clip_covar: float = 0.00001):
    residuals = []
    keys = sorted(poses_opt)
    for (ts1, cam_id1), (ts2, cam_id2) in zip(keys[:-1], keys[1:]):
        qt1 = poses_opt[ts1, cam_id1]
        qt2 = poses_opt[ts2, cam_id2]
        T_1tow_track = poses_relative[ts1, cam_id1]
        T_2tow_track = poses_relative[ts2, cam_id2]
        T_2to1_track = T_1tow_track.inverse() * T_2tow_track
        sigma_pose = np.abs(np.r_[T_2to1_track.r.as_rotvec(), T_2to1_track.t]) * relative_noise
        sigma_pose = sigma_pose.clip(min=clip_covar)
        cov_tracking = np.diag(sigma_pose)**2
        factor = pyceres.factors.PoseGraphRelativeCost(*T_2to1_track.inverse().qt, cov_tracking)
        residuals.append(problem.add_residual_block(factor, loss, [*qt1, *qt2]))
        problem.set_parameterization(qt1[0], pyceres.QuaternionParameterization())
        problem.set_parameterization(qt2[0], pyceres.QuaternionParameterization())
    return residuals


def get_rig_id_at_timestamp(poses: QTDictType, ts: int) -> str:
    return next(filter(lambda ts_id: ts_id[0] == ts, poses))[1]  # dirty


def qt_dict_from_trajectories(poses: Trajectories) -> QTDictType:
    return {k: deepcopy(poses[k].inverse().qt) for k in poses.key_pairs()}


class RefinementCallback(pyceres.IterationCallback):
    def __init__(self, pose_parameters: List[Tuple[np.ndarray]],
                 min_pose_change: Tuple[float, float] = (0.01, 0.001),
                 min_iterations: int = 2):
        pyceres.IterationCallback.__init__(self)
        self.qts = pose_parameters
        self.qts_previous = deepcopy(self.qts)
        self.min_pose_change = min_pose_change
        self.min_iterations = min_iterations
        self.pose_changes = []

    def __call__(self, summary: pyceres.IterationSummary):
        if not summary.step_is_successful:
            return pyceres.CallbackReturnType.SOLVER_CONTINUE
        diff = []
        for qt_prev, qt_new in zip(self.qts_previous, self.qts):
            q_rel, t_rel = pycolmap.relative_pose(*qt_prev, *qt_new)
            dr = np.rad2deg(np.abs(2 * np.arctan2(np.linalg.norm(q_rel[1:]), q_rel[0])))
            dt = np.linalg.norm(t_rel)
            diff.append((dr, dt))
        diff = np.array(diff)
        self.qts_previous = deepcopy(self.qts)
        med, q99, max_ = np.quantile(diff, [0.5, 0.99, 1.0], axis=0)
        logger.info('%d Pose update: med/q99/max dR=%.3f/%.3f/%.3f deg, dt=%.3f/%.3f/%.3f cm',
                    summary.iteration, med[0], q99[0], max_[0], med[1], q99[1], max_[1])
        self.pose_changes.append((med, q99, max_))
        if summary.iteration >= self.min_iterations and np.all(q99 <= self.min_pose_change):
            return pyceres.CallbackReturnType.SOLVER_TERMINATE_SUCCESSFULLY
        return pyceres.CallbackReturnType.SOLVER_CONTINUE


class GlobalRefiner:
    def __init__(self,
                 reconstruction: pycolmap.Reconstruction,
                 key2imageid: Dict[KeyType, int],
                 conf: RefinementConf):
        self.reconstruction = reconstruction
        self.key2imageid = key2imageid
        self.conf = conf
        self.problem = pyceres.Problem()
        self.loss_bundle = pyceres.TrivialLoss()
        self.loss_tracking = pyceres.TrivialLoss()
        self.rigs = {}
        self.residuals = {'bundle': {}, 'pose_graph': {}}
        self.callback = None
        self.pose_parameters = []
        self.points3d = set()

    def solve(self):
        options = pyceres.SolverOptions()
        options.linear_solver_type = pyceres.LinearSolverType.ITERATIVE_SCHUR
        options.preconditioner_type = pyceres.PreconditionerType.SCHUR_JACOBI
        options.max_num_iterations = self.conf.num_solver_iterations
        options.max_linear_solver_iterations = 100
        options.minimizer_progress_to_stdout = True
        options.function_tolerance = 0
        options.gradient_tolerance = 0
        options.parameter_tolerance = 0
        options.num_threads = -1
        self.callback = RefinementCallback(self.pose_parameters)
        options.callbacks = [self.callback]
        options.update_state_every_iteration = True
        summary = pyceres.SolverSummary()
        pyceres.solve(options, self.problem, summary)
        return summary

    def update_reconstruction_poses(self, poses: Dict[str, QTDictType]):
        for sid, key2imageid in self.key2imageid.items():
            for key, image_id in key2imageid.items():
                image = self.reconstruction.images[image_id]
                qt = poses[sid].get(key)
                if qt is None:  # it's a rig
                    ts, sensor_id = key
                    rig_id = get_rig_id_at_timestamp(poses[sid], ts)
                    T_cam2rig = self.rigs[sid][rig_id, sensor_id]
                    qt = (T_cam2rig.inv * Pose(*poses[sid][ts, rig_id])).qt
                image.qvec, image.tvec = qt

    def add_bundle_factors_to_image(
            self,
            image_id: int,
            qt: Tuple[np.ndarray],
            stddev: float,
            constant_pose: bool = False,
            qt_rig: Optional[Tuple[np.ndarray]] = None):
        image = self.reconstruction.images[image_id]
        camera = self.reconstruction.cameras[image.camera_id]
        qt_constant = qt if constant_pose else ()
        qt_variable = () if constant_pose else qt
        residuals = []

        p2Ds = image.get_valid_points2D()
        for p2D in p2Ds:
            p3D = self.reconstruction.points3D[p2D.point3D_id]
            if qt_rig is None:
                factor = pyceres.factors.BundleAdjustmentCost(
                    camera.model_id, p2D.xy, *qt_constant, stddev)
            else:
                assert not constant_pose
                factor = pyceres.factors.BundleAdjustmentRigCost(
                    camera.model_id, p2D.xy, *qt_rig, stddev)
            residuals.append(self.problem.add_residual_block(
                factor, self.loss_bundle, [*qt_variable, p3D.xyz, camera.params]))
            self.points3d.add(p2D.point3D_id)
        if len(p2Ds) > 0:
            if not constant_pose:
                self.problem.set_parameterization(qt[0], pyceres.QuaternionParameterization())
            self.problem.set_parameter_block_constant(camera.params)
        return residuals

    def add_pose_graph_factors_to_sequence(
            self,
            poses_relative: Trajectories,
            poses_opt: QTDictType,
            rel_noise_tracking: float):
        return add_pose_graph_factors_to_sequence(
            self.problem, self.loss_tracking, poses_relative, poses_opt, rel_noise_tracking)

    def add_session_to_problem(
            self,
            session: Session,
            poses_opt: QTDictType,
            poses_tracking: Trajectories,
            is_reference: bool = False,
            detection_noise: Dict[KeyType, float] = None):
        sid = session.id
        self.rigs[sid] = session.rigs
        self.pose_parameters.extend(poses_opt.values())
        self.residuals['bundle'][sid] = {}
        for key, image_id in self.key2imageid[sid].items():
            qt_rig = None
            qt = poses_opt.get(key)
            if qt is None:  # it's a rig
                ts, sensor_id = key
                rig_id = get_rig_id_at_timestamp(poses_opt, ts)
                T_cam2rig = session.rigs[rig_id, sensor_id]
                qt = poses_opt[ts, rig_id]
                qt_rig = T_cam2rig.inverse().qt
            noise = detection_noise[key] if detection_noise else 1.0
            self.residuals['bundle'][sid][key] = self.add_bundle_factors_to_image(
                image_id, qt, noise, constant_pose=is_reference, qt_rig=qt_rig)

        if not is_reference:
            self.residuals['pose_graph'][sid] = self.add_pose_graph_factors_to_sequence(
                poses_tracking, poses_opt, self.conf.rel_noise_tracking[session.device])

    def make_cameras_constant(self):
        for cam in self.reconstruction.cameras.values():
            if self.problem.has_parameter_block(cam.params):
                self.problem.set_parameter_block_constant(cam.params)
                assert self.problem.is_parameter_block_constant(cam.params)

    def make_points_constant(self):
        for i in self.points3d:
            p = self.reconstruction.points3D[i]
            if self.problem.has_parameter_block(p.xyz):
                self.problem.set_parameter_block_constant(p.xyz)
