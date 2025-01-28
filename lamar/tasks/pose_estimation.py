import logging
import uuid
from pathlib import Path
from typing import List, Tuple
from copy import deepcopy
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
import numpy as np

from scantools.capture import Trajectories, Rigs

from .feature_extraction import FeatureExtraction
from .feature_matching import FeatureMatching
from .dense_matching import DenseMatching
from .mapping import Mapping
from ..utils.capture import list_images_for_session, list_trajectory_keys_for_session
from ..utils.misc import same_configs, write_config
from ..utils.localization import (
    estimate_camera_pose, estimate_camera_pose_rig,
    recover_matches_2d3d, compute_pose_errors)
from ..utils.retrieval import get_retrieval

logger = logging.getLogger(__name__)
KeyType = Tuple[int, str]


class PoseEstimationPaths:
    def __init__(self, root, config, query_id, ref_id, override_workdir_root=None):
        self.root = root
        if override_workdir_root:
            root = override_workdir_root
        self.workdir = (
            root / 'pose_estimation' / query_id / ref_id
            / config['features']['name'] / config['matches']['name']
            / config['pairs']['name'] / config['mapping']['name'] / config['name']
        )
        self.poses = self.workdir / 'poses.txt'
        self.config = self.workdir / 'configuration.json'


def rig_to_image_trajectory(T_c2w_rig: Trajectories, rigs: Rigs):
    T_c2w_image = Trajectories()
    for ts, rig_id in T_c2w_rig.key_pairs():
        for cam_id in rigs[rig_id]:
            T_c2w_image[ts, cam_id] = T_c2w_rig[ts, rig_id] * rigs[rig_id, cam_id]
    return T_c2w_image


class PoseEstimation:
    methods = {}
    method2class = {}
    method = None
    evaluation = {
        'Rt_thresholds': [(5, .5), (10, .25), (10, .5), (10, 1), (10, 2.5), (10, 5), (10, 10)],
    }

    def __init_subclass__(cls):
        '''Register the child classes into the parent'''
        if cls.method is None:  # abstract class
            return
        name = cls.method['name']
        cls.methods[name] = cls.method
        cls.method2class[name] = cls

    def __new__(cls, config, *_, **__):
        '''Instanciate the object from the child class'''
        return super().__new__(cls.method2class[config['name']])

    def __init__(self, config, outputs, capture, query_id,
                 extraction: FeatureExtraction,
                 matching: FeatureMatching,
                 mapping: Mapping,
                 query_keys: list = None,
                 parallel: bool = False,
                 return_covariance: bool = False,
                 override_workdir_root: Path = None):
        if extraction.config['name'] != mapping.extraction.config['name']:
            raise ValueError('Mapping and query features are different:'
                             f'{mapping.extraction.config} vs {extraction.config}')
        assert query_id == extraction.session_id
        assert query_id == matching.query_id
        ref_id = mapping.session_id
        assert ref_id == matching.ref_id

        assert query_id == extraction.session_id
        self.config = config = {
            **deepcopy(config),
            'features': extraction.config,
            'matches': matching.config,
            'pairs': matching.pair_selection.config.to_dict(),
            'mapping': mapping.config,
        }
        self.query_id = query_id
        self.ref_id = ref_id
        self.extraction = extraction
        self.matching = matching
        self.mapping = mapping
        self.paths = PoseEstimationPaths(outputs, config, query_id, ref_id, override_workdir_root)
        self.query_keys = query_keys
        self.query_rigs = capture.sessions[query_id].rigs
        self.parallel = parallel
        self.return_covariance = return_covariance

        self.paths.workdir.mkdir(parents=True, exist_ok=True)
        overwrite = not same_configs(config, self.paths.config)
        if overwrite:
            logger.info('Localizing (%s) session %s with features %s.',
                        config['name'], query_id, config['features']['name'])
            self.poses = self.run(capture)
            self.poses.save(self.paths.poses)
            write_config(config, self.paths.config)
        else:
            self.poses = Trajectories().load(self.paths.poses)

    def run(self, capture):
        raise NotImplementedError

    def recover_matches_2d3d(self, query: str, ref_key_names: List[Tuple[KeyType, str]]):
        return recover_matches_2d3d(
            query,
            ref_key_names,
            self.mapping,
            self.extraction.paths.features,
            self.matching.paths.matches,
        )

    def evaluate(self, T_c2w_gt: Trajectories):
        if self.query_keys:
            T_c2w_gt_filtered = Trajectories()
            for key in self.query_keys:
                T_c2w_gt_filtered[key] = T_c2w_gt[key]
            T_c2w_gt = T_c2w_gt_filtered
        T_c2w_gt = self.convert_poses_for_eval(T_c2w_gt)
        query_keys = T_c2w_gt.key_pairs()
        T_c2w = self.convert_poses_for_eval(self.poses)
        err_r, err_t = compute_pose_errors(query_keys, T_c2w, T_c2w_gt)
        threshs = self.evaluation['Rt_thresholds']
        recalls = [np.mean((err_r < th_r) & (err_t < th_t)) for th_r, th_t in threshs]
        return {'recall': recalls, 'Rt_thresholds': threshs}

    def convert_poses_for_eval(self, T_c2w):
        raise NotImplementedError


class SingleImagePoseEstimation(PoseEstimation):
    method = {
        'name': 'single_image',
        'pnp_error_multiplier': 3.0,
    }

    def run(self, capture):
        retrieval = self.matching.pair_selection.retrieval

        keys, names, _ = list_images_for_session(
            capture, self.query_id, self.query_keys)
        session = capture.sessions[self.query_id]

        poses = Trajectories()

        def _worker_fn(idx: int):
            query_name = names[idx]
            key = keys[idx]
            ref_key_names = get_retrieval(key, retrieval, self.ref_id, capture)
            camera = session.sensors[key[1]]
            pose, _ = estimate_camera_pose(
                query_name,
                camera,
                ref_key_names,
                self.recover_matches_2d3d,
                self.config['pnp_error_multiplier'],
                return_covariance=self.return_covariance
            )
            if pose is not None:
                poses[key] = pose

        map_ = thread_map if self.parallel else lambda f, x: list(map(f, tqdm(x)))
        map_(_worker_fn, range(len(keys)))

        return poses

    def convert_poses_for_eval(self, T_c2w):
        return T_c2w


class RigPoseEstimation(PoseEstimation):
    method = {
        'name': 'rig',
        'pnp_error_multiplier': 1.0
    }

    def run(self, capture):
        retrieval = self.matching.pair_selection.retrieval

        keys = list_trajectory_keys_for_session(capture, self.query_id, self.query_keys)
        session = capture.sessions[self.query_id]
        prefix = capture.data_path(self.query_id).relative_to(capture.sessions_path())

        poses = Trajectories()

        def _worker_fn(idx: int):
            key = keys[idx]
            ts, rig_id = key
            rig = session.rigs[rig_id]
            query_keys = [(ts, camera_id) for camera_id in rig]
            query_names = [str(prefix / session.images[k]) for k in query_keys]
            ref_key_names = [
                get_retrieval(key, retrieval, self.ref_id, capture) for key in query_keys]
            cameras = [session.cameras[camera_id] for _, camera_id in query_keys]
            T_cams2rig = [rig[camera_id] for _, camera_id in query_keys]
            pose, _ = estimate_camera_pose_rig(
                query_names, cameras, T_cams2rig,
                ref_key_names,
                self.recover_matches_2d3d,
                self.config['pnp_error_multiplier'],
                return_covariance=self.return_covariance
            )
            if pose is not None:
                poses[key] = pose

        map_ = thread_map if self.parallel else lambda f, x: list(map(f, tqdm(x)))
        map_(_worker_fn, range(len(keys)))

        return poses

    def convert_poses_for_eval(self, T_c2w):
        return rig_to_image_trajectory(T_c2w, self.query_rigs)


class RigSinglePoseEstimation(SingleImagePoseEstimation):
    method = {
        'name': 'rig_single',
        'pnp_error_multiplier': 1.0
    }


class DensePoseEstimation(PoseEstimation):
    method = None

    def __init__(self, config, outputs, capture, query_id,
                 matching: DenseMatching, mapping: Mapping,
                 query_keys: list = None, parallel: bool = True,
                 return_covariance: bool = False):

        assert query_id == matching.query_id
        ref_id = mapping.session_id
        assert ref_id == matching.ref_id

        self.config = config = {
            **deepcopy(config),
            'features': {'name': ''},  # dummy for paths
            'matches': matching.config,
            'pairs': matching.pair_selection.config.to_dict(),
            'mapping': mapping.config,
        }
        self.query_id = query_id
        self.ref_id = ref_id
        self.matching = matching
        self.mapping = mapping
        self.paths = PoseEstimationPaths(outputs, config, query_id, ref_id)
        self.query_keys = query_keys
        self.query_rigs = capture.sessions[query_id].rigs
        self.parallel = parallel
        self.return_covariance = return_covariance

        self.paths.workdir.mkdir(parents=True, exist_ok=True)
        overwrite = not same_configs(config, self.paths.config)
        if overwrite:
            self.poses = self.run(capture)
            self.poses.save(self.paths.poses)
            write_config(config, self.paths.config)
        else:
            self.poses = Trajectories().load(self.paths.poses)

    def recover_matches_2d3d(self, query: str, ref_key_names: List[Tuple[KeyType, str]]):
        if len(ref_key_names) == 0:
            ref_keys = ref_names = []
        else:
            ref_keys, ref_names = zip(*ref_key_names)
        matches = self.matching.get_matches_pairs(zip([query]*len(ref_names), ref_names))
        ret = {
            'kp_q': [np.empty((0, 2))],
            'p3d': [np.empty((0, 3))],
            'indices': [np.empty((0,), int)],
            'node_ids_ref': [np.empty((0, 2), object)]
        }
        noise = None
        for idx, ref_key in enumerate(ref_keys):
            kp_q, kp_r, noise, _ = matches[idx]
            if len(kp_q) == 0:
                continue
            valid, p3ds = self.mapping.lift_points2D(ref_key, kp_r)
            if len(p3ds) == 0:
                continue
            node_ids_ref = [(ref_key, uuid.uuid4().int) for _ in p3ds]  # unique ID
            ret['kp_q'].append(kp_q[valid])
            ret['p3d'].append(np.asarray(p3ds))
            ret['indices'].append(np.array([idx]*len(p3ds)))
            ret['node_ids_ref'].append(np.array(node_ids_ref, dtype=object))
        ret = {k: np.concatenate(v, 0) for k, v in ret.items()}
        return {**ret, 'keypoint_noise': noise}


class SingleImageDensePoseEstimation(DensePoseEstimation, SingleImagePoseEstimation):
    method = {
        'name': 'dense_single_image',
        'pnp_error_multiplier': 3.0,
    }

class RigDensePoseEstimation(DensePoseEstimation, RigPoseEstimation):
    method = {
        'name': 'dense_rig',
        'pnp_error_multiplier': 1.0,
    }
