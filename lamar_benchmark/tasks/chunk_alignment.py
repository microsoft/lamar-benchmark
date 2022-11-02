import logging
from copy import deepcopy
from itertools import chain
import numpy as np
from tqdm import tqdm

from scantools.capture import Trajectories
from scantools.proc.alignment.sequence import (
    logger as seq_logger, InitializerConf, PGOConf, BAConf,
    align_trajectories_with_voting,
    optimize_sequence_pose_graph_gnc,
    optimize_sequence_bundle)

from ..utils.localization import compute_pose_errors
from ..utils.misc import same_configs, write_config
from ..utils.retrieval import get_retrieval
from .pair_selection import PairSelection, PairSelectionConf
from .pose_estimation import PoseEstimation, rig_to_image_trajectory
from .mapping import Mapping
from .feature_matching import FeatureMatching
from .feature_extraction import FeatureExtraction

seq_logger.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def keys_from_chunks(chunks):
    return list(chain.from_iterable(chunks))


class ChunkAlignmentPaths:
    def __init__(self, root, config, query_id, ref_id, chunk_length_s):
        self.root = root
        name = config['name']
        if chunk_length_s:
            name += f"-{chunk_length_s}s"
        self.workdir = (
            root / 'chunk_alignment' / query_id / ref_id
            / config['features']['name'] / config['matches']['name']
            / config['pairs']['name'] / config['pairs_reloc']['name']
            / config['mapping']['name'] / name
        )
        self.loc_root = self.workdir / 'loc'
        self.poses_loc = self.workdir / 'poses_loc.txt'
        self.poses_init = self.workdir / 'poses_init.txt'
        self.poses_pgo_loc = self.workdir / 'poses_pgo_loc.txt'
        self.reloc_root = self.workdir / 'reloc'
        self.poses_reloc = self.workdir / 'poses_reloc.txt'
        self.poses_pgo_reloc = self.workdir / 'poses_pgo_reloc.txt'
        self.poses = self.workdir / 'poses.txt'
        self.config = self.workdir / 'configuration.json'


class ChunkAlignment:
    methods = {}
    method2class = {}
    method = {}
    evaluation = {
        'Rt_thresholds': [(1, 0.1), (5, 1.)],
    }

    def __init_subclass__(cls):
        '''Register the child classes into the parent'''
        name = cls.method['name']
        cls.methods[name] = cls.method
        cls.method2class[name] = cls

    def __new__(cls, configs, *_, **__):
        '''Instanciate the object from the child class'''
        return super().__new__(cls.method2class[configs['chunks']['name']])

    def __init__(self, configs, outputs, capture, query_id,
                 extraction: FeatureExtraction,
                 mapping: Mapping,
                 query_chunks,
                 chunk_length_s: int):
        if extraction.config['name'] != mapping.extraction.config['name']:
            raise ValueError('Mapping and query features are different:'
                             f'{mapping.extraction.config} vs {extraction.config}')
        assert query_id == extraction.session_id
        ref_id = mapping.session_id

        self.outputs = outputs

        self.query_chunks = query_chunks

        self.query_id = query_id
        self.ref_id = ref_id

        self.extraction = extraction
        self.mapping = mapping

        self.query_rigs = capture.sessions[query_id].rigs

        config_pairs_loc = configs['pairs_loc']
        pairs_loc = PairSelectionConf.from_dict(config_pairs_loc)
        config_pairs_reloc = {**configs['pairs_loc'], **configs['extra_pairs_reloc']}
        # Deactivate radios for second reloc since we have stronger priors.
        pairs_reloc = PairSelectionConf.from_dict(config_pairs_reloc)
        pairs_reloc.filter_radio.do = False
        self.config = config = {
            **deepcopy(configs['chunks']),
            'features': extraction.config,
            'matches': configs['matching'],
            'matches_query': configs['matching_query'],
            'pairs': pairs_loc.to_dict(),
            'pairs_reloc': pairs_reloc.to_dict(),
            'mapping': self.mapping.config,
            'pose_estimation': configs['poses']
        }
        self.paths = ChunkAlignmentPaths(
            outputs, config, query_id, ref_id, chunk_length_s)
        self.paths.workdir.mkdir(parents=True, exist_ok=True)

        overwrite = not same_configs(config, self.paths.config)
        if overwrite:
            logger.info('Chunk alignment (%s) for session %s with features %s.',
                        config['name'], query_id, config['features']['name'])
            self.run(capture)
            write_config(config, self.paths.config)
        else:
            self.poses_loc = Trajectories.load(self.paths.poses_loc)
            self.poses_init = Trajectories.load(self.paths.poses_init)
            self.poses_pgo_loc = Trajectories.load(self.paths.poses_pgo_loc)
            self.poses_reloc = Trajectories.load(self.paths.poses_reloc)
            self.poses_pgo_reloc = Trajectories.load(self.paths.poses_pgo_reloc)
            self.poses = Trajectories.load(self.paths.poses)

    def run(self, capture):
        raise NotImplementedError

    def evaluate(self, T_c2w_gt: Trajectories, query_keys=None):
        if query_keys:
            T_c2w_gt_filtered = Trajectories()
            for key in query_keys:
                T_c2w_gt_filtered[key] = T_c2w_gt[key[0], key[1].split('$')[0]]
            T_c2w_gt = T_c2w_gt_filtered
        T_c2w_gt = self.convert_poses_for_eval(T_c2w_gt)
        query_keys = T_c2w_gt.key_pairs()
        eval_poses = {
            'loc': self.poses_loc, 'init': self.poses_init,
            'pgo_loc': self.poses_pgo_loc, 'reloc': self.poses_reloc,
            'pgo_reloc': self.poses_pgo_reloc, 'final': self.poses}
        recalls = {}
        for key, T_c2w in eval_poses.items():
            T_c2w = self.convert_poses_for_eval(T_c2w)
            err_r, err_t = compute_pose_errors(query_keys, T_c2w, T_c2w_gt)
            threshs = self.evaluation['Rt_thresholds']
            recalls_ = [np.mean((err_r < th_r) & (err_t < th_t))
                        for th_r, th_t in threshs]
            recalls[key] = recalls_
        return {'recall': recalls, 'Rt_thresholds': threshs}

    def convert_poses_for_eval(self, T_c2w):
        raise NotImplementedError


class SingleImageChunkAlignment(ChunkAlignment):
    method = {
        'name': 'single_image',
        'init': {
            'distance_thresh': 2.,
            'angle_thresh': 20.,
            'min_num_inliers': 1
        },
        'pgo': {
            'rel_noise_tracking': 0.05,
            'cost_loc': ['Arctan', 10.0],
            'num_threads': -1,
        },
        'ba': {
            'noise_point3d': None,
            'rel_noise_tracking': 0.05,
            'num_threads': -1,
        },
    }

    def run(self, capture):
        query_chunks = self.query_chunks
        session = capture.sessions[self.query_id]

        # Configs.
        conf_init = InitializerConf.from_dict(self.config['init'])
        conf_pgo = PGOConf.from_dict(self.config['pgo'])
        conf_ba = BAConf.from_dict(self.config['ba'])

        # First retrieval and matching.
        query_keys = keys_from_chunks(self.query_chunks)
        pair_selection = PairSelection(
            self.paths.root, capture, self.query_id, self.ref_id,
            self.config['pairs'], query_keys, override_workdir_root=self.paths.loc_root)
        matching = FeatureMatching(
            self.outputs, capture, self.query_id, self.ref_id,
            self.config['matches_query'], pair_selection,
            self.extraction, self.mapping.extraction)

        # First localization.
        pose_estimation = PoseEstimation(
            self.config['pose_estimation'], self.outputs, capture, self.query_id,
            self.extraction, matching, self.mapping,
            query_keys, return_covariance=True, override_workdir_root=self.paths.loc_root)
        poses_loc = pose_estimation.poses
        poses_loc.save(self.paths.poses_loc)
        self.poses_loc = poses_loc

        # Recover tracking poses.
        poses_tracking = []
        for query_keys in query_chunks:
            poses_tracking.append(Trajectories())
            for key in query_keys:
                poses_tracking[-1][key] = session.trajectories[key]

        # Rigid alignment + first PGO.
        logger.info('Rigid alignment and first PGO.')
        poses_init, poses_pgo_loc = run_pgo_with_init(
            poses_tracking, poses_loc, conf_init, conf_pgo)
        poses_init.save(self.paths.poses_init)
        self.poses_init = poses_init
        poses_pgo_loc.save(self.paths.poses_pgo_loc)
        self.poses_pgo_loc = poses_pgo_loc

        # Ignore queries that didn't survive initialization.
        query_keys = list(self.poses_pgo_loc.key_pairs())

        # Retrival with prior poses.
        logger.info('Guided retrieval and relocalization.')
        if session.rigs:
            poses_pair_selection_reloc = rig_to_image_trajectory(
                poses_pgo_loc, session.rigs)
        else:
            poses_pair_selection_reloc = poses_pgo_loc
        pair_selection_reloc = PairSelection(
            self.paths.root, capture, self.query_id, self.ref_id, self.config['pairs_reloc'],
            query_keys, query_poses=poses_pair_selection_reloc,
            override_workdir_root=self.paths.reloc_root)
        # Reuse matches.
        rematching = FeatureMatching(
            self.paths.root, capture, self.query_id, self.ref_id, self.config['matches_query'],
            pair_selection_reloc, self.extraction, self.mapping.extraction)
        # Infer what pose estimation to use.
        pose_estimation2 = PoseEstimation(
            self.config['pose_estimation'], self.paths.root, capture, self.query_id,
            self.extraction, rematching, self.mapping, query_keys,
            return_covariance=True, override_workdir_root=self.paths.reloc_root)
        poses_reloc = pose_estimation2.poses
        poses_reloc.save(self.paths.poses_reloc)
        self.poses_reloc = poses_reloc

        # Second PGO.
        logger.info('Second PGO.')
        poses_pgo_reloc = run_pgo(poses_tracking, poses_reloc, poses_pgo_loc, conf_pgo)
        poses_pgo_reloc.save(self.paths.poses_pgo_reloc)
        self.poses_pgo_reloc = poses_pgo_reloc

        # BA.
        logger.info('Aggregating matches for BA.')
        matches_2d3d = aggregate_matches_for_ba(
            capture, self.query_id, self.ref_id, self.query_chunks, rematching, pose_estimation2)
        logger.info('Bundle adjustment.')
        poses_ba = run_ba(poses_tracking, poses_pgo_reloc, session, matches_2d3d, conf_ba)
        poses_ba.save(self.paths.poses)
        self.poses = poses_ba

    def convert_poses_for_eval(self, T_c2w):
        return T_c2w


class RigChunkAlignment(SingleImageChunkAlignment):
    method = {
        'name': 'rig',
        'init': {
            'distance_thresh': 2.,
            'angle_thresh': 20.,
            'min_num_inliers': 1
        },
        'pgo': {
            'rel_noise_tracking': 0.03,
            'cost_loc': ['Arctan', 10.0],
            'num_threads': -1,
        },
        'ba': {
            'noise_point3d': None,
            'rel_noise_tracking': 0.03,
            'num_threads': -1,
        }
    }

    def convert_poses_for_eval(self, T_c2w):
        return rig_to_image_trajectory(T_c2w, self.query_rigs)


def run_pgo_with_init(poses_tracking, poses_loc, conf_init, conf_pgo):
    poses_init = Trajectories()
    poses_pgo = Trajectories()
    def _worker_fn_pgo_loc(idx):
        poses_tracking_ = poses_tracking[idx]
        poses_loc_ = Trajectories()
        for key in poses_tracking_.key_pairs():
            if key in poses_loc:
                poses_loc_[key] = poses_loc[key]
        poses_init_, _ = align_trajectories_with_voting(poses_tracking_, poses_loc_, conf_init)
        if poses_init_ is None:
            return
        for key in poses_init_.key_pairs():
            poses_init[key] = poses_init_[key]
        poses_pgo_, _ = optimize_sequence_pose_graph_gnc(
            poses_tracking_, poses_loc_, poses_init_, conf_pgo)
        for key in poses_pgo_.key_pairs():
            poses_pgo[key] = poses_pgo_[key]
    # Iterative is faster.
    list(map(_worker_fn_pgo_loc, tqdm(range(len(poses_tracking)))))
    return poses_init, poses_pgo


def run_pgo(poses_tracking, poses_loc, poses_init, conf_pgo):
    poses_pgo = Trajectories()
    def _worker_fn_pgo_reloc(idx):
        poses_tracking_ = poses_tracking[idx]
        poses_init_ = Trajectories()
        poses_loc_ = Trajectories()
        for key in poses_tracking_.key_pairs():
            if key not in poses_init:
                logging.warning(
                    'First PGO failed for (%d, %s). Skipping second PGO.', key[0], key[1])
                return
            poses_init_[key] = poses_init[key]
            if key in poses_loc:
                poses_loc_[key] = poses_loc[key]
        poses_pgo_, _ = optimize_sequence_pose_graph_gnc(
            poses_tracking_, poses_loc_, poses_init_, conf_pgo)
        for key in poses_pgo_.key_pairs():
            poses_pgo[key] = poses_pgo_[key]
    # Iterative is faster.
    list(map(_worker_fn_pgo_reloc, tqdm(range(len(poses_tracking)))))
    return poses_pgo


def aggregate_matches_for_ba(capture, query_id, ref_id, query_chunks, matching, pose_estimation):
    session_q = capture.sessions[query_id]
    query_rigs = session_q.rigs
    prefix = capture.data_path(query_id).relative_to(capture.sessions_path())
    retrieval = matching.pair_selection.retrieval
    matches_2d3d = [None] * len(query_chunks)
    def _worker_fn_aggregation(idx):
        query_chunk = query_chunks[idx]
        matches_2d3d_ = {}
        for key in query_chunk:
            if query_rigs:
                ts, rig_id = key
                for cam_id in query_rigs[rig_id]:
                    query_name = str(prefix / session_q.images[ts, cam_id])
                    ref_key_names = get_retrieval((ts, cam_id), retrieval, ref_id, capture)
                    matches_2d3d_[ts, cam_id] = pose_estimation.recover_matches_2d3d(
                        query_name, ref_key_names)
            else:
                query_name = str(prefix / session_q.images[key])
                ref_key_names = get_retrieval(key, retrieval, ref_id, capture)
                matches_2d3d_[key] = pose_estimation.recover_matches_2d3d(
                    query_name, ref_key_names)
        matches_2d3d[idx] = matches_2d3d_
    # Iterative is faster.
    list(map(_worker_fn_aggregation, tqdm(range(len(query_chunks)))))
    return matches_2d3d


def run_ba(poses_tracking, poses_init, session, matches_2d3d, conf_ba):
    poses_ba = Trajectories()
    def _worker_fn_ba(idx):
        poses_tracking_ = poses_tracking[idx]
        poses_init_ = Trajectories()
        for key in poses_tracking_.key_pairs():
            if key not in poses_init:
                logging.warning('First PGO failed for (%d, %s). Skipping BA.', key[0], key[1])
                return
            poses_init_[key] = poses_init[key]
        # No need for reference tracks since points are fixed.
        poses_ba_, _ = optimize_sequence_bundle(
            poses_tracking_, poses_init_, session,
            matches_2d3d[idx], None, conf_ba,
            use_reference_tracks=False,
            compute_stats=False,
            compute_covariances=False)
        for key in poses_ba_.key_pairs():
            poses_ba[key] = poses_ba_[key]
    # Iterative is faster.
    list(map(_worker_fn_ba, tqdm(range(len(poses_tracking)))))
    return poses_ba
