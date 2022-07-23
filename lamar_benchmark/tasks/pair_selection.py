from typing import Dict
import logging
from collections import defaultdict
import dataclasses
import numpy as np

from hloc.pairs_from_retrieval import pairs_from_score_matrix

from scantools.utils.configuration import BaseConf

from .feature_extraction import RetrievalFeatureExtraction
from ..utils.capture import list_images_for_session
from ..utils.misc import same_configs, write_config
from ..utils.retrieval import (
    FrustumFilterConf, RadioFilterConf, PoseFilterConf,
    compute_overlap_pairs, compute_similarity_matrix,
    filter_by_frustum, filter_by_pose, filter_by_radio,
    fuse_similarities)

logger = logging.getLogger(__name__)


class PairSelectionConf(BaseConf):
    # pylint: disable=no-member
    method: Dict
    filter_frustum: FrustumFilterConf = dataclasses.field(default_factory=FrustumFilterConf)
    filter_radio: RadioFilterConf = dataclasses.field(default_factory=RadioFilterConf)
    filter_pose: PoseFilterConf = dataclasses.field(default_factory=PoseFilterConf)
    num_pairs: int = 10
    name: str = None

    def __post_init__(self):
        self.name = self.method_name()
        if self.filter_radio.do:
            assert self.filter_radio.num_pairs_filter is not None

    def method_name(self):
        name = self.method['name']
        if name == 'fusion':
            name += '-' + '-'.join(c['name'] for c in self.method['retrieval'])
        name += f'-{self.num_pairs}'
        if self.filter_frustum.do:
            name += '_frustum'
        if self.filter_pose.do:
            name += '_pose'
            name += f'-{self.filter_pose.max_rotation:.0f}'
            name += f'-{self.filter_pose.max_translation:.0f}'
            if self.filter_pose.num_pairs_filter is not None:
                name += f'-{self.filter_pose.num_pairs_filter}'
        if self.filter_radio.do:
            name += f'_radio-{self.filter_radio.window_us}-{self.filter_radio.num_pairs_filter}'
        return name


class PairSelectionPaths:
    def __init__(self, root, config, query_id, ref_id, override_workdir_root=None):
        self.root = root
        if override_workdir_root:
            root = override_workdir_root
        self.workdir = root / 'pair_selection' / query_id / ref_id / config.name
        self.retrieval = self.workdir / 'retrieval.txt'
        self.pairs_hloc = self.workdir / 'pairs.txt'
        self.config = self.workdir / 'configuration.json'


class PairSelection:
    methods = {
        **RetrievalFeatureExtraction.methods,
        'overlap': {
            'name': 'overlap',
            'mesh_id': 'mesh_simplified',
            'num_rays': 60,
        },
        'fusion': {
            'name': 'fusion',
            'retrieval': [
                RetrievalFeatureExtraction.methods['netvlad'],
                RetrievalFeatureExtraction.methods['ap-gem'],
            ],
        }
    }

    def __init__(self, outputs, capture, query_id, ref_id, config,
                 query_keys=None, query_poses=None, override_workdir_root=None):
        config = PairSelectionConf.from_dict(config)
        self.config = config
        self.query_id = query_id
        self.ref_id = ref_id
        self.paths = PairSelectionPaths(outputs, config, query_id, ref_id, override_workdir_root)
        self.query_keys = query_keys
        self.paths.workdir.mkdir(parents=True, exist_ok=True)
        overwrite = not same_configs(config.to_dict(), self.paths.config)
        if overwrite:
            logger.info('Selecting image pairs with %s for sessions (%s, %s).',
                        config.name, query_id, ref_id)
            self.retrieval, self.pairs = self.run(capture, query_poses)
            save_retrieval(self.retrieval, self.paths.retrieval)
            save_pairs(self.pairs, self.paths.pairs_hloc)
            write_config(config.to_dict(), self.paths.config)
        else:
            self.retrieval = load_retrieval(self.paths.retrieval)
            self.pairs = load_pairs(self.paths.pairs_hloc)

    def run(self, capture, poses_q=None):
        config = self.config
        session_q = capture.sessions[self.query_id]
        session_r = capture.sessions[self.ref_id]

        if poses_q is None and self.query_id == self.ref_id:
            # Only map sessions have GT absolute poses stored in trajectories
            poses_q = session_q.trajectories
        poses_r = session_r.trajectories

        keys_q, names_q, _ = list_images_for_session(capture, self.query_id, self.query_keys)
        keys_r, names_r, _ = list_images_for_session(capture, self.ref_id)

        # Prevent self-matching
        discard = np.array(names_q)[:, None] == np.array(names_r)[None]
        if config.filter_frustum.do:
            logger.info('Filtering pairs by frustums.')
            discard |= filter_by_frustum(
                session_q, session_r, keys_q, keys_r, poses_q, poses_r, config.filter_frustum)
        if config.filter_radio.do:
            logger.info('Filtering pairs by radios.')
            discard |= filter_by_radio(
                session_q, session_r, keys_q, keys_r, config.filter_radio)
        if config.filter_pose.do:
            logger.info('Filtering pairs by poses.')
            discard |= filter_by_pose(
                session_q, session_r, keys_q, keys_r, poses_q, poses_r, config.filter_pose, discard)

        if config.method['name'] == 'overlap':
            logger.info('Computing pairs from overlaps via mesh.')
            pairs_ij = compute_overlap_pairs(
                session_q, session_r, keys_q, keys_r, poses_q, poses_r,
                discard, capture.proc_path(self.ref_id), config)
        else:  # retrieval
            logger.info('Computing pairs from visual similarity.')
            do_fusion = config.method['name'] == 'fusion'
            sim = []
            for m in config.method['retrieval'] if do_fusion else [config.method]:
                sim.append(compute_similarity_matrix(
                    self.paths.root, capture, self.query_id, self.ref_id, m,
                    names_q, names_r, self.query_keys))
            sim = fuse_similarities(sim) if do_fusion else sim[0]
            pairs_ij = pairs_from_score_matrix(sim, discard, config.num_pairs)

        if len(pairs_ij) == 0:
            raise ValueError('No pair found!')
        retrieval = defaultdict(list)
        pairs = []
        for idx_q, idx_r in pairs_ij:
            retrieval[keys_q[idx_q]].append(keys_r[idx_r])
            pairs.append((names_q[idx_q], names_r[idx_r]))
        return retrieval, pairs


def save_pairs(pairs, output_path):
    with open(output_path, 'w') as fid:
        fid.write('\n'.join(' '.join(p) for p in pairs))


def load_pairs(input_path):
    with open(input_path, 'r') as fid:
        lines = fid.readlines()
    pairs = []
    for line in lines:
        pairs.append(line.strip('\n').split(' '))
    return pairs


def save_retrieval(retrieval, output_path):
    with open(output_path, 'w') as fid:
        for ts_q, cam_q in retrieval:
            for ts_r, cam_r in retrieval[ts_q, cam_q]:
                fid.write(','.join(map(str, [ts_q, cam_q, ts_r, cam_r])) + '\n')


def load_retrieval(input_path):
    with open(input_path, 'r') as fid:
        lines = fid.readlines()
    retrieval = defaultdict(list)
    for line in lines:
        ts_q, cam_q, ts_r, cam_r = line.strip('\n').split(',')
        retrieval[int(ts_q), cam_q].append((int(ts_r), cam_r))
    return retrieval
