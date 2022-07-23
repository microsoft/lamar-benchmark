import logging
from pathlib import Path
from typing import List, Optional
import numpy as np
from tqdm import tqdm
import torch
import h5py

from hloc.pairs_from_retrieval import pairs_from_score_matrix

from scantools.utils.configuration import BaseConf
from scantools.proc.overlap import OverlapTracer
from scantools.proc.alignment.image_matching import get_pairwise_distances
from scantools.proc.rendering import Renderer
from scantools.utils import radio_mapping
from scantools.utils.io import read_mesh
from scantools.utils.frustum import frustum_intersections

from ..tasks.feature_extraction import RetrievalFeatureExtraction

logger = logging.getLogger(__name__)


class FrustumFilterConf(BaseConf):
    do: bool = False
    max_depth: float = 20.0


class RadioFilterConf(BaseConf):
    do: bool = False
    window_us: int = 10_000_000
    num_pairs_filter: int = None


class PoseFilterConf(BaseConf):
    do: bool = False
    max_rotation: float = 120.0
    max_translation: float = 20.0
    num_pairs_filter: Optional[int] = None


def filter_by_frustum(session_q, session_r, keys_q, keys_r, poses_q, poses_r,
                      conf: FrustumFilterConf):
    intersects = frustum_intersections(
        keys_q, session_q, poses_q, keys_r, session_r, poses_r, max_depth=conf.max_depth)
    return ~intersects


def filter_by_radio(session_q, session_r, keys_q, keys_r, conf: RadioFilterConf):
    assert conf.num_pairs_filter is not None
    radio_map = radio_mapping.build_radio_map(session_r, conf.window_us)
    keep = np.full((len(keys_q), len(keys_r)), False, np.bool)
    keyr2idx = {k: i for i, k in enumerate(keys_r)}
    num_without_radios = 0
    for i, key_q in enumerate(tqdm(keys_q)):
        d = radio_mapping.build_query_descriptor(key_q, session_q, conf.window_us)
        if len(d.radio_ids) > 0:
            keys_match, _ = radio_mapping.retrieve_relevant_map_images(
                d, radio_map, conf.num_pairs_filter)
            if len(keys_match) == 0:
                keep[i, :] = True
            else:
                for key_r in keys_match:
                    keep[i, keyr2idx[key_r]] = True
        else:
            # No filtering if there are no radios.
            num_without_radios += 1
            keep[i, :] = True
    logger.info('%d out of %d with no radios in range.', num_without_radios, len(keys_q))
    return ~keep


def filter_by_pose(session_q, session_r, keys_q, keys_r, poses_q, poses_r, conf: PoseFilterConf,
                   mask=None):
    poses_q = [session_q.get_pose(*k, poses_q) for k in keys_q]
    poses_r = [session_r.get_pose(*k, poses_r) for k in keys_r]
    dR, dt = get_pairwise_distances(poses_q, poses_r)
    discard = (dR > conf.max_rotation) | (dt > conf.max_translation)
    if mask is not None:
        discard |= mask
    if conf.num_pairs_filter is not None:
        dt_top_k = np.partition(dt, conf.num_pairs_filter)[:, conf.num_pairs_filter]
        discard |= (dt > dt_top_k[:, np.newaxis])
    return discard


def get_descriptors(features_path: Path, names: List[str], device: torch.device) -> torch.Tensor:
    with h5py.File(str(features_path), 'r') as fd:
        desc = [fd[n]['global_descriptor'].__array__() for n in names]
    return torch.from_numpy(np.stack(desc, 0)).to(device).float()


def compute_similarity_matrix(root, capture, query_id, ref_id, config, names_q, names_r, keys_q):
    features_q = RetrievalFeatureExtraction(root, capture, query_id, config, keys_q)
    features_r = RetrievalFeatureExtraction(root, capture, ref_id, config)
    device = 'cpu'
    desc_q = get_descriptors(features_q.paths.features, names_q, device)
    desc_r = get_descriptors(features_r.paths.features, names_r, device)
    sim = torch.einsum('id,jd->ij', desc_q, desc_r)
    return sim


def fuse_similarities(similarities: List[torch.Tensor], gamma=0.5):
    '''Fusion via generalized harmonic mean.
       Reference implementation:
       https://github.com/naver/kapture-localization/blob/main/kapture_localization/image_retrieval/fusion.py
    '''
    n = len(similarities)
    similarities = torch.stack(similarities, 0)
    fused = torch.reciprocal(torch.mean(torch.reciprocal(gamma + similarities/n), 0)) - gamma
    return fused


def compute_overlap_pairs(session_q, session_r, keys_q, keys_r, poses_q, poses_r,
                          discard, proc_path, conf):
    mesh = read_mesh(proc_path / session_r.proc.meshes[conf.method['mesh_id']])
    tracer = OverlapTracer(Renderer(mesh), num_rays=conf.method['num_rays'])
    keep = ~discard
    overlaps_q2r = tracer.trajectory_overlap(
        keys_q, session_q, poses_q, keys_r, session_r, poses_r, mask=keep)
    keep &= overlaps_q2r > 0.01
    overlaps_r2q = tracer.trajectory_overlap(
        keys_r, session_r, poses_r, keys_q, session_q, poses_q, mask=keep.T)
    overlaps = (overlaps_q2r + overlaps_r2q.T) / 2
    pairs_ij = pairs_from_score_matrix(torch.from_numpy(overlaps), ~keep, conf.num_pairs)
    return pairs_ij


def get_retrieval(key, retrieval, ref_id, capture):
    session_ref = capture.sessions[ref_id]
    prefix_ref = capture.data_path(ref_id).relative_to(capture.sessions_path())
    ref_keys = retrieval[key]
    ref_key_names = [(k, str(prefix_ref / session_ref.images[k])) for k in ref_keys]
    return ref_key_names
