import logging
from pathlib import Path
from typing import List, Optional
import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
import torch
import h5py

from hloc.pairs_from_retrieval import pairs_from_score_matrix

from scantools.utils.configuration import BaseConf
from scantools.proc.alignment.image_matching import get_pairwise_distances
from scantools.utils import radio_mapping
from scantools.utils.frustum import frustum_intersections

logger = logging.getLogger(__name__)


class FrustumFilterConf(BaseConf):
    do: bool = False
    max_depth: float = 20.0  # meter


class RadioFilterConf(BaseConf):
    do: bool = False
    window_us: int = 10_000_000
    frac_pairs_filter: float = None


class PoseFilterConf(BaseConf):
    do: bool = False
    max_rotation: float = 120.0  # degree
    max_translation: float = 20.0  # meter
    num_pairs_filter: Optional[int] = None


def filter_by_frustum(session_q, session_r, keys_q, keys_r, poses_q, poses_r,
                      conf: FrustumFilterConf):
    intersects = frustum_intersections(
        keys_q, session_q, poses_q, keys_r, session_r, poses_r, max_depth=conf.max_depth)
    return ~intersects


def filter_by_radio(session_q, session_r, keys_q, keys_r, conf: RadioFilterConf):
    assert conf.frac_pairs_filter is not None
    num_pairs_filter = int(np.ceil(conf.frac_pairs_filter * len(keys_r)))
    radio_map = radio_mapping.build_radio_map(session_r, conf.window_us)
    keep = np.full((len(keys_q), len(keys_r)), False, bool)
    keyr2idx = {k: i for i, k in enumerate(keys_r)}
    without_radios = []
    def _worker_fn(i):
        key_q = keys_q[i]
        d = radio_mapping.build_query_descriptor(key_q, session_q, conf.window_us)
        if len(d.radio_ids) > 0:
            keys_match, _ = radio_mapping.retrieve_relevant_map_images(
                d, radio_map, num_pairs_filter)
            if len(keys_match) == 0:
                keep[i, :] = True
            else:
                for key_r in keys_match:
                    keep[i, keyr2idx[tuple(key_r)]] = True
        else:
            # No filtering if there are no radios.
            without_radios.append(key_q)
            keep[i, :] = True
    thread_map(_worker_fn, range(len(keys_q)))
    logger.info('%d out of %d with no radios in range.', len(without_radios), len(keys_q))
    return ~keep


def filter_by_pose(session_q, session_r, keys_q, keys_r, poses_q, poses_r, conf: PoseFilterConf,
                   mask=None):
    poses_q = [session_q.get_pose(*k, poses_q) for k in keys_q]
    poses_r = [session_r.get_pose(*k, poses_r) for k in keys_r]
    num_poses_q = len(poses_q)
    discard_full = np.empty((num_poses_q, len(poses_r)), dtype=bool)
    BATCH_SIZE = 4096
    for batch_start_idx in tqdm(range(0, num_poses_q, BATCH_SIZE)):
        batch_end_idx = min(num_poses_q, batch_start_idx + BATCH_SIZE)
        dR, dt = get_pairwise_distances(
            poses_q[batch_start_idx : batch_end_idx], poses_r)
        discard = (dR > conf.max_rotation) | (dt > conf.max_translation)
        del dR
        if mask is not None:
            discard |= mask[batch_start_idx : batch_end_idx]
        if conf.num_pairs_filter is not None:
            dt[discard] = np.inf
            dt_top_k = torch.from_numpy(dt).topk(conf.num_pairs_filter+1, dim=1, largest=False)
            discard = dt >= dt_top_k.values[:, -1].numpy()[:, np.newaxis]
        discard_full[batch_start_idx : batch_end_idx] = discard
    return discard_full


def get_descriptors(features_path: Path, names: List[str], device: torch.device) -> torch.Tensor:
    with h5py.File(str(features_path), 'r') as fd:
        desc = [fd[n]['global_descriptor'].__array__() for n in names]
    return torch.from_numpy(np.stack(desc, 0)).to(device).float()


def fused_retrieval(root, capture, query_id, ref_id, config, names_q, names_r, keys_q, discard):
    # avoid circular import
    from ..tasks.feature_extraction import RetrievalFeatureExtraction  # pylint: disable=import-outside-toplevel

    do_fusion = config.method['name'] == 'fusion'
    descs_q = []
    descs_r = []
    methods = config.method['retrieval'] if do_fusion else [config.method]
    for method in methods:
        features_q = RetrievalFeatureExtraction(root, capture, query_id, method, keys_q)
        features_r = RetrievalFeatureExtraction(root, capture, ref_id, method)
        device = 'cpu'
        descs_q.append(get_descriptors(features_q.paths.features, names_q, device))
        descs_r.append(get_descriptors(features_r.paths.features, names_r, device))

    batch_size = 1024
    pairs_ij = []
    num_q = len(names_q)
    for batch_start_idx in range(0, num_q, batch_size):
        batch_end_idx = min(batch_start_idx + batch_size, num_q)
        sim = []
        for m_idx in range(len(methods)):
            sim.append(torch.einsum(
                'id,jd->ij',
                descs_q[m_idx][batch_start_idx : batch_end_idx],
                descs_r[m_idx]))
        sim = fuse_similarities(sim) if do_fusion else sim[0]
        raw_pairs_ij = pairs_from_score_matrix(
            sim, discard[batch_start_idx : batch_end_idx], config.num_pairs)
        pairs_ij.extend([
            (batch_start_idx + i, j) for i, j in raw_pairs_ij])
    return pairs_ij


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
                          discard, proc_path, conf, agg='avg'):
    # pylint: disable=import-outside-toplevel
    from scantools.proc.overlap import OverlapTracer
    from scantools.proc.rendering import Renderer

    mesh_path = proc_path / session_r.proc.meshes[conf.method['mesh_id']]
    tracer = OverlapTracer(Renderer(mesh_path), num_rays=conf.method['num_rays'])
    keep = ~discard
    # TODO: get rid of the frustum filter in trajectory_overlap
    overlaps_q2r = tracer.trajectory_overlap(
        keys_q, session_q, poses_q, keys_r, session_r, poses_r, mask=keep)
    keep &= overlaps_q2r > 0.01
    overlaps_r2q = tracer.trajectory_overlap(
        keys_r, session_r, poses_r, keys_q, session_q, poses_q, mask=keep.T)
    if agg == 'avg':
        overlaps = (overlaps_q2r + overlaps_r2q.T) / 2
    elif agg == 'min':
        overlaps = np.mininimum(overlaps_q2r, overlaps_r2q.T)
    else:
        raise ValueError(agg)
    pairs_ij = pairs_from_score_matrix(torch.from_numpy(overlaps), ~keep, conf.num_pairs)
    return pairs_ij


def get_retrieval(key, retrieval, ref_id, capture):
    session_ref = capture.sessions[ref_id]
    prefix_ref = capture.data_path(ref_id).relative_to(capture.sessions_path())
    ref_keys = retrieval[key]
    ref_key_names = [(k, str(prefix_ref / session_ref.images[k])) for k in ref_keys]
    return ref_key_names
