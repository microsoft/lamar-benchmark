from pathlib import Path
from typing import List, Tuple, Iterator
import numpy as np
import h5py
import scipy.spatial

from hloc.utils.io import find_pair

from ...capture import Pose


def get_pairwise_distances(T_q2w: List[Pose], T_r2w: List[Pose]):
    R_q2w = np.stack([T.r.as_matrix() for T in T_q2w]).astype(np.float32)
    t_q2w = np.stack([T.t for T in T_q2w]).astype(np.float32)
    R_r2w = np.stack([T.r.as_matrix() for T in T_r2w]).astype(np.float32)
    t_r2w = np.stack([T.t for T in T_r2w]).astype(np.float32)
    dt = scipy.spatial.distance.cdist(t_q2w, t_r2w)
    trace = np.einsum('nji,mji->nm', R_q2w, R_r2w, optimize=True)
    dR = np.clip((trace - 1) / 2, -1., 1.)
    dR = np.rad2deg(np.abs(np.arccos(dR)))
    return dR, dt


def get_keypoints(feats_path: Path, keys: Iterator[str]) -> List[np.ndarray]:
    keypoints, uncertainties = [], []
    with h5py.File(feats_path, 'r') as fid:
        for k in keys:
            dset = fid[str(k)]['keypoints']
            keypoints.append(dset.__array__() + 0.5)  # to COLMAP coordinates
            uncertainties.append(dset.attrs.get('uncertainty'))
    return keypoints, uncertainties


def get_matches(matches_path: Path, key_pairs: Iterator[Tuple[str]]) -> List[np.ndarray]:
    matches = []
    with h5py.File(matches_path, 'r') as fid:
        for k1, k2 in key_pairs:
            pair, reverse = find_pair(fid, k1, k2)
            m = fid[pair]['matches0'].__array__()
            idx = np.where(m != -1)[0]
            m = np.stack([idx, m[idx]], -1)
            if reverse:
                m = np.flip(m, -1)
            matches.append(m)
    return matches
