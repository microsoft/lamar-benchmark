import logging
from pathlib import Path
from typing import Dict, List, Union, Tuple, Any, Iterator, Optional
import pprint
import numpy as np
from tqdm import tqdm
import h5py
import torch

from hloc import extract_features, match_features, pairs_from_retrieval, matchers
from hloc.match_features import find_unique_new_pairs, WorkQueue
from hloc.utils.base_model import dynamic_load
from hloc.utils.io import find_pair
from hloc.utils.parsers import parse_retrieval, names_to_pair

from ...capture import Session, Pose, Trajectories
from ...utils.configuration import BaseConf

logger = logging.getLogger(__name__)


class MatchingConf(BaseConf):
    global_features: Union[str, Dict[str, Any]]
    local_features: Union[str, Dict[str, Any]]
    matcher: Union[str, Dict[str, Any]]
    skip_cameras: Tuple[str, ...] = ('cam5_center',)  # discard some images (top)
    num_pairs: int = 5
    Rt_thresh: Tuple[float, float] = (45., np.inf)  # for spatial matching [degrees, meters]
    pairs_file: Optional[str] = None

    def __post_init__(self):
        if self.pairs_file is None:
            self.pairs_file = 'pairs_{}_{}.txt'.format(
                self.global_features['output'], self.num_pairs)

    def __setattr__(self, name, val):
        if name in ['global_features', 'local_features'] and isinstance(val, str):
            val = extract_features.confs[val]
        elif name == 'matcher' and isinstance(val, str):
            val = match_features.confs[val]
        super().__setattr__(name, val)

    @property
    def gfeats_file(self) -> str:
        return '{}.h5'.format(self.global_features['output'])

    @property
    def lfeats_file(self) -> str:
        return '{}.h5'.format(self.local_features['output'])

    @property
    def matches_file(self) -> str:
        return '{}_{}.h5'.format(self.local_features['output'], self.matcher['output'])


class KeyFramingConf(BaseConf):
    num: Optional[Union[int, float]] = None  # subsample the query images
    max_rotation: float = 20.0  # degrees
    max_distance: float = 1.0  # meters
    max_elapsed: float = 1.0  # seconds


def subsample_list(keys: List, num: Union[int, float]) -> List:
    if isinstance(num, float):
        assert 0 < num <= 1.
        num = max(int(num * len(keys)), 1)
    else:
        assert 1 <= num <= len(keys)
    idxs = np.round(np.linspace(0, len(keys)-1, num)).astype(int)
    keys = [keys[i] for i in idxs]
    return keys


def subsample_poses(keys, T_c2w: Trajectories, conf: KeyFramingConf):
    max_elapsed_us = conf.max_elapsed * 1e6
    selected = [keys[0]]
    dr_dt = np.array([0., 0.])
    dts = 0.
    for k_i, k_prev in zip(keys[1:], keys[:-1]):
        T_prev2i = T_c2w[k_i].inverse() * T_c2w[k_prev]
        dr_dt += np.array(T_prev2i.magnitude())
        dts += k_i[0] - k_prev[0]
        if dr_dt[0] > conf.max_rotation or dr_dt[1] > conf.max_distance or dts > max_elapsed_us:
            dr_dt = np.array([0., 0.])
            dts = 0.
            selected.append(k_i)
    if selected[-1] != keys[-1]:  # always add the last frame
        selected.append(keys[-1])
    return selected


def list_images_for_matching(session: Session,
                             prefix: Path,
                             conf: MatchingConf,
                             keyframing: Optional[KeyFramingConf] = None,
                             poses: Optional[Trajectories] = None):
    skip_cameras = set(conf.skip_cameras)
    skip_cameras = {i for i in session.sensors
                    if any(i.endswith(skip_camera) for skip_camera in skip_cameras)}

    # TODO: support exclusion via regexp to handle tiles
    keys = [
        (ts, sensor_id) for ts, sensor_id in session.trajectories.key_pairs()
        if sensor_id not in skip_cameras]

    # Subsample the images to speed up the matching
    if keyframing is not None:
        if keyframing.num is not None:
            keys = subsample_list(keys, keyframing.num)
        else:
            assert poses is not None
            keys = subsample_poses(keys, poses, keyframing)

    # Recover the image keys
    images = []
    for ts, cam_or_rig_id in keys:
        if cam_or_rig_id in session.sensors:
            images.append((ts, cam_or_rig_id))
        elif session.rigs is not None and cam_or_rig_id in session.rigs:
            for cam_id in session.images[ts]:
                if cam_id in session.rigs[cam_or_rig_id] and cam_id not in skip_cameras:
                    images.append((ts, cam_id))
        else:
            raise ValueError(cam_or_rig_id, session.sensors.keys())

    impath2key = {str(prefix / session.images[k]): k for k in images}
    return impath2key


def extract_session_features(images: List[str], image_root: Path, session_dir: Path,
                             conf: MatchingConf):
    extract_features.main(conf.global_features, image_root,
                          feature_path=session_dir / conf.gfeats_file,
                          image_list=images)
    extract_features.main(conf.local_features, image_root,
                          feature_path=session_dir / conf.lfeats_file,
                          image_list=images, as_half=True)


def match_from_retrieval(images_q: List[str], images_r: List[str],
                         dir_q: Path, dir_r: Path,
                         outputs: Path, conf: MatchingConf) -> List[Tuple[str, str]]:
    pairs = outputs / conf.pairs_file
    gfeats = dir_r / conf.gfeats_file
    lfeats = dir_r / conf.lfeats_file

    # TODO: images_r could be infered from the entries of the descriptor file.
    pairs_from_retrieval.main(
        dir_q / conf.gfeats_file, pairs, conf.num_pairs,
        query_list=images_q, db_list=images_r, db_descriptors=gfeats)

    retrieval = parse_retrieval(pairs)
    assert len(set(images_q) - retrieval.keys()) == 0

    matches_path = outputs / conf.matches_file
    match_features.main(
        conf.matcher, pairs, features=dir_q / conf.lfeats_file,
        matches=matches_path, features_ref=lfeats)

    return retrieval


def get_pairwise_distances(T_q2w: List[Pose], T_r2w: List[Pose]):
    R_q2w = np.stack([T.r.as_matrix() for T in T_q2w]).astype(np.float32)
    t_q2w = np.stack([T.t for T in T_q2w]).astype(np.float32)
    R_r2w = np.stack([T.r.as_matrix() for T in T_r2w]).astype(np.float32)
    t_r2w = np.stack([T.t for T in T_r2w]).astype(np.float32)

    # equivalent to scipy.spatial.distance.cdist but supports fp32
    dt = t_q2w.dot(t_r2w.T)
    dt *= -2
    dt += np.einsum('ij,ij->i', t_q2w, t_q2w)[:, None]
    dt += np.einsum('ij,ij->i', t_r2w, t_r2w)[None]
    np.clip(dt, a_min=0, a_max=None, out=dt)  # numerical errors
    np.sqrt(dt, out=dt)

    trace = np.einsum('nji,mji->nm', R_q2w, R_r2w, optimize=True)
    dR = np.clip((trace - 1) / 2, -1., 1.)
    dR = np.rad2deg(np.abs(np.arccos(dR)))
    return dR, dt


def pairs_from_poses(T_q2w: List[Pose], T_r2w: List[Pose],
                     Rt_thresh: Tuple[float], num_pairs: int) -> List[List[int]]:
    dR, dt = get_pairwise_distances(T_q2w, T_r2w)
    R_thresh, t_thresh = Rt_thresh
    valid = (dR < R_thresh) & (dt < t_thresh)
    dt = np.where(valid, dt, np.inf)

    pairs = []
    for dt_i, valid_i in zip(dt, valid):
        idx = np.argpartition(dt_i, num_pairs)[:num_pairs]  # not sorted
        idx = idx[np.argsort(dt_i[idx])]
        idx = idx[valid_i[idx]]
        pairs.append(idx)
    return pairs


def match_from_poses(T_q2w: Dict[str, Pose], T_r2w: Dict[str, Pose],
                     dir_q: Path, dir_r: Path,
                     outputs: Path, conf: MatchingConf) -> List[Tuple[str, str]]:
    pairs_path = outputs / conf.pairs_file
    lfeats = dir_r / conf.lfeats_file

    images_q = list(T_q2w.keys())
    images_r = list(T_r2w.keys())
    pairs = pairs_from_poses([T_q2w[i] for i in images_q],
                             [T_r2w[i] for i in images_r], conf.Rt_thresh, conf.num_pairs)
    pairs = {images_q[i]: [images_r[j] for j in js] for i, js in enumerate(pairs)}
    with open(pairs_path, 'w') as fid:
        fid.write('\n'.join(' '.join((n1, n2)) for n1, ns2 in pairs.items() for n2 in ns2))

    matches_path = outputs / conf.matches_file
    match_features.main(
        conf.matcher, pairs_path, features=dir_q / conf.lfeats_file,
        matches=matches_path, features_ref=lfeats)

    return pairs


# TODO: maybe compute the pairs outside of this module to limit dependencies
def match_from_overlap(impath2key_q, impath2key_ref, id_q, id_ref, capture, T_q, outputs, conf,
                       dir_q, dir_r) -> List[Tuple[str, str]]:
    pairs_path = outputs / conf.pairs_file
    lfeats_r = dir_r / conf.lfeats_file

    images_q, keys_q = zip(*impath2key_q.items())
    images_ref, keys_ref = zip(*impath2key_ref.items())
    from .. import overlap
    pairs = overlap.pairs_for_sequence(capture, id_q, id_ref, conf.num_pairs,
                                       keys_q, keys_ref, T_q)
    pairs = {images_q[i]: [images_ref[j] for j in js] for i, js in enumerate(pairs)}
    with open(pairs_path, 'w') as fid:
        fid.write('\n'.join(' '.join((n1, n2)) for n1, ns2 in pairs.items() for n2 in ns2))

    matches_path = outputs / conf.matches_file
    match_features.main(
        conf.matcher, pairs_path, features=dir_q / conf.lfeats_file,
        matches=matches_path, features_ref=lfeats_r)
    return pairs


def pairwise_matching(images_q: List[str], images_r: List[str], image_root: Path,
                      dir_q: Path, dir_r: Path, dir_out: Path, conf: MatchingConf):
    extract_session_features(images_q, image_root, dir_q, conf)
    extract_session_features(images_r, image_root, dir_r, conf)
    match_from_retrieval(images_q, images_r, dir_q, dir_r, dir_out, conf)


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


def write_inliers(path: Path, inliers: Dict[Tuple[str, str], np.ndarray]):
    with h5py.File(path, 'w') as fid:
        for (k1, k2), inl in inliers.items():
            pair = names_to_pair(str(k1), str(k2))
            fid.create_dataset(pair, data=inl)


class FeaturePairsDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, feature_paths):
        self.pairs = pairs
        self.feature_paths = feature_paths

    def __getitem__(self, idx):
        id0, id1, name0, name1 = self.pairs[idx]
        data = {}
        with h5py.File(self.feature_paths[id0], 'r') as fd:
            grp = fd[name0]
            for k, v in grp.items():
                data[k+'0'] = torch.from_numpy(v.__array__()).float()
            # some matchers might expect an image but only use its size
            data['image0'] = torch.empty((1,)+tuple(grp['image_size'])[::-1])
        with h5py.File(self.feature_paths[id1], 'r') as fd:
            grp = fd[name1]
            for k, v in grp.items():
                data[k+'1'] = torch.from_numpy(v.__array__()).float()
            data['image1'] = torch.empty((1,)+tuple(grp['image_size'])[::-1])
        return data

    def __len__(self):
        return len(self.pairs)


def _match_writer_fn(inp):
    path, pair, matches = inp
    with h5py.File(str(path), 'a') as fd:
        if pair in fd:
            del fd[pair]
        grp = fd.create_group(pair)
        grp.create_dataset('matches0', data=matches.short().cpu().numpy())


def match_multi_sequences(conf: Dict,
                          pairwise_paths: Dict[Tuple[str, str], 'Paths'],
                          feature_paths: Dict[str, Path]):
    all_pairs = []
    for (id0, id1), paths in tqdm(pairwise_paths.items()):
        pairs = parse_retrieval(paths.pairs)
        pairs = [(i, j) for i, js in pairs.items() for j in js]
        pairs = find_unique_new_pairs(pairs, paths.matches)
        pairs = [(id0, id1, i, j) for i, j in pairs]
        all_pairs.extend(pairs)

    if len(all_pairs) == 0:
        logger.info('Skipping the matching.')
        return
    logger.info('Matching local features with configuration:\n%s', pprint.pformat(conf))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = dynamic_load(matchers, conf['model']['name'])(conf['model']).eval().to(device)

    dataset = FeaturePairsDataset(all_pairs, feature_paths)
    loader = torch.utils.data.DataLoader(
        dataset, num_workers=5, batch_size=1, shuffle=False, pin_memory=True)
    writer_queue = WorkQueue(_match_writer_fn, 5)

    for idx, data in enumerate(tqdm(loader)):
        data = {k: v if k.startswith('image') else v.to(device, non_blocking=True)
                for k, v in data.items()}
        matches = model(data)['matches0'].squeeze(0)
        id0, id1, name0, name1 = all_pairs[idx]
        pair = names_to_pair(name0, name1)
        writer_queue.put((pairwise_paths[id0, id1].matches, pair, matches))
    writer_queue.join()
