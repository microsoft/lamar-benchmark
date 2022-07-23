import logging
import warnings
from typing import Tuple, Iterator
from types import SimpleNamespace
from tqdm import tqdm
import numpy as np
import h5py
import torch
from kornia.feature import LoFTR

from hloc.utils.parsers import parse_retrieval, names_to_pair
from hloc.match_features import find_unique_new_pairs
from hloc.extract_features import read_image, resize_image

from .pair_selection import PairSelection
from ..utils.misc import same_configs, write_config

logger = logging.getLogger(__name__)


class DenseMatchingPaths:
    def __init__(self, root, config, query_id, ref_id):
        self.root = root
        self.workdir = root / 'dense_matching' / query_id / ref_id /  config['name']
        self.matches = self.workdir / 'matches.h5'
        self.config = self.workdir / 'configuration.json'


class ImagePairDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, conf, pairs):
        self.image_dir = image_dir
        self.conf = conf = SimpleNamespace(**conf)
        self.pairs = pairs

    def preprocess(self, image: np.ndarray):
        image = image.astype(np.float32, copy=False)
        size = image.shape[:2][::-1]
        scale = np.array([1.0, 1.0])
        if self.conf.resize_max:
            scale = self.conf.resize_max / max(size)
            if scale < 1.0:
                size_new = tuple(int(round(x*scale)) for x in size)
                image = resize_image(image, size_new, 'cv2_area')
                scale = np.array(size) / np.array(size_new)
        if self.conf.grayscale:
            assert image.ndim == 2, image.shape
            image = image[None]
        else:
            image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        image = torch.from_numpy(image / 255.0).float()
        return image, scale

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        name0, name1 = self.pairs[idx]
        image0 = read_image(self.image_dir / name0, self.conf.grayscale)
        image1 = read_image(self.image_dir / name1, self.conf.grayscale)
        image0, scale0 = self.preprocess(image0)
        image1, scale1 = self.preprocess(image1)
        return image0, image1, scale0, scale1, names_to_pair(name0, name1)


def scale_keypoints(kpts: np.ndarray, scale: np.ndarray):
    if np.any(scale != 1.0):
        kpts *= scale.astype(kpts.dtype, copy=False)
    return kpts


class DenseMatching:
    methods = {
        'loftr': {
            'name': 'loftr',
            'max_num_matches': 2048,
            'pretrained': 'outdoor',
            'grayscale': True,
            'resize_max': 640,
        },
    }

    def __init__(self, outputs, capture, query_id, ref_id, config,
                 pair_selection: PairSelection, device=None, overwrite: bool = False):
        assert query_id == pair_selection.query_id
        assert ref_id == pair_selection.ref_id
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.config = config
        self.query_id = query_id
        self.ref_id = ref_id
        self.session_id = ref_id
        self.pair_selection = pair_selection
        self.paths = DenseMatchingPaths(outputs, config, query_id, ref_id)
        self.paths.workdir.mkdir(parents=True, exist_ok=True)
        if not same_configs(config, self.paths.config):
            overwrite = True

        pairs = parse_retrieval(pair_selection.paths.pairs_hloc)
        pairs = [(q, r) for q, rs in pairs.items() for r in rs]
        pairs = find_unique_new_pairs(pairs, None if overwrite else self.paths.matches)
        dataset = ImagePairDataset(capture.sessions_path(), config, pairs)
        logger.info('Dense matching %s with for sessions (%s, %s).',
                    config['name'], query_id, ref_id)
        if len(dataset) > 0:
            self.run(dataset)
        write_config(config, self.paths.config)

    def run(self, dataset):
        if self.config['name'] == 'loftr':
            self.matcher = LoFTR(pretrained=self.config['pretrained']).to(self.device)
            detection_noise = 1.0
        else:
            raise ValueError(self.config['name'])
        loader = torch.utils.data.DataLoader(
            dataset, num_workers=2, batch_size=1, shuffle=False, pin_memory=True)

        for data in tqdm(loader):
            image0, image1, scale0, scale1, (pair,) = data
            scale0, scale1 = scale0[0].numpy(), scale1[0].numpy()
            kpts0, kpts1 = self.match_pair(image0, image1)
            kpts0 += 0.5  # to COLMAP coordinates
            kpts1 += 0.5
            kpts0 = scale_keypoints(kpts0, scale0)
            kpts1 = scale_keypoints(kpts1, scale1)
            with h5py.File(str(self.paths.matches), 'a') as fd:
                if pair in fd:
                    del fd[pair]
                grp = fd.create_group(pair)
                dset = grp.create_dataset('keypoints0', data=kpts0)
                dset.attrs['uncertainty'] = detection_noise * scale0.mean()
                dset = grp.create_dataset('keypoints1', data=kpts1)
                dset.attrs['uncertainty'] = detection_noise * scale1.mean()

    @torch.no_grad()
    def match_pair(self, image0: torch.Tensor, image1: torch.Tensor):
        data = {"image0": image0.to(self.device), "image1": image1.to(self.device)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = self.matcher(data)
        kpts0 = pred['keypoints0'].cpu().numpy()
        kpts1 = pred['keypoints1'].cpu().numpy()
        scores = pred['confidence'].cpu().numpy()

        top_k = self.config['max_num_matches']
        if top_k is not None and len(scores) > top_k:
            keep = np.argpartition(-scores, top_k)[:top_k]
            kpts0, kpts1 = kpts0[keep], kpts1[keep]

        return kpts0, kpts1

    def get_matches_pairs(self, key_pairs: Iterator[Tuple[str]]):
        matches = []
        with h5py.File(str(self.paths.matches), 'r') as fid:
            for n0, n1 in key_pairs:
                pair = names_to_pair(str(n0), str(n1))
                k0, k1 = 'keypoints0', 'keypoints1'
                if pair not in fid:
                    pair = names_to_pair(str(n1), str(n0))
                    k0, k1 = k1, k0
                    if pair not in fid:
                        raise ValueError(f'Cannot find pair {n0} {n1} in {self.paths.matches}')
                dset = fid[pair][k0]
                keypoints0 = dset.__array__()
                noise0 = dset.attrs.get('uncertainty')
                dset = fid[pair][k1]
                keypoints1 = dset.__array__()
                noise1 = dset.attrs.get('uncertainty')
                matches.append((keypoints0, keypoints1, noise0, noise1))
        return matches
