import argparse
from pathlib import Path
from typing import Optional, List, Dict
from collections import Counter, defaultdict
import functools
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
import numpy as np

from . import logger
from .capture import Capture, Session, KeyType
from .proc.anonymization import BaseAnonymizer, EgoBlurAnonymizer, BrighterAIAnonymizer
from .viz.image import plot_images


def plot_blurring(blurred: np.ndarray, mask: np.ndarray, detections: List):
    if len(detections) == 0:
        return
    patches = []
    titles = []
    hi, wi = blurred.shape[:2]
    for detection in detections:
        x1, y1, x2, y2 = list(map(int, detection.bounding_box))
        h, w = x2-x1, y2-y1
        c = np.array([x1+w//2, y1+h//2])
        s = int(max(h, w) * 2.5) // 2
        px1, py1 = np.maximum(c - s, 0)
        px2, py2 = np.minimum(c + s + 1, np.array([wi, hi]))
        sli = np.s_[py1:py2, px1:px2]
        patches.extend([blurred[sli], mask[sli]])
        titles.extend(['', f'{detection.score:.3f}'])
    plot_images(patches, titles=titles)


def blur_image_group(capture: Capture, session: Session, keys: List[KeyType],
                     tmp_dir: Path, anonymizer: BaseAnonymizer,
                     output_path: Optional[Path] = None) -> int:
    subpaths = [session.images[key] for key in sorted(keys)]
    tmp_dir.mkdir(exist_ok=True, parents=True)
    (tmp_dir / 'list.txt').write_text('\n'.join(subpaths))
    input_paths = [capture.data_path(session.id) / s for s in subpaths]
    output_paths = None if output_path is None else [output_path / s for s in subpaths]
    return anonymizer.blur_image_group(input_paths, tmp_dir, output_paths)


def split_sequences(groups: Dict[str, List[KeyType]], session: Session,
                    capture: Capture, sequential: bool) -> Dict[str, List[KeyType]]:
    # Split the keys by camera if sequential mode
    if sequential and session.device != session.Device.PHONE:
        groups_camera = defaultdict(list)
        for id_, keys in groups.items():
            for k in keys:
                _, sensor_id = k
                groups_camera[f'{id_}/{sensor_id}'].append(k)
        groups = dict(groups_camera)
    # Make sure that each group of images is smaller than the maximum size (2GB)
    groups_out = dict()
    for id_ in list(groups):
        group_index = 0
        group_size = 0
        group_keys = []
        image_size = None
        for k in sorted(groups[id_]):
            # We also catch changes in image orientation due to the gravity correction.
            rotated = False
            if session.device == session.Device.PHONE:
                image_size_ = tuple(session.sensors[k[1]].size)
                if image_size is not None and image_size != image_size_:
                    rotated = True
                image_size = image_size_

            size = (capture.data_path(session.id) / session.images[k]).stat().st_size
            if rotated or (group_size + size) > 1.8e9:  # 1.8GB
                groups_out[f'{id_}/{group_index}'] = group_keys
                group_size = 0
                group_keys = []
                group_index += 1
            group_size += size
            group_keys.append(k)
        groups_out[f'{id_}/{group_index}'] = group_keys
    return groups_out


def run(capture: Capture, session_id: str, apikey: Optional[str] = None,
        output_path: Optional[Path] = None, num_parallel: int = 16,
        sequential: bool = False, device: str = None):
    session = capture.sessions[session_id]
    if session.images is None:
        return

    inplace = output_path is None
    if inplace:
        logger.info('Will run image anonymization in place.')

    if apikey is None:
        if sequential:
            raise ValueError('Sequential mode is not supported by Ego Blur.')
        anonymizer = EgoBlurAnonymizer(device=device)
        num_parallel = 1
        anon_dirname = 'anonymization_egoblur'
    else:
        anonymizer = BrighterAIAnonymizer(apikey, single_frame_optimized=not sequential)
        anon_dirname = 'anonymization_brighterai'

    all_keys = list(session.images.key_pairs())
    # Split the keys by subsequence.
    key_groups = {}
    if session.proc is None or session.proc.subsessions is None:
        key_groups[session_id] = all_keys
    else:
        for subid in session.proc.subsessions:
            key_groups[subid] = []
            for k in all_keys:
                if k[1].startswith(subid):
                    key_groups[subid].append(k)
    if not isinstance(anonymizer, EgoBlurAnonymizer):
        key_groups = split_sequences(key_groups, session, capture, sequential)
    assert len(all_keys) == sum(map(len, key_groups.values()))

    worker_args = []
    for id_, keys in key_groups.items():
        if len(keys) == 0:
            continue
        tmp_dir = capture.path / anon_dirname / session_id / id_
        worker_args.append((keys, tmp_dir))

    def _worker_fn(_args):
        _keys, _tmp_dir = _args
        return blur_image_group(
            capture, session, _keys, _tmp_dir, anonymizer, output_path)

    if num_parallel > 1 and len(key_groups) > 1:
        map_ = functools.partial(thread_map, max_workers=num_parallel)
    elif len(key_groups) > 1:
        map_ = lambda f, x: list(map(f, tqdm(x)))
    else:
        map_ = lambda f, x: list(map(f, x))
    counts = map_(_worker_fn, worker_args)
    counter = Counter()
    for c, a in zip(counts, worker_args):
        if c is None:
            logger.warning('Anynonymization failed for %s', a[1])
            continue
        counter += c
    logger.info('Detected %s in %d images.', str(dict(counter)), len(all_keys))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     argument_default=argparse.SUPPRESS)
    parser.add_argument('--capture_path', type=Path, required=True)
    parser.add_argument('--session_id', type=str, required=True)
    parser.add_argument('--apikey', type=str)
    parser.add_argument('--sequential', action='store_true')
    parser.add_argument('--output_path', type=Path)
    args = parser.parse_args().__dict__
    args['capture'] = Capture.load(args.pop('capture_path'), session_ids=[args['session_id']])
    run(**args)
