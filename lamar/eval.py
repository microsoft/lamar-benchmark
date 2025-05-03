import argparse
from pathlib import Path
from typing import Optional
from pprint import pformat

from scantools.capture import Pose

from . import logger

import numpy as np


CSV_COMMENT_CHAR = '#'
THRESHOLDS_DEG_M = [(1, 0.1), (5, 1.)]


def read_csv(path: Path, expected_columns: Optional[list[str]] = None) -> list[list[str]]:
    if not path.exists():
        raise IOError(f'CSV file does not exsit: {path}')

    data = []
    check_header = expected_columns is not None
    with open(str(path), 'r') as fid:
        for line in fid:
            line = line.strip()
            if len(line) == 0:
                continue
            if line[0] == CSV_COMMENT_CHAR:
                if check_header and len(data) == 0:
                    columns = [w.strip() for w in line[1:].split(',')]
                    if columns != expected_columns:
                        raise ValueError(
                            f'Got CSV columns {columns} but expected {expected_columns}.')
                check_header = False
            else:
                words = [w.strip() for w in line.split(',')]
                data.append(words)
    return data


Trajectories = dict[tuple[int, str], Pose]
def load_trajectories(path: Path) -> Trajectories:
    table = read_csv(path)
    trajectories = Trajectories()
    for timestamp, device_id, *qt in table:
        timestamp = int(timestamp)
        trajectories[timestamp, device_id] = Pose.from_list(qt)
    return trajectories


Rigs = dict[str, dict[str, Pose]]
def load_rigs(path: Path) -> Optional[Rigs]:
    if not path.exists():
        return None
    table = read_csv(path)
    rigs = Rigs()
    for rig_id, sensor_id, *qt in table:
        if rig_id not in rigs:
            rigs[rig_id] = {}
        rigs[rig_id][sensor_id] = Pose.from_list(qt)
    return rigs


def compute_pose_errors(query_keys: list, T_c2w: Trajectories, T_c2w_gt: Trajectories) -> tuple[np.ndarray, np.ndarray]:
    err_r, err_t = [], []
    for key in query_keys:
        if key in T_c2w:
            dr, dt = (T_c2w[key].inverse() * T_c2w_gt[key]).magnitude()
        else:
            dr = np.inf
            dt = np.inf
        err_r.append(dr)
        err_t.append(dt)
    err_r = np.stack(err_r)
    err_t = np.stack(err_t)
    return err_r, err_t


def read_query_list(path:Path) -> list[tuple[int, str]]:
    queries = []
    with open(path, 'r') as fid:
        for line in fid.readlines():
            ts, sensor_id = line.strip('\n').split(', ')
            ts = int(ts)
            queries.append((ts, sensor_id))
    return queries


def convert_poses_for_eval(T_c2w_sensor: Trajectories, rigs: Optional[Rigs]) -> Trajectories:
    if rigs is None:
        return T_c2w_sensor
    T_c2w_image = Trajectories()
    for ts, rig_id in T_c2w_sensor:
        for cam_id in rigs[rig_id]:
            T_c2w_image[ts, cam_id] = T_c2w_sensor[ts, rig_id] * rigs[rig_id][cam_id]
    return T_c2w_image


def run(gt_dir: Path, scene: str, query_id: str, eval_file: Path):
    logger.info('Processing scene %s, query_id %s', scene, query_id)
    query_keys = read_query_list(gt_dir / f'{scene}_{query_id}_list.txt')
    logger.info('Loaded %d queries', len(query_keys))

    rigs_file = gt_dir / f'{scene}_{query_id}_rigs.txt'
    rigs = load_rigs(rigs_file)
        
    T_c2w_gt = load_trajectories(gt_dir / f'{scene}_{query_id}.txt')
    T_c2w_gt_filtered = Trajectories()
    for key in query_keys:
        T_c2w_gt_filtered[key] = T_c2w_gt[key]
    T_c2w_gt = T_c2w_gt_filtered
    T_c2w_gt = convert_poses_for_eval(T_c2w_gt, rigs)

    T_c2w = load_trajectories(eval_file)
    T_c2w = convert_poses_for_eval(T_c2w, rigs)

    image_query_keys = list(T_c2w_gt.keys())
    err_r, err_t = compute_pose_errors(image_query_keys, T_c2w, T_c2w_gt)
    recalls = [np.mean((err_r < th_r) & (err_t < th_t)) for th_r, th_t in THRESHOLDS_DEG_M]
    return {'recall': recalls, 'Rt_thresholds': THRESHOLDS_DEG_M}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--scene', type=str, required=True)
    parser.add_argument('--query_id', type=str, required=True)
    parser.add_argument('--gt_dir', type=Path, required=True, help="Path to directory containing GT")
    parser.add_argument('--eval_file', type=Path, required=True, help="Path to evaluation file")
    args = parser.parse_args().__dict__

    results = run(**args)
    logger.info('Results:\n%s', pformat(results))
