from typing import List, Tuple, Dict, Set, Optional
import argparse
import functools
import time
import random
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map, process_map
import pycolmap

from . import logger
from .capture import Capture, KeyType, Pose
from .proc.overlap import OverlapTracer, Renderer, compute_rays
from .run_joint_refinement import SFM_OUTPUT
from .utils.frustum import frustum_intersection_multisessions
from .utils.tagging import is_session_night
from .utils.io import read_mesh


def compute_sfm_overlaps_worker(idx1_start: int, num: int, p3d_observed_sets: List[Set[int]],
                                batch_size: int, min_common: int = 5) -> np.ndarray:
    idx1_end = min(num, idx1_start+batch_size)
    nums_common_row = np.full((idx1_end-idx1_start, num), 0, np.uint16)
    for i, idx1 in enumerate(range(idx1_start, idx1_end)):
        for idx2 in range(idx1+1, num):
            n = len(p3d_observed_sets[idx1] & p3d_observed_sets[idx2])
            if n < min_common:
                continue
            nums_common_row[i, idx2] = n
    return nums_common_row


def compute_sfm_overlaps(rec: pycolmap.Reconstruction, image_ids: List[int],
                         parallel: bool = True, batch_size: int = 100):
    p3d_obs = [[p.point3D_id for p in rec.images[i].points2D if p.has_point3D()] for i in image_ids]
    p3d_observed_sets = [set(p) for p in p3d_obs]
    num_observed = np.array([len(p) for p in p3d_obs])
    num = len(image_ids)

    fn = functools.partial(
        compute_sfm_overlaps_worker, num=num,
        p3d_observed_sets=p3d_observed_sets,
        batch_size=batch_size)
    if parallel:
        map_ = functools.partial(process_map, max_workers=16, chunksize=10)
    else:
        map_ = lambda f, x: map(f, tqdm(x))
    ret = map_(fn, range(0, num, batch_size))
    nums_common = np.concatenate(ret, 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        overlaps = nums_common * (1/num_observed[None] + 1/num_observed[:, None]) / 2
    overlaps[~np.isfinite(overlaps)] = 0
    overlaps = np.maximum(overlaps, overlaps.T)
    return overlaps.astype(np.float16)


def get_pairwise_distances(T_q2w: List[Pose], T_r2w: List[Pose]):
    t_q2w = np.stack([T.t for T in T_q2w]).astype(np.float32)
    t_r2w = np.stack([T.t for T in T_r2w]).astype(np.float32)
    # equivalent to scipy.spatial.distance.cdist but supports fp32
    dt = t_q2w.dot(t_r2w.T)
    dt *= -2
    dt += np.einsum('ij,ij->i', t_q2w, t_q2w)[:, None]
    dt += np.einsum('ij,ij->i', t_r2w, t_r2w)[None]
    np.clip(dt, a_min=0, a_max=None, out=dt)  # numerical errors
    np.sqrt(dt, out=dt)
    dz = np.abs(t_q2w[:, None, 2] - t_r2w[None, :, 2])
    return dt, dz


def compute_mesh_overlaps(overlaps_sfm: np.ndarray, capture: Capture,
                          keys: List[Tuple[str, KeyType]], ref_id: str) -> np.ndarray:
    num = len(keys)
    intersections = frustum_intersection_multisessions(capture, keys, num_threads=24)
    poses = []
    for sid, (ts, cam_id) in keys:
        session = capture.sessions[sid]
        poses.append(session.get_pose(ts, cam_id, session.proc.alignment_trajectories))
    dt, dz = get_pairwise_distances(poses, poses)
    session_ids = np.array([sid for sid, _ in keys])
    different_session = session_ids[:, None] != session_ids[None, :]
    mask = intersections & (dt < 10) & (dz < 2) & (overlaps_sfm < 0.001) & different_session
    mask = mask | mask.T

    proc = capture.sessions[ref_id].proc
    mesh_path = proc.meshes.get('mesh_simplified', proc.meshes.get('mesh'))
    mesh = read_mesh(capture.proc_path(ref_id) / mesh_path)
    tracer = OverlapTracer(Renderer(mesh), num_rays=60)

    pbar = tqdm(total=np.count_nonzero(mask))
    overlaps_mesh = np.full((num,)*2, -1, np.float16)
    for i in range(num):
        sid_q, (_, id_q) = keys[i]
        session_q = capture.sessions[sid_q]
        T_q = poses[i]
        cam_q = session_q.sensors[id_q]
        rays_q = compute_rays(T_q, cam_q, stride=tracer.get_stride(cam_q))
        intersect_q = tracer.renderer.compute_intersections(rays_q)
        for j in np.where(mask[i])[0]:
            pbar.update()
            if overlaps_mesh[j, i] != -1 and overlaps_mesh[j, i] < 0.1:
                continue
            sid_r, (_, id_r) = keys[j]
            session_r = capture.sessions[sid_r]
            T_r = poses[j]
            cam_r = session_r.sensors[id_r]
            ov = tracer.compute_overlap_from_rays(rays_q, *intersect_q, T_r, cam_q, cam_r)
            if ov is not None:
                ov = np.mean(ov)
                overlaps_mesh[i, j] = ov

    overlaps_mesh = np.minimum(overlaps_mesh, overlaps_mesh.T)
    return overlaps_mesh


class MapQuerySplitObjective:
    """Minimize the overlap between map sequences while ensuring that each query sequence
       overlaps with the map by at least X% (by default 80%).
       The overlap of a map sequence to all other map sequences is the ratio of sequence images
       that are covisibible with at least one other map image of a different sequence.
       Two images are covisible if the ratio of common 3D points is larger than X% (by default 2%).
    """
    def __init__(self, sequences: List[str], sequence2imageidxs: Dict[str, List[int]],
                 overlap_mask: np.ndarray, timestamps: List[np.ndarray],
                 masks: Optional[np.ndarray] = None,
                 min_query_coverage: float = 5):
        self.overlaps_seq = {}

        def _worker_fn(idx1: int):
            seq1 = sequences[idx1]
            for idx2, seq2 in enumerate(sequences):
                if idx1 == idx2:
                    continue
                ov = overlap_mask[sequence2imageidxs[seq1]][:, sequence2imageidxs[seq2]]
                self.overlaps_seq[idx1, idx2] = np.any(ov, -1)
                if masks is not None:
                    self.overlaps_seq[idx1, idx2] = self.overlaps_seq[idx1, idx2][masks[idx1]]

        thread_map(_worker_fn, range(len(sequences)))
        self.min_query_coverage = min_query_coverage
        self.timestamps = timestamps

    def cost(self, idx, others):
        return np.stack([self.overlaps_seq[idx, s] for s in others - {idx}], -1).any(-1).mean()

    def cost_solution(self, idxs):
        return sum(self.cost(i, idxs) for i in idxs) / len(idxs)

    def is_feasible(self, queries, maps):
        for q in queries:
            is_overlap = np.stack([self.overlaps_seq[q, s] for s in maps], -1).any(-1)
            diff = np.diff(np.r_[1, is_overlap, 1])
            start = np.where(diff == -1)[0]
            end = np.where(diff == 1)[0]
            ts = self.timestamps[q]
            duration = ts[end.clip(max=len(ts)-1)] - ts[start]
            if len(duration) > 0 and np.max(duration) > self.min_query_coverage:
                return False
        return True


def solve_dfs(objective: MapQuerySplitObjective, sequences: List[str], lengths: List[int],
              timeout_min: int, keep_top_n: int = 3, shuffle_interval: int = 1_000_000,
              exclude_map: List[str] = None, initial_sol: Optional[List[str]] = None):
    """A greedy algorithm that explores the solution by space with a depth-first search.
       Stop growing a branch when the leaf solution becomes feasible.
       Heuristic: explores only a subset of most promising branches to ensure diversity.
       Here most promising = lowest overlap with the current map.
       Stops after a given number of minutes - the solution is then usually quickly good enough.
    """
    exclude_map = set(exclude_map or [])
    exclude_map = {i for i, s in enumerate(sequences) if s in exclude_map}

    best_sols = [set()]
    best_scores = [np.inf]
    sequence_set = set(range(len(sequences)))

    if initial_sol is None:
        stack = [{s} for s in sorted(sequence_set-exclude_map, key=lambda i: lengths[i])]
    else:
        initial_sol = set(initial_sol)
        stack = [{i for i, s in enumerate(sequences) if s in initial_sol}]

    start_time = time.time()
    visited = set()
    it = 0
    pbar = tqdm()
    while len(stack) > 0:
        pbar.update(it)
        if (it % 1000) == 0:
            pbar.set_postfix(
                best=best_scores[-1], solution_size=len(best_sols[-1]),
                stack_size=len(stack), solutions=len(best_sols))
            if (time.time() - start_time) > (timeout_min * 60):
                break
        it += 1
        if (it % shuffle_interval) == 0:
            random.Random(it).shuffle(stack)

        sol = stack.pop()
        visited.add(frozenset(sol))
        qs = sequence_set - sol - exclude_map
        if objective.is_feasible(qs, sol):
            score = objective.cost_solution(sol)
            if score < best_scores[-1]:
                best_scores.append(score)
                best_sols.append(sol)
        else:
            # sort by score and sequence length
            scores = [(objective.cost(q, sol), -lengths[q], q) for q in qs]
            for score, _, q in sorted(scores)[:keep_top_n]:
                candidate = sol | {q}
                if candidate not in visited:
                    stack.append(candidate)

    return best_sols, best_scores


def split_map_queries_for_evaluation(session_ids: List[str],
                                     capture: Capture,
                                     rec: pycolmap.Reconstruction,
                                     overlaps_path: Path = None,
                                     ref_id: Optional[str] = None,
                                     timeout_min: int = 10,
                                     no_night_in_map: bool = True,
                                     exclude_from_map: List[str] = None,
                                     include_in_map: List[str] = None,
                                     min_overlap_score: Tuple[float] = (0.01, 0.15),
                                     **objective_args):
    if not isinstance(rec, pycolmap.Reconstruction):
        rec = pycolmap.Reconstruction(rec)
    session_ids = sorted(session_ids)

    image_ids = []
    sequence2imageids = defaultdict(list)
    for i, image in rec.images.items():
        sid = image.name.split('/')[0]
        image_ids.append(i)
        sequence2imageids[sid].append(i)
    name2key = {}
    for sid in sequence2imageids:
        prefix = capture.data_path(sid).relative_to(capture.sessions_path())
        images = capture.sessions[sid].images
        for ts, cam_id in images.key_pairs():
            name2key[str(prefix / images[ts, cam_id])] = (sid, (ts, cam_id))

    overlaps = None
    if overlaps_path is not None and overlaps_path.exists():
        data = np.load(str(overlaps_path))
        overlaps = (data['overlaps_sfm'], data['overlaps_mesh'])
        image_ids = data['image_ids']
    else:
        logger.info('Computing overlaps via SfM covisibility and Mesh tracing.')
        image_ids = sorted(image_ids)
        overlaps_sfm = compute_sfm_overlaps(rec, image_ids)
        keys = [name2key[rec.images[i].name] for i in image_ids]
        overlaps_mesh = compute_mesh_overlaps(overlaps_sfm, capture, keys, ref_id)
        overlaps = (overlaps_sfm, overlaps_mesh)
        if overlaps_path is not None:
            np.savez(
                str(overlaps_path),
                overlaps_sfm=overlaps_sfm,
                overlaps_mesh=overlaps_mesh,
                image_ids=image_ids)
    overlap_mask = (overlaps[0] > min_overlap_score[0]) | (overlaps[1] > min_overlap_score[1])
    del overlaps

    imageid2idx = {i: idx for idx, i in enumerate(image_ids)}
    sequence2imageidxs = {sid: np.sort([imageid2idx[i] for i in sequence2imageids[sid]])
                          for sid in session_ids}

    # Recover the image timestamps
    timestamps = []
    for sid in session_ids:
        ts = np.array([name2key[rec.images[image_ids[i]].name][1][0]
                       for i in sequence2imageidxs[sid]])
        assert np.all(np.diff(ts) >= 0), sid
        timestamps.append(ts / 1e6)

    exclude_from_map = exclude_from_map or []
    if no_night_in_map:
        exclude_from_map = list(set(exclude_from_map) | set(filter(is_session_night, session_ids)))

    logger.info('Starting to solve the problem...')
    lengths = [len(sequence2imageids[i]) for i in session_ids]
    objective = MapQuerySplitObjective(
        session_ids, sequence2imageidxs, overlap_mask, timestamps, **objective_args)
    best_solutions, best_scores = solve_dfs(
        objective, session_ids, lengths, timeout_min,
        exclude_map=exclude_from_map, initial_sol=include_in_map)

    score = best_scores[-1]
    sessions_map = sorted(session_ids[i] for i in best_solutions[-1])
    sessions_query = sorted(set(session_ids) - set(sessions_map))
    logger.info('Found solution with score %s and map/query split %d/%d',
                score, len(sessions_map), len(sessions_query))

    other_solutions = [[session_ids[i] for i in s] for s in best_solutions[-5:-1]]
    image_ids_map = [i for s in sessions_map for i in sequence2imageids[s]]
    image_ids_query = [i for s in sessions_query for i in sequence2imageids[s]]
    debug = (image_ids_map, image_ids_query, other_solutions)

    return sessions_map, sessions_query, debug


def run(capture: Capture,
        session_ids: List[str],
        split_filename: Optional[str] = None,
        overwrite: bool = False,
        ref_id: Optional[str] = None,
        **objective_args):

    if split_filename is None:
        split_filename = 'query_map_split.json'
    split_path = capture.path / split_filename
    if split_path.exists():
        with open(split_path, 'r') as fid:
            split = json.load(fid)
        map_ids, query_ids = split['map'], split['query']
        if set(session_ids) != set(map_ids+query_ids):
            logger.warning('Incompatible split file %s, will recompute the split.', split_path)
        elif not overwrite:
            return map_ids, query_ids

    reconstruction = capture.registration_path() / SFM_OUTPUT
    overlaps = reconstruction.parent / 'overlaps.npz'
    map_ids, query_ids, _ = split_map_queries_for_evaluation(
        session_ids, capture, reconstruction, overlaps, ref_id, **objective_args)

    logger.info('Writing map/query split to %s.', split_path)
    with open(split_path, 'w') as fid:
        json.dump({'map': map_ids, 'query': query_ids}, fid, indent=4)

    return map_ids, query_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--capture_path', type=Path, required=True)
    parser.add_argument('--session_ids', type=str, nargs='+', required=True)
    parser.add_argument('--split_filename', type=str)
    parser.add_argument('--ref_id', type=str)
    args = parser.parse_args().__dict__
    args['capture'] = Capture.load(args.pop('capture_path'))
    run(**args)
