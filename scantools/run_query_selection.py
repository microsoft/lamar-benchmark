from typing import List, Tuple, Optional
import argparse
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

from . import logger
from .capture import Capture, Trajectories, KeyType
from .proc.rendering import Renderer
from .viz.qualitymap import LidarMapPlotter, plot_legend
from .utils.tagging import is_session_night


def filter_queries_by_visibility(capture: Capture, ref_id: str, queries: List[Tuple[str, KeyType]],
                                 trajectory: Trajectories, max_distance: float = 10):
    ''' Only retain queries that are in the direct line-of-sight of a scan image
        that is less than Xm away. This is much cheaper than explicitly computing
        the mesh overlap but nevertheless a good approximation of it,
        since scans cover 360 degrees.
    '''
    session_ref = capture.sessions[ref_id]
    keys_ref = sorted(session_ref.trajectories.key_pairs())
    centers_ref = np.array([session_ref.trajectories[k].t for k in keys_ref])
    centers = np.stack([trajectory[k].t for _, k in queries])
    renderer = Renderer(capture.proc_path(ref_id) / session_ref.proc.meshes['mesh_simplified'])

    directions = (centers_ref[None] - centers[:, None]).reshape(-1, 3)
    directions = np.ascontiguousarray(directions, dtype=np.float32)
    origins = np.ascontiguousarray(np.repeat(centers, len(centers_ref), axis=0), dtype=np.float32)
    intersections, intersected = renderer.compute_intersections((origins, directions))

    dist_intersect = np.linalg.norm(intersections - origins[intersected], axis=-1)
    dist_centers = np.linalg.norm(directions, axis=-1)
    visible = ~intersected
    visible[intersected] = dist_intersect > dist_centers[intersected]
    visible &= dist_centers < max_distance
    visible = visible.reshape(len(centers), len(centers_ref))
    query_is_in_area = visible.any(1)
    selected = [queries[i] for i in np.where(query_is_in_area)[0]]
    return selected


def farthest_point_sampling(points: np.ndarray, num_samples: int,
                            session_ids: Optional[List[str]] = None,
                            seed: int = 0, count_multiplier: float = 1.2,
                            max_ratio_night: float = 1/4):
    '''Sample a given number of points using farthest point sampling, which ensure a good spatial
       coverage of the entire scene. Preserve the original distribution by capping the number of
       queries sampled from each sequence.
    '''
    assert num_samples <= len(points)
    dist = np.full(len(points), np.inf)
    selected_indices = np.empty(num_samples, dtype=np.int64)
    if session_ids is not None:
        count = Counter()
        count_all = Counter(session_ids)
        count_night = {k: v for k, v in count_all.items() if is_session_night(k)}
        num_night = sum(count_night.values())
        num_day = len(session_ids) - num_night
        ratio_night = min(max_ratio_night, num_night/sum(count_all.values()))
        ratio_day = 1 - ratio_night
        target_counts = {k: int(count_multiplier*num_samples*v
                                *(ratio_night/num_night if k in count_night else ratio_day/num_day))
                         for k, v in count_all.items()}
        print(target_counts, ratio_night)
    for i in range(num_samples):
        if i == 0:
            idx = np.random.RandomState(seed).choice(len(points))
        else:
            idx = np.argmax(dist)
            assert dist[idx] != 0
        selected_indices[i] = idx
        np.minimum(dist, np.linalg.norm(points - points[idx], axis=1), out=dist)
        if session_ids is not None:
            sid = session_ids[idx]
            count[sid] += 1
            if count[sid] > target_counts[sid]:
                for idx2, sid2 in enumerate(session_ids):
                    if sid2 == sid:
                        dist[idx2] = 0
    return selected_indices


def sample_queries(queries: List[Tuple[str, KeyType]], trajectory: Trajectories, num_samples: int,
                   weight_rotation: float = 2):
    '''Sample queries to uniformly cover the scene in terms of position and orientation.
       Do the search if the 6D pose space, effectively augmenting spatial distances
       with 2*(1-cos(angle)).
    '''
    centers = np.stack([trajectory[k].t for _, k in queries])
    view_directions = np.stack([trajectory[k].R[:, -1] for _, k in queries])
    pose_vect = np.concatenate([centers, view_directions*weight_rotation], 1)
    session_ids = [i for i, _ in queries]
    indices = farthest_point_sampling(pose_vect, num_samples, session_ids=session_ids)
    selected = [queries[i] for i in indices]
    return selected


def plot_queries(capture, ref_id, trajectory, q_all, q_pool, q_filtered, q_selected):
    centers_all = np.stack([trajectory[k].t for _, k in q_all])
    centers_pool = np.stack([trajectory[k].t for _, k in q_pool])
    centers_filtered = np.stack([trajectory[k].t for _, k in q_filtered])
    centers_selected = np.stack([trajectory[k].t for _, k in q_selected])
    plotter = LidarMapPlotter.from_scan_session(capture, ref_id, centers_all)
    ax = plotter.plot_2d()
    ax.scatter(*centers_all.T[plotter.masks[0]], c='k', label='all', s=0.5, linewidth=0)
    ax.scatter(*centers_pool.T[plotter.masks[0]], c='r', label='pool', s=1, linewidth=0)
    ax.scatter(*centers_filtered.T[plotter.masks[0]], c='b', label='filtered', s=1.5, linewidth=0)
    ax.scatter(*centers_selected.T[plotter.masks[0]], c='lime', label='selected', s=2, linewidth=0)
    plot_legend(ax, prop={'size': 6})


def run(capture: Capture,
        session_id: str,
        num_queries: int,
        max_chunk_size_us: int,
        ref_id: str,
        max_uncertainty_meter: float = 0.0333,
        query_filename: str = 'queries.txt',
        visualize: bool = False):
    assert session_id in capture.sessions, session_id
    output_path = capture.session_path(session_id) / query_filename
    # assert not output_path.exists()

    session = capture.sessions[session_id]
    if session.proc.subsessions:
        prefixes = [f'{subsession}/' for subsession in session.proc.subsessions]
    else:
        prefixes = ['']

    trajectory = session.proc.alignment_trajectories
    keys = sorted(trajectory.key_pairs())
    covariances = []
    has_valid_covariance = []
    for k in keys:
        covar = trajectory[k].covar
        has_valid_covariance.append(covar is not None)
        covariances.append(np.eye(6) if covar is None else covar)
    uncertainty = np.sqrt(np.linalg.eig(np.stack(covariances))[0].max(1))
    uncertainty_dict = dict(zip(keys, np.where(has_valid_covariance, uncertainty, np.nan)))
    if max_uncertainty_meter:
        certain = uncertainty[has_valid_covariance] <= max_uncertainty_meter
        invalid = ~np.array(has_valid_covariance)
        logger.info('Session %s: %d/%d (%.2f%%) poses with uncertainty < %.2fcm'
                    ', %d (%.2f%%) poses without covariance.',
                    session_id, np.count_nonzero(certain), len(certain), np.mean(certain)*100,
                    max_uncertainty_meter*100, np.count_nonzero(invalid), 100*np.mean(invalid))

    # First select query candidates in each sequence
    queries_pool = []
    for prefix in prefixes:
        # Recover all camera timestamps from current session.
        keys_select = []
        start_timestamp = None
        for ts, sensor_id in keys:
            if not sensor_id.startswith(prefix):
                continue
            if start_timestamp is None:
                start_timestamp = ts
            if max_uncertainty_meter is not None:
                unc = uncertainty_dict[ts, sensor_id]
                if np.isfinite(unc) and unc > max_uncertainty_meter:
                    continue
            if ts > start_timestamp + max_chunk_size_us:
                keys_select.append((ts, sensor_id))
        queries_pool.extend([(prefix, k) for k in keys_select])

    # find out-of-area queries
    queries_filtered = filter_queries_by_visibility(capture, ref_id, queries_pool, trajectory)

    # sample based on density
    if len(queries_filtered) > num_queries:
        queries = sample_queries(queries_filtered, trajectory, num_queries)
    else:
        queries = queries_filtered
    is_query_night = [is_session_night(i) for i, _ in queries]
    logger.info('Selected %d queries with %.1f%% night.', len(queries), np.mean(is_query_night)*100)

    if visualize:
        plot_queries(capture, ref_id, trajectory,
                     [(None, k) for k in keys], queries_pool, queries_filtered, queries)
        plt.savefig(capture.session_path(session_id) / f'{session_id}_{query_filename}.pdf')
        plt.close()

    # Save queries to disk, one query per line.
    queries = sorted([q for _, q in queries])
    with open(output_path, 'w') as f:
        for ts, sensor_id in queries:
            f.write(f'{ts}, {sensor_id}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--capture_path', type=Path, required=True)
    parser.add_argument('--session_id', type=str, required=True)
    parser.add_argument('--num_queries', type=float, default=1_000)
    parser.add_argument('--max_chunk_size_us', type=float, default=20_000_000)
    parser.add_argument('--ref_id', type=str, required=True)
    args = parser.parse_args().__dict__
    args['capture'] = Capture.load(args.pop('capture_path'), session_ids=[args['session_id']])
    run(**args)
