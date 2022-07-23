import logging

import numpy as np

logger = logging.getLogger(__name__)


def list_images_for_query_session(capture, session_id, query_keys):
    # Here we assume that query sessions are rigs only or non-rigs only.
    session = capture.sessions[session_id]
    image_root = capture.sessions_path()
    image_prefix = capture.data_path(session_id).relative_to(image_root)
    rigs = session.rigs is not None
    keys = []
    image_names = []
    for ts, sensor_id in query_keys:
        if rigs:
            for camera_id in session.rigs[sensor_id]:
                keys.append((ts, camera_id))
                image_names.append(str(image_prefix / session.images[ts, camera_id]))
        else:
            keys.append((ts, sensor_id))
            image_names.append(str(image_prefix / session.images[ts, sensor_id]))
    return keys, image_names, image_root


def list_images_for_session(capture, session_id, query_keys=None):
    if query_keys:
        return list_images_for_query_session(capture, session_id, query_keys)
    session = capture.sessions[session_id]
    image_root = capture.sessions_path()
    image_prefix = capture.data_path(session_id).relative_to(image_root)
    keys = sorted(session.images.key_pairs())
    image_names = [str(image_prefix / session.images[k]) for k in keys]
    # cams = [k[1] for k in keys]
    return keys, image_names, image_root


def list_trajectory_keys_for_session(capture, session_id, query_keys=None):
    if query_keys:
        return query_keys
    return capture.session[session_id].trajectories.key_pairs()


def read_query_list(path):
    queries = []
    with open(path, 'r') as fid:
        for line in fid.readlines():
            ts, sensor_id = line.strip('\n').split(', ')
            ts = int(ts)
            queries.append((ts, sensor_id))
    return queries


def build_chunks(capture, session_id, query_keys, chunk_length_s):
    session = capture.sessions[session_id]
    trajectory = session.trajectories
    if session.proc.subsessions:
        prefixes = [f'{subsession}/' for subsession in session.proc.subsessions]
    else:
        prefixes = ['']
    queries = []
    keys = []
    for prefix in prefixes:
        # Recover all camera timestamps from current session.
        timestamps = set()
        ts_to_sensor_id = {}
        for ts, sensor_id in trajectory.key_pairs():
            if sensor_id.startswith(prefix):
                timestamps.add(ts)
                ts_to_sensor_id[ts] = sensor_id
        timestamps = np.array(list(sorted(timestamps)))
        current_query_keys = [k for k in query_keys if k[1].startswith(prefix)]
        queries.extend(current_query_keys)
        for qts, qsensor_id in current_query_keys:
            assert qsensor_id == ts_to_sensor_id[qts]
            chunk_keys = []
            idx = np.searchsorted(
                timestamps, qts - chunk_length_s * 1_000_000, side='right')
            for ts in timestamps[max(0, idx - 1) :]:
                ts = int(ts)
                if ts > qts:
                    break
                sensor_id = ts_to_sensor_id[ts]
                chunk_keys.append((ts, ts_to_sensor_id[ts]))
            keys.append(chunk_keys)
    lengths_s = []
    for chunk_keys in keys:
        lengths_s.append(
            (chunk_keys[-1][0] - chunk_keys[0][0]) / 1_000_000)
    logging.info('Built %d chunks with mean/med/min/q1/q9/max '
                 'durations %.2f/%.2f/%.2f/%.2f/%.2f/%.2fs.',
                 len(lengths_s), np.mean(lengths_s), np.median(lengths_s),
                 np.min(lengths_s), np.quantile(lengths_s, 0.1),
                 np.quantile(lengths_s, 0.9), np.max(lengths_s))
    return queries, keys


def avoid_duplicate_keys_in_chunks(session, query_list, query_chunks):
    def obfuscate_sensor_id(sensor_id, chunk_idx):
        return f'{sensor_id}$seq_{chunk_idx}'
    final_query_list = []
    for chunk_idx, key in enumerate(query_list):
        ts, sensor_id = key
        new_sensor_id = obfuscate_sensor_id(sensor_id, chunk_idx)
        final_query_list.append((ts, new_sensor_id))
    final_query_chunks = []
    for chunk_idx, query_chunk in enumerate(query_chunks):
        final_query_chunk = []
        for key in query_chunk:
            ts, sensor_id = key
            new_sensor_id = obfuscate_sensor_id(sensor_id, chunk_idx)
            new_key = (ts, new_sensor_id)
            final_query_chunk.append(new_key)
            session.trajectories[new_key] = session.trajectories[key]
            if session.rigs:
                session.rigs[new_sensor_id] = {}
                for cam_id in session.rigs[sensor_id]:
                    new_cam_id = obfuscate_sensor_id(cam_id, chunk_idx)
                    session.rigs[new_sensor_id][new_cam_id] = session.rigs[sensor_id][cam_id]
                    session.sensors[new_cam_id] = session.sensors[cam_id]
                    session.images[ts, new_cam_id] = session.images[ts, cam_id]
            else:
                session.images[new_key] = session.images[key]
                session.sensors[new_sensor_id] = session.sensors[sensor_id]
        final_query_chunks.append(final_query_chunk)
    return final_query_list, final_query_chunks
