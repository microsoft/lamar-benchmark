from typing import Optional, List
from pathlib import Path
import os.path as osp
import argparse

from . import logger
from .capture import (
    Capture, Session, Proc, Sensors, Trajectories, Rigs,
    RecordsCamera, RecordsDepth, RecordsLidar, RecordsWifi, RecordsBluetooth)
from .proc.alignment.image_matching import KeyFramingConf, subsample_poses


def prefixed_id(sensor_or_rig_id, session_id):
    return f'{session_id}/{sensor_or_rig_id}'

def copy_session(session: Session, session_id: str, dst: Session,
                 overwrite_poses: bool, keyframing: KeyFramingConf):

    aligned = True
    poses = session.proc.alignment_trajectories
    if poses is None:
        logger.warning('No alignment for session %s.', session_id)
        poses = session.trajectories
        aligned = False
    pose_keys = poses.key_pairs()
    if keyframing is not None:
        pose_keys = subsample_poses(pose_keys, poses, keyframing)

    # Cache sensor and rig ids of the selected poses
    selected_data_keys = set(pose_keys)
    for ts, sensor_or_rig_id in pose_keys:
        if sensor_or_rig_id not in session.sensors:
            assert session.rigs is not None
            assert sensor_or_rig_id in session.rigs
            for sensor_id in session.rigs[sensor_or_rig_id]:
                selected_data_keys.add((ts, sensor_id))
    if session.pointclouds is not None:
        selected_data_keys |= set(session.pointclouds.key_pairs())
    selected_sensor_and_rig_ids = {i for _, i in selected_data_keys}
    if session.wifi is not None:
        selected_sensor_and_rig_ids |= {i for _, i in session.wifi.key_pairs()}
    if session.bt is not None:
        selected_sensor_and_rig_ids |= {i for _, i in session.bt.key_pairs()}
    new_ids = {i: prefixed_id(i, session_id) for i in selected_sensor_and_rig_ids}

    # Copy trajectories
    if dst.trajectories is None:
        dst.trajectories = Trajectories()
    for ts, sensor_or_rig_id in pose_keys:
        if overwrite_poses:
            pose = poses[ts, sensor_or_rig_id]
        else:
            pose = session.trajectories[ts, sensor_or_rig_id]
        dst.trajectories[ts, new_ids[sensor_or_rig_id]] = pose

    # Copy ground truth trajectories
    if dst.proc is None:
        dst.proc = Proc()
    if not overwrite_poses and aligned:
        if dst.proc.alignment_trajectories is None:
            dst.proc.alignment_trajectories = Trajectories()
        for ts, sensor_or_rig_id in pose_keys:
            pose = poses[ts, sensor_or_rig_id]
            dst.proc.alignment_trajectories[ts, new_ids[sensor_or_rig_id]] = pose

    # Copy sensors
    for sensor_id, sensor in session.sensors.items():
        if sensor_id not in selected_sensor_and_rig_ids:
            continue
        # Here we don't try to update the sensor name
        dst.sensors[new_ids[sensor_id]] = sensor

    # Copy rigs
    if session.rigs is not None:
        if dst.rigs is None:
            dst.rigs = Rigs()
        for rig_id, sensor_id in session.rigs.key_pairs():
            if rig_id not in selected_sensor_and_rig_ids:
                continue
            pose = session.rigs[rig_id, sensor_id]
            dst.rigs[new_ids[rig_id], new_ids[sensor_id]] = pose

    # Copy images
    if dst.images is None:
        dst.images = RecordsCamera()
    for ts, sensor_id in session.images.key_pairs():
        if (ts, sensor_id) not in selected_data_keys:
            continue
        rpath = Path(session_id, session.images[ts, sensor_id])
        dst.images[ts, new_ids[sensor_id]] = str(rpath)

    # Copy depth
    if session.depths is not None:
        if dst.depths is None:
            dst.depths = RecordsDepth()
        for ts, sensor_id in session.depths.key_pairs():
            rpath = Path(session_id, session.depths[ts, sensor_id])
            if (ts, sensor_id) in selected_data_keys:
                new_sensor_id = new_ids[sensor_id]
            else:
                depth_prefix, *sensor_id = sensor_id.split('/')
                sensor_id = '/'.join(sensor_id)
                if (ts, sensor_id) in selected_data_keys:
                    new_sensor_id = depth_prefix + '/' + new_ids[sensor_id]
                else:
                    continue
            dst.depths[ts, new_sensor_id] = str(rpath)

    # Copy pointclouds
    if session.pointclouds is not None:
        if dst.pointclouds is None:
            dst.pointclouds = RecordsLidar()
        for ts, sensor_id in session.pointclouds.key_pairs():
            rpath = Path(session_id, session.pointclouds[ts, sensor_id])
            dst.pointclouds[ts, new_ids[sensor_id]] = str(rpath)

    # Copy radios
    if session.wifi is not None:
        if dst.wifi is None:
            dst.wifi = RecordsWifi()
        for ts, sensor_id in session.wifi.key_pairs():
            dst.wifi[ts, new_ids[sensor_id]] = session.wifi[ts, sensor_id]
    if session.bt is not None:
        if dst.bt is None:
            dst.bt = RecordsBluetooth()
        for ts, sensor_id in session.bt.key_pairs():
            dst.bt[ts, new_ids[sensor_id]] = session.bt[ts, sensor_id]


def run(capture: Capture,
        session_ids: List[str],
        output_id: Optional[str] = None,
        overwrite_poses: bool = False,
        keyframing: Optional[KeyFramingConf] = None,
        reference_id: Optional[str] = None):

    session_ids = sorted(session_ids)
    for session_id in session_ids:
        if session_id not in capture.sessions:
            raise ValueError(f'Unknown session {session_id}')
    if output_id is None:
        output_id = '+'.join(session_ids)

    session_new = Session(Sensors(), proc=Proc())
    session_new.proc.subsessions = []
    for session_id in session_ids:
        session = capture.sessions[session_id]
        copy_session(session, session_id, session_new, overwrite_poses, keyframing)

        if session.proc.subsessions:
            subsessions = ['/'.join((session_id, i)) for i in session.proc.subsessions]
        else:
            subsessions = [session_id]
        session_new.proc.subsessions += subsessions

    logger.info('Combined %d sequences into %s, with %d images',
                len(session_ids), output_id, len(session_new.images.key_pairs()))
    capture.sessions[output_id] = session_new
    capture.save(capture.path, session_ids=[output_id])

    output_data_path = capture.data_path(output_id)
    output_data_path.mkdir(parents=True, exist_ok=True)
    raw_data_paths = []
    for k in session_new.images.key_pairs():
        raw_data_paths.append(session_new.images[k])
    if session_new.depths is not None:
        for k in session_new.depths.key_pairs():
            raw_data_paths.append(session_new.depths[k])
    if session_new.pointclouds is not None:
        for k in session_new.pointclouds.key_pairs():
            raw_data_paths.append(session_new.pointclouds[k])
    for p in raw_data_paths:
        session_id, rpath = str(p).split("/", 1)
        target = output_data_path / p
        target.parent.mkdir(exist_ok=True, parents=True)
        target.symlink_to(
            osp.relpath(capture.data_path(session_id) / rpath, target.parent))

    if reference_id is not None and capture.sessions[reference_id].proc is not None:
        if session_new.proc is None:
            session_new.proc = Proc()
        session_new.proc.meshes = {}
        for mesh_id, mesh_subpath in capture.sessions[reference_id].proc.meshes.items():
            session_new.proc.meshes[mesh_id] = mesh_subpath
            dst_path = capture.proc_path(output_id) / mesh_subpath
            dst_path.parent.mkdir(exist_ok=True, parents=True)
            dst_path.symlink_to(
                osp.relpath(capture.proc_path(reference_id) / mesh_subpath, dst_path.parent))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--capture_path', type=Path, required=True)
    parser.add_argument('--session_ids', type=str, nargs='+', required=True)
    parser.add_argument('--output_id', type=str)
    parser.add_argument('--overwrite_poses', action='store_true')
    parser.add_argument('--keyframing', action='store_true')
    args = parser.parse_args().__dict__
    args['capture'] = Capture.load(args.pop('capture_path'))
    args['keyframing'] = KeyFramingConf() if args.pop('keyframing') else None
    run(**args)
