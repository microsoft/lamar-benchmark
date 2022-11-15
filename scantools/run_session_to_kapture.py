import argparse
from typing import Optional, Set
from pathlib import Path

import kapture
import kapture.io
from kapture.io.csv import kapture_to_dir

from .capture import Capture, Session


def convert_to_kapture(session: Session, query_list: Optional[Set] = None):
    # Convert sensors.
    sensors = kapture.Sensors()
    if session.sensors is not None:
        for key, sensor_info in session.sensors.items():
            sensors[key] = kapture.create_sensor(
                sensor_info.sensor_type, sensor_info.sensor_params, key)

    # Conver rigs.
    rigs = kapture.Rigs()
    sensor_to_rig = {}
    if session.rigs is not None:
        for key, rig in session.rigs.items():
            for sensor, pose in rig.items():
                sensor_to_rig[sensor] = key
                # Kapture is rig->camera.
                inv_pose = pose.inverse()
                rigs[key, sensor] = kapture.PoseTransform(r=inv_pose.qvec, t=inv_pose.t)

    # Convert trajectories.
    trajectories = kapture.Trajectories()
    if session.trajectories is not None:
        for ts, sensor in session.trajectories.key_pairs():
            if query_list is None or (ts, sensor) in query_list:
                inv_pose = session.trajectories[ts, sensor].inverse()
                trajectories[ts, sensor] = kapture.PoseTransform(r=inv_pose.qvec, t=inv_pose.t)

    # Conver records camera.
    records_camera = kapture.RecordsCamera()
    if session.images is not None:
        for ts, sensor in session.images.key_pairs():
            if query_list is None or (ts, sensor) in query_list or (
                    sensor in sensor_to_rig and (ts, sensor_to_rig[sensor]) in query_list):
                records_camera[ts, sensor] = session.images[ts, sensor]

    # Convert records depth.
    records_depth = kapture.RecordsDepth()
    if session.depths is not None:
        for ts, sensor in session.depths.key_pairs():
            records_depth[ts, sensor] = session.depths[ts, sensor]

    # Convert records lidar.
    records_lidar = kapture.RecordsLidar()
    if session.pointclouds is not None:
        for ts, sensor in session.pointclouds.key_pairs():
            records_lidar[ts, sensor] = session.pointclouds[ts, sensor]

    return kapture.Kapture(
        sensors=sensors,
        rigs=rigs,
        trajectories=trajectories,
        records_camera=records_camera,
        records_depth=records_depth,
        records_lidar=records_lidar
    )


def run(capture: Capture, session_id: str, output_path: Path):
    kapture_ = convert_to_kapture(capture.sessions[session_id])

    # Save to disk.
    output_path.mkdir(exist_ok=True, parents=True)
    kapture_to_dir(output_path, kapture_)
    # Symlink data.
    (output_path / 'sensors' / 'records_data').symlink_to(capture.data_path(session_id).resolve())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--capture_path', type=Path, required=True)
    parser.add_argument('--session_id', type=str, required=True)
    parser.add_argument('--output_path', type=Path, required=True)
    args = parser.parse_args().__dict__
    args['capture'] = Capture.load(args.pop('capture_path'))
    run(**args)
