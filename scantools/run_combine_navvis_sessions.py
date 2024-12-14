import argparse
import os.path as osp
from pathlib import Path
from typing import List
import open3d as o3d

from . import logger
from .capture import (
    Capture, Session, Sensors, create_sensor, Trajectories, Proc,
    RecordsLidar, RecordsCamera, RecordsDepth, RecordsWifi, RecordsBluetooth)
from .utils.io import read_pointcloud, write_pointcloud, read_mesh, write_mesh


def run(capture: Capture, session_ids: List[str], skip: int = None,
        export_combined_pointcloud: bool = False, export_pointclouds: bool = False,
        export_depths: bool = False, export_meshes: bool = False):
    session_ids = sorted(session_ids)
    output_id = '+'.join(session_ids)
    if skip is not None:
        output_id += f'+skip-{skip}'
        temporal_stride = skip + 1
    else:
        temporal_stride = 1
    for session_id in session_ids:
        if session_id not in capture.sessions:
            raise ValueError(f'Unknown session {session_id}')

    # Create data path.
    capture.data_path(output_id).mkdir(exist_ok=True, parents=True)

    sensors = Sensors()
    pointclouds = RecordsLidar()
    if export_combined_pointcloud or export_pointclouds:
        pointcloud_input_id = 'point_cloud_final'
        if export_combined_pointcloud:
            pcd_combined = o3d.geometry.PointCloud()
        for session_id in session_ids:
            session = capture.sessions[session_id]
            pcd_path = capture.data_path(session_id) / session.pointclouds[0, pointcloud_input_id]
            pcd = read_pointcloud(pcd_path)

            T_s2w = session.proc.alignment_global.get_abs_pose('pose_graph_optimized')
            if T_s2w is None:
                logger.warning(
                    'Session %s does not have a transform, make sure that it is the reference.',
                    session_id)
            else:
                pcd.transform(T_s2w.to_4x4mat())

            if export_pointclouds:
                # Save aligned sub-pointclouds.
                pointcloud_id = f'{session_id}/{pointcloud_input_id}'
                # Hacky, to avoid creating directories in case of recursive merging
                session_id_noslash = session_id.replace('/', '_')
                pointcloud_filename = f'pointcloud_{session_id_noslash}.ply'
                sensors[pointcloud_id] = create_sensor(
                    'lidar', name=f'NavVis point cloud {session_id}')
                pointclouds[0, pointcloud_id] = pointcloud_filename

                pcd_path = capture.data_path(output_id) / pointcloud_filename
                write_pointcloud(pcd_path, pcd)

            if export_combined_pointcloud:
                # Concatenate pointclouds.
                pcd_combined += pcd
                del pcd

        if export_combined_pointcloud:
            pointcloud_id = 'point_cloud_combined'
            pointcloud_filename = 'pointcloud.ply'
            sensors[pointcloud_id] = create_sensor('lidar', name='combined NavVis point cloud')
            pointclouds[0, pointcloud_id] = pointcloud_filename

            pcd_path = capture.data_path(output_id) / pointcloud_filename
            write_pointcloud(pcd_path, pcd_combined)

    # Subsession, depths and meshes.
    proc = Proc()
    proc.subsessions = []
    for session_id in session_ids:
        session = capture.sessions[session_id]
        # Support recursive merging of trajectories.
        if session.proc.subsessions:
            for subsubsession_id in session.proc.subsessions:
                proc.subsessions.append(session_id + '/' + subsubsession_id)
        else:
            proc.subsessions.append(session_id)
    if export_meshes:
        proc.meshes = {}
        session_mesh_path = capture.proc_path(output_id) / proc.meshes_dirname
        session_mesh_path.mkdir(exist_ok=True, parents=True)
        for session_id in session_ids:
            session = capture.sessions[session_id]
            for mesh_id in session.proc.meshes:
                mesh_path = capture.proc_path(session_id) / session.proc.meshes[mesh_id]
                mesh = read_mesh(mesh_path)

                T_s2w = session.proc.alignment_global.get_abs_pose('pose_graph_optimized')
                if T_s2w is None:
                    logger.warning(
                        'Session %s does not have a transform, make sure that it is the reference.',
                        session_id)
                else:
                    mesh.transform(T_s2w.to_4x4mat())

                # Again, hacky, to avoid creating directories in case of recursive merging
                session_id_noslash = session_id.replace('/', '_')
                new_mesh_id = f'{mesh_id}_{session_id_noslash}'
                proc.meshes[new_mesh_id] = Path(proc.meshes_dirname) / f'{new_mesh_id}.ply'
                mesh_path = session_mesh_path / f'{new_mesh_id}.ply'
                print(mesh_path)
                write_mesh(mesh_path, mesh)

    # Add other sensors and trajectories.
    trajectories = Trajectories()
    images = RecordsCamera()
    depths = RecordsDepth()
    wifi = RecordsWifi()
    bt = RecordsBluetooth()
    for session_id in session_ids:
        # Sensors.
        session = capture.sessions[session_id]
        new_sensor_ids = {}
        for sensor_id in session.sensors:
            new_sensor_id = f'{session_id}/{sensor_id}'
            new_sensor_ids[sensor_id] = new_sensor_id
            sensors[new_sensor_id] = session.sensors[sensor_id]

        # Temporal downsampling of each session.
        timestamps = []
        for ts in session.trajectories:
            timestamps.append(ts)
        timestamps = set(timestamps[::temporal_stride])

        # Trajectories.
        old_trajectories = session.trajectories
        T_s2w = session.proc.alignment_global.get_abs_pose('pose_graph_optimized')
        if T_s2w is None:
            logger.warning(
                'Session %s does not have a transform, make sure that it is the reference.',
                session_id)
        else:
            old_trajectories = T_s2w * old_trajectories
        for ts, cam_id in old_trajectories.key_pairs():
            if ts not in timestamps:
                continue
            trajectories[ts, new_sensor_ids[cam_id]] = old_trajectories[ts, cam_id]

        # Images.
        session_data_path = capture.data_path(output_id) / session_id
        # relative symlink between combined and original sessions
        session_data_path.symlink_to(
            osp.relpath(capture.data_path(session_id), session_data_path.parent))
        for ts, cam_id in session.images.key_pairs():
            if ts not in timestamps:
                continue
            rpath = Path(session_id) / session.images[ts, cam_id]
            images[ts, new_sensor_ids[cam_id]] = str(rpath)

        # Depths.
        if export_depths and session.depths is not None:
            for ts, cam_id in session.depths.key_pairs():
                if ts not in timestamps:
                    continue
                rpath = Path(session_id) / session.depths[ts, cam_id]
                depths[ts, new_sensor_ids[cam_id]] = str(rpath)

        # Radios
        if session.wifi is not None:
            for ts, sensor_id in session.wifi.key_pairs():
                wifi[ts, new_sensor_ids[sensor_id]] = session.wifi[ts, sensor_id]
        if session.bt is not None:
            for ts, sensor_id in session.bt.key_pairs():
                bt[ts, new_sensor_ids[sensor_id]] = session.bt[ts, sensor_id]

    if not export_depths:
        depths = None
    if len(wifi.key_pairs()) == 0:
        wifi = None
    if len(bt.key_pairs()) == 0:
        bt = None
    session = Session(
        sensors=sensors, pointclouds=pointclouds, trajectories=trajectories,
        images=images, depths=depths, wifi=wifi, bt=bt, proc=proc)
    capture.sessions[output_id] = session
    capture.save(capture.path, session_ids=[output_id])

    return output_id


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--capture_path', type=Path, required=True)
    parser.add_argument('--session_ids', type=str, nargs='+', required=True)
    parser.add_argument('--skip', type=int, default=None)
    args = parser.parse_args().__dict__
    args['capture'] = Capture.load(args.pop('capture_path'))
    run(**args)
