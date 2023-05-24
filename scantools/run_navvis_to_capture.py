import argparse
from typing import Optional, Tuple
from pathlib import Path
import shutil
import logging
from tqdm import tqdm
import cv2
import numpy as np

from . import logger
from .scanners import NavVis
from .scanners.navvis.camera_tiles import TileFormat
from .capture import (
        Capture, Session, Sensors, create_sensor, Trajectories, Rigs, Pose,
        RecordsCamera, RecordsLidar, RecordBluetooth, RecordBluetoothSignal,
        RecordsBluetooth, RecordWifi, RecordWifiSignal, RecordsWifi)
from .utils.misc import add_bool_arg
from .utils.io import read_image, write_image

TILE_CHOICES = sorted([attr.name.split('_')[1] for attr in TileFormat])


def compute_downsampling_size(size: Tuple[int], max_edge: int):
    scale = max_edge / max(size)
    new_size = tuple([int(round(edge * scale)) for edge in size])
    return new_size

def get_pose(nv: NavVis,
             upright: bool,
             frame_id: int,
             cam_id: Optional[int] = 0,
             tile_id: Optional[int] = 0):
    qvec, tvec = nv.get_pose(frame_id, cam_id, tile_id)
    pose = Pose(r=qvec, t=tvec)
    if upright and nv.get_device() == 'VLX' and (cam_id == 0 or cam_id == 'cam0'):
        pose = fix_vlx_extrinsics(pose)
    return pose

def fix_vlx_extrinsics(pose: Pose):
    # Camera 0 is (physically) mounted upside down on VLX.
    # Intrinsics stay the same since they are the image center.
    # Extrinsics should be rotated by 180 deg counter-clockwise around z.
    fix_matrix = np.array([
        [-1,  0,  0],
        [ 0, -1,  0],
        [ 0,  0,  1]
    ])
    new_rotmat = pose.r.as_matrix() @ fix_matrix
    pose = Pose(r=new_rotmat, t=pose.t)
    return pose

def convert_to_us(time_s):
    return int(round(time_s * 1_000_000))

def run(input_path: Path, capture: Capture, tiles_format: str, session_id: Optional[str] = None,
        downsample_max_edge: int = None, upright: bool = True, export_as_rig: bool = False,
        copy_pointcloud: bool = False):

    if session_id is None:
        session_id = input_path.name
    assert session_id not in capture.sessions

    output_path = capture.data_path(session_id)
    nv = NavVis(input_path, output_path, tiles_format, upright)

    frame_ids = nv.get_frame_ids()
    camera_ids = nv.get_camera_indexes()
    tiles = nv.get_tiles()

    num_frames = len(frame_ids)
    num_cameras = len(camera_ids)
    num_tiles = nv.get_num_tiles()

    K = nv.get_camera_intrinsics()
    fx, fy, cx, cy = K[[0, 1, 0, 1], [0, 1, 2, 2]]
    assert fx == fy
    if downsample_max_edge is None:
        camera_params = ('SIMPLE_PINHOLE', tiles.width, tiles.height, fx, cx, cy)
    else:
        w = tiles.width
        h = tiles.height
        new_w, new_h = compute_downsampling_size((w, h), downsample_max_edge)
        fx = fx / w * new_w
        fy = fy / h * new_h
        cx = cx / w * new_w
        cy = cy / h * new_h
        camera_params = ('PINHOLE', new_w, new_h, fx, fy, cx, cy)

    device = nv.get_device()

    sensors = Sensors()
    trajectory = Trajectories()
    images = RecordsCamera()
    rigs = Rigs() if export_as_rig else None

    if export_as_rig:
        # This code assumes NavVis produces consistent rigs across all frames,
        # using `cam_id=0` as the rig base.
        frame_id_0 = frame_ids[0]
        rig_from_world = get_pose(nv, upright, frame_id_0, cam_id=0, tile_id=0).inverse()
        rig_id = "navvis_rig"

    for camera_id in camera_ids:
        for tile_id in range(num_tiles):
            sensor_id = f'cam{camera_id}_{tiles_format}'
            sensor_id += f'-{tile_id}' if num_tiles > 1 else ''
            sensor = create_sensor(
                'camera', sensor_params=camera_params,
                name=f'NavVis {device} camera-cam{camera_id} tile-{tiles_format} id-{tile_id}')
            sensors[sensor_id] = sensor

            if export_as_rig:
                world_from_cam = get_pose(nv, upright, frame_id_0, camera_id, tile_id)
                rig_from_cam = rig_from_world * world_from_cam
                rigs[rig_id, sensor_id] = rig_from_cam

    # Process frames (enumerate is needed since frame_id is not sequential).
    for frame_idx, frame_id in enumerate(frame_ids):
        if not nv.get_frame_valid(frame_id):
            if tile_id == 0:
                logging.warning('Invalid frame %d.', frame_id)
            continue
        time_s = nv.get_frame_timestamp(frame_id)
        timestamp_us = convert_to_us(time_s)
        if export_as_rig:
            pose = get_pose(nv, upright, frame_id, cam_id=0)
            trajectory[timestamp_us, rig_id] = pose

        for camera_id in camera_ids:
            for tile_id in range(num_tiles):
                sensor_id = f'cam{camera_id}_{tiles_format}'
                sensor_id += f'-{tile_id}' if num_tiles > 1 else ''
                logging.info("Processing :: frame_id: %d/%d "
                        "- cam_id: %d/%d "
                        "- tile_id: %d/%d",
                        frame_idx + 1, num_frames,
                        camera_id + 1, num_cameras,
                        tile_id + 1, num_tiles)
                image_path = nv.get_output_image_path(frame_id, camera_id, tile_id)
                image_subpath = image_path.resolve().relative_to(output_path.resolve())
                images[timestamp_us, sensor_id] = str(image_subpath)
                if not export_as_rig:
                    pose = get_pose(nv, upright, frame_id, camera_id, tile_id)
                    trajectory[timestamp_us, sensor_id] = pose

    pointcloud_id = 'point_cloud_final'
    sensors[pointcloud_id] = create_sensor('lidar', name='final NavVis point cloud')
    pointclouds = RecordsLidar()
    pointcloud_filename = nv.get_pointcloud_path().name
    pointclouds[0, pointcloud_id] = pointcloud_filename

    wifi_signals = RecordsWifi()
    sensor_id = 'wifi_sensor'
    sensor = create_sensor('wifi', sensor_params=[], name='NavVis M6 WiFi sensor')
    sensors[sensor_id] = sensor
    for measurement in nv.read_wifi():
        timestamp_us = convert_to_us(measurement.timestamp_s)
        mac_addr = measurement.mac_address
        freq_khz = measurement.center_channel_freq_khz
        rssi_dbm = measurement.signal_strength_dbm
        time_offset_us = int(measurement.time_offset_ms) / 1_000
        if (timestamp_us, sensor_id) not in wifi_signals:
            wifi_signals[timestamp_us, sensor_id] = RecordWifi()
        wifi_signals[timestamp_us, sensor_id][mac_addr] = RecordWifiSignal(
            frequency_khz=freq_khz, rssi_dbm=rssi_dbm, scan_time_start_us=(timestamp_us - time_offset_us)
        )

    bluetooth_signals = RecordsBluetooth()
    sensor_id = 'bt_sensor'
    sensor = create_sensor('bluetooth', sensor_params=[], name='NavVis M6 bluetooth sensor')
    sensors[sensor_id] = sensor
    for measurement in nv.read_bluetooth():
        timestamp_us = convert_to_us(measurement.timestamp_s)
        id = f'{measurement.guid}:{measurement.major_version}:{measurement.minor_version}'
        rssi_dbm = measurement.signal_strength_dbm
        if (timestamp_us, sensor_id) not in bluetooth_signals:
            bluetooth_signals[timestamp_us, sensor_id] = RecordBluetooth()
        bluetooth_signals[timestamp_us, sensor_id][id] = RecordBluetoothSignal(rssi_dbm=rssi_dbm)

    session = Session(
        sensors=sensors, rigs=rigs, trajectories=trajectory,
        images=images, pointclouds=pointclouds, wifi=wifi_signals, bt=bluetooth_signals)
    capture.sessions[session_id] = session
    capture.save(capture.path, session_ids=[session_id])

    logger.info('Generating raw data for session %s.', session_id)
    nv.undistort()
    for ts, cam in tqdm(session.images.key_pairs()):
        downsample = downsample_max_edge is not None
        # Camera 0 is (physically) mounted upside down on VLX.
        flip = (upright and device == 'VLX' and cam.startswith('cam0'))
        if downsample or flip:
            image_path = capture.data_path(session_id) / session.images[ts, cam]
            image = read_image(image_path)
            if downsample:
                new_size = (session.sensors[cam].width, session.sensors[cam].height)
                image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
            if flip:
                image = image[:: -1, :: -1, :]
            write_image(image_path, image)
    if copy_pointcloud:
        shutil.copy(str(nv.get_pointcloud_path()), str(output_path))
    else:
        (output_path / pointcloud_filename).symlink_to(nv.get_pointcloud_path())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_path', type=Path, required=True)
    parser.add_argument('--capture_path', type=Path, required=True)
    parser.add_argument('--tiles_format', type=str, default='none', choices=TILE_CHOICES)
    parser.add_argument('--session_id', type=str)
    parser.add_argument('--downsample_max_edge', type=int, default=None)
    add_bool_arg(parser, 'upright', default=True)
    add_bool_arg(parser, 'export_as_rig', default=False)
    parser.add_argument('--copy_pointcloud', action='store_true')
    args = parser.parse_args().__dict__

    capture_path = args.pop('capture_path')
    if capture_path.exists():
        args['capture'] = Capture.load(capture_path)
    else:
        args['capture'] = Capture(sessions={}, path=capture_path)

    run(**args)
