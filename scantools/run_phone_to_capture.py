import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import shutil
import subprocess

import numpy as np
import cv2
from tqdm import tqdm
from scipy.spatial.transform import Rotation

from . import logger
from .capture import (
        Capture, Session, Camera, Sensors, create_sensor, Trajectories, Pose,
        RecordsCamera, RecordBluetooth, RecordBluetoothSignal, RecordsBluetooth,
        RecordsDepth)
from .utils.io import read_image, write_image, read_csv, write_depth
from .viz.meshlab import MeshlabProject


def extract_frames_from_video(input_dir: Path, images_dir: Path):
    assert images_dir.exists()
    video_path = input_dir / 'images.mp4'

    # Extract frames.
    frames_format = 'out-%012d.jpg'
    cmd = [
        'ffmpeg',
        '-hide_banner', '-loglevel', 'warning', '-nostats',
        '-i', video_path.as_posix(),
        '-vsync', '0',
        '-qmin', '1',
        '-q:v', '1',
        (images_dir / frames_format).as_posix(),
    ]
    subprocess.run(cmd, check=True)

    # Extract timestamps.
    cmd = [
        'ffprobe',
        '-hide_banner', '-loglevel', 'warning',
        '-f', 'lavfi',
        '-i', f'movie={video_path.as_posix()}',
        '-show_entries', 'frame=pkt_pts',
        '-of', 'csv=p=0',
    ]
    result = subprocess.run(cmd,
                            check=True,
                            capture_output=True,
                            text=True)
    # Convert list of newline separated chars to list of strings.
    timestamps = ''.join(result.stdout).split()

    # Extract time origin (timestamp of the first pose).
    poses = read_csv(input_dir / 'poses.txt')
    assert len(poses) == len(timestamps)
    time_origin = int(poses[0][0])

    # Rename all image data.
    for idx, timestamp in enumerate(timestamps):
        image_path = images_dir / (frames_format % (idx + 1))
        output_path = images_dir / (str(time_origin + int(timestamp)) + '.jpg')
        image_path.rename(output_path)


def rotate_camera(camera: Camera, num_rot90: int) -> Camera:
    assert camera.model_name == 'PINHOLE'
    w, h = camera.width, camera.height
    fx, fy, cx, cy = camera.params

    num_rot90 = num_rot90 % 4
    if num_rot90 == 0:
        return camera
    if num_rot90 == 1:
        cx2, cy2 = cy, w-cx
    elif num_rot90 == 2:
        cx2, cy2 = w-cx, h-cy
    elif num_rot90 == 3:
        cx2, cy2 = h-cy, cx
    else:
        raise ValueError

    perm = slice(None, None, -1 if (num_rot90 % 2) else 1)
    fx2, fy2 = [fx, fy][perm]
    w2, h2 = [w, h][perm]
    params2 = [w2, h2, fx2, fy2, cx2, cy2]
    return Camera(camera.model, params2, camera.name, camera.sensor_type)


def get_rot90(pose_cam2world: Pose) -> int:
    '''In ARKit & ARCore, the y axis always points up along the gravity direction.
       We use this to autorotate the images so that they are upright.
    '''
    gravity_world = np.array([0, -1, 0])
    gravity_cam = pose_cam2world.r.as_matrix().T @ gravity_world
    angle = np.rad2deg(np.arctan2(gravity_cam[1], gravity_cam[0]))
    binned = np.round(angle / 90) % (360/90)  # 0=0, 1=90, 2=180, 3=270
    num_rot90 = int((binned - 1) % 4)
    return num_rot90


def parse_pose_file(path: Path) -> Tuple[Dict[int, Pose], Dict[int, Camera], Dict[int, int]]:
    poses = {}
    cameras = {}
    rots90 = {}

    # Poses from ARKit & ARCore use the computer graphics conventions (inverted y and z)
    rot_cg_to_cv = Rotation.from_matrix(np.diag([1, -1, -1]))

    rows = read_csv(path)
    for ts, status, tx, ty, tz, qx, qy, qz, qw, w, h, fx, fy, cx, cy, *_ in tqdm(rows):
        ts = int(ts)
        if status != 'normal':
            logger.warning('Tracking of %d with abnormal status: %s.', ts, status)

        tvec = np.array([tx, ty, tz], float)
        qvec = np.array([qw, qx, qy, qz], float)
        rot = Pose(qvec).r * rot_cg_to_cv
        pose = Pose(rot, tvec)

        # Here we assume that the principal point is already given in COLMAP coordinates
        # From https://developer.apple.com/documentation/arkit/arcamera/2875730-intrinsics :
        # "The values ox and oy are the offsets of the principal point
        #  from the top-left corner of the image frame."
        camera = create_sensor('camera', ['PINHOLE', w, h, fx, fy, cx, cy])

        num_rot90 = get_rot90(pose)
        if num_rot90 != 0:
            camera = rotate_camera(camera, num_rot90)
            rot_upright = Rotation.from_euler('z', 90*num_rot90, degrees=True)
            pose = Pose(pose.r * rot_upright, pose.t)

        poses[ts] = pose
        cameras[ts] = camera
        rots90[ts] = num_rot90

    return poses, cameras, rots90


def parse_bluetooth_file(bt_path: Path,
                         timestamps: List[int],
                         sensors: Sensors,
                         sensor_id: str = 'bt_sensor'):
    bluetooth_signals = RecordsBluetooth()
    sensor = create_sensor('bluetooth', sensor_params=[], name='Apple bluetooth sensor')
    sensors[sensor_id] = sensor
    for timestamp_us, _, guid, rssi_dbm in read_csv(bt_path):
        timestamp_us = int(timestamp_us)
        if not timestamps[0] <= timestamp_us <= timestamps[-1]:
            continue
        id_ = f'{guid}:0:0'
        rssi_dbm = float(rssi_dbm)
        if (timestamp_us, sensor_id) not in bluetooth_signals:
            bluetooth_signals[timestamp_us, sensor_id] = RecordBluetooth()
        bluetooth_signals[timestamp_us, sensor_id][id_] = RecordBluetoothSignal(rssi_dbm=rssi_dbm)
    return bluetooth_signals


def parse_depth_files(input_dir: Path,
                      data_dir: Path,
                      sensors: Sensors,
                      images: RecordsCamera,
                      rots90: Dict[int, int]) -> RecordsDepth:
    records = RecordsDepth()
    paths = list(input_dir.glob('*.bin'))
    for depth_path in paths:
        timestamp = int(depth_path.stem)
        if timestamp not in images:
            continue

        confidence = cv2.imread(
            depth_path.with_suffix('.confidence.png').as_posix(), cv2.IMREAD_ANYDEPTH)
        depth = np.fromfile(depth_path, dtype=np.float32).reshape(confidence.shape)

        confidence = np.rot90(confidence, rots90[timestamp])
        depth = np.rot90(depth, rots90[timestamp])

        camera_id, = images[timestamp].keys()
        depth_id = camera_id.replace('cam', 'depth')
        if depth_id not in sensors:
            camera = sensors[camera_id]
            assert camera.model_name == 'PINHOLE'
            h, w = depth.shape
            scale = np.array([w / camera.width, h / camera.height] * 2)
            params = np.array(camera.projection_params) * scale
            depth_camera = create_sensor('depth', ['PINHOLE', w, h] + params.tolist())
            sensors[depth_id] = depth_camera

        subpath = f'depth/{timestamp}.png'
        out_path = data_dir / subpath
        out_path.parent.mkdir(exist_ok=True, parents=True)
        write_depth(out_path, depth)
        write_image(out_path.with_suffix('.confidence.png'), confidence)
        records[timestamp, depth_id] = subpath
    return records


def chunk_tracking_failures(T_c2w: Dict[int, Pose], window_size: int = 5,
                            max_relative_error: float = 10, max_error: float = 1.0,
                            min_chunk_duration: float = 10) -> List[List[int]]:
    timestamps = sorted(T_c2w.keys())
    translation = []
    for t1, t2 in zip(timestamps[:-1], timestamps[1:]):
        T_1to2 = T_c2w[t2].inverse() * T_c2w[t1]
        translation.append(T_1to2.t)
    translation = np.array(translation)

    # We assume a constant velocity model and detect large deviations.
    velocity = translation / np.diff(timestamps)[:, None]
    error_relative = []
    error = []
    for i, t in enumerate(translation):
        if i == 0:
            error_relative.append(0)
            error.append(0)
        else:
            window = np.r_[velocity[max(0, i-window_size):i], velocity[i+1:i+window_size+1]]
            v_predicted = np.median(window, 0)
            v_observed = t / (timestamps[i+1] - timestamps[i])
            diff = np.linalg.norm(v_predicted - v_observed)
            v_predicted = np.linalg.norm(v_predicted)
            error_relative.append(diff / v_predicted)
            error.append(diff*1e6)  # us to s
    error_relative = np.array(error_relative)
    error = np.array(error)
    outlier = (error > max_error) & (error_relative > max_relative_error)
    print(error_relative[outlier], np.where(outlier)[0])

    chunks = []
    cuts = np.where(outlier)[0]
    cuts = np.stack([np.r_[0, cuts+1], np.r_[cuts, len(timestamps)-1]], 1)
    durations = []
    for start, end in cuts:
        duration = (timestamps[end] - timestamps[start]) * 1e-6
        if duration > min_chunk_duration:
            chunks.append(timestamps[start:end+1])
            durations.append(duration)
    logger.info('Chunked the phone sequence into durations %s', durations)
    return chunks


def keyframe_selection(timestamps: List[float], target_framerate: float, slack: float = 0.99):
    keyframes = [timestamps[0]]
    for t in timestamps[1:]:
        if (t - keyframes[-1]) > (1e6 / target_framerate)*slack:
            keyframes.append(t)
    return keyframes


def timestamps_to_session(timestamps: List[int],
                          session_id: str,
                          capture: Capture,
                          cameras: Dict[int, Camera],
                          rots90: Dict[int, int],
                          poses: Dict[int, Pose],
                          input_path: Path,
                          image_dir: Path):
    sensors = Sensors()
    trajectory = Trajectories()
    images = RecordsCamera()

    # Check if the intrinsics are constant throughout the sequence
    is_camera_shared = len(set(tuple(c.sensor_params) for c in cameras.values())) == 1
    if is_camera_shared:
        camera = next(iter(cameras.values()))
        camera.name = 'phone camera shared across all frames'
        camera_id = 'cam_phone'
        sensors[camera_id] = camera

    for timestamp in tqdm(timestamps):
        if not is_camera_shared:
            camera = cameras[timestamp]
            camera.name = f'phone camera for timestamp {timestamp}'
            camera_id = f'cam_phone_{timestamp}'
            sensors[camera_id] = camera

        trajectory[timestamp, camera_id] = poses[timestamp]
        input_image_path = image_dir / f'{timestamp}.jpg'
        image_subpath = f'images/{timestamp}.jpg'
        output_image_path = capture.data_path(session_id) / image_subpath
        output_image_path.parent.mkdir(exist_ok=True, parents=True)

        num_rot90 = rots90[timestamp]
        if num_rot90 == 0:
            shutil.copy(str(input_image_path), str(output_image_path))
        else:
            image = read_image(input_image_path)
            image = np.rot90(image, num_rot90)
            write_image(output_image_path, image)
        images[timestamp, camera_id] = image_subpath

    depth_dir = input_path / 'depth'
    depths = None
    if depth_dir.exists():
        depths = parse_depth_files(
            depth_dir, capture.data_path(session_id), sensors, images, rots90)

    bt_path = input_path / 'bluetooth.txt'
    bluetooth_signals = None
    if bt_path.exists():
        bluetooth_signals = parse_bluetooth_file(bt_path, timestamps, sensors)

    for filename in [
            'accelerometer.txt',
            'gyroscope.txt',
            'magnetometer.txt',
            'fused_imu.txt',
            'location.txt',
        ]:
        if (input_path / filename).exists():
            shutil.copy(input_path / filename, capture.session_path(session_id))

    session = Session(
        sensors=sensors, trajectories=trajectory,
        images=images, bt=bluetooth_signals, depths=depths)
    capture.sessions[session_id] = session


def run(input_path: Path,
        capture: Capture,
        session_id: str,
        visualize: bool = False,
        downsample_framerate: Optional[float] = 5) -> List[str]:
    assert session_id not in capture.sessions, session_id

    images_as_video = (input_path / 'images.mp4').exists()
    if images_as_video:  # new format
        image_dir = input_path / 'tmp/'
        image_dir.mkdir(exist_ok=True, parents=True)
        logger.info('Extracting phone data')
        extract_frames_from_video(input_path, image_dir)
    else:  # old format
        image_dir = input_path / 'images'
        assert image_dir.exists()

    if visualize:
        mlp_path = capture.viz_path() / f'phone_trajectory_{session_id}.mlp'
        mlp = MeshlabProject()

    poses, cameras, rots90 = parse_pose_file(input_path / 'poses.txt')
    timestamp_chunks = chunk_tracking_failures(poses)
    chunk_ids = []
    for i, timestamps in enumerate(timestamp_chunks):
        chunk_id = f'{session_id}_{i:03}'
        if downsample_framerate is not None:
            timestamps = keyframe_selection(timestamps, downsample_framerate)
        logger.info('Importing sub-session %s', chunk_id)
        chunk_ids.append(chunk_id)
        timestamps_to_session(
            timestamps, chunk_id, capture, cameras, rots90, poses, input_path, image_dir)

        if visualize:
            session = capture.sessions[chunk_id]
            for ts, camera_id in tqdm(sorted(session.trajectories.key_pairs())):
                pose = session.trajectories[ts, camera_id]
                camera = session.sensors[camera_id]
                mlp.add_camera(f'{ts}/{camera_id}', camera, pose)
                mlp.add_trajectory_point(chunk_id, pose)

    capture.save(capture.path, session_ids=chunk_ids)
    if visualize:
        mlp.write(mlp_path)
    if images_as_video:
        shutil.rmtree(image_dir)
    return chunk_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_path', type=Path, required=True)
    parser.add_argument('--capture_path', type=Path, required=True)
    parser.add_argument('--session_id', type=str, required=True)
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args().__dict__
    args['capture'] = Capture.load(args.pop('capture_path'))
    run(**args)
