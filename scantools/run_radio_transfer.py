import argparse
import copy
from pathlib import Path
import shutil
from typing import List

import numpy as np

from . import logger
from .capture import (
    Capture, RecordBluetooth, RecordBluetoothSignal, RecordsBluetooth,
    RecordWifi, RecordWifiSignal, RecordsWifi)
from .utils.radio_mapping import RadioDescriptor


TRANSFER_MAX_DELAY_US = 1_000_000
TRANSFER_TOP_K = 5
TRANSFER_MAX_DISTANCE = 3
TRANSFER_MAX_HEIGHT_DISTANCE = 1.5
DECAY_COEFF = .25


def interpolate_trajectory_at_timestamp(ts, timestamps, tvecs):
    idx = np.searchsorted(timestamps, ts, side='left')
    if idx == 0:
        return tvecs[0]
    if idx == len(timestamps):
        return tvecs[-1]
    # Linear interpolation.
    alpha = (ts - timestamps[idx - 1]) / (timestamps[idx] - timestamps[idx - 1])
    assert 0 <= alpha <= 1
    return tvecs[idx - 1] + alpha * (tvecs[idx] - tvecs[idx - 1])


def loc_radios_for_subsession(subsession_id, radio, trajectory):
    traj_timestamps = []
    traj_tvecs = []
    for ts, sensor_id in sorted(trajectory.key_pairs()):
        if sensor_id.startswith(subsession_id):
            traj_timestamps.append(ts)
            traj_tvecs.append(trajectory[ts, sensor_id].t)
    traj_timestamps = np.array(traj_timestamps)
    traj_tvecs = np.array(traj_tvecs)

    radio_tvecs = []
    radio_addrs = []
    radio_dbms = []
    for ts, sensor_id in sorted(radio.key_pairs()):
        if not sensor_id.startswith(subsession_id):
            continue
        tvec = interpolate_trajectory_at_timestamp(
            ts, traj_timestamps, traj_tvecs)
        for mac, signal in radio[ts, sensor_id].items():
            radio_tvecs.append(tvec)
            radio_addrs.append(mac)
            radio_dbms.append(signal.rssi_dbm)
    return radio_tvecs, radio_addrs, radio_dbms


def find_nearby_radios(d, tvec, radio_tvecs, radio_addrs, radio_dbms):
    dists = np.linalg.norm(radio_tvecs - tvec[np.newaxis, :], axis=-1)
    height_dists = np.abs(radio_tvecs[:, -1] - tvec[np.newaxis, -1])
    indices, = np.where(np.logical_and(
        dists <= TRANSFER_MAX_DISTANCE,
        height_dists <= TRANSFER_MAX_HEIGHT_DISTANCE))
    for dist, idx in sorted(zip(dists[indices], indices)):
        addr = radio_addrs[idx]
        rssi = radio_dbms[idx]
        if addr in d.descriptor and len(d.descriptor[addr]) >= TRANSFER_TOP_K:
            continue
        d.add_measurement(addr, (rssi, dist))


def run(capture: Capture, session_ids: List[str]):
    # Preprocess existing radios.
    valid_sessions = set()
    wifi_tvecs = []
    wifi_addrs = []
    wifi_dbms = []
    bt_tvecs = []
    bt_addrs = []
    bt_dbms = []
    for sess_id in sorted(session_ids):
        session = capture.sessions[sess_id]
        logger.info(f'Processions session {sess_id} with {len(session.proc.subsessions)} subsessions.')

        # Remove sessions with partial radios.
        valid_bt_subsessions = set()
        if session.bt:
            for key in session.bt.key_pairs():
                valid_bt_subsessions.add(key[1].split('/')[0])
        valid_wifi_subsessions = set()
        if session.wifi:
            for key in session.wifi.key_pairs():
                valid_wifi_subsessions.add(key[1].split('/')[0])
        valid_subsessions = valid_wifi_subsessions
        logger.info(f'Found {len(valid_bt_subsessions)} sessions with BT in {sess_id}.')
        logger.info(f'Found {len(valid_wifi_subsessions)} sessions with WiFi in {sess_id}.')
        logger.info(f'Keeping all sessions with WiFi.')

        trajectory = session.proc.alignment_trajectories
        if trajectory is None:
            logger.warning(f'No alignment trajectories found for {sess_id}. '
                           'Defaulting to trajectories.')
            trajectory = session.trajectories
       
        for subsession_id in sorted(valid_subsessions):
            if subsession_id in valid_sessions:
                # Already processed.
                continue
            radio_tvecs, radio_addrs, radio_dbms = loc_radios_for_subsession(
                subsession_id, session.wifi, trajectory)
            wifi_tvecs.extend(radio_tvecs)
            wifi_addrs.extend(radio_addrs)
            wifi_dbms.extend(radio_dbms)
            radio_tvecs, radio_addrs, radio_dbms = loc_radios_for_subsession(
                subsession_id, session.bt, trajectory)
            bt_tvecs.extend(radio_tvecs)
            bt_addrs.extend(radio_addrs)
            bt_dbms.extend(radio_dbms)
        valid_sessions = valid_sessions.union(valid_subsessions)
    wifi_tvecs = np.array(wifi_tvecs)
    bt_tvecs = np.array(bt_tvecs)

    for target_id in session_ids:
        session = capture.sessions[target_id]
        logger.info(f'Transferring radios to session {target_id}.')
        trajectory = session.proc.alignment_trajectories
        if trajectory is None:
            logger.warning(f'No alignment trajectories found for {target_id}. '
                        'Defaulting to trajectories.')
            trajectory = session.trajectories

        if session.bt:
            bt = copy.deepcopy(session.bt)
        else:
            bt = RecordsBluetooth()
        if session.wifi:
            wifi = copy.deepcopy(session.wifi)
        else:
            wifi = RecordsWifi()

        # Remove session with partial radios.
        for key in bt.key_pairs():
            subsession = key[1].split('/')[0]
            if subsession not in valid_sessions:
                del(bt[key])
        for key in wifi.key_pairs():
            subsession = key[1].split('/')[0]
            if subsession not in valid_sessions:
                del(wifi[key])

        # Transfer radios.
        np.random.seed(0) 
        num_invalid_images = 0
        for key in sorted(trajectory.key_pairs()):
            subsession = key[1].split('/')[0]
            prefix = f'{subsession}/'
            if subsession in valid_sessions:
                continue
            wifi_sensor_id = f'{prefix}wifi_sensor'
            bt_sensor_id = f'{prefix}bt_sensor'
            tvec = trajectory[key].t

            # Find nearby radios.
            d = RadioDescriptor()
            find_nearby_radios(d, tvec, wifi_tvecs, wifi_addrs, wifi_dbms)
            find_nearby_radios(d, tvec, bt_tvecs, bt_addrs, bt_dbms)
            # Save.
            for radio_id in d.radio_ids:
                is_bt = '-' in radio_id
                # Weighted average of RSSI with exponential distance decay.
                strengths = [s for s, _ in d.strength(radio_id)]
                dists = np.array([dist for _, dist in d.strength(radio_id)])
                weights = np.exp(-dists / DECAY_COEFF) / np.sum(np.exp(-dists / DECAY_COEFF))
                rssi_dbm = np.average(strengths, weights=weights)
                if is_bt:
                    scan_key = (key[0], bt_sensor_id)
                    if scan_key not in bt:
                        bt[scan_key] = RecordBluetooth()
                    bt[scan_key][radio_id] = RecordBluetoothSignal(rssi_dbm, '')
                else:
                    scan_key = (key[0], wifi_sensor_id)
                    if scan_key not in wifi:
                        wifi[scan_key] = RecordWifi()
                    wifi[scan_key][radio_id] = RecordWifiSignal(
                        -1, rssi_dbm, '', -1, -1)
            if len(d.radio_ids) == 0:
                num_invalid_images += 1
        logger.info(f'Could not find radios for {num_invalid_images} images in {target_id}.')

        path = capture.session_path(target_id)
        assert not (path / 'bt_old.txt').exists()
        if (path / 'bt.txt').exists():
            shutil.move(path / 'bt.txt', path / 'bt_old.txt')
        assert not (path / 'wifi_old.txt').exists()
        if (path / 'wifi.txt').exists():
            shutil.move(path / 'wifi.txt', path / 'wifi_old.txt')
        bt.save(path / 'bt.txt')
        wifi.save(path / 'wifi.txt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS)
    parser.add_argument('--capture_path', type=Path, required=True)
    parser.add_argument('--session_ids',  type=str, nargs='+', required=True)
    args = parser.parse_args().__dict__
    args['capture'] = Capture.load(args.pop('capture_path'))

    run(**args)
