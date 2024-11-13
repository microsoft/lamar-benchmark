from pathlib import Path
from typing import List, Optional
import argparse
import numpy as np
import shutil

from . import logger
from .capture import (
    Capture, RecordBluetooth, RecordsBluetooth, RecordWifi, RecordsWifi)


def anonymize_wifi_mac_addr(addr, mapping):
    anon_addr = []
    for h in addr.split(':'):
        anon_addr.append(mapping[h])
    return ':'.join(anon_addr)


def anonymize_bt_guid(guid, mapping):
    guid, minor, major = guid.split(':')
    anon_guid = []
    for part in guid.split('-'):
        anon_part = []
        for idx in range(0, len(part), 2):
            anon_part.append(mapping[part[idx : idx + 2]])
        anon_guid.append(''.join(anon_part))
    return ':'.join(['-'.join(anon_guid), minor, major])


def convert_uchar_to_hex(val):
    assert 0 <= val <= 255
    h = hex(val)[2 :]
    if len(h) == 1:
        h = '0' + h
    return h


def run(capture: Capture, session_ids: List[str], seed: Optional[int] = 0):
    hex_mapping = {
        convert_uchar_to_hex(idx): convert_uchar_to_hex(val)
        for idx, val in enumerate(np.random.RandomState(seed).permutation(256))
    }

    logger.info('Will run radio anonymization in place.')

    for target_id in session_ids:
        session = capture.sessions[target_id]
        path = capture.session_path(target_id)

        if session.wifi:
            wifi = RecordsWifi()
            for key in session.wifi.key_pairs():
                wifi[key] = RecordWifi()
                for mac_addr, record in session.wifi[key].items():
                    mac_addr = anonymize_wifi_mac_addr(mac_addr, hex_mapping)
                    wifi[key][mac_addr] = record
            assert not (path / 'wifi.backup.txt').exists()
            shutil.move(path / 'wifi.txt', path / 'wifi.backup.txt')
            wifi.save(path / 'wifi.txt')

        if session.bt:
            bt = RecordsBluetooth()
            for key in session.bt.key_pairs():
                bt[key] = RecordBluetooth()
                for guid, record in session.bt[key].items():
                    guid = anonymize_bt_guid(guid, hex_mapping)
                    bt[key][guid] = record
            assert not (path / 'bt.backup.txt').exists()
            shutil.move(path / 'bt.txt', path / 'bt.backup.txt')
            bt.save(path / 'bt.txt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS)
    parser.add_argument('--capture_path', type=Path, required=True)
    parser.add_argument('--session_ids', type=str, nargs='+', required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args().__dict__
    args['capture'] = Capture.load(args.pop('capture_path'), session_ids=args['session_ids'])

    run(**args)
