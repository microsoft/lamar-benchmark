# Copyright 2020-present NAVER Corp. Under BSD 3-clause license
# Modified by Paul-Edouard Sarlin (ETH Zurich)

from functools import cached_property
from pathlib import Path
from dataclasses import dataclass, fields, astuple
from abc import ABC, abstractproperty, abstractmethod
from typing import Dict, List, Set, TypeVar, Union
import numpy as np

from ..utils.io import read_csv, write_csv
from .misc import KeyType


T = TypeVar('T')  # Declare generic type variable

class RecordsBase(Dict[int, Dict[str, T]], dict, ABC):
    @abstractproperty
    def record_type(self):
        raise NotImplementedError('Child classes must define a record_type attribute.')

    @abstractproperty
    def field_names(self) -> List[str]:
        raise NotImplementedError('Child classes must define a field_names attribute.')

    @abstractmethod
    def record_to_list(self, record: T) -> List[str]:
        raise NotImplementedError

    def __setitem__(self,
                    key: Union[int, KeyType],
                    value: Union[Dict[str, T], T]):
        # enforce type checking
        if isinstance(key, tuple):
            timestamp, device_id = key
            if not isinstance(value, self.record_type):
                raise TypeError(f'invalid record type of {type(value)} (expect {self.record_type})')
            self.setdefault(timestamp, {})[device_id] = value
        elif isinstance(key, int):
            timestamp = key
            if not isinstance(value, dict):
                raise TypeError('invalid value for data (expect dict)')
            if not all(isinstance(k, str) for k in value.keys()):
                raise TypeError('invalid device_id')
            if not all(isinstance(k, self.record_type) for k in value.values()):
                raise TypeError(f'invalid value for record (expect all {self.record_type})')
            super().__setitem__(timestamp, value)
        else:
            raise TypeError('key must be either int or Tuple[int, str]')

    def __getitem__(self, key: Union[int, KeyType]) -> Union[Dict[str, T], T]:
        if isinstance(key, tuple):  # pylint: disable=no-else-return
            timestamp, device_id = key
            return super().__getitem__(timestamp)[device_id]
        elif isinstance(key, int):
            return super().__getitem__(key)
        else:
            raise TypeError('key must be either int or Tuple[int, str]')

    def __delitem__(self, key: Union[int, KeyType]):
        if isinstance(key, tuple):
            timestamp, device_id = key
            super().__getitem__(timestamp).__delitem__(device_id)
            if len(super(RecordsBase, self).__getitem__(timestamp)) == 0:
                super().__delitem__(timestamp)
        elif isinstance(key, int):
            super().__delitem__(key)
        else:
            raise TypeError('key must be either int or Tuple[int, str]')

    def key_pairs(self) -> List[KeyType]:
        return [
            (timestamp, sensor_id)
            for timestamp, sensors in self.items()
            for sensor_id in sensors.keys()
        ]

    @property
    def sensors_ids(self) -> Set[str]:
        return set(
            sensor_id
            for timestamp, sensors in self.items()
            for sensor_id in sensors.keys()
        )

    def __contains__(self, key: Union[int, KeyType]):
        if isinstance(key, tuple):  # pylint: disable=no-else-return
            timestamp, device_id = key
            return super().__contains__(timestamp) and device_id in self[timestamp]
        elif isinstance(key, int):
            return super().__contains__(key)
        else:
            raise TypeError('key must be either int or Tuple[int, str]')

    @classmethod
    def load(cls, path: Path) -> 'RecordsBase':
        table = read_csv(path)
        records = cls()
        for timestamp, sensor_id, *data in table:
            records[int(timestamp), sensor_id] = records.record_type(*data)
        return records

    def save(self, path: Path):
        table = ([str(ts), sensor_id] + self.record_to_list(self[ts, sensor_id])
                 for ts, sensor_id in self.key_pairs())
        columns = ['timestamp', 'sensor_id'] + list(self.field_names)
        write_csv(path, table, columns=columns)

    def __repr__(self) -> str:
        # [timestamp, sensor_id] = str
        lines = [f'[ {timestamp:010}, {sensor_id:5}] = {data}'
                 for timestamp, sensors in self.items()
                 for sensor_id, data in sensors.items()]
        return '\n'.join(lines)


class RecordsFilePath(RecordsBase[str]):
    record_type = str
    field_names = ['file_path']

    def record_to_list(self, record: str) -> List[str]:
        return [record]

    @classmethod
    def load(cls, records_path: Path, data_dir: Path) -> 'RecordsFilePath':
        records = super().load(records_path)
        # check that all files exist
        for key in records.key_pairs():
            if not (data_dir / records[key]).exists():
                raise FileNotFoundError(f'{records[key]} does not exist in {data_dir}.')
        return records


class RecordsCamera(RecordsFilePath):
    field_names = ['image_path']


class RecordsDepth(RecordsFilePath):
    field_names = ['depth_path']


class RecordsLidar(RecordsFilePath):
    field_names = ['point_cloud_path']



# New data types inherit from RecordEntry (a record) and RecordsArray (mapping of records)
@dataclass
class RecordEntry:
    def __post_init__(self):
        # force cast to expected types
        for field in fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, field.type):
                setattr(self, field.name, field.type(value))

    def astuple(self):
        return astuple(self)


class RecordsArray(RecordsBase[T]):
    def record_to_list(self, record: T) -> List[List[str]]:
        return [[key] + list(map(str, data.astuple())) for key, data in record.items()]

    @property
    def field_names(self) -> List[str]:
        return self.record_type.field_names

    @cached_property
    def sorted_unique_timestamps(self) -> np.ndarray:
        timestamps = set()
        for ts, _ in self.key_pairs():
            timestamps.add(ts)
        return np.sort(list(timestamps))

    @classmethod
    def load(cls, path: Path) -> 'RecordsArray':
        table = read_csv(path)
        records = cls()
        for timestamp, sensor_id, key, *data in table:
            if (int(timestamp), sensor_id) not in records:
                records[int(timestamp), sensor_id] = records.record_type()
            records[int(timestamp), sensor_id][key] = records.record_type.record_entry(*data)
        return records

    def save(self, path: Path):
        table = []
        for ts, sensor_id in self.key_pairs():
            for record_entry in self.record_to_list(self[ts, sensor_id]):
                table.append([str(ts), sensor_id] + list(record_entry))
        columns = ['timestamp', 'sensor_id'] + list(self.field_names)
        write_csv(path, table, columns=columns)


# wifi recordings is made of dict of wifi devices with frequency, signal strengths
# and additional optional information optional: WiFi name - SSID and scan time start / end.
@dataclass
class RecordWifiSignal(RecordEntry):
    frequency_khz: int
    rssi_dbm: float
    name: str = ''
    scan_time_start_us: int = -1
    scan_time_end_us: int = -1


class RecordWifi(dict):
    record_entry = RecordWifiSignal
    field_names = ['mac_addr'] + [f.name for f in fields(record_entry)]

    def __setitem__(self, mac_addr: str, data: RecordWifiSignal):
        if not isinstance(mac_addr, str):
            raise TypeError(f'{mac_addr} is not expected type str.')
        if not isinstance(data, RecordWifiSignal):
            raise TypeError(f'{data} is not expected type RecordWifiHotspot.')
        # Convert to lower case.
        mac_addr = mac_addr.lower()
        # Check format.
        if len(mac_addr) != 17:
            raise ValueError(
                f'WiFi MAC address {mac_addr} has length {len(mac_addr)}. '
                'Expected length 17.')
        if ''.join(filter(lambda ch: not ch.isalnum(), mac_addr)) != ':::::':
            raise ValueError(
                f'WiFi MAC address {mac_addr} is expected to be separated by ":".')
        # Add to dictionary.
        super().__setitem__(mac_addr, data)


class RecordsWifi(RecordsArray[RecordWifi]):
    """
    brief: Records wifi
            records[timestamp][sensor_id] = <RecordWifi> = {mac_addr: RecordWifiHotspot}
            or
            records[timestamp, sensor_id] = <RecordWifi>
    """
    record_type = RecordWifi


# bluetooth recordings is made of dict of bt devices and signal strengths.
@dataclass
class RecordBluetoothSignal(RecordEntry):
    rssi_dbm: float
    name: str = ''


class RecordBluetooth(dict):
    record_entry = RecordBluetoothSignal
    field_names = ['id'] + [f.name for f in fields(record_entry)]

    def __setitem__(self, id: str, data: RecordBluetoothSignal):
        if not isinstance(id, str):
            raise TypeError(f'{id} is not expected type str.')
        if not isinstance(data, RecordBluetoothSignal):
            raise TypeError(f'{data} is not expected type RecordBluetoothDevice.')
        # Convert to lower case.
        id = id.lower()
        # Check format.
        split_id = id.split(':')
        if len(split_id) != 3 or not split_id[1].isnumeric() or not split_id[2].isnumeric():
            raise ValueError(
                f'BT id {id} does not follow expected format: '
                '"guid-with-hyphens:major:minor".')
        if list(map(len, split_id[0].split('-'))) != [8, 4, 4, 4, 12]:
            raise ValueError(
                f'BT id {id} guid does not follow expected format '
                '"XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX".')
        # Add to dictionary.
        super().__setitem__(id, data)


class RecordsBluetooth(RecordsArray[RecordBluetooth]):
    """
    brief: Records wifi
            records[timestamp][sensor_id] = <RecordBluetooth> = {address: <RecordBluetoothDevice>}
            or
            records[timestamp, sensor_id] = <RecordBluetooth>
    """
    record_type = RecordBluetooth
