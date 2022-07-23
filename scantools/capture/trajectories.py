# Copyright 2020-present NAVER Corp. Under BSD 3-clause license
# Modified by Paul-Edouard Sarlin (ETH Zurich)

from pathlib import Path
from typing import Union, Dict, List, Set, Tuple

from ..utils.io import read_csv, write_csv
from .pose import Pose


class Trajectories(Dict[int, Dict[str, Pose]], dict):
    def __setitem__(self,
                    key: Union[int, Tuple[int, str]],
                    value: Union[Dict[str, Pose], Pose]):
        # enforce type checking
        if isinstance(key, tuple):
            timestamp, device_id = key
            if not isinstance(value, Pose):
                raise TypeError('invalid Pose type as value')
            self.setdefault(timestamp, {})[device_id] = value
        elif isinstance(key, int):
            timestamp = key
            if not isinstance(value, dict):
                raise TypeError('invalid value for trajectory timestamp')
            if not all(isinstance(k, str) for k in value.keys()):
                raise TypeError('invalid device_id')
            if not all(isinstance(v, Pose) for v in value.values()):
                raise TypeError('invalid Pose')
            super(Trajectories, self).__setitem__(timestamp, value)
        else:
            raise TypeError('key must be either int or Tuple[int, str]')

    def __getitem__(self, key: Union[int, Tuple[int, str]]) -> Union[Dict[str, Pose], Pose]:
        if isinstance(key, tuple):  # pylint: disable=no-else-return
            timestamp, device_id = key
            return super().__getitem__(timestamp)[device_id]
        elif isinstance(key, int):
            return super().__getitem__(key)
        else:
            raise TypeError('key must be either int or Tuple[int, str]')

    def __delitem__(self, key: Union[int, Tuple[int, str]]):
        if isinstance(key, tuple):
            timestamp, device_id = key
            super().__getitem__(timestamp).__delitem__(device_id)
            if len(super().__getitem__(timestamp)) == 0:
                super().__delitem__(timestamp)
        elif isinstance(key, int):
            super().__delitem__(key)
        else:
            raise TypeError('key must be either int or Tuple[int, str]')

    def key_pairs(self) -> List[Tuple[int, str]]:
        return [
            (timestamp, device_id)
            for timestamp, devices in self.items()
            for device_id in devices.keys()
        ]

    def transform(self, T: Pose, left=True) -> 'Trajectories':
        transformed = self.__class__()
        for k in self.key_pairs():
            transformed[k] = (T * self[k]) if left else (self[k] * T)
        return transformed

    def __mul__(self, T: Pose):
        return self.transform(T, left=False)

    def __rmul__(self, T: Pose):
        return self.transform(T, left=True)

    @property
    def device_ids(self) -> Set[str]:
        return set(device_id for devices in self.values() for device_id in devices.keys())

    @classmethod
    def load(cls, path: Path) -> 'Trajectories':
        table = read_csv(path)
        trajectories = cls()
        for timestamp, device_id, *qt in table:
            timestamp = int(timestamp)
            trajectories[timestamp, device_id] = Pose.from_list(qt)
        return trajectories

    def save(self, path: Path):
        columns = ['timestamp', 'device_id', 'qw', 'qx', 'qy', 'qz', 'tx', 'ty', 'tz', '*covar']
        table = ([str(ts), device_id] + self[ts, device_id].to_list()
                 for ts, device_id in self.key_pairs())
        write_csv(path, table, columns=columns)

    def __contains__(self, key: Union[int, Tuple[int, str]]):
        if isinstance(key, tuple):  # pylint: disable=no-else-return
            timestamp, device_id = key
            return super().__contains__(timestamp) and self[timestamp].__contains__(device_id)
        elif isinstance(key, int):
            return super().__contains__(key)
        else:
            raise TypeError('key must be Union[int, Tuple[int, str]]')

    def __repr__(self) -> str:
        # [timestamp, device_id] = qw, qx, qy, qz, tx, ty, tz
        lines = [f'[ {timestamp:010}, {device_id:5}] = {pose}'
                 for timestamp, devices in self.items()
                 for device_id, pose in devices.items()]
        return '\n'.join(lines)
