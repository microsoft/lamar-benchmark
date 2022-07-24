# Copyright 2020-present NAVER Corp. Under BSD 3-clause license
# Modified by Paul-Edouard Sarlin (ETH Zurich)

from pathlib import Path
from typing import Dict, Union, Tuple, List, Set

from .pose import Pose
from ..utils.io import read_csv, write_csv


class Rigs(Dict[str, Dict[str, Pose]], dict):
    def __setitem__(self,
                    key: Union[str, Tuple[str, str]],
                    value: Union[Dict[str, Pose], Pose]):
        if isinstance(key, tuple):
            rig_device_id, sensor_id = key
            if not isinstance(value, Pose):
                raise TypeError('expect Pose type as value')
            self.setdefault(rig_device_id, {})[sensor_id] = value

        elif isinstance(key, str):
            rig_id = key
            if not isinstance(value, dict):
                raise TypeError('invalid value for rig id.')
            if not all(isinstance(k, str) for k in value.keys()):
                raise TypeError('invalid device_id')
            if not all(isinstance(v, Pose) for v in value.values()):
                raise TypeError('invalid Pose')
            super(Rigs, self).__setitem__(rig_id, value)
        else:
            raise TypeError('invalid key type for Rigs')

    def __getitem__(self, key: Union[str, Tuple[str, str]]) -> Union[Dict[str, Pose], Pose]:
        if isinstance(key, str):  # pylint: disable=no-else-return
            return super().__getitem__(key)
        elif isinstance(key, tuple):
            rig_id, sensor_id = key
            return super().__getitem__(rig_id)[sensor_id]
        else:
            raise TypeError('key must be either str or Tuple[str, str]')

    def key_pairs(self) -> List[Tuple[str, str]]:
        return [
            (rig_id, sensor_id)
            for rig_id, sensors in self.items()
            for sensor_id in sensors.keys()
        ]

    @property
    def sensor_ids(self) -> Set[str]:
        return set(sensor_id for sensors in self.values() for sensor_id in sensors.keys())

    @classmethod
    def load(cls, path: Path) -> 'Rigs':
        table = read_csv(path)
        rigs = cls()
        for rig_id, sensor_id, *qt in table:
            rigs[rig_id, sensor_id] = Pose.from_list(qt)
        return rigs

    def save(self, path: Path):
        columns = ['rig_id', 'sensor_id', 'qw', 'qx', 'qy', 'qz', 'tx', 'ty', 'tz']
        table = ([rig_id, sensor_id] + self[rig_id, sensor_id].to_list()
                 for rig_id, sensor_id in self.key_pairs())
        write_csv(path, table, columns=columns)

    def __repr__(self) -> str:
        poses = [f'[{rig_id:5}, {sensor_id:5}] = {pose}'
                 for rig_id, rig in self.items()
                 for sensor_id, pose in rig.items()]
        return '\n'.join(poses)
