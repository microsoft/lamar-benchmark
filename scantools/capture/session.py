import logging
from pathlib import Path
from dataclasses import dataclass, fields, Field
from typing import Dict, Optional, Union, get_origin, get_args
from enum import Enum

from .sensors import Sensors, Camera
from .rigs import Rigs
from .trajectories import Trajectories
from .records import RecordsBluetooth, RecordsCamera, RecordsDepth, RecordsLidar, RecordsWifi
from .proc import Proc
from .pose import Pose

logger = logging.getLogger(__name__)

class Device(Enum):
    NAVVIS = 'NAVVIS'
    HOLOLENS = 'HOLOLENS'
    PHONE = 'PHONE'
    UNDEFINED = 'UNDEFINED'

    @classmethod
    def from_id(cls, id_: str):
        if id_ is not None:
            if id_.startswith('hl'):
                return Device.HOLOLENS
            if id_.startswith('ios'):
                return Device.PHONE
        return Device.UNDEFINED


# TODO: inherit Rigs, Trajectories, and Records from a common abstract class
@dataclass
class Session:
    sensors: Sensors
    rigs: Optional[Rigs] = None
    trajectories: Optional[Trajectories] = None
    images: Optional[RecordsCamera] = None
    depths: Optional[RecordsDepth] = None
    pointclouds: Optional[RecordsLidar] = None
    wifi: Optional[RecordsWifi] = None
    bt: Optional[RecordsBluetooth] = None
    proc: Optional[Proc] = None
    id: Optional[str] = None

    data_dirname = 'raw_data'
    proc_dirname = 'proc'
    Device = Device  # pylint: disable=invalid-name

    def __post_init__(self):
        all_devices = set(self.sensors.keys())
        if self.rigs is not None:
            assert len(self.sensors.keys() & self.rigs.keys()) == 0
            all_devices |= self.rigs.keys()
        if self.trajectories is not None:
            assert len(self.trajectories.device_ids - all_devices) == 0

    @property
    def cameras(self) -> Dict[str, Camera]:
        if self.sensors is None:
            return {}
        return {k: v for k, v in self.sensors.items() if isinstance(v, Camera)}

    @property
    def device(self) -> Device:
        return Device.from_id(self.id)

    @classmethod
    def filename(cls, attr: Union[Field, str]) -> str:
        name = attr.name if isinstance(attr, Field) else attr
        if name == 'proc':
            return 'proc/'
        return f'{name}.txt'

    @classmethod
    def load(cls, path: Path, wireless=True) -> 'Session':
        if not path.exists():
            raise IOError(f'Session directory does not exists: {path}')
        data = {}
        for attr in fields(cls):
            if attr.name == 'id':
                continue
            filepath = path / cls.filename(attr)
            if not filepath.exists():
                continue

            # Hack to get the classes from the attributes
            type_ = attr.type
            if get_origin(type_) is Union:
                type_ = (get_args(type_))[0]
            if attr.name in ['images', 'depths', 'pointclouds']:
                obj = type_.load(filepath, path / cls.data_dirname)
            else:
                if attr.name in ['bt', 'wifi'] and not wireless:
                    continue
                obj = type_.load(filepath)
            data[attr.name] = obj
        if 'sensors' not in data:
            raise ValueError(f'No sensor file for session at path {path}.')
        return cls(**data, id=path.name)

    def get_pose(self, ts: int, sensor_id: str, poses: Optional[Trajectories] = None) -> Pose:
        if poses is None:
            poses = self.trajectories
        try:
            ids_to_pose = poses[ts]
        except KeyError:
            raise ValueError(f'No pose for timestamp {ts}.')
        try:
            pose = ids_to_pose[sensor_id]
        except KeyError:
            try:
                valid_rig_id = None
                for rig_id in ids_to_pose:
                    if sensor_id in self.rigs[rig_id]:
                        # Here we assume that a sensor is part of at most one rig at a timestamp.
                        T_sensor2rig = self.rigs[rig_id, sensor_id]
                        valid_rig_id = rig_id
                        break
                T_rig2world = ids_to_pose[valid_rig_id]
            except KeyError:
                raise ValueError(f'No pose or rig for {sensor_id} at timestamp {ts}.')
            pose = T_rig2world * T_sensor2rig
        return pose

    def save(self, path: Path, overwrite : bool = True):
        path.mkdir(exist_ok=True, parents=True)
        for attr in fields(self):
            if attr.name == 'id':
                continue
            data = getattr(self, attr.name)
            if data is None:
                continue
            filepath = path / self.filename(attr)
            if not overwrite and filepath.exists() and attr.name != 'proc':
                raise IOError(f'File exists: {filepath}')
            data.save(filepath)
        self.id = path.name

    def __repr__(self) -> str:
        strs = []
        for attr in fields(self):
            data = getattr(self, attr.name)
            if data is None:
                continue
            strs.append(f'[{attr.name}] = {data}')
        return '\n'.join(strs)
