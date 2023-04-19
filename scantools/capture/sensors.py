# Copyright 2020-present NAVER Corp. Under BSD 3-clause license
# Modified by Paul-Edouard Sarlin (ETH Zurich)

from pathlib import Path
from typing import Union, Optional, List, Dict
from functools import cached_property
import numpy as np

from ..utils.io import read_csv, write_csv
from ..utils.colmap import CameraModel, CAMERA_MODEL_NAMES


class Sensor:
    def __init__(self,
                 sensor_type: str,
                 sensor_params: Optional[list] = None,
                 name: Optional[str] = None):
        self.name = name
        self._sensor_type = sensor_type
        self.sensor_params = sensor_params or []

    @property
    def sensor_type(self) -> str:
        return self._sensor_type

    def to_list(self) -> List[str]:
        fields = [self.name or '']
        fields += [self.sensor_type]
        fields += [str(v) for v in self.sensor_params]
        return fields

    def __repr__(self) -> str:
        representation = ''
        representation += f'name: {self.name or "--":5}, '
        representation += f'type: {self.sensor_type:6}, '
        representation += '[{}]'.format(', '.join(f'{i:3}' for i in self.sensor_params))
        return representation


# Details at https://github.com/colmap/colmap/blob/master/src/base/camera_models.h.
CAMERA_MODEL_PARAM_NAMES = {
    'SIMPLE_PINHOLE': ['f', 'cx', 'cy'],
    'PINHOLE': ['fx', 'fy', 'cx', 'cy'],
    'SIMPLE_RADIAL': ['f', 'cx', 'cy', 'k'],
    'RADIAL': ['f', 'cx', 'cy', 'k1', 'k2'],
    'OPENCV': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2'],
    'OPENCV_FISHEYE': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'k3', 'k4'],
    'FULL_OPENCV': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6'],
    'FOV': ['fx', 'fy', 'cx', 'cy', 'omega'],
    'SIMPLE_RADIAL_FISHEYE': ['f', 'cx', 'cy', 'k'],
    'RADIAL_FISHEYE': ['f', 'cx', 'cy', 'k1', 'k2'],
    'THIN_PRISM_FISHEYE': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'sx1', 'sy1']
}


# TODO: add method to (un)distort points, get camera matrix
class Camera(Sensor):
    def __init__(self,
                 camera_model: Union[CameraModel, str],
                 camera_params: List,
                 name: Optional[str] = None,
                 sensor_type: str = 'camera'):
        if isinstance(camera_model, str):
            camera_model = CAMERA_MODEL_NAMES[camera_model]
        # check params are consistent with model
        assert isinstance(camera_params, list)
        assert len(camera_params) == camera_model.num_params + 2  # extra width and height
        assert sensor_type in ['camera', 'depth']
        camera_params = list(map(str, camera_params))  # convert to str before next checks
        assert camera_params[0].isnumeric()  # width
        assert camera_params[1].isnumeric()  # height
        assert len(camera_params[2:]) == len(CAMERA_MODEL_PARAM_NAMES[camera_model.model_name])

        # make sure it crashes if camera_params cannot be cast to float, store as string
        camera_params = [float(v) for v in camera_params]
        camera_params = [str(int(v)) if v.is_integer() else str(v) for v in camera_params]
        sensor_params = [camera_model.model_name] + camera_params
        super().__init__(sensor_type=sensor_type, sensor_params=sensor_params,
                         name=name)

    @cached_property
    def model_name(self) -> str:
        return self.sensor_params[0]

    @cached_property
    def model(self) -> CameraModel:
        return CAMERA_MODEL_NAMES[self.sensor_params[0]]

    @cached_property
    def width(self) -> int:
        return int(self.sensor_params[1])

    @cached_property
    def height(self) -> int:
        return int(self.sensor_params[2])

    @cached_property
    def params(self) -> List[float]:
        return [float(p) for p in self.sensor_params[3:]]

    @cached_property
    def params_dict(self) -> Dict[str, float]:
        return dict(zip(CAMERA_MODEL_PARAM_NAMES[self.model_name()],
                        self.params))

    @cached_property
    def projection_params(self) -> List[float]:
        if self.model_name in {'SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'RADIAL',
                               'SIMPLE_RADIAL_FISHEYE', 'RADIAL_FISHEYE'}:
            f, cx, cy, *_ = self.params
            fx = fy = f
        elif self.model_name in {'PINHOLE', 'OPENCV', 'OPENCV_FISHEYE',
                                 'FULL_OPENCV', 'FOV', 'THIN_PRISM_FISHEYE'}:
            fx, fy, cx, cy, *_ = self.params
        else:
            raise ValueError('Unsupported camera type.')
        return [fx, fy, cx, cy]

    @cached_property
    def f(self) -> np.ndarray:
        return np.array(self.projection_params[:2])

    @cached_property
    def c(self) -> np.ndarray:
        return np.array(self.projection_params[2:])

    @cached_property
    def size(self) -> np.ndarray:
        return np.array([self.width, self.height])

    @property
    def K(self) -> np.ndarray:
        fx, fy, cx, cy = self.projection_params
        return np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])

    @property
    def asdict(self) -> dict:
        return {
            'model': self.model_name,
            'width': self.width,
            'height': self.height,
            'params': self.params
        }

    def world2image(self, pts: np.ndarray) -> np.ndarray:
        if self.model_name not in {'SIMPLE_PINHOLE', 'PINHOLE'}:
            raise ValueError('Unsupported camera type.')
        return (pts * self.f) + self.c

    def image2world(self, pts: np.ndarray) -> np.ndarray:
        if self.model_name not in {'SIMPLE_PINHOLE', 'PINHOLE'}:
            raise ValueError('Unsupported camera type.')
        return (pts - self.c) / self.f

    def in_image(self, pts: np.ndarray):
        return np.all((pts >= 0) & (pts <= (self.size - 1)), -1)


def create_sensor(sensor_type: str,
                  sensor_params: Optional[list] = None,
                  name: Optional[str] = None):
    if sensor_type in ['camera', 'depth']:
        if sensor_params is None or len(sensor_params) < 1:
            raise ValueError('sensor_params is requried for camera or depth sensor.')
        camera_model, *camera_params = sensor_params
        return Camera(camera_model, camera_params, name=name, sensor_type=sensor_type)

    return Sensor(sensor_type, sensor_params=sensor_params, name=name)


class Sensors(Dict[str, Sensor], dict):
    def __setitem__(self, sensor_id: str, sensor: Sensor):
        # enforce type checking
        if not isinstance(sensor_id, str):
            raise TypeError('invalid type for sensor_id')
        if not isinstance(sensor, Sensor):
            raise TypeError('invalid type of sensor')
        super(Sensors, self).__setitem__(sensor_id, sensor)

    @classmethod
    def load(cls, path: Path) -> 'Sensors':
        table = read_csv(path)
        sensors = cls()
        for sensor_id, name, sensor_type, *sensor_params in table:
            sensors[sensor_id] = create_sensor(sensor_type, sensor_params, name)
        return sensors

    def save(self, path: Path):
        columns = ['sensor_id', 'name', 'sensor_type', '[sensor_params]+']
        table = ([sensor_id] + sensor.to_list() for sensor_id, sensor in self.items())
        write_csv(path, table, columns=columns)

    def __repr__(self) -> str:
        representation = '\n'.join(
            f'[{sensor_id:5}] = {sensor}' for sensor_id, sensor in self.items()
        )
        return representation
