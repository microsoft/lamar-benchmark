from pathlib import Path
from .pose import Pose
from ..utils.io import read_csv, write_csv


class NamedPoses(dict):
    def __setitem__(self, name: str, pose: Pose):
        if not isinstance(name, str):
            raise TypeError('expect str type as name')
        if not isinstance(pose, Pose):
            raise TypeError('expect Pose type as pose')
        super().__setitem__(name, pose)

    def load(self, path: Path) -> 'NamedPoses':
        table = read_csv(path)
        for name, *qt in table:
            self[name] = Pose.from_list(qt)

    def save(self, path: Path):
        columns = ['name', 'qw', 'qx', 'qy', 'qz', 'tx', 'ty', 'tz']
        table = []
        for name, pose in self.items():
            table.append([name] + pose.to_list())
        write_csv(path, table, columns=columns)
