from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
import logging

from .pose import Pose
from .trajectories import Trajectories
from ..utils.io import read_csv, write_csv


logger = logging.getLogger(__name__)


class GlobalAlignment(Dict[Tuple[str], Tuple[Pose, List[str]]], dict):
    no_ref = '__absolute__'  # after global alignment

    def get_abs_pose(self, label: str) -> Pose:
        key = (label, self.no_ref)
        if key in self:
            T_session2w, _ = self[key]
        else:
            T_session2w = None
        return T_session2w

    @classmethod
    def load(cls, path: Path) -> 'GlobalAlignment':
        table = read_csv(path)
        alignment = GlobalAlignment()
        for label, ref_id, *qt_info in table:
            qt, info = qt_info[:7], qt_info[7:]
            alignment[label, ref_id] = (Pose.from_list(qt), info)
        return alignment

    def save(self, path: Path):
        columns = ['label', 'reference_id', 'qw', 'qx', 'qy', 'qz', 'tx', 'ty', 'tz', '[info]+']
        table = (list(key) + pose.to_list() + list(map(str, info))
                 for key, (pose, info) in self.items())
        write_csv(path, table, columns=columns)


@dataclass
class Proc:
    meshes: Optional[Dict[str, Path]] = field(default_factory=dict)
    alignment_global: GlobalAlignment = field(default_factory=GlobalAlignment)
    alignment_trajectories: Optional[Trajectories] = None

    subsessions: Optional[List[str]] = None

    meshes_dirname = 'meshes'
    subsessions_filename = 'subsessions.txt'
    alignment_global_filename = 'alignment_global.txt'
    alignment_traj_filename = 'alignment_trajectories.txt'
    overlap_filename = 'overlaps.h5'

    @classmethod
    def load(cls, path: Path) -> 'Proc':
        args = {}

        meshes_path = path / cls.meshes_dirname
        if meshes_path.exists():
            args['meshes'] = {p.stem: p.relative_to(path) for p in meshes_path.glob('*.ply')}

        align_global_path = path / cls.alignment_global_filename
        if align_global_path.exists():
            args['alignment_global'] = GlobalAlignment.load(align_global_path)
        trajectories_path = path / cls.alignment_traj_filename
        if trajectories_path.exists():
            args['alignment_trajectories'] = Trajectories.load(trajectories_path)

        subsessions_path = path / cls.subsessions_filename
        if subsessions_path.exists():
            with open(subsessions_path, 'r') as f:
                args['subsessions'] = [line.strip('\n') for line in f.readlines()]

        return cls(**args)

    def save(self, path: Path):
        path.mkdir(exist_ok=True, parents=True)

        if self.meshes:
            for mesh_path in self.meshes.values():
                if not (path / mesh_path).exists():
                    logger.warning('%s does not exist in %s.', mesh_path, path)

        if self.alignment_global:
            self.alignment_global.save(path / self.alignment_global_filename)
        if self.alignment_trajectories:
            self.alignment_trajectories.save(path / self.alignment_traj_filename)

        if self.subsessions:
            subsessions_path = path / self.subsessions_filename
            with open(subsessions_path, 'w') as f:
                f.write('\n'.join(self.subsessions))
