import pytest
import numpy as np
import os
from scipy.spatial.transform import Rotation

from ..capture import Pose
from ..capture import NamedPoses

def Rz(theta):
    return Rotation.from_euler('z', theta, degrees=True)


@pytest.mark.parametrize(
    "rot,tvec",
    [
        (['1', '0', '0', '0'], ['0', '1', '1']),
        (Rz(30), ['1', '1', '0'])
    ])
def test_pose_valid(rot, tvec, tmp_path):
    temp_file_path = tmp_path / 'named_poses.csv'
    pose = Pose(rot, tvec)
    poses = NamedPoses()
    poses['pose1'] = pose
    poses['pose2'] = pose
    poses.save(temp_file_path)
    loaded_poses = NamedPoses()
    loaded_poses.load(temp_file_path)
    assert len(poses) == len(loaded_poses)
    assert poses.keys() == loaded_poses.keys()
    for k in poses.keys():
        assert poses[k].to_list() == loaded_poses[k].to_list()
    os.remove(temp_file_path)



