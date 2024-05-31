import pytest
import numpy as np
import os
from scipy.spatial.transform import Rotation

from ..capture import Pose
from ..proc import GlobalAlignment

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
    global_alignment = GlobalAlignment()
    

    os.remove(temp_file_path)



