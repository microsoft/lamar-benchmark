import pytest
import numpy as np
from scipy.spatial.transform import Rotation

from ..capture import Pose

def Rz(theta):
    return Rotation.from_euler('z', theta, degrees=True)


@pytest.mark.parametrize(
    "rot,tvec",
    [
        (None, None),
        ([1, 0, 0, 0], None),
        (np.array([1, 0, 0, 0]), None),
        (None, [0, 0, 0]),
        (None, np.array([0, 0, 0])),
        (['1', '0', '0', '0'], ['1', '1', '1']),
        (np.eye(3), None),
        (Rz(30), None),
    ])
def test_pose_valid(rot, tvec):
    Pose(rot, tvec)


@pytest.mark.parametrize(
    "rot,tvec",
    [
        ([1, 0, 0, 0, 0], None),  # qvec shape != 4
        (np.ones(5), None),
        (np.array([]), None),
        (np.array([[], [], [], []]), None),
        (np.eye(4), None),  # rotmat shape != (3, 3)
        (np.zeros(4), None),  # invalid qvec
        (np.array([1, 0, 0, np.nan]), None),
        ([1, 0, 0, None], None),
        (None, np.ones(4)),  # tvec shape != 3
        (None, [1, 0, None]),  # invalid tvec
        (None, [1, 0, float('nan')]),
        (None, np.array([1, 0, np.nan])),
        (None, ['1', '2', 'error']),  # tvec cannot be cast to float
        (1, 2),
    ])
def test_pose_invalid(rot, tvec):
    with pytest.raises(ValueError):
        Pose(rot, tvec)


def test_pose_4x4mat():
    theta_z = np.deg2rad(68)
    diag = np.cos(theta_z)
    off_diag = np.sin(theta_z)
    #pylint: disable=bad-whitespace
    T = np.array([[diag,     -off_diag,  0,   0.5],
                  [off_diag,      diag,  0, -12.123],
                  [0,                 0, 1,   0],
                  [0,                 0, 0,   1]])

    pose = Pose.from_4x4mat(T)
    T_ret = pose.to_4x4mat()
    np.testing.assert_allclose(T, T_ret)


def test_pose_compose():
    z1 = 90
    z2 = 15
    pose1 = Pose(Rz(z1), [1, 0, 0])
    pose2 = Pose(Rz(z2), [2, 0, 3])
    composed = pose1 * pose2

    np.testing.assert_allclose(composed.t, [1, 2, 3])
    np.testing.assert_allclose(composed.r.as_euler('xyz', degrees=True), [0, 0, z1+z2])


def test_pose_inverse():
    z = 90
    pose = Pose(Rz(z), [1, 0, 0])
    inv = pose.inverse()

    np.testing.assert_allclose(inv.t, [0, 1, 0], atol=1e-10)
    np.testing.assert_allclose(inv.r.as_euler('xyz', degrees=True), [0, 0, -z])


@pytest.mark.parametrize(
    "p3d,rot,tvec,expected",
    [
        ([[0, 0, 0]], None, [1, 2, 3], [[1, 2, 3]]),
        ([[1, 2, 3]], None, [4, 5, 6], [[5, 7, 9]]),
        ([[1, 2, 3]], Rz(90), None, [[-2, 1, 3]]),
    ])
def test_transform_points(p3d, rot, tvec, expected):
    pose = Pose(rot, tvec)
    result = pose.transform_points(np.array(p3d))
    np.testing.assert_allclose(result, expected)
