import pytest
import numpy as np

from ..capture import Camera, Pose
from ..run_phone_to_capture import get_rot90, rotate_camera
from .test_capture_pose import Rz


def test_rotate_camera():
    state = np.random.RandomState(0)
    h = state.randint(100, 300)
    w = state.randint(100, 300)

    cx = state.randint(0, w)
    cy = state.randint(0, h)

    fx = state.randint(100, 1000)
    fy = state.randint(100, 1000)

    # to COLMAP coordinates: shift the principal points
    camera = Camera('PINHOLE', [w, h, fx, fy, cx+0.5, cy+0.5], None, 'camera')
    image = state.rand(h, w, 3)

    for rot in [-1, 0, 1, 2, 3, 4]:
        camera2 = rotate_camera(camera, rot)
        image2 = np.rot90(image, rot)

        w2 = camera2.width
        h2 = camera2.height
        fx2, fy2, cx2, cy2 = camera2.params

        v = image[cy, cx]
        v_rot = image2[int(cy2-0.5), int(cx2-0.5)]  # from COLMAP coordinates

        np.testing.assert_allclose(image2.shape[:2], (h2, w2))
        np.testing.assert_allclose(fx*fy, fx2*fy2)
        np.testing.assert_allclose(v, v_rot)


@pytest.mark.parametrize(
    "rot_cam2world,expected",
    [
        (Rz(0), 2),
        (Rz(90), 1),
        (Rz(180), 0),
        (Rz(270), 3),
    ])
def test_get_rot90(rot_cam2world, expected):
    pose_cam2world = Pose(rot_cam2world)
    result = get_rot90(pose_cam2world)
    np.testing.assert_allclose(expected, result)
