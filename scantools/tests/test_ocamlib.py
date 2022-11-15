import pytest

import numpy as np

from ..scanners.navvis.ocamlib import (
    cam2world,
    world2cam,
    create_undistortion_LUT,
    undistort_point,
    distort_point
)
from ..scanners.navvis.camera_tiles import Tiles, TileFormat


##### TEST CAM2WORLD #####
# TODO get expected values for more points
def cam2world_input():
    return [400.0, 300.0]

def cam2world_expected_res():
    return [0.733011271294813, 0.549758453471110, -0.400574735838165]

# correct input dimensions, correct output values
@pytest.mark.parametrize(
    "input_point_2D, exp_point_backprojected",
    [
        # input dim: (1, 2); output dim: (1, 3);
        (np.array([cam2world_input()]), np.array([cam2world_expected_res()])),
        # input dim: (10, 2); output dim: (20, 3);
        (np.tile(cam2world_input(), (20, 1)), np.tile(cam2world_expected_res(), (20, 1))),
    ])
def test_cam2world(ocam_model, input_point_2D, exp_point_backprojected):

    res_point_backprojected = cam2world(input_point_2D, ocam_model)
    assert np.allclose(res_point_backprojected, exp_point_backprojected, 1e-5)


# correct input dimensions, incorrect output values
@pytest.mark.parametrize(
    "input_point_2D,exp_point_backprojected",
    [
        # input dim: (1, 2); incorrect output value;
        (np.array([[500.0, 300.0]]), np.array(
            [[0.56365445, 0.9284783, 0.128487345]])),

        # input dim: (2, 2);
        # output for first point correct, for second point incorrect (duplicated output of first);
        (np.array([cam2world_input(), [670.0, 400.0]]),
         np.tile(cam2world_expected_res(), (2, 1))),

        # input dim: (2, 2);
        # output for first point correct, for second incorrect: 0.3 difference in z;
        (np.tile(cam2world_input(), (2, 1)),
         np.array([cam2world_expected_res(), cam2world_expected_res() - 0.3 * np.array([0, 0, 1])])),
])
def test_cam2world_incorrect_input(
        ocam_model, input_point_2D, exp_point_backprojected):

    res_point_backprojected = cam2world(input_point_2D, ocam_model)
    assert not np.allclose(res_point_backprojected,
                           exp_point_backprojected, 1e-5)


@pytest.mark.parametrize(
    "input_point_2D",
    [
        # NoneType
        None,
        # empty list
        [],
        # correct dims, list instead of array
        [[1, 2], [3, 4]],
        # list of numpy array
        [np.array([[1, 2]])],
    ])
def test_cam2world_invalid_type(ocam_model, input_point_2D):
    with pytest.raises(TypeError):
        cam2world(input_point_2D, ocam_model)


@pytest.mark.parametrize(
    "input_point_2D",
    [
        # empty numpy array; dim: (0,)
        np.array([]),
        # incorrect dim length: 1, dim: (-1,)
        np.array([1, 2, 3, 4, 5]),
        # incorrect dim length: 3, dim: (-1, 2, 1)
        np.array([[1, 2]])[:, :, np.newaxis],
        # incorrect dim axis 0: (0, 2)
        np.array([[], []]).T,
        # incorrect dim axis 1: (2, 0)
        np.array([[], []]),
        # non uniform dim axis on 1, results in array of list: dim (-1,)
        # warning by numpy if dtype=object is ommited
        np.array([[1, 2], [1], [1, 2]], dtype=object),
    ])
def test_cam2world_incorrect_dims(ocam_model, input_point_2D):
    with pytest.raises(ValueError):
        cam2world(input_point_2D, ocam_model)


##### TEST WORLD2CAM #####
# TODO get expected values for more points and test correctness
def world2cam_input():
    return [1.0, 1.0, 1.0]

def world2cam_expected_res():
    return [4.729118411664447e+02, 3.929118411664447e+02]

@pytest.mark.parametrize(
    "input_point_3d, exp_point_projected",
    [
        # input dim: (1, 3); output dim: (1, 2);
        (np.array([world2cam_input()]), np.array([world2cam_expected_res()])),

        # input dim: (20, 2); output dim: (20, 3);
        (np.tile(world2cam_input(), (20, 1)), np.tile(world2cam_expected_res(), (20, 1))),
    ])
def test_world2cam(ocam_model, input_point_3d, exp_point_projected):

    res_point_projected = world2cam(input_point_3d, ocam_model)
    assert np.allclose(res_point_projected, exp_point_projected, 1e-4)


# correct input dimensions, incorrect output values
@pytest.mark.parametrize(
    "input_point_3d, exp_point_projected",
    [
        # input dim: (1, 3); incorrect output value;
        (np.array([[10.0, 20.0, 50.8]]), np.array([[0.982173, 5.19274823]])),

        # input dim: (2, 2);
        # output for first point correct,
        # output for second point incorrect
        # (duplicated output of first);
        (np.array([world2cam_input(), world2cam_input() + 4 * np.array([0, 0, 1])]),
         np.tile(world2cam_expected_res(), (2, 1))),

        # input dim: (2, 2);
        # output for first point correct, for second incorrect: 0.3 difference in y;
        (np.tile(world2cam_input(), (2, 1)),
         np.array([world2cam_expected_res(), world2cam_expected_res() - 0.3 * np.array([0, 1])])),
    ])
def test_world2cam_incorrect_input(
        ocam_model, input_point_3d, exp_point_projected):

    res_point_projected = world2cam(input_point_3d, ocam_model)
    assert not np.allclose(res_point_projected, exp_point_projected, 1e-4)

@pytest.mark.parametrize(
    "input_point_3d",
    [
        # NoneType
        None,
        # empty list
        [],
        # correct dims, list instead of array
        [[1, 2, 3], [4, 5, 6]],
        # list of numpy array
        [np.array([[1, 2, 3]])],
])
def test_world2cam_invalid_type(ocam_model, input_point_3d):
    with pytest.raises(TypeError):
        world2cam(input_point_3d, ocam_model)


@pytest.mark.parametrize(
    "input_point_3d",
    [
        # empty numpy array; dim: (0,)
        np.array([]),
        # incorrect dim length: 1, dim: (-1,)
        np.array([1, 2, 3, 4, 5]),
        # incorrect dim length: 3, dim: (-1, 3, 1)
        np.array([[1, 2, 3]])[:, :, np.newaxis],
        # incorrect dim axis 0: (0, 3)
        np.array([[], [], []]).T,
        # incorrect dim axis 1: (3, 0)
        np.array([[], [], []]),
        # non uniform dim axis on 1, results in array of list: dim (-1,)
        # warning by numpy if dtype=object is ommited
        np.array([[1, 2, 3], [1, 2], [1, 2, 3]], dtype=object),
])
def test_world2cam_incorrect_dims(ocam_model, input_point_3d):
    with pytest.raises(ValueError):
        world2cam(input_point_3d, ocam_model)


@pytest.fixture
def world2cam_testdata_norm_zero(ocam_model):

    zero_norm_expected_res = [ocam_model['xc'], ocam_model['yc']]

    testdata = [
        # one point with zero norm
        (np.zeros((1, 3)), np.array(zero_norm_expected_res)),
        # multiple points, all with zero norms
        (np.zeros((20, 3)), np.tile(zero_norm_expected_res, (20, 1))),
    ]

    # multiple points, some zero and some non-zero norm
    multiple_pts_input = np.zeros((5, 3))
    multiple_pts_expected_res = np.tile(zero_norm_expected_res, (5, 1))
    non_zero_norm_idx = [1, 3]
    multiple_pts_input[non_zero_norm_idx, :] = world2cam_input()
    multiple_pts_expected_res[non_zero_norm_idx, :] = world2cam_expected_res()

    # add input to test data
    testdata.append((multiple_pts_input, multiple_pts_expected_res))

    return testdata


def test_world2cam_norm_zero(ocam_model, world2cam_testdata_norm_zero):

    for (input_point_3d, exp_point_projected) in world2cam_testdata_norm_zero:
        res_point_projected = world2cam(input_point_3d, ocam_model)
        assert np.allclose(res_point_projected, exp_point_projected, 1e-6)


##### TEST CREATE_UNDISTORTION_LUT #####

def test_create_undistortion_LUT(ocam_model):

    tiles = Tiles("M6", ocam_model['width'], ocam_model['height'], tile_format=TileFormat.TILES_3x3)

    exp_filename = "./test_data/ocamlib_create_undistortion_lut_expected/lut_" + tiles.format + "_angles_"
    for angles_id, angles in enumerate(tiles.angles):
        exp_m = np.load(exp_filename + str(angles_id) + ".npy")

        exp_mapx = exp_m[0, :, :]
        exp_mapy = exp_m[1, :, :]

        res_mapx, res_mapy = create_undistortion_LUT(
            ocam_model, tiles.width, tiles.height, tiles.zoom_factor, angles)

        assert np.allclose(res_mapx, exp_mapx, 1e-6)
        assert np.allclose(res_mapy, exp_mapy, 1e-6)


def test_single_undist_point(ocam_model_navvis):
    point2D = np.zeros(2)
    point2D[0] = 2396.5
    point2D[1] = 1680.5

    # distort point
    point2D_dist = distort_point(point2D, ocam_model_navvis)

    # undistort point
    point2D_new = undistort_point(point2D_dist, ocam_model_navvis)

    assert np.allclose(point2D, point2D_new, 1e-6)


def test_single_undist_point_zoom_factor(ocam_model_navvis):
    # different zoom_factor
    point2D_dist = np.zeros(2)
    point2D_dist[0] = 2449.5
    point2D_dist[1] = 1647.5

    point2D_undist = undistort_point(point2D_dist,
                                     ocam_model_navvis,
                                     output_height=862,
                                     output_width=1148,
                                     zoom_factor=0.75)

    point2D_dist_new = distort_point(point2D_undist,
                                     ocam_model_navvis,
                                     output_height=862,
                                     output_width=1148,
                                     zoom_factor=0.75)

    assert np.allclose(point2D_dist, point2D_dist_new, 1e-6)
