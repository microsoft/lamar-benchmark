""" Tests for navvis.py """
from pathlib import Path
import numpy as np
import pytest

from ..scanners.navvis.navvis import NavVis


##### Tests for M6 class in scanners/navvis/M6.py. #####

# Test data hardcoded in a M6Data class, in order to test parsing
# of the ground truth NavVis test files using M6.py.

class M6Data:
    """ Represents NavVis ground truth data. Data copied from test files in: test_data/navvis_files.

        Attributes:
            frame       TestFrame object containing test frame data.

            cameras     List of TestCamera objects containing camera test data.
                        cameras stores data for 2 cameras: "cam0" and "cam1"
    """
    def __init__(self):
        self.frame = self.TestFrame()
        self.cameras = [# values of ocam model with id "cam0"
                        self.TestOCam(camera_id=0,
                                      coeff_c=1.000512,
                                      coeff_d=-0.001413,
                                      coeff_e=0.001454,
                                      cx=1745.644423,
                                      cy=2289.580521,
                                      cam2world=[-1983.406, 0, 0.0002165983,
                                                 -4.003405e-08, 1.616231e-11],
                                      world2cam=[2897.294809, 1592.573894, -76.269834,
                                                 253.324578, 115.369911, -10.80424,
                                                 61.389893, 22.740515, -17.282497,
                                                 14.429784, 3.529446, -0.179122,
                                                 12.644676, -0.663886, -8.260312000000001,
                                                 -0.208706, 2.623593, 0.723404]),
                        # values of ocam model with id "cam1"
                        self.TestOCam(camera_id=1,
                                      coeff_c=1.000293,
                                      coeff_d=-0.000795,
                                      coeff_e=0.000857,
                                      cx=1740.703503,
                                      cy=2291.373174,
                                      cam2world=[-1984.06, 0, 0.0002122601,
                                                 -3.591422e-08, 1.507185e-11],
                                      world2cam=[2904.926365, 1612.176004, -60.696503,
                                                 252.13943, 117.566538, -5.873288,
                                                 57.694283, 25.18299, -13.740347,
                                                 9.38533, 2.828129, 3.966487,
                                                 12.586478, -2.415703, -8.248135,
                                                 0.186093, 2.656615, 0.70158])
                        ]

    def get_camera_ids(self):

        res = []
        for camera_id in range(6):
            res.append("cam" + str(camera_id))
        return res


    def get_camera_indexes(self):
        return range(6)


    class TestOCam:
        """
        Represents NavVis camera ground truth data. Data copied from test file 'sensor_frame.xml'.

        Attributes:

            camera_id:    value of tag <SensorName>

            All camera object attributes contain values
            with respect to the same camera_id.

            coeff_c         float, value of tag <c> in <OCamModel>
            coeff_d         float, value of tag <d> in <OCamModel>
            coeff_e         float, value of tag <e> in <OCamModel>
            cx              float, value of tag <cx> in <OCamModel>
            cy              float, value of tag <cy> in <OCamModel>
            cam2world       list of all <coeff> in <cam2world> in <OCamModel>
            world2cam       list of all <coeff> in <world2cam> in <OCamModel>

        """
        def __init__(self, camera_id, coeff_c, coeff_d, coeff_e, cx, cy, cam2world, world2cam):
            self.id = camera_id
            self.coeff_c = coeff_c
            self.coeff_d = coeff_d
            self.coeff_e = coeff_e
            self.cx = cx
            self.cy = cy

            self.cam2world = cam2world
            self.world2cam = world2cam

            self.width = 4592
            self.height = 3448

            self.upright = False


    class TestFrame:
        """
        Represents NavVis frame ground truth data. Data copied from file: 'info/00013-info.json'.

        Attributes:

            id              Integer with value 13, copied from test filename
            timestamp       Float, value copied from file contents
            valid           String, value copied from file contents
            values          List of all json keys contained in test file
            pose            TestPose Object

        """

        def __init__(self):
            self.id = 13
            self.timestamp = 1596013970.590877
            self.valid = 'true'
            self.values = ['cam0', 'cam1', 'cam2', 'cam3', 'cam4', 'cam5',
                           'cam_head', 'footprint', 'timestamp', 'valid']

            self.pose = self.TestPose()


        class TestPose:
            """
            Represents NavVis camera pose ground truth data.
            Contains copied data from json key "cam0".

            Attributes:

            camera_id       String with value "cam0"
            qvec            Numpy ndarray of shape (4,)
                            Value of "cam0"["quaternion"]
            tvec            Numpy ndarray of shape (3,)
                            Value of "cam0"["position"]
            rotmat          Numpy ndarray of shape (3, 3)
                            Rotation matrix calculated from attribute qvec
            """
            def __init__(self):
                self.camera_id = 0
                self.image_filename = "00013-cam0.jpg"
                self.qvec = np.array([0.08952181860377329, -0.6952457665956079,
                                      0.09034806860960908, 0.7074294272386978])
                self.tvec = np.array([-1.2299637315581074, 2.962799296764929, 1.8473998446586065])
                # Computed from qvec
                self.rotmat = np.array([[-0.01723834, -0.25228896, -0.96749838],
                                        [0.00103251, -0.96764614, 0.2523091],
                                        [-0.99985088, 0.00335043, 0.0169411]])


# M6 objects and test M6Data objects are instantiated per
# test function through fixtures (no variable sharing).

@pytest.fixture(name="m6_testdata")
def fixture_m6_testdata():
    return M6Data()


@pytest.fixture(name="m6_input_path")
def fixture_m6_input_path():
    return Path("./test_data/navvis_files/")


@pytest.fixture(name="m6_output_path")
def fixture_m6_output_path(tmp_path):
    return tmp_path


@pytest.fixture(name="m6_object")
def fixture_m6_object(m6_input_path, m6_output_path):
    m6 = NavVis(m6_input_path, m6_output_path)

    m6.load_cameras()
    m6.load_frames()
    # m6.load_pointcloud()

    return m6

####### TESTS #######


def test_get_frames(m6_object, m6_testdata):
    test_frame = m6_testdata.frame
    # testing of frame values done in test_get_frame
    res_frames = m6_object.get_frames()
    assert isinstance(res_frames, dict)
    assert list(res_frames.keys())[0] == test_frame.id

    res_frame = res_frames[test_frame.id]
    assert isinstance(res_frame, dict)
    assert len(res_frame.keys()) == len(test_frame.values)


def test_get_frame(m6_object, m6_testdata):
    test_frame = m6_testdata.frame
    test_camera_id = "cam" + str(test_frame.pose.camera_id)
    res_frame = m6_object.get_frame(test_frame.id)[test_camera_id]

    assert len(res_frame['position']) == test_frame.pose.tvec.shape[0]
    assert np.array_equal(np.array(res_frame['position']), test_frame.pose.tvec)
    assert len(res_frame['quaternion']) == test_frame.pose.qvec.shape[0]
    assert np.array_equal(np.array(res_frame['quaternion']), test_frame.pose.qvec)


def test_get_frame_ids(m6_object, m6_testdata):
    exp_frame_id = m6_testdata.frame.id

    res_frame_ids = m6_object.get_frame_ids()
    assert isinstance(res_frame_ids, list)
    assert len(res_frame_ids) == 1
    assert all([isinstance(res_frame_id, int) for res_frame_id in res_frame_ids])
    assert exp_frame_id in res_frame_ids


def test_get_frame_timestamp(m6_object, m6_testdata):
    exp_frame_id = m6_testdata.frame.id
    exp_timestamp = m6_testdata.frame.timestamp

    res_timestamp = m6_object.get_frame_timestamp(exp_frame_id)
    assert isinstance(res_timestamp, float)
    assert exp_timestamp == res_timestamp


def test_get_frame_valid(m6_object, m6_testdata):
    exp_frame_valid = m6_testdata.frame.valid

    res_frame_valid = m6_object.get_frame_valid(m6_testdata.frame.id)
    assert isinstance(res_frame_valid, str)
    assert exp_frame_valid == res_frame_valid


def test_get_frame_values(m6_object, m6_testdata):
    exp_frame_values = m6_testdata.frame.values

    res_frame_values = m6_object.get_frame_values()
    assert isinstance(res_frame_values, list)
    assert len(exp_frame_values) == len(res_frame_values)
    assert all([res_frame_val in exp_frame_values for res_frame_val in res_frame_values])


def test_get_cameras(m6_object, m6_testdata):
    # testing of camera values done in test_get_camera
    res_cameras = m6_object.get_cameras()
    assert isinstance(res_cameras, dict)
    assert len(res_cameras) == 6
    assert all([res_cam_id in m6_testdata.get_camera_ids() for res_cam_id in res_cameras.keys()])
    assert all([isinstance(res_cam_val, dict) for res_cam_val in res_cameras.values()])


def test_get_camera(m6_object, m6_testdata):
    test_cameras = m6_testdata.cameras

    for test_camera in test_cameras:
        res_camera = m6_object.get_camera(test_camera.id)

        assert len(res_camera.keys()) == 12
        assert res_camera['width'] == test_camera.width
        assert res_camera['height'] == test_camera.height
        assert res_camera['c'] == test_camera.coeff_c
        assert res_camera['d'] == test_camera.coeff_d
        assert res_camera['e'] == test_camera.coeff_e
        assert res_camera['xc'] == test_camera.cx
        assert res_camera['yc'] == test_camera.cy
        assert res_camera['length_pol'] == len(test_camera.cam2world)
        assert res_camera['length_invpol'] == len(test_camera.world2cam)
        assert res_camera['upright'] == test_camera.upright
        assert all([res_coeff == exp_coeff
                    for (res_coeff, exp_coeff) in zip(res_camera['pol'], test_camera.cam2world)])
        assert all([res_coeff == exp_coeff
                    for (res_coeff, exp_coeff) in zip(res_camera['invpol'], test_camera.world2cam)])


def test_get_camera_ids(m6_object, m6_testdata):
    exp_camera_ids = m6_testdata.get_camera_ids()

    res_camera_ids = m6_object.get_camera_ids()
    assert isinstance(res_camera_ids, list)
    assert len(exp_camera_ids) == len(res_camera_ids)
    assert all([res_cam_id in exp_camera_ids for res_cam_id in res_camera_ids])


def test_get_camera_indexes(m6_object, m6_testdata):
    exp_camera_indexes = m6_testdata.get_camera_indexes()

    res_camera_indexes = m6_object.get_camera_indexes()
    assert isinstance(res_camera_indexes, range)
    assert len(exp_camera_indexes) == len(res_camera_indexes)
    assert all([res_cam_idx in exp_camera_indexes for res_cam_idx in res_camera_indexes])


def test_get_camera_intrinsics(m6_object, m6_testdata):

    input_camera = m6_testdata.cameras[0]

    input_camera_cx = (input_camera.width - 1.0) / 2
    input_camera_cy = (input_camera.height - 1.0) / 2

    zoom_factor = 4
    input_camera_f = input_camera.width / zoom_factor

    exp_intrinsic_matrix = np.array([[input_camera_f, 0., input_camera_cx],
                                     [0., input_camera_f, input_camera_cy],
                                     [0., 0., 1.]])
    res_intrinsic_matrix = m6_object.get_camera_intrinsics()

    assert isinstance(res_intrinsic_matrix, np.ndarray)
    assert np.array_equal(res_intrinsic_matrix, exp_intrinsic_matrix)


def test_get_pose(m6_object, m6_testdata):

    exp_pose_qvec = m6_testdata.frame.pose.qvec
    exp_pose_tvec = m6_testdata.frame.pose.tvec

    res_pose_qvec, res_pose_tvec = m6_object.get_pose(m6_testdata.frame.id,
                                                      m6_testdata.frame.pose.camera_id)

    # check returned type
    assert isinstance(res_pose_qvec, np.ndarray)
    assert isinstance(res_pose_qvec, np.ndarray)

    # check returned values
    assert np.allclose(exp_pose_qvec, res_pose_qvec, 1e-10)
    assert np.allclose(exp_pose_tvec, res_pose_tvec, 1e-10)


# TODO
# def test_get_pose_tile():
#     pass


def test_get_pose_as_matrix(m6_object, m6_testdata):

    test_frame = m6_testdata.frame
    exp_pose_tvec = test_frame.pose.tvec
    exp_pose_rotmat = test_frame.pose.rotmat

    exp_pose_matrix = np.zeros((4, 4))
    for i in range(3):
        for j in range(3):
            exp_pose_matrix[i][j] = exp_pose_rotmat[i][j]

    exp_pose_matrix[0][3] = exp_pose_tvec[0]
    exp_pose_matrix[1][3] = exp_pose_tvec[1]
    exp_pose_matrix[2][3] = exp_pose_tvec[2]
    exp_pose_matrix[3][3] = 1.

    res_pose_matrix = m6_object.get_pose_as_matrix(test_frame.id, test_frame.pose.camera_id)

    assert isinstance(res_pose_matrix, np.ndarray)
    assert res_pose_matrix.dtype == np.float64
    assert np.allclose(res_pose_matrix, exp_pose_matrix, 1e-10)


def test_get_input_image_path(m6_object, m6_input_path, m6_testdata):
    test_frame = m6_testdata.frame
    exp_input_image_path = m6_input_path / "cam" /test_frame.pose.image_filename

    res_input_image_path = m6_object.get_input_image_path(test_frame.id, test_frame.pose.camera_id)
    assert exp_input_image_path.resolve() == res_input_image_path.resolve()


def test_get_output_image_path(m6_object, m6_output_path, m6_testdata):
    test_frame = m6_testdata.frame
    exp_output_image_path = m6_output_path / "images_undistr" / test_frame.pose.image_filename

    res_output_image_path = m6_object.get_output_image_path(test_frame.id,
                                                            test_frame.pose.camera_id)
    assert exp_output_image_path.resolve() == res_output_image_path.resolve()


# TODO test undistortion functions

# def test_undistort():
#     pass

# def test_undistort_tiles():
#     pass

# def test_get_tiles():
#     pass

# def test_set_tiles_format():
#     pass


def test_get_number_of_images(m6_object, m6_testdata):
    exp_number_of_images = len(m6_testdata.get_camera_ids()) # * num_frames, but 1 test frame

    res_number_of_images = m6_object.get_number_of_images()
    assert isinstance(res_number_of_images, int)
    assert exp_number_of_images == res_number_of_images


def test_get_image_filename(m6_object, m6_testdata):
    test_frame = m6_testdata.frame
    exp_image_filename = test_frame.pose.image_filename

    res_image_filename = m6_object.get_image_filename(test_frame.id, test_frame.pose.camera_id)
    assert res_image_filename == exp_image_filename
