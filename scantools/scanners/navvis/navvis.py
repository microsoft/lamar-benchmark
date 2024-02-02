import json
import logging
import multiprocessing
from pathlib import Path, PurePath
from typing import Optional

from bs4 import BeautifulSoup
import numpy as np

from .camera_tiles import Tiles, TileFormat
from .ibeacon_parser import parse_navvis_ibeacon_packet, BluetoothMeasurement
from .iwconfig_parser import parse_iwconfig, WifiMeasurement
from . import ocamlib
from ...utils import transform
from ...utils.io import read_csv

logger = logging.getLogger(__name__)


class NavVis:
    def __init__(self, input_path: Path, output_path: Optional[Path] = None,
                 tiles_format: str = "none", upright: bool = False,
                 number_processes: int = multiprocessing.cpu_count()):

        self._input_path = None
        self._input_image_path = None
        self._input_json_path = None
        self._camera_file_path = None
        self._pointcloud_file_path = None
        self._trace_path = None

        self._output_path = None
        self._output_image_path = None
        self._output_LUT_path = None

        self.__cameras = {}
        self.__frames = {}
        self.__trace = {}

        # upright fix
        self.__upright = upright

        # number of processes used in parallel computing
        self.set_processes(number_processes)

        # paths
        self._set_dataset_paths(input_path, output_path, tiles_format)

        # load cameras (sensor_frame.xml)
        self.load_cameras()
        num_cameras = len(self.__cameras)
        if num_cameras == 4:
            self.__device = 'VLX'
        elif num_cameras == 6:
            self.__device = 'M6'
        else:
            raise ValueError(f'Unknown NavVis device with {num_cameras} cameras.')

        # load frames (*-info.json)
        self.load_frames()

        # set tiles format
        self.set_tiles_format(tiles_format)

        # mapping path: position, orientation, and magnetic field information in
        # frequent intervals
        self.load_trace()

    def _set_dataset_paths(self, input_path: Path, output_path: Optional[Path], tiles_format: str):
        # Dataset path
        self._input_path = Path(input_path).absolute()
        if not self._input_path.exists():
            raise FileNotFoundError(f'Input path {self._input_path}.')

        # Images path
        self._input_image_path = self._input_path / "cam"
        if not self._input_image_path.exists():
            raise FileNotFoundError(f'Input image path {self._input_image_path}.')

        # Json files path
        self._input_json_path = self._input_path / "info"
        if not self._input_json_path.exists():
            raise FileNotFoundError(f'Input json path {self._input_json_path}.')

        # Mapping Path: file trace.csv contains position, orientation, and
        # magnetic field information in frequent intervals
        self._trace_path = self._input_path / "artifacts" / "trace.csv"

        self._camera_file_path = self._input_path / 'sensor_frame.xml'

        # Pointcloud file path (default)
        pointcloud_file_path = self._input_path / "pointcloud.ply"
        if pointcloud_file_path.exists():
            self._pointcloud_file_path = pointcloud_file_path

        # Output path. If empty, set to same level as <input_path>
        if not output_path:
            output_path = self._input_path.parent / \
                (self._input_path.name + "_ms_eth_dataset")
        self._output_path = Path(output_path).absolute()
        self._output_path.mkdir(exist_ok=True, parents=True)

        # Undistorted images output path
        output_dir_postfix = "" if tiles_format.lower() == "none" else ("_" + tiles_format.lower())
        self._output_image_path = self._output_path / ("images_undistr" + output_dir_postfix)

        # LUT output path
        # TODO: move the LUTs somewhere else (not in sensors/)
        self._output_LUT_path = self._output_path / "LUT"

        logger.info(
            "NavVis dataset paths:\n"
            "  -- input path        = %s\n"
            "  -- input image path  = %s\n"
            "  -- input pointcloud  = %s\n"
            "  -- output path       = %s\n"
            "  -- output LUT path   = %s\n"
            "  -- output image path = %s",
            self._input_path, self._input_image_path, self._pointcloud_file_path,
            self._output_path, self._output_LUT_path, self._output_image_path)

    def load_frames(self):
        """ Load valid frames from: `<input_path>/info/XXXXX-info.json`.
            See NavVis documentation:
            https://docs.navvis.com/mapping/v2.8.2/en/html/dataset_description_m6.html?#capture-location-info-files
        """
        # get valid frame_ids parsing image folder
        valid_ids = sorted(self._input_image_path.glob("*-cam0.jpg"))
        for file_path in valid_ids:
            frame_id = int(file_path.name.split('-')[0])

            # verify if `frame_id` exists for all the cameras
            for cam in self.get_camera_ids():
                path = self._input_image_path / f'{frame_id:05d}-{cam}.jpg'
                if not path.exists():
                    raise FileNotFoundError(f'{path}.')

            # load json files
            json_path = self._input_json_path / f'{frame_id:05d}-info.json'
            with open(json_path, 'r') as json_fid:
                logger.debug("-- reading frame id: %d", frame_id)
                self.__frames[frame_id] = json.load(json_fid)


    def load_cameras(self):
        # read xml file 'sensor_frame.xml'
        with open(self._camera_file_path) as xml_file:
            xml = BeautifulSoup(xml_file, features="lxml")

            # cameras dict
            cameras = {}

            # get camera models
            camera_models = xml.find_all("cameramodel")
            for cam in camera_models:
                # current camera dict
                ocam_model = {}

                # cam2world
                coeff = cam.cam2world.find_all("coeff")
                ocam_model['pol'] = [float(d.string) for d in coeff]
                ocam_model['length_pol'] = len(coeff)

                # world2cam
                coeff = cam.world2cam.find_all("coeff")
                ocam_model['invpol'] = [float(d.string) for d in coeff]
                ocam_model['length_invpol'] = len(coeff)

                ocam_model['xc'] = float(cam.cx.string)
                ocam_model['yc'] = float(cam.cy.string)
                ocam_model['c'] = float(cam.c.string)
                ocam_model['d'] = float(cam.d.string)
                ocam_model['e'] = float(cam.e.string)
                ocam_model['height'] = int(cam.height.string)
                ocam_model['width'] = int(cam.width.string)
                if self.__upright:
                    # only switch height and width to update undistortion sizes
                    # rest stays the same since we pre-rotate the target coordinates
                    ocam_model['height'], ocam_model['width'] = (
                        ocam_model['width'], ocam_model['height'])
                ocam_model['upright'] = self.__upright

                sensorname = cam.sensorname.string
                cameras[sensorname] = ocam_model

        # save metadata inside the class
        self.__cameras = cameras

    def load_trace(self):
        expected_columns = [
            "nsecs",
            "x",
            "y",
            "z",
            "ori_w",
            "ori_x",
            "ori_y",
            "ori_z",
            "mag_x",
            "mag_y",
            "mag_z",
        ]
        input_filepath = self._input_path / "artifacts" / "trace.csv"
        rows = read_csv(input_filepath)
        rows = rows[1:]  # remove header

        # convert to dict
        trace = []
        for row in rows:
            row_dict = {column: value for column, value in zip(expected_columns, row)}
            trace.append(row_dict)

        self.__trace = trace

    def get_input_path(self):
        return self._input_path

    def get_pointcloud_path(self):
        return self._pointcloud_file_path

    def get_output_path(self):
        return self._output_path
    
    def get_device(self):
        return self.__device

    def get_frames(self):
        return self.__frames

    def get_frame(self, frame_id):
        return self.__frames.get(frame_id)

    def get_frame_ids(self):
        return list(self.__frames.keys())

    def get_frame_timestamp(self, frame_id):
        return self.__frames[frame_id]['timestamp']

    def get_frame_valid(self, frame_id):
        return self.__frames[frame_id]['valid']

    def get_frame_values(self):
        # possible values: cam0, ..., cam5, cam_head, footprint, timestamp, valid
        return list(self.__frames[self.get_frame_ids()[0]].keys())

    def get_cameras(self):
        return self.__cameras

    def get_trace(self):
        return self.__trace

    def get_camera(self, camera_id):
        cam_id = self._convert_cam_id_to_str(camera_id)
        return self.__cameras[cam_id]

    # return camera IDs as key values: ["cam0", "cam1", ...]
    def get_camera_ids(self):
        return list(self.__cameras.keys())

    # return camera IDs as numerical indices: [0, 1, ...]
    def get_camera_indexes(self):
        return range(0, len(self.__cameras))

    # auxiliary function:
    #   generates intrisic matrix given width, heigh and zoom_factor.
    def __build_intrinsic_matrix(self, width, height, zoom_factor):
        #pylint: disable=invalid-name
        f = (height if self.__upright else width) / zoom_factor
        cx = (width - 1.0) / 2
        cy = (height - 1.0) / 2

        return np.array([[f, 0., cx],
                         [0., f, cy],
                         [0., 0., 1.]])

    # intrisics matrix (for the tiles. All the tiles share the same intrinsics)
    def get_camera_intrinsics(self):
        tiles = self.get_tiles()

        return self.__build_intrinsic_matrix(tiles.width,
                                             tiles.height,
                                             tiles.zoom_factor)

    # auxiliary function:
    #   get raw pose of a particular frame and camera id
    #   this is not valid for upright images!
    def __get_raw_pose(self, frame_id, cam_id):
        cam_id = self._convert_cam_id_to_str(cam_id)
        data = self.__frames[frame_id][cam_id]

        # get pose
        qvec = np.array(data["quaternion"])
        tvec = np.array(data["position"])

        return qvec, tvec

    # auxiliary function:
    #   fixes a camera-to-world qvec for upright fix
    def __upright_fix(self, qvec):
        R = transform.qvec2rotmat(qvec)

        # images are rotated by 90 degrees clockwise
        # rotate coordinates counter-clockwise before camera-to-world pose
        # sin(-pi / 2) = -1, cos(-pi / 2) = 0
        R_fix = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1]
        ])

        R = R @ R_fix

        qvec = transform.rotmat2qvec(R)

        return qvec

    # get pose of a particular frame and camera id
    #
    # Example:
    #   frame_id = 1
    #   get_pose(frame_id, "cam1")
    #   get_pose(frame_id, 1)
    #
    def get_pose(self, frame_id, cam_id, tile_id=0):
        tiles = self.get_tiles()
        angles = tiles.angles[tile_id]

        # angles to rotation
        Rx = transform.Rx(angles[1])
        Ry = transform.Ry(angles[0])
        Rz = transform.Rz(angles[2])

        #
        # axis convertion
        #
        #   y <- x
        #   x <- y
        #   z <- z
        #
        # The order applies like: Ry * Rx * Rz
        # (mirror in ocam model is rotated in this order)
        #
        R_tile = Ry @ Rx @ Rz   # even though it looks like a bug it is correct!

        # get Rotation from tile angles

        T_rot_only = transform.create_transform_4x4(
            R_tile, np.array([0, 0, 0]))

        # inverse of tile rotations
        T_rot_only_inv = np.linalg.inv(T_rot_only)

        # extrinsics: [R t]
        T = self.__pose_as_matrix(*self.__get_raw_pose(frame_id, cam_id))
        T_world_cam = T @ T_rot_only_inv
        extrinsics = T_world_cam[0:3, 0:4]

        # extrinsics: [R t]
        R = extrinsics[0:3, 0:3]
        tvec = extrinsics[0:3, 3]

        # convert to quaternion
        qvec = transform.rotmat2qvec(R)

        if self.__upright:
            qvec = self.__upright_fix(qvec)

        return qvec, tvec

    def __pose_as_matrix(self, qvec, tvec):
        # quaternion to matrix
        R = transform.qvec2rotmat(qvec)

        # 4x4 transformation
        T = transform.create_transform_4x4(R, tvec)

        return T

    def get_pose_as_matrix(self, frame_id, cam_id, tile_id=0):
        """ Get pose as 4x4 transformation matrix: T = [R t; 0 0 0 1]

        Args:
            frame_id (int): frame id
            cam_id (int): camera id
            tile_id (int, optional): tile id. Defaults to 0.

        Returns:
            np.matrix: 4x4 transformation
        """
        qvec, tvec = self.get_pose(frame_id, cam_id, tile_id)

        return self.__pose_as_matrix(qvec, tvec)

    def get_image_filename(self, frame_id, cam_id):
        return "{:05d}-".format(frame_id) + \
            self._convert_cam_id_to_str(cam_id) + ".jpg"

    def get_input_image_path(self, frame_id, cam_id):
        filename = self.get_image_filename(frame_id, cam_id)
        path = self._input_image_path / filename
        return path

    def get_output_image_path(self, frame_id, cam_id, tile_id=0):
        tiles = self.get_tiles()
        filename = self.get_image_filename(frame_id, cam_id)
        image_path = self._output_image_path / filename

        # postfix with tile instance
        postfix_tile = tiles.postfix(tile_id)

        # output image tile path
        path = str(image_path)[0:-4] + postfix_tile + ".jpg"
        path = Path(path)

        return path

    def get_processes(self):
        return self.__processes

    def set_processes(self, processes):
        # if negative use all processes
        if processes <= 0:
            processes = multiprocessing.cpu_count()
        self.__processes = processes

    def compute_LUT(self):
        # get tiles
        tiles = self.get_tiles()

        # create LUT output folder
        self._output_LUT_path.mkdir(exist_ok=True, parents=True)

        # pool of workers
        num_processes = self.get_processes()
        logger.debug("Number of processes used: %d", num_processes)
        pool = multiprocessing.Pool(processes=num_processes)

        for cam_id in self.get_camera_ids():
            ocam_model = self.get_camera(cam_id)

            for tile_id, angles in enumerate(tiles.angles):
                # postfix with tile instance
                postfix_tile = tiles.postfix(tile_id)

                # camera LUT path
                cam_LUT_path = str(self._output_LUT_path /
                                   (cam_id + postfix_tile + '.pkl'))

                if Path(cam_LUT_path).is_file():
                    continue

                pool.apply_async(_create_LUT,
                                 args=(cam_LUT_path,
                                       ocam_model,
                                       tiles.width,
                                       tiles.height,
                                       tiles.zoom_factor,
                                       angles))

        pool.close()
        pool.join()

    def undistort(self):
        # paths
        input_image_path = self._input_image_path

        # get tiles
        tiles = self.get_tiles()

        # compute LUT
        self.compute_LUT()

        # pool of workers
        pool = multiprocessing.Pool(processes=self.get_processes())

        # create a separed folder for the tiles
        self._output_image_path.mkdir(exist_ok=True, parents=True)

        # undistort images
        logger.info("Starting the undistortion...")
        for frame_id in self.get_frame_ids():
            for cam_id in self.get_camera_ids():
                ocam_model = self.get_camera(cam_id)
                filename = self.get_image_filename(frame_id, cam_id)
                input_image = str(input_image_path / filename)
                output_image = str(self._output_image_path / filename)

                for tile_id, angles in enumerate(tiles.angles):
                    # postfix with tile instance
                    postfix_tile = tiles.postfix(tile_id)

                    # output image tile path
                    output_image_tile = output_image[0:-4] + \
                        postfix_tile + ".jpg"

                    # camera LUT path
                    cam_LUT_path = str(self._output_LUT_path /
                                       (cam_id + postfix_tile + '.pkl'))

                    # undistort
                    pool.apply_async(ocamlib.undistort,
                                     args=(ocam_model,
                                           input_image,
                                           output_image_tile,
                                           cam_LUT_path,
                                           tiles.width,
                                           tiles.height,
                                           tiles.zoom_factor,
                                           angles))
        pool.close()
        pool.join()
        logger.info("Done with the undistortion.")

    def get_tiles(self):
        return self.__tiles

    def set_tiles_format(self, tiles_format):
        cam = self.get_camera(0)
        self.__tiles = Tiles(self.__device, cam['width'], cam['height'],
                             TileFormat["TILES_" + str(tiles_format).lower()])

    def get_num_tiles(self):
        return len(self.get_tiles().angles)

    def _convert_cam_id_to_str(self, cam_id):
        if isinstance(cam_id, int):
            cam_id = 'cam{:d}'.format(cam_id)
        return cam_id

    def get_number_of_images(self):
        return len(self.get_camera_ids()) * len(self.get_frame_ids())


    # -------
    # Radio data
    # -------
    def read_bluetooth(self):
        """Returns list of bluetooth measurements provided in NavVis CSV.

        Returns
        -------
        list
            List of BluetoothMeasurements.
        """

        # mapping of csv header keys to indices
        expected_columns = ["TimeStamp",
                            "PosX",
                            "PosY",
                            "PosZ",
                            "RotW",
                            "RotX",
                            "RotY",
                            "RotZ",
                            "Rssi",
                            "Id"]

        input_filepath = Path(
            self._input_path) / "artifacts" / "bluetooth_beacons.csv"
        rows = read_csv(input_filepath)

        # mapping from column names to indices
        header = {c: i for i, c in enumerate(expected_columns)}

        # convert to protobuf
        bt_measurements = []
        for row in rows:

            timestamp_s = float(row[header["TimeStamp"]])
            signal_strength_dbm = int(row[header["Rssi"]])
            ibeacon = parse_navvis_ibeacon_packet(row[header["Id"]])

            bt_measurement = BluetoothMeasurement(timestamp_s,
                                                  signal_strength_dbm,
                                                  ibeacon.uuid,
                                                  ibeacon.major_version,
                                                  ibeacon.minor_version,
                                                  ibeacon.broadcasting_power_dbm)

            bt_measurements.append(bt_measurement)

        return bt_measurements

    def read_wifi(self):
        """Returns list of wifi measurements provided in NavVis wifi log files.

        Returns
        -------
        list
            List of WifiMeasurements.
        """

        # get filenames
        paths = list(Path(Path(self._input_path) / "wifi").glob("*-wifi.log"))

        # get frame ids
        frame_ids = self.get_frame_ids()
        assert len(paths) <= len(frame_ids)

        wifi_measurements = []
        for file_path in paths:

            # extract id_s from filenames
            basename = PurePath(file_path).name
            frame_id = int(basename.split('-')[0])
            assert frame_id in frame_ids

            timestamp_s = self.get_frame_timestamp(frame_id)

            with open(file_path, 'r') as f:
                data = f.readlines()
                if len(data) == 0:
                    continue

                samples = parse_iwconfig(data)
                for sample in samples:
                    wifi_measurement = WifiMeasurement(timestamp_s,
                                                       sample.mac_address,
                                                       sample.signal_strength_dbm,
                                                       sample.frequency_khz,
                                                       sample.time_offset_ms)

                    wifi_measurements.append(wifi_measurement)

        return wifi_measurements


#
# auxiliary function for parallel computing
#
def _create_LUT(cam_LUT_path,
                ocam_model,
                output_width,
                output_height,
                zoom_factor=4,
                angles=None):

    # using list as default argument is dangerous
    if angles is None:
        angles = [0, 0, 0]

    logger.debug("Computing camera LUT: %s", cam_LUT_path)
    mapx, mapy = ocamlib.create_undistortion_LUT(ocam_model,
                                                 output_width,
                                                 output_height,
                                                 zoom_factor,
                                                 angles)

    logger.info("Save pre-computed LUT: %s", cam_LUT_path)
    ocamlib.save_cam_LUT(mapx, mapy, cam_LUT_path)
