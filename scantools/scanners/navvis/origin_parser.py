import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

UNKNOWN_CRS_NAME = 'UNKNOWN'

def is_navvis_origin_valid(navvis_origin : dict):
    """
    Check if the NavVis origin dictionary is valid.
    CRS is optional. Pose is required.
    :param navvis_origin: NavVis origin dictionary
    :return: True if valid, False otherwise
    :rtype: bool
    """
    if  navvis_origin['Pose']['position']['x'] is None or \
        navvis_origin['Pose']['position']['y'] is None or \
        navvis_origin['Pose']['position']['z'] is None or \
        navvis_origin['Pose']['orientation']['w'] is None or \
        navvis_origin['Pose']['orientation']['x'] is None or \
        navvis_origin['Pose']['orientation']['y'] is None or \
        navvis_origin['Pose']['orientation']['z'] is None:
        return False
    return True

def parse_navvis_origin_file(file_path : Path):
    """
    The origin.json file is optional and if present it can be found 
    in the anchors folder.
    
    The origin.json file contains two important values: pose and CRS.
    
    The CRS value stands for Coordinate Reference System (CRS) and explains in which 
    coordinate system the origin itself is defined. Example: EPSG:25834 https://epsg.io/25834
    
    The pose transforms dataset entities into the origin. The origin 
        of the dataset can be created in many different ways:
        0 - The origin is the NavVis 'dataset' origin, where a dataset equals a NavVis session. 
            The origin then defaults to identity and the origin.json file might not be even present.
        1 - NavVis software allows relative alignment between dataset via the NavVis IVION Dataset Web Editor 
            but also via the NavVis local processing software which is soon to be deprecated.
        2 - The origin is the NavVis Site origin. NavVis organizes datasets in the same physical location
            via Sites. The origin file contains then the transformation which moves all the entities of a 
            NavVis dataset into the Site origin. Additionally NavVis IVION allows to register the Site origin
            to a global coordinate system. Hence, many NavVis sessions can be registered then to the same 
            global coordinate system. Note that this is achieved via the NavVis IVION Dataset Web Editor.
        3 - The origin lies in a Coordinate Reference System (CRS) like EPSG:25834 https://epsg.io/25834.
            The transformation is computed via geo-referenced Control Points which are registered during
            capture. More information about control points and the origin can be found here:
            https://knowledge.navvis.com/v1/docs/creating-the-control-point-poses-file 
            https://knowledge.navvis.com/docs/what-coordinate-system-do-we-use-for-the-control-points-related-tasks

    :param file_path: Path to the file
    :return: NavVis anchor origin dictionary
    :rtype: Dict
    """
    if not file_path.exists():
        print(f"Warning: Origin '{file_path}' does not exist.")
        return {}

    try:
        with file_path.open() as f:
            origin = json.load(f)
            if not is_navvis_origin_valid(origin):
                print("Invalid origin.json file", json.dumps(origin, indent=4))
            return origin
    except Exception as e:
        logger.warning(
            "Failed reading origin.json file. %s", e)
    return {}


def get_crs_from_navvis_origin(navvis_origin : dict):
    """
    Get the label from the NavVis origin
    :param navvis_origin: NavVis origin dictionary
    :return: Label
    :rtype: str
    """
    
    return navvis_origin.get('CRS', UNKNOWN_CRS_NAME)


def get_pose_from_navvis_origin(navvis_origin : dict):
    """
    Extract the pose from the NavVis origin dictionary
    :param navvis_origin: NavVis origin dictionary
    :return: Quaternion and translation vector
    :rtype: qvec, tvec
    """

    qvec = [1, 0, 0, 0]
    tvec = [0, 0, 0]
    if navvis_origin:
        orientation = navvis_origin['Pose']['orientation']
        position = navvis_origin['Pose']['position']
        qvec = [orientation['w'], orientation['x'], orientation['y'], orientation['z']]
        tvec = [position['x'], position['y'], position['z']]
    return qvec, tvec


def convert_navvis_origin_to_csv(navvis_origin : dict):
    csv_str = "# CRS, qw, qx, qy, qz, tx, ty, tz\n"
    if 'CRS' in navvis_origin:
        crs = navvis_origin['CRS']
    else:
        crs = UNKNOWN_CRS_NAME
    position = navvis_origin['Pose']['position']
    orientation = navvis_origin['Pose']['orientation']
    csv_str += (f"{crs},"
                f"{orientation['w']},"
                f"{orientation['x']},"
                f"{orientation['y']},"
                f"{orientation['z']},"
                f"{position['x']},"
                f"{position['y']},"
                f"{position['z']}\n")
    return csv_str
