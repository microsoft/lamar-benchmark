import json
from pathlib import Path

UNKNOWN_CRS_NAME = 'UNKNOWN'

def is_navvis_origin_valid(navvis_origin : dict):
    """
    Check if the NavVis origin dictionary is valid
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
    Read NavVis Origin File Format Version 1.0
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
        print("Warning Failed reading origin.json file.", e)
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
        
        
    



