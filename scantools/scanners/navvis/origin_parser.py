import json
from pathlib import Path


def convert_navvis_origin_to_csv(navvis_origin : dict):
    csv_str = "# CRS, qw, qx, qy, qz, tx, ty, tz\n"

    #CRS stands for Coordinate Reference System (CRS)
    #Example: EPSG:25834 https://epsg.io/25834
        
    if 'CRS' in navvis_origin:
        crs = navvis_origin['CRS']
    else:
        crs = "unknown"

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


def convert_navvis_origin_file_to_csv(file_path : Path):
    """
    Read NavVis SLAM Anchor Origin File Format Version 1.0 and write as csv file.
    The navvis origin file is a json file stored in the 'anchors' folder.
    :param file_path: Path to the file
    :return: NavVis anchor origin
    :rtype: Dict
    """

    if not file_path.exists():
        return

    with file_path.open() as f:
        navvis_origin = json.load(f)
        return convert_navvis_origin_to_csv(navvis_origin)
        
        
    



