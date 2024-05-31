""" Tests for origin_parse.py """
import pytest
import json
import os

from scipy.spatial.transform import Rotation
from ..capture import Pose, GlobalAlignment

from ..scanners.navvis import origin_parser

@pytest.mark.parametrize("nominal_origin, expected_output_csv", [({
    "CRS": "EPSG:25834",
    "Pose": {
        "orientation": {
            "w": 0.8,
            "x": 0,
            "y": 0,
            "z": -0.5
        },
        "position": {
            "x": 6.3,
            "y": 2.4,
            "z": 99.95
        }
    }},
     "# CRS, qw, qx, qy, qz, \
        tx, ty, tz\n\
        EPSG:25834,0.8,0,0,-0.5,\
        6.3,2.4,99.95\n"),
    ({
        "Pose": {  
            "orientation": {
                "w": 0.5,
                "x": 0,
                "y": 0,
                "z": 0
            },
            "position": {
                "x": 0,
                "y": 0,
                "z": 0
            }
        }
    }, "# CRS, qw, qx, qy, qz, tx, ty, tz\n" + origin_parser.UNKNOWN_CRS_NAME + ",0.5,0,0,0,0,0,0\n"),
])
def test_parse_navvis_origin(nominal_origin, expected_output_csv, tmp_path):
    temp_origin_path = tmp_path / "input_data.json"
    with open(temp_origin_path, 'w') as file:
        json.dump(nominal_origin, file)
    origin = origin_parser.parse_navvis_origin_file(temp_origin_path)
    assert origin == nominal_origin
    assert expected_output_csv.replace(" ","") == origin_parser.convert_navvis_origin_to_csv(origin).replace(" ","")
    os.remove(temp_origin_path)

    global_alignment_path = tmp_path / 'origin.txt'
    global_alignment = GlobalAlignment()
    crs = origin_parser.get_crs_from_navvis_origin(origin)
    qvec, tvec = origin_parser.get_pose_from_navvis_origin(origin)
    global_alignment[crs, global_alignment.no_ref] = (
            Pose(qvec, tvec), [])
    global_alignment.save(global_alignment_path)    
    os.remove(global_alignment_path)    

@pytest.mark.parametrize("bad_json_keys_origin", [{
    "CRS": "EPSG:25834",
    "Pose": {
        "orientation": {
            "w": 0.8,
            "x": 0,
            "y": 0,
            "z": -0.5
        },
        "positon": { # misspelled key
            "x": 6.3,
            "y": 2.4,
            "z": 99.95
        }
    }},
    {
        "Pose": {  
            "orentation": { # misspelled key
                "w": 0.5,
                "x": 0,
                "y": 0,
                "z": 0
            },
            "position": {
                "x": 0,
                "y": 0,
                "z": 0
            }
        }
    }
])
def test_parse_navvis_origin_bad_input(bad_json_keys_origin, tmp_path):
    temp_origin_path = tmp_path / "bad_json_keys_origin.json"
    with open(temp_origin_path, 'w') as file:
        json.dump(bad_json_keys_origin, file)
    assert not origin_parser.parse_navvis_origin_file(temp_origin_path)
    with pytest.raises(KeyError):
        origin_parser.convert_navvis_origin_to_csv(bad_json_keys_origin)
    os.remove(temp_origin_path)

