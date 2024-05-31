""" Tests for origin_parse.py """
import pytest
import json
import os
import numpy as np

from ..capture import Pose, GlobalAlignment
from ..scanners.navvis import origin_parser

@pytest.mark.parametrize("navvis_origin, expected_csv_output", [({
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
     "# label, reference_id, qw, qx, qy, qz, \
        tx, ty, tz\n\
        EPSG:25834,__absolute__,0.8,0,0,-0.5,\
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
    }, "# label, reference_id, qw, qx, qy, qz, tx, ty, tz\n" +
      origin_parser.UNKNOWN_CRS_NAME + ",__absolute__,0.5,0,0,0,0,0,0\n"),
])
def test_parse_navvis_origin(navvis_origin, expected_csv_output, tmp_path):
    navvis_origin_path = tmp_path / "navvis_origin.json"
    with open(navvis_origin_path, 'w') as file:
        json.dump(navvis_origin, file)

    navvis_origin_loaded = origin_parser.parse_navvis_origin_file(navvis_origin_path)
    assert navvis_origin_loaded == navvis_origin
    os.remove(navvis_origin_path)

    alignment = GlobalAlignment()
    crs = origin_parser.get_crs_from_navvis_origin(navvis_origin_loaded)
    qvec, tvec = origin_parser.get_pose_from_navvis_origin(navvis_origin_loaded)
    alignment_pose = Pose(qvec, tvec)
    alignment[crs, alignment.no_ref] = (
            alignment_pose, [])
    alignment_path = tmp_path / 'origin.txt'
    alignment.save(alignment_path)

    alignment_loaded = GlobalAlignment().load(alignment_path)
    os.remove(alignment_path)
    
    alignment_pose_loaded = alignment_loaded.get_abs_pose(crs)
    assert np.allclose(alignment_pose_loaded.qvec, 
                       alignment_pose.qvec, 1e-10)
    assert np.allclose(alignment_pose_loaded.t, 
                       alignment_pose.t, 1e-10)


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
    os.remove(temp_origin_path)
