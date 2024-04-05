""" Tests for origin_parse.py """
import pytest

from ..scanners.navvis import origin_parser

@pytest.mark.parametrize("input_data, expected_output_csv", [({
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
    }, "# CRS, qw, qx, qy, qz, tx, ty, tz\nunknown,0.5,0,0,0,0,0,0\n"),
])
def test_parse_navvis_origin(input_data, expected_output_csv):
    assert expected_output_csv.replace(" ","") == origin_parser.convert_navvis_origin_to_csv(input_data).replace(" ","")


@pytest.mark.parametrize("bad_input_data", [{
    "CRS": "EPSG:25834",
    "Pose": {
        "orientation": {
            "w": 0.8,
            "x": 0,
            "y": 0,
            "z": -0.5
        },
        "positon": {
            "x": 6.3,
            "y": 2.4,
            "z": 99.95
        }
    }},
    {
        "Pose": {  
            "orentation": {
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
    },
    {
        "Pose": {  
            "orentation": {
                "w": 0.522222221212121212112121212121212121223,
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
    },
    
])
def test_parse_navvis_origin_bad_input(bad_input_data):
    with pytest.raises(KeyError):
        origin_parser.convert_navvis_origin_to_csv(bad_input_data)


