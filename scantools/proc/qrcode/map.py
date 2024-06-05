import csv
import itertools
import json
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from scantools import logger
from scantools.capture import Capture, Pose
from scantools.proc.qrcode.detector import QRCodeDetector
from scantools.proc.rendering import Renderer, compute_rays
from scantools.utils.io import read_mesh


def create_qr_map(
    capture: Capture,
    session_id: str,
    mesh_id: str,
) -> List[dict]:
    """
    Create a QR map from the detected QR codes.

    Parameters:
    capture (Capture): Capture object containing the images and sessions.
    session_id (str): ID of the session to process.
    mesh_id (str): ID of the mesh to use.

    Returns:
    List[dict]: A list of dictionaries representing the QR codes.

    A QR code dictionary contains the following fields:
    - id (str): The string in the QR code.
    - timestamp (int): The timestamp of the image.
    - cam_id (str): The camera ID.
    - points2D (list): A list of 2D points representing the corners of the QR code.
    - points3D_world (list): A list of 3D points representing the corners of the
                             QR code in world coordinates.
    - qvec_qr2world (list):  A list of 4 elements representing the quaternion
                             rotation from the QR code coordinate system to the
                             world coordinate system.
    - tvec_qr2world (list):  A list of 3 elements representing the translation
                             from the QR code coordinate system to the world
                             coordinate system.
    - qvec_cam2world (list): A list of 4 elements representing the quaternion
                             rotation from the camera coordinate system to the
                             world coordinate system.
    - tvec_cam2world (list): A list of 3 elements representing the translation
                             from the camera coordinate system to the world
                             coordinate system.

    where qvec is a quaternion in the form [qw, qx, qy, qz],
    and tvec is a translation vector in the form [tx, ty, tz].
    """
    session = capture.sessions[session_id]
    output_dir = capture.proc_path(session_id)

    assert session.proc is not None
    assert session.proc.meshes is not None
    assert mesh_id in session.proc.meshes
    assert session.images is not None

    mesh_path = output_dir / session.proc.meshes[mesh_id]
    mesh = read_mesh(mesh_path)
    renderer = Renderer(mesh)
    image_dir = capture.data_path(session_id)
    qrcode_dir = output_dir / "qrcodes"
    suffix = ".qrcodes.txt"

    logger.info("Create QR map from detected QR codes.")
    qr_map = []
    for ts, cam_id in tqdm(session.images.key_pairs()):
        pose_cam2w = session.trajectories[ts, cam_id]
        camera = session.sensors[cam_id]

        # Load QR codes.
        filename = session.images[ts, cam_id]
        image_path = image_dir / filename
        qrcode_path = (qrcode_dir / filename).with_suffix(suffix)
        qrcodes = QRCodeDetector(image_path)
        qrcodes.load(qrcode_path)

        for qr in qrcodes:
            points2D = np.asarray(qr["points2D"])

            # Ray casting.
            origins, directions = compute_rays(pose_cam2w, camera, p2d=points2D)
            intersections, intersected = renderer.compute_intersections(
                (origins, directions)
            )

            # Verify all rays intersect the mesh.
            if not intersected.all() and len(intersected) == 4:
                logger.warning(
                    "QR code %s doesn't intersected in all points.", qr["id"]
                )
                continue

            # 3D points from ray casting, intersection with mesh.
            points3D_world = intersections

            # QR code indices:
            #
            #     0. top-left,
            #     1. bottom-left,
            #     2. bottom-right,
            #     3. top-right
            #
            #
            # QR code coordinate system:
            #
            #            ^
            #           / z-axis
            #          /
            #         /
            #
            #       0.  --- x-axis --->   3.
            #
            #       |
            #       | y-axis
            #       |
            #       v
            #
            #       1.                    2.
            #

            # Rotation (QR to World).
            rotmat_qr2w = np.zeros((3, 3))

            # x-axis.
            v = points3D_world[3] - points3D_world[0]
            x_axis = v / np.linalg.norm(v)
            rotmat_qr2w[0:3, 0] = x_axis

            # y-axis.
            v = points3D_world[1] - points3D_world[0]
            y_axis = v / np.linalg.norm(v)
            rotmat_qr2w[0:3, 1] = y_axis

            # z-axis (cross product, right-hand coordinate system).
            z_axis = np.cross(x_axis, y_axis)
            rotmat_qr2w[0:3, 2] = z_axis

            pose_qr2w = Pose(r=rotmat_qr2w, t=points3D_world[0])
            if math.isnan(np.linalg.det(pose_qr2w.R)):
                continue

            # Append current QR to the QR map.
            #  - qvec: qw, qx, qy, qz
            #  - tvec: tx, ty, tz
            QR = {
                "id": qr["id"],  # String in the QR code.
                "timestamp": ts,
                "cam_id": cam_id,
                "points2D": points2D.tolist(),
                "points3D_world": points3D_world.tolist(),
                "qvec_qr2world": pose_qr2w.qvec.tolist(),
                "tvec_qr2world": pose_qr2w.t.tolist(),
                "qvec_cam2world": pose_cam2w.qvec.tolist(),
                "tvec_cam2world": pose_cam2w.t.tolist(),
            }
            logger.info(QR)
            qr_map.append(QR)

    return qr_map


def save_qr_maps(
    qr_map: List[dict],
    qr_map_filtered: List[dict],
    qrcode_dir: Path,
    json_format: bool,
    txt_format: bool,
):
    """
    Save QR map and filtered QR map in the specified formats.

    Parameters:
    qr_map (List[Dict]): The QR map to save.
    qr_map_filtered (List[Dict]): The filtered QR map to save.
    qrcode_dir (Path): Directory path to save the files.
    json_format (bool): Whether to save in JSON format.
    txt_format (bool): Whether to save in TXT format.
    """
    qr_map_path = qrcode_dir / "qr_map"
    if json_format:
        save_json(qr_map, qr_map_path.with_suffix(".json"))
    if txt_format:
        save_txt(qr_map, qr_map_path.with_suffix(".txt"))

    # Save filtered QR map.
    qr_map_filtered = filter_qr_codes_by_area(qr_map)
    qr_map_filtered_path = qrcode_dir / "qr_map_filtered_by_area"
    if json_format:
        save_json(qr_map_filtered, qr_map_filtered_path.with_suffix(".json"))
    if txt_format:
        save_txt(qr_map_filtered, qr_map_filtered_path.with_suffix(".txt"))


# Load QR map from json file.
def load_json(path):
    with open(path) as json_file:
        logger.info(f"Loading QR code poses from file: {path}.")
        qr_map = json.load(json_file)
        return qr_map


# Save QR map to json file.
def save_json(qr_map: list[dict], path: Path):
    with open(path, "w") as json_file:
        logger.info(f"Saving qr_map to file: {path}")
        json.dump(qr_map, json_file, indent=2)


# Save QR map to txt file.
def save_txt(qr_map: list[dict], path: Path):
    """
    Save a QR map to a text file.

    Parameters:
    - qr_map (list): A list of dictionaries representing QR data.
    - path (str): The file path where the QR map will be saved.
    """
    try:
        logger.info(f"Saving qr_map to file: {path}")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)

            # Generate the header row.
            header = generate_csv_header(qr_map[0])
            writer.writerow(header)

            # Write each QR code to a row in the file.
            for qr in qr_map:
                row = []
                for value in qr.values():
                    # Handle integers, strings, and lists
                    if isinstance(value, (int, str)):
                        row.append(value)
                    elif isinstance(value, list):
                        # Flatten the list if it contains nested lists
                        flattened_value = (
                            list(itertools.chain.from_iterable(value))
                            if all(isinstance(i, list) for i in value)
                            else value
                        )
                        row.extend(flattened_value)
                writer.writerow(row)
    except Exception as e:
        logger.info(f"An error occurred while saving the QR map: {e}")


def generate_csv_header(sample_qr: dict):
    """
    Generates a CSV header based on the structure of a sample QR dictionary.

    Parameters:
    - sample_qr (dict): A sample QR dictionary from the qr_map.

    Returns:
    list: A list of header strings for the CSV file.
    """
    header = []

    # Function to add header fields for list-type values
    def add_list_fields(field_name, dim, length_list):
        for i in range(length_list):
            index = f"[{i}]" if length_list > 1 else ""
            if dim == 2:  # 2D point.
                header.append(f"{field_name}{index}_x")
                header.append(f"{field_name}{index}_y")
            elif dim == 3:  # 3D point.
                header.append(f"{field_name}{index}_x")
                header.append(f"{field_name}{index}_y")
                header.append(f"{field_name}{index}_z")
            elif dim == 4:  # Quaternion.
                header.append(f"{field_name}{index}_w")
                header.append(f"{field_name}{index}_x")
                header.append(f"{field_name}{index}_y")
                header.append(f"{field_name}{index}_z")

    # Iterate over all keys in the sample QR dictionary.
    for key, value in sample_qr.items():
        if isinstance(value, list):
            dim = len(value[0]) if isinstance(value[0], list) else len(value)
            length_list = len(value) if isinstance(value[0], list) else 1
            add_list_fields(key, dim, length_list)
        else:
            # Directly add the key for scalar values.
            header.append(key)

    # Add a comment character to the first header field.
    header[0] = "# " + header[0]
    return header


def filter_qr_codes_by_area(qr_codes: list[dict]) -> list[dict]:
    """
    Filter the QR codes by area, keeping only the largest one for each ID.

    Parameters:
    - qr_map (list): A list of dictionaries representing the QR codes.

    Returns:
    - qr_map_filtered (list): A filtered list of QR codes, with only the
                              largest QR code for each unique ID.
    """
    # Extract unique IDs from the QR codes.
    qr_ids = {qr["id"] for qr in qr_codes}

    # Iterate over unique IDs and filter the largest QR code for each ID.
    qr_map_filtered = [
        max(
            [qr for qr in qr_codes if qr["id"] == qr_id],  # QRs with same id.
            key=lambda qr: calculate_polygon_area(qr["points2D"]),
        )
        for qr_id in qr_ids
    ]

    # Sort the filtered list by ID for consistent ordering.
    qr_map_filtered.sort(key=lambda qr: qr["id"])

    return qr_map_filtered


def calculate_polygon_area(vertices: List[Tuple[float, float]]) -> float:
    """
    Calculate the area of a polygon given its vertices using the Shoelace
    formula. The Shoelace formula, also known as Gauss's area formula, sums the
    cross-products of pairs of sequential vertices and divides by 2. It works
    for any non-self-intersecting polygon.

    Parameters:
    - vertices (list of tuples): A list of (x, y) tuples representing the
                                 vertices of the polygon.

    Returns:
    - float: The area of the polygon.
    """
    n = len(vertices)

    area = 0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]  # Ensure last vertex connects to first.
        area += x1 * y2 - x2 * y1

    return abs(area) / 2.0
