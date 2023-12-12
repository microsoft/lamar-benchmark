import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pyzbar.pyzbar import ZBarSymbol, decode  # pip install pyzbar-upright
from tqdm import tqdm

from scantools import (
    logger,
    run_meshing,
    run_navvis_to_capture,
    to_meshlab_visualization,
)
from scantools.capture import Capture
from scantools.proc.rendering import Renderer, compute_rays
from scantools.utils.io import read_mesh


@dataclass
class QRCodeDetector:
    image_path: str
    qrcodes: list = field(default_factory=list)

    def __post_init__(self):
        if not Path(self.image_path).is_file():
            raise FileNotFoundError(str(self.image_path))

    def __getitem__(self, key):
        return self.qrcodes[key]

    def __iter__(self):
        return iter(self.qrcodes)

    def __len__(self):
        return len(self.qrcodes)

    def is_empty(self):
        return len(self.qrcodes) == 0

    # Detect QR codes.
    def detect(self):
        img = cv2.imread(str(self.image_path))
        detected_qrcodes = decode(img, symbols=[ZBarSymbol.QRCODE])
        # Loop over the detected QR codes.
        for qr in detected_qrcodes:
            qr_code = {
                "id": qr.data.decode("utf-8"),
                "points2D": np.asarray(qr.polygon, dtype=float).tolist(),
            }
            self.qrcodes.append(qr_code)

    def load(self, path):
        if not Path(path).is_file():
            raise FileNotFoundError(str(path))
        if Path(path).stat().st_size == 0:
            return []
        with open(path) as json_file:
            self.qrcodes = json.load(json_file)

    def save(self, path):
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        if not self.qrcodes:
            Path(path).touch()
        else:
            with open(path, "w") as json_file:
                json.dump(self.qrcodes, json_file, indent=2)

    def show(self, markersize=1):
        img = cv2.imread(str(self.image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(0, figsize=(30, 70))
        plt.imshow(img)

        for qr in self.qrcodes:
            # ! pyzbar returns points in the following order:
            # !     1. top-left, 2. bottom-left, 3. bottom-right, 4. top-right
            print("[INFO] Found {}: {}".format("QR Code", qr["id"]))
            print(qr["points2D"])
            (x, y) = qr["points2D"][0]
            plt.plot(x, y, "m.", markersize)
            (x, y) = qr["points2D"][1]
            plt.plot(x, y, "g.", markersize)
            (x, y) = qr["points2D"][2]
            plt.plot(x, y, "b.", markersize)
            (x, y) = qr["points2D"][3]
            plt.plot(x, y, "r.", markersize)
        plt.show()


# Load QR map from json file.
def load_qr_map(path):
    with open(path) as json_file:
        print("Loading QR code poses from file:", path)
        qr_map = json.load(json_file)
        return qr_map


# Save QR map to json file.
def save_qr_map(qr_map, path):
    with open(path, "w") as json_file:
        print("Saving qr_map to file:", path)
        json.dump(qr_map, json_file, indent=2)


def run_qrcode_detection(
    capture: Capture, session_id: str, mesh_id: str = "mesh"
):
    session = capture.sessions[session_id]
    output_dir = capture.data_path(session_id)

    assert session.proc is not None
    assert session.proc.meshes is not None
    assert mesh_id in session.proc.meshes
    assert session.images is not None

    mesh_path = capture.proc_path(session_id) / session.proc.meshes[mesh_id]
    mesh = read_mesh(mesh_path)
    renderer = Renderer(mesh)

    qrcode_dir = output_dir / "qrcodes"
    qrcode_dir.mkdir(exist_ok=True, parents=True)

    qr_map = []
    for ts, cam_id in tqdm(session.images.key_pairs()):
        pose_cam2w = session.trajectories[ts, cam_id]
        camera = session.sensors[cam_id]

        image_path = output_dir / session.images[ts, cam_id]
        qrcodes = QRCodeDetector(image_path)

        qrcode_path = qrcode_dir / session.images[ts, cam_id]
        qrcode_path = qrcode_path.with_suffix(".qrcode.json")

        if qrcode_path.is_file():
            qrcodes.load(qrcode_path)
        else:
            qrcodes.detect()
            qrcodes.save(qrcode_path)
        # qrcodes.show(markersize=2)
        logger.info(qrcodes)

        # Create QR map from detected QR codes.
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
            #   0. top-left,
            #   1. bottom-left,
            #   2. bottom-right,
            #   3. top-right
            #
            #
            # QR code coordinate system:
            #        ^
            #       /
            #      / z-axis
            #     /
            #   0. ---- x-axis --->  3.
            #   |
            #   |
            #   y-axis
            #   |
            #   |
            #   v
            #   1.                   2.
            #

            world_T_qr = np.zeros((4, 4))
            world_T_qr[3, 3] = 1

            # Translation (QR to World).
            world_T_qr[0:3, 3] = points3D_world[0]

            # Rotation (QR to World).
            # x-axis.
            v = points3D_world[3] - points3D_world[0]
            x_axis = v / np.linalg.norm(v)
            world_T_qr[0:3, 0] = x_axis

            # y-axis.
            v = points3D_world[1] - points3D_world[0]
            y_axis = v / np.linalg.norm(v)
            world_T_qr[0:3, 1] = y_axis

            # z-axis (cross product, right-hand coordinate system).
            z_axis = np.cross(x_axis, y_axis)
            world_T_qr[0:3, 2] = z_axis

            R = world_T_qr[0:3, 0:3]

            if math.isnan(np.linalg.det(R)):
                continue

            # Append current QR to the QR map.
            QR = {
                "id": qr["id"],  # String in the QR code.
                "timestamp": ts,
                "cam_id": cam_id,
                "points2D": points2D.tolist(),
                "points3D": points3D_world.tolist(),
                "world_T_qr": world_T_qr.reshape(1, 16).tolist(),
                "world_T_cam": pose_cam2w.to_4x4mat().reshape(1, 16).tolist(),
            }

            print(QR)
            qr_map.append(QR)
    save_qr_map(qr_map, qrcode_dir / "qr_map.json")


def run(
    capture_path: Path,
    sessions: Optional[List[str]] = None,
    navvis_dir: Optional[Path] = None,
    visualization: bool = True,
):
    if capture_path.exists():
        capture = Capture.load(capture_path)
    else:
        capture = Capture(sessions={}, path=capture_path)

    tiles_format = "none"
    mesh_id = "mesh"

    # If `sessions` is not provided, run for all sessions in the `capture_path`.
    if sessions is None:
        sessions = capture.sessions.keys()

    for session in sessions:
        if session not in capture.sessions:
            logger.info("Exporting NavVis session %s.", session)
            run_navvis_to_capture.run(
                navvis_dir / session,
                capture,
                tiles_format,
                session,
            )

        if (
            not capture.sessions[session].proc
            or mesh_id not in capture.sessions[session].proc.meshes
        ):
            logger.info("Meshing session %s.", session)
            run_meshing.run(
                capture,
                session,
            )

        run_qrcode_detection(capture, session)

        if visualization:
            to_meshlab_visualization.run(
                capture,
                session,
                f"trajectory_{session}",
                export_mesh=True,
                export_poses=True,
                mesh_id=mesh_id,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--capture_path",
        type=Path,
        required=True,
        help="Path to the capture. If it doesn't exist it will process with "
        "tile format `none` and export the capture to this path.",
    )
    parser.add_argument(
        "--sessions",
        nargs="*",
        type=str,
        default=None,
        required=False,
        help="List of sessions to process. If not provided, it will process all "
        "sessions in the `capture_path`. Useful when we want to process only "
        "some sessions.",
    )
    parser.add_argument(
        "--navvis_dir",
        type=Path,
        default=None,
        required=False,
        help="Input NavVis data path, used if `--capture_path` doesn't exist. "
        "This could be useful when we have already converted to capture format "
        "and we don't have the original NavVis data anymore.",
    )
    parser.add_argument(
        "--visualization",
        type=Path,
        default=True,
        required=False,
        help="Write out MeshLab visualization.",
    )
    args = parser.parse_args().__dict__

    run(**args)
