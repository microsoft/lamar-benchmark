# %%
import argparse
import json
from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pyzbar.pyzbar import ZBarSymbol, decode  # pip install pyzbar-upright
from tqdm import tqdm

from scantools import (
    logger,
    run_meshing,
    run_navvis_to_capture,
    run_rendering,
    to_meshlab_visualization,
)
from scantools.capture import Capture

# from scantools.utils.io import read_image, write_image

# from .qrcode.detection import find_qr_codes, show_qr_codes


def find_qr_codes(image_path):
    if not Path(image_path).is_file():
        raise FileNotFoundError(str(image_path))

    img = cv2.imread(str(image_path))

    # Detect QR codes.
    detected_qr_code = decode(img, symbols=[ZBarSymbol.QRCODE])

    # Loop over the detected QR codes.
    qr_codes = []
    for qr in detected_qr_code:
        qr_code = {
            "id": qr.data.decode("utf-8"),
            "points2D": np.asarray(qr.polygon, dtype=float).tolist(),
        }
        qr_codes.append(qr_code)

    return qr_codes


def show_qr_codes(image_path, qr_codes, markersize=1):
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(0, figsize=(30, 70))
    plt.imshow(img)

    # loop over the detected QR codes
    for qr in qr_codes:
        # data is a bytes object so if we want to draw it on
        # our output image we need to convert it to a string first
        qr_code_data = qr.data.decode("utf-8")
        qr_code_type = qr.type

        # print the qr_code type and data to the terminal
        print("[INFO] Found {}: {}".format(qr_code_type, qr_code_data))

        # ! pyzbar returns points in the following order:
        # !     1. top-left, 2. bottom-left, 3. bottom-right, 4. top-right

        # top-left
        (x, y) = qr.polygon[0]
        print("x = {}, y = {}".format(x, y))
        plt.plot(x, y, "m.", markersize)

        # bottom-left
        (x, y) = qr.polygon[1]
        print("x = {}, y = {}".format(x, y))
        plt.plot(x, y, "g.", markersize)

        # bottom-right
        (x, y) = qr.polygon[2]
        print("x = {}, y = {}".format(x, y))
        plt.plot(x, y, "b.", markersize)

        # top-right
        (x, y) = qr.polygon[3]
        print("x = {}, y = {}".format(x, y))
        plt.plot(x, y, "r.", markersize)

    plt.show()


def load_qr_poses(path):
    if not Path(path).is_file():
        raise FileNotFoundError(str(path))
    # Ignore empty files.
    if Path(path).stat().st_size == 0:
        return []
    with open(path) as json_file:
        return json.load(json_file)

def save_qr_codes(qrcodes, path):
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    if qrcodes is None or len(qrcodes) == 0:
        Path(path).touch()
    else:
        with open(path, 'w') as json_file:
            json.dump(qrcodes, json_file, indent=2)


def run_qrcode_detection(capture: Capture, mesh_id: str = "mesh"):
    for session_id in capture.sessions:
        session = capture.sessions[session_id]
        output_dir = capture.data_path(session_id)

        assert session.proc is not None
        assert session.proc.meshes is not None
        assert mesh_id in session.proc.meshes
        assert session.images is not None

        mesh_path = capture.proc_path(session_id) / session.proc.meshes[mesh_id]
        # mesh = read_mesh(mesh_path)
        # renderer = Renderer(mesh)

        qrcode_dir = output_dir / "qrcodes"
        qrcode_dir.mkdir(exist_ok=True, parents=True)

        for ts, cam_id in tqdm(session.images.key_pairs()):
            pose_cam2w = session.trajectories[ts, cam_id]
            camera = session.sensors[cam_id]

            image_path = output_dir / session.images[ts, cam_id]
            qrcodes = find_qr_codes(image_path)
            # show_qr_codes(image_path, qrcodes, markersize=1)
            print(qrcodes)

            qrcode_path = qrcode_dir / session.images[ts, cam_id]
            qrcode_path = qrcode_path.with_suffix('.qrcode.json')
            save_qr_codes(qrcodes, qrcode_path)

            # Create QR map from detected QR codes.
            for qr in qrcodes:
                points2D = np.asarray(qr["points2D"])

                # 3D points from depth
                point3D_cam = utils.transform.get_point3D_from_depth(
                    points2D, depth, K
                )

                points3D_world = utils.transform.apply(world_T_cam, point3D_cam)

                world_T_qr = np.zeros((4, 4))
                world_T_qr[3, 3] = 1

                # translation (QR -> World)
                world_T_qr[0:3, 3] = points3D_world[0]

                # rotation (QR -> World)
                # x-axis
                v = points3D_world[1] - points3D_world[0]
                x_axis = v / np.linalg.norm(v)
                world_T_qr[0:3, 0] = x_axis

                # y-axis
                y_id = 2 if (use_qr_detector) else 3
                v = points3D_world[y_id] - points3D_world[0]
                y_axis = v / np.linalg.norm(v)
                world_T_qr[0:3, 1] = y_axis

                # z-axis (cross product, right-hand coordinate system)
                z_axis = np.cross(y_axis, x_axis)
                world_T_qr[0:3, 2] = z_axis

                R = world_T_qr[0:3, 0:3]

                if math.isnan(np.linalg.det(R)):
                    continue

                # add current QR code instance
                QR = {
                    "id": pose["id"],  # "string in the QR code"
                    "frame_id": frame_id,
                    "cam_id": cam_id,
                    "points2D": points2D.tolist(),
                    "points3D": points3D_world.tolist(),
                    "world_T_qr": world_T_qr.reshape(1, 16).tolist(),
                    "world_T_cam": world_T_cam.reshape(1, 16).tolist(),
                }

                print(QR)
                qr_map.append(QR)

            # rays = ...
            # directions = (centers_ref[None] - centers[:, None]).reshape(-1, 3)
            # directions = np.ascontiguousarray(directions, dtype=np.float32)
            # origins = np.ascontiguousarray(np.repeat(centers, len(centers_ref), axis=0), dtype=np.float32)
            # intersections, intersected = renderer.compute_intersections((origins, directions))
            # intersections, valid = renderer.compute_intersections(rays)

        #     image_path = session.images[ts, camera_id]
        #     depth_path = Path(prefix, image_path).as_posix() + '.depth.png'
        #     render_id = f'{prefix}/{camera_id}'
        #     depths[ts, render_id] = depth_path

        #     output_path = output_dir / depth_path
        #     output_path.parent.mkdir(exist_ok=True, parents=True)
        #     write_depth(output_path, depth_map)

        #     if render_images:
        #         im = PIL.Image.fromarray((rgb * 255).astype(np.uint8))
        #         im.save(output_dir / prefix / image_path)

        # logger.info('Wrote the depth renderings to %s.', output_dir)
        # # Hacky but safer than rewriting the whole session data
        # depths.save(capture.session_path(session_id) / session.filename('depths'))


def run(capture_path: Path, session_ids: List[str], navvis_dir: Path):
    # if capture_path.exists():
    #     capture = Capture.load(capture_path)
    # else:
    #     capture = Capture(sessions={}, path=capture_path)

    #
    capture = Capture.load(capture_path)
    run_qrcode_detection(capture, mesh_id="mesh")

    tiles_format = "none"
    mesh_id = "mesh"
    # meshing_method = "advancing_front"
    meshing_method = "poisson"

    for session in session_ids:
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
                "point_cloud_final",
                mesh_id,
                method=meshing_method,
            )

        if not capture.sessions[session].depths:
            logger.info("Rendering session %s.", session)
            run_rendering.run(capture, session, mesh_id=mesh_id + "_simplified")

        to_meshlab_visualization.run(
            capture,
            session,
            f"trajectory_{session}",
            export_mesh=True,
            export_poses=True,
            mesh_id=mesh_id,
        )


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--capture_path", type=Path, required=True)
#     parser.add_argument("--input_path", type=Path, required=True)
#     parser.add_argument("--sessions", nargs="*", type=str, default=[])
#     args = parser.parse_args()

#     run(args.capture_path, args.sessions, args.input_path)


# %%
capture_path = Path(
    "/home/pablo/MS_data/locopt/talstrasse-2023-04-19/converted/lamar-format-qrcode"
)
sessions = ["2023-04-19-13-32-03-proc-euler"]
input_path = Path(
    "/home/pablo/MS_data/locopt/talstrasse-2023-04-19/converted/navvisvlx-sun-proc-v2.15.1"
)

run(capture_path, sessions, input_path)
