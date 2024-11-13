import argparse
import multiprocessing
from pathlib import Path
from typing import List, Optional

from scantools import (
    logger,
    run_meshing,
    run_navvis_to_capture,
    to_meshlab_visualization,
)
from scantools.capture import Capture
from scantools.proc.qrcode.detector import QRCodeDetector
from scantools.proc.qrcode.map import (
    create_qr_map,
    filter_qr_codes_by_area,
    save_qr_maps,
)

description = """
This script is used for QR code detection. The script saves the QR map (list of
all QR codes) in the `CAPTURE_PATH/proc/qrcodes/` directory. The QR map can be
saved in both TXT and JSON formats. The TXT format is used for compatibility
with the rest of CAPTURE format. The JSON format is used for convince but
disabled by default. The script also saves the detected QR codes for each image.
These are saved in the `CAPTURE_PATH/proc/qrcodes/` directory with the same name
as the image file, but with the `.qrcodes.txt` suffix. These files are used to
load the QR codes if they have already been detected for the image. This is
useful when we want to re-run the script without having to detect the QR codes
again.
"""


def run(
    capture_path: Path,
    navvis_path: Optional[Path] = None,
    sessions: Optional[List[str]] = None,
    use_simplified_mesh: bool = False,
    visualization: bool = True,
    **kargs,
):
    if capture_path.exists():
        capture = Capture.load(capture_path)
    else:
        capture = Capture(sessions={}, path=capture_path)

        logger.info("Convert NavVis data to CAPTURE format: %s", navvis_path)
        # Typically, QR codes are captured intentionally by approaching them
        # closely. As a result, we don't need tiling, which could potentially
        # split the QR code across multiple tiles.
        tiles_format = "none"
        run_navvis_to_capture.run(navvis_path, capture, tiles_format)

    # If `sessions` is not provided, run for all sessions in the `capture_path`.
    if sessions is None:
        sessions = capture.sessions.keys()

    mesh_id = "mesh"
    if use_simplified_mesh:
        mesh_id += "_simplified"
    for session in sessions:
        if (
            not capture.sessions[session].proc
            or mesh_id not in capture.sessions[session].proc.meshes
        ):
            logger.info("Meshing session %s.", session)
            run_meshing.run(capture, session)

        # Detect QR codes in the session.
        run_qrcode_detection_session(capture, session, mesh_id, **kargs)

        if visualization:
            to_meshlab_visualization.run(
                capture,
                session,
                f"trajectory_{session}",
                export_mesh=True,
                export_poses=True,
                mesh_id=mesh_id,
            )


def run_qrcode_detection_session(
    capture: Capture,
    session_id: str,
    mesh_id: str,
    txt_format: bool = True,
    json_format: bool = False,
):
    """
    Detect QR codes in the images of a session and save them to a file (qr_map).

    Parameters:
     - capture (Capture): Capture object containing the images and sessions.
     - session_id (str): ID of the session to process.
     - mesh_id (str, optional): ID of the mesh to use. Defaults to "mesh".
     - txt_format (bool, optional): Whether to save the QR map in TXT format.
       Defaults to True.
     - json_format (bool, optional): Whether to save the QR map in JSON format.
       Defaults to False.

    Returns: None
    """
    # Detect QR codes in images (parallel). If a file already exists at
    # `qrcode_path`, the function will load the QR codes from it instead of
    # detecting them again in the image. Otherwise, it detects the QR codes in
    # the image and saves them to `qrcode_path`.
    detect_qr_codes_parallel(capture, session_id)

    # Get list of all QR codes (qr_map).
    qr_map = create_qr_map(capture, session_id, mesh_id)

    # Filtering QR codes, keeping the one with the largest area per ID.
    qr_map_filtered = filter_qr_codes_by_area(qr_map)

    # Save list of QR codes.
    qrcode_dir = capture.proc_path(session_id) / "qrcodes"
    qrcode_dir.mkdir(exist_ok=True, parents=True)
    save_qr_maps(qr_map, qr_map_filtered, qrcode_dir, json_format, txt_format)


def detect_qr_codes_parallel(capture: Capture, session_id: str):
    """
    Detect QR codes in images in parallel and save them to files.

    Parameters:
    capture (Capture): Capture object containing the images and sessions.
    session_id (str): ID of the session to process.
    """
    # Prepare paths.
    output_dir = capture.proc_path(session_id)
    session = capture.sessions[session_id]
    assert session.images is not None
    image_dir = capture.data_path(session_id)
    qrcode_dir = output_dir / "qrcodes"
    qrcode_dir.mkdir(exist_ok=True, parents=True)
    suffix = ".qrcodes.txt"

    logger.info("Detecting QR codes in parallel.")
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    for ts, cam_id in session.images.key_pairs():
        filename = session.images[ts, cam_id]
        image_path = image_dir / filename
        qrcode_path = (qrcode_dir / filename).with_suffix(suffix)
        pool.apply_async(
            _detect_or_load_qr_codes, args=(image_path, qrcode_path)
        )
    pool.close()
    pool.join()


def _detect_or_load_qr_codes(image_path: Path, qrcode_path: Path):
    """
    Detect QR codes in an image and save them to a file, or load them if they
    already exist.

    This function creates a QRCodeDetector object for the given image. If a file
    already exists at the specified QR code path, it loads the QR codes from
    this file. Otherwise, it detects the QR codes in the image and saves them to
    the file. After processing the QR codes, it logs the name of the image file.

    Parameters:
    - image_path (Path): The path to the image file.
    - qrcode_path (Path): The path to the QR codes file (saved or loaded from).
    """
    qrcodes = QRCodeDetector(image_path)
    if qrcode_path.is_file():
        qrcodes.load(qrcode_path)
    else:
        qrcodes.detect()
        qrcodes.save(qrcode_path)
    logger.info(f"Processed QR codes for {image_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=description)
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
        help="[Optional] List of sessions to process. If not provided, all "
        "sessions in the `capture_path` are processed. Useful when "
        "processing only specific sessions.",
    )
    parser.add_argument(
        "--navvis_path",
        type=Path,
        default=None,
        required=False,
        help="[Optional] Specifies NavVis data path. This argument is ignored "
        "if `--capture_path` exists, which is useful when the data has already "
        "been converted to the CAPTURE format and the original NavVis data is no "
        "longer available.",
    )
    parser.add_argument(
        "--use_simplified_mesh",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use simplified mesh. Default: False. Pass --use_simplified_mesh "
        "to set to True. This is useful for large scenes.",
    )
    parser.add_argument(
        "--visualization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write out MeshLab visualization. Default: True. "
        "Pass --no-visualization to set to False.",
    )
    parser.add_argument(
        "--txt_format",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write out QR maps in txt format. Default: True.",
    )
    parser.add_argument(
        "--json_format",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write out QR maps in json format. Default: False.",
    )
    args = parser.parse_args().__dict__

    run(**args)
