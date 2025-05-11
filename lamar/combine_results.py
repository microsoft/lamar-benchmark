import argparse
import datetime
import zipfile
from pathlib import Path

from . import logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Combine estimated poses from multiple scenes / devices in a zip file for evaluation."
    )
    parser.add_argument(
        "--cab_phone_path",
        type=Path,
        default=None,
        help="File containing the CAB phone estimated poses.",
    )
    parser.add_argument(
        "--lin_phone_path",
        type=Path,
        default=None,
        help="File containing the LIN phone estimated poses.",
    )
    parser.add_argument(
        "--hge_phone_path",
        type=Path,
        default=None,
        help="File containing the HGE phone estimated poses.",
    )
    parser.add_argument(
        "--cab_hololens_path",
        type=Path,
        default=None,
        help="File containing the CAB HoloLens estimated poses.",
    )
    parser.add_argument(
        "--lin_hololens_path",
        type=Path,
        default=None,
        help="File containing the LIN HoloLens estimated poses.",
    )
    parser.add_argument(
        "--hge_hololens_path",
        type=Path,
        default=None,
        help="File containing the HGE HoloLens estimated poses.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Output directory where the zip will be saved.",
        required=True,
    )
    args = parser.parse_args().__dict__
    output_dir = args.pop("output_dir")

    # Generate timestamp for the zip file name.
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    zip_filename = output_dir / f"result_{timestamp}.zip"
    
    # Create the zip file with existing paths.
    logger.info(f"Creating zip file at {zip_filename}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for name, path in args.items():
            split = name.split("_")
            assert len(split) == 3
            assert split[-1] == "path"
            split = split[:-1]
            if path is None:
                logger.warning(f"No path provided for [{split}], skipping...")
                continue
            assert path.exists()
            assert path.is_file()
            assert path.suffix == ".txt", f"Expected .txt file, got {path.suffix}"
            logger.info(f"Adding [{split}] file from {path} to zip")
            zipf.write(path, arcname=f"{split[0].upper()}_query_{split[1]}.txt")
    logger.info(f"Successfully created zip file at {zip_filename}")
