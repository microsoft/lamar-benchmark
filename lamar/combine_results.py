import argparse
import datetime
import zipfile
from pathlib import Path

from . import logger

def assert_valid_txt_path(path: Path):
    """Assert that the txt path is valid."""
    assert path.exists()
    assert path.is_file()
    assert path.suffix == ".txt"


def combine_results(description_path: Path, results_paths: dict[str, Path], output_dir: Path):
    # Generate timestamp for the zip file name.
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    zip_filename = output_dir / f"submission_{timestamp}.zip"
    
    # Create the zip file with existing paths.
    logger.info(f"Creating zip file at {zip_filename}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add the description file to the zip.
        logger.info(f"Adding description file from {description_path} to zip")
        assert_valid_txt_path(description_path)
        zipf.write(description_path, arcname="description.txt")
        for name, path in results_paths.items():
            split = name.split("_")
            assert len(split) == 3
            assert split[-1] == "path"
            split = split[:-1]
            if path is None:
                logger.warning(f"No path provided for [{split}], skipping...")
                continue
            logger.info(f"Adding [{split}] file from {path} to zip")
            assert_valid_txt_path(path)
            zipf.write(path, arcname=f"{split[0].upper()}_query_{split[1]}.txt")
    logger.info(f"Successfully created zip file at {zip_filename}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Combine estimated poses from multiple scenes / devices in a zip file for evaluation."
    )
    parser.add_argument(
        "--description_path",
        type=Path,
        required=True,
        help="Path to a text file containing description of the submission.",
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
        required=True,
        help="Output directory where the zip will be saved.",
    )
    args = parser.parse_args().__dict__
    output_dir = args.pop("output_dir")
    description_path = args.pop("description_path")

    combine_results(
        description_path=description_path,
        results_paths=args,
        output_dir=output_dir,
    )
