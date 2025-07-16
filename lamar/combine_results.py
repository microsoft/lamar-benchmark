import argparse
import datetime
import zipfile
from pathlib import Path

from . import logger

LOCATIONS = ["CAB", "LIN", "HGE"]
DEVICES = ["hololens", "phone"]


def assert_valid_txt_path(path: Path):
    """Assert that the txt path is valid."""
    assert path.exists()
    assert path.is_file()
    assert path.suffix == ".txt"


def combine_results(
    description_path: Path, results_paths: dict[str, Path | None], output_dir: Path
):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Combine the estimated poses from multiple scenes / devices in a "
            "zip file for evaluation on CodaBench."
        )
    )
    parser.add_argument(
        "--description_path",
        type=Path,
        required=True,
        help="Path to a text file containing the description of the submission.",
    )
    for location in LOCATIONS:
        for device in DEVICES:
            parser.add_argument(
                f"--{location.lower()}_{device}_path",
                type=Path,
                default=None,
                help=f"File containing the estimated poses for {location} {device}.",
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
    assert len(args) == len(LOCATIONS) * len(DEVICES)

    combine_results(
        description_path=description_path,
        results_paths=args,
        output_dir=output_dir,
    )
