import csv
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from scantools import logger

try:
    from pyzbar.pyzbar import ZBarSymbol, decode  # pip install pyzbar-upright
except ImportError as error:
    logger.info(
        "Optional dependency not installed: pyzbar-upright. Please install it "
        "with 'pip install pyzbar-upright' to enable QR code detection."
    )
    raise error


@dataclass
class QRCodeDetector:
    """
    A class for detecting QR codes in an image.

    This class uses pyzbar (`pip install pyzbar-upright`) libraries to detect QR
    codes in an image file. Detected QR codes are stored in a list, with each QR
    code represented as a dictionary containing its data and 2D points of the
    corners.

    Attributes
    ----------
    - image_path : str
        The path to the image file where QR codes will be detected.
    - qrcodes : list, optional
        A list to store detected QR code information (default is an empty list).

    Methods
    -------
    - detect()
        Detects QR codes in the specified image and populates the 'qrcodes'
        attribute.
    - load(path)
        Loads QR code data from a CSV file.
    - save(path)
        Saves detected QR code data to a CSV file.
    - show(markersize=1)
        Displays the image with detected QR codes marked.
    """

    image_path: str
    qrcodes: list = field(default_factory=list)

    def __post_init__(self):
        if not Path(self.image_path).is_file():
            raise FileNotFoundError(
                f"The file {self.image_path} was not found."
            )

    def __getitem__(self, key):
        return self.qrcodes[key]

    def __iter__(self):
        return iter(self.qrcodes)

    def __len__(self):
        return len(self.qrcodes)

    def is_empty(self):
        return len(self.qrcodes) == 0

    def detect(self):
        try:
            img = cv2.imread(str(self.image_path))
            if img is None:
                raise ValueError("Unable to read the image file.")
            detected_qrcodes = decode(img, symbols=[ZBarSymbol.QRCODE])

            for qr in detected_qrcodes:
                qr_code = {
                    "id": qr.data.decode("utf-8"),
                    "points2D": np.asarray(qr.polygon, dtype=float).tolist(),
                }
                self.qrcodes.append(qr_code)
        except Exception as e:
            raise RuntimeError(
                f"An error occurred during QR code detection: {e}"
            )

    def __csv_header(self):
        return [
            "# qrcode_id",
            "top-left-corner.x",
            "top-left-corner.y",
            "bottom-left-corner.x",
            "bottom-left-corner.y",
            "bottom-right-corner.x",
            "bottom-right-corner.y",
            "top-right-corner.x",
            "top-right-corner.y",
        ]

    def load(self, path):
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)

            header = next(reader, None)
            if header != self.__csv_header():
                raise ValueError(
                    "The CSV header file does not match the expected format."
                )

            self.qrcodes = []
            for row in reader:
                qr_code = {
                    "id": row[0],
                    "points2D": [
                        [float(row[1]), float(row[2])],
                        [float(row[3]), float(row[4])],
                        [float(row[5]), float(row[6])],
                        [float(row[7]), float(row[8])],
                    ],
                }
                self.qrcodes.append(qr_code)

    def save(self, path):
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.__csv_header())
            for qr in self.qrcodes:
                row = [qr["id"]]
                for point in qr["points2D"]:
                    row.extend(point)
                writer.writerow(row)

    def show(self, markersize: int = 1):
        """
        Display the image with detected QR codes.

        This function reads an image from `self.image_path`, displays it using
        matplotlib, and overlays the detected QR codes. Each corner of the QR
        codes is marked with a different color.

        Parameters:
        - markersize (int, optional): The size of the markers that indicate the
                                      corners of the QR codes. Defaults to 1.

        Raises:
        - ValueError: If the image file cannot be read.
        """
        try:
            img = cv2.imread(str(self.image_path))
            if img is None:
                raise ValueError("Unable to read the image file.")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.figure(0, figsize=(30, 70))
            plt.imshow(img)

            colors = ["m.", "g.", "b.", "r."]
            for qr in self.qrcodes:
                # pyzbar returns points in the following order:
                #   1. top-left, 2. bottom-left, 3. bottom-right, 4. top-right
                logger.info(f"Found QR Code: {qr['id']}")
                logger.info(qr["points2D"])
                for i, point in enumerate(qr["points2D"]):
                    x, y = point
                    plt.plot(x, y, colors[i], markersize)
            plt.show()
        except Exception as e:
            logger.info(f"Error displaying the image: {e}")
