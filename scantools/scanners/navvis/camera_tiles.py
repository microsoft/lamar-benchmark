""" Generation of tiled images from images of the native (wide-angle) NavVis M6 cameras.  """
from enum import Enum
import math


class TileFormat(Enum):
    """ Tile Formats for generating undistorted image tiles.
        Valid formats: {TILES_none, TILES_2x2, TILES_3x3, TILES_5x5, TILES_center, TILES_cross}
    """
    TILES_none = 0
    TILES_2x2 = 1
    TILES_3x3 = 2
    TILES_5x5 = 3
    TILES_center = 4
    TILES_cross = 5


class Tiles:
    """ Creates Tiles (metadata ocam model: angles, zoom_factor).
        It requires the width and height of the original image that will be divided by # tiles.
    """
    def __init__(self, device, img_width, img_height, tile_format):

        # Typechecks
        if not isinstance(device, str):
            raise ValueError("Invalid device:", device)

        if not isinstance(tile_format, TileFormat):
            raise ValueError("Invalid tile format")

        if not isinstance(img_width, int):
            raise ValueError("Invalid image width:", img_width)

        if not isinstance(img_height, int):
            raise ValueError("Invalid image width:", img_height)

        self.device = device

        self.tile_format = tile_format

        self.width = None
        self.height = None

        self.angles = []
        self.zoom_factor = None

        self._config(img_width, img_height, tile_format)


    def _config(self, img_width, img_height, tile_format):

        if self.device == 'M6':
            # no tiles
            if tile_format is TileFormat.TILES_none:
                self.zoom_factor = 4
                self.width = img_width
                self.height = img_height
                self.angles.append([0, 0, 0])

            # tiles 2x2
            elif tile_format == TileFormat.TILES_2x2:
                self.zoom_factor = 1
                size_reduction = 2
                self.width = int(img_width / size_reduction)
                self.height = int(img_height / size_reduction)
                rad_30 = math.radians(30)
                rad_15 = math.radians(15)
                self.angles.append([0, rad_15, 0])
                self.angles.append([0, -rad_15, 0])
                self.angles.append([rad_30, rad_15, 0])
                self.angles.append([rad_30, -rad_15, 0])

            # tiles 3x3
            elif tile_format == TileFormat.TILES_3x3:
                self.zoom_factor = 0.75
                size_reduction = 4
                self.width = int(img_width / size_reduction)
                self.height = int(img_height / size_reduction)
                tile_angle_jump_x = math.radians(41)
                tile_angle_jump_y = math.radians(31)
                for i in [-1, 0, 1]:
                    for j in [-1, 0, 1]:
                        current_tile = [tile_angle_jump_x * i,   # x-axis angle
                                        tile_angle_jump_y * j,   # y-axis angle
                                        0]                       # z-axis angle
                        self.angles.append(current_tile)

            # tiles 5x5
            elif tile_format == TileFormat.TILES_5x5:
                self.zoom_factor = 0.5
                size_reduction = 4
                self.width = int(img_width / size_reduction)
                self.height = int(img_height / size_reduction)
                tile_angle_jump_x = math.radians(25)
                tile_angle_jump_y = math.radians(16)
                for i in [-2, -1, 0, 1, 2]:
                    for j in [-2, -1, 0, 1, 2]:
                        current_tile = [tile_angle_jump_x * i,   # x-axis angle
                                        tile_angle_jump_y * j,   # y-axis angle
                                        0]                       # z-axis angle
                        self.angles.append(current_tile)

            # center crop
            elif tile_format == TileFormat.TILES_center:
                self.zoom_factor = 3
                self.width = img_width
                self.height = img_height
                self.angles.append([0, 0, 0])

            else:
                raise ValueError("Non-implemented tile format:", tile_format, "for device:", self.device)

        elif self.device == 'VLX':
            # tiles 2x2
            if tile_format == TileFormat.TILES_2x2:
                self.zoom_factor = 1.5
                size_reduction = 2
                self.width = int(img_width / size_reduction)
                self.height = int(img_height / size_reduction)
                rad_30 = math.radians(30)
                rad_48 = math.radians(48)
                self.angles.append([-rad_30,  rad_48, 0])
                self.angles.append([-rad_30, -rad_48, 0])
                self.angles.append([ rad_30,  rad_48, 0])
                self.angles.append([ rad_30, -rad_48, 0])

            # tiles 3x3
            elif tile_format == TileFormat.TILES_3x3:
                self.zoom_factor = 1.5
                size_reduction = 4.56
                self.width = int(img_width / size_reduction)
                self.height = int(img_height / size_reduction)
                tile_angle_jump_x = math.radians(40)
                tile_angle_jump_y = math.radians(35)
                for i in [-1, 0, 1]:
                    for j in [-1, 0, 1]:
                        current_tile = [tile_angle_jump_x * i,   # x-axis angle
                                        tile_angle_jump_y * j,   # y-axis angle
                                        0]                       # z-axis angle
                        self.angles.append(current_tile)

            # tiles 5x5
            elif tile_format == TileFormat.TILES_5x5:
                self.zoom_factor = 1
                size_reduction = 7.6
                self.width = int(img_width / size_reduction)
                self.height = int(img_height / size_reduction)
                tile_angle_jump_x = math.radians(28)
                tile_angle_jump_y = math.radians(29)
                for i in [-2, -1, 0, 1, 2]:
                    for j in [-2, -1, 0, 1, 2]:
                        current_tile = [tile_angle_jump_x * i,   # x-axis angle
                                        tile_angle_jump_y * j,   # y-axis angle
                                        0]                       # z-axis angle
                        self.angles.append(current_tile)

            # center crop
            elif tile_format == TileFormat.TILES_center:
                self.zoom_factor = 2
                self.width = img_width
                self.height = img_height
                self.angles.append([0, 0, 0])

            # cross vlx
            elif tile_format == TileFormat.TILES_cross:
                self.zoom_factor = 1.2
                size_reduction = 4.56
                self.width = int(img_width / size_reduction)
                self.height = int(img_height / size_reduction)
                self.angles.append([0.0, 0.0, 0.0])
                self.angles.append([1.1, 0.0, 0.0])
                self.angles.append([-1.1, 0.0, 0.0])
                self.angles.append([0.0, -0.9, 0.0])
                self.angles.append([0.0, 0.9, 0.0])

            else:
                raise ValueError("Non-implemented tile format:", tile_format, "for device:", self.device)

        else:
            raise ValueError("Non-implemented tiling for device:", self.device)

    @property
    def format(self):
        """ TileFormat returned as string

        Returns
        -------
        str
            Tile format, empty string if no tiles
        """
        if self.tile_format is TileFormat.TILES_none:
            return ""
        if self.tile_format is TileFormat.TILES_center:
            return "center"
        return self.tile_format.name.lower()


    def postfix(self, tile_id=0):
        if self.tile_format is TileFormat.TILES_none:
            return ""
        if self.tile_format is TileFormat.TILES_center:
            return "__center"
        return "__" + self.format + "_" + "{:02d}".format(tile_id)
