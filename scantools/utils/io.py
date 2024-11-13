from pathlib import Path
import logging
from typing import List, Iterator, Optional
import numpy as np

logger = logging.getLogger(__name__)

CSV_COMMENT_CHAR = '#'
DEPTH_SCALE = 1000.


def read_csv(path: Path, expected_columns: Optional[List[str]] = None) -> List[List[str]]:
    if not path.exists():
        raise IOError(f'CSV file does not exsit: {path}')

    data = []
    check_header = expected_columns is not None
    with open(str(path), 'r') as fid:
        for line in fid:
            line = line.strip()
            if len(line) == 0:
                continue
            if line[0] == CSV_COMMENT_CHAR:
                if check_header and len(data) == 0:
                    columns = [w.strip() for w in line[1:].split(',')]
                    if columns != expected_columns:
                        raise ValueError(
                            f'Got CSV columns {columns} but expected {expected_columns}.')
                check_header = False
            else:
                words = [w.strip() for w in line.split(',')]
                data.append(words)
    return data


def write_csv(path: Path, table: Iterator[List[str]], columns: Optional[List[str]] = None):
    if not path.parent.exists():
        raise IOError(f'Parent directory does not exsit: {path}')

    with open(str(path), 'w') as fid:
        if columns is not None:
            header = CSV_COMMENT_CHAR + ' ' + ', '.join(columns) + '\n'
            fid.write(header)
        for row in table:
            data = ', '.join(row) + '\n'
            fid.write(data)


try:
    import PIL.Image
    import cv2
except ImportError:
    logger.info('Optional dependency not installed: pillow')
else:
    def read_image(path: Path) -> np.ndarray:
        return np.asarray(PIL.Image.open(path))

    def write_image(path: Path, image: np.ndarray):
        PIL.Image.fromarray(image).save(path)

    def write_depth(path: Path, depth: np.ndarray):
        depth = depth * DEPTH_SCALE
        dtype = np.uint16
        mask = (depth > np.iinfo(dtype).max) | (depth < 0)
        depth[mask] = 0  # avoid overflow
        im = PIL.Image.fromarray(depth.round().astype(dtype))
        im.save(path, format='PNG', compress_level=9)

    def convert_dng_to_jpg(dng_path: Path):
        try:
            import rawpy  # pylint: disable=import-outside-toplevel
        except ImportError:
            raise ImportError("rawpy not installed. Please run: pip install rawpy")

        if not dng_path.exists():
            raise FileNotFoundError(f'Input DNG file does not exist: {dng_path}')
        with rawpy.imread(str(dng_path)) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                output_color=rawpy.ColorSpace.sRGB,
                no_auto_bright=True,
                user_flip=0,  # Ignore any flip information; 0 means no flip
            )
            image = PIL.Image.fromarray(rgb)
            image.save(str(dng_path.with_suffix(".jpg")), format='JPEG', quality=100)

try:
    import cv2
except ImportError:
    logger.info('Optional dependency not installed: opencv')
else:
    def read_depth(path: Path) -> np.ndarray:
        # Much faster than PIL.Image for high-res images
        depth = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH) / DEPTH_SCALE
        return depth

try:
    import open3d as o3d
except ImportError:
    logger.info('Optional dependency not installed: open3d')
else:
    def read_pointcloud(path: Path) -> o3d.geometry.PointCloud:
        logger.info('Reading point cloud %s.', path.resolve())
        return o3d.io.read_point_cloud(str(path))

    def read_mesh(path: Path) -> o3d.geometry.TriangleMesh:
        logger.info('Reading mesh %s.', path.resolve())
        return o3d.io.read_triangle_mesh(str(path))

    def write_mesh(path: Path, mesh: o3d.geometry.TriangleMesh):
        logger.info('Writing mesh to %s.', path.resolve())
        o3d.io.write_triangle_mesh(
            str(path), mesh, compressed=True, print_progress=True)

try:
    import plyfile
except ImportError:
    logger.info('Optional dependency not installed: plyfile')
else:
    def write_pointcloud(path: Path, pcd: o3d.geometry.PointCloud,
                         write_normals: bool = True, xyz_dtype: float = 'float32'):
        logger.info('Writing point cloud to %s.', path.resolve())
        assert pcd.has_points()
        write_normals = write_normals and pcd.has_normals()
        dtypes = [('x', xyz_dtype), ('y', xyz_dtype), ('z', xyz_dtype)]
        if write_normals:
            dtypes.extend([('nx', xyz_dtype), ('ny', xyz_dtype), ('nz', xyz_dtype)])
        if pcd.has_colors():
            dtypes.extend([('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        data = np.empty(len(pcd.points), dtype=dtypes)
        data['x'], data['y'], data['z'] = np.asarray(pcd.points).T
        if write_normals:
            data['nx'], data['ny'], data['nz'] = np.asarray(pcd.normals).T
        if pcd.has_colors():
            colors = (np.asarray(pcd.colors)*255).astype(np.uint8)
            data['red'], data['green'], data['blue'] = colors.T
        with open(str(path), mode='wb') as f:
            plyfile.PlyData([plyfile.PlyElement.describe(data, 'vertex')]).write(f)
