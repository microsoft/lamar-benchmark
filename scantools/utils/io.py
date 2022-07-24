from pathlib import Path
import logging
from typing import List, Iterator, Optional

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
    import open3d as o3d
except ImportError:
    logger.info('Optional dependency not installed: open3d')
else:
    def read_mesh(path: Path) -> o3d.geometry.TriangleMesh:
        logger.info('Reading mesh %s.', path.resolve())
        return o3d.io.read_triangle_mesh(str(path))
