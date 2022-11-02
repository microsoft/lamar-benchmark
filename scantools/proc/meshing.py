import logging
from pathlib import Path
from typing import Tuple, Dict, Optional, Iterable, Callable
import open3d as o3d
import numpy as np

try:
    import pcdmeshing
except ImportError:
    pcdmeshing = None

logger = logging.getLogger(__name__)

def poisson_meshing(pcd: o3d.geometry.PointCloud,
                    depth: int = 12,
                    resolution: Optional[int] = None,
                    densities_thresh_resolution: Optional[float] = None,
                    densities_thresh_ratio: float = 0.01,
                    ) -> Tuple[o3d.geometry.TriangleMesh, Dict]:
    debug = {}

    bbox = pcd.get_axis_aligned_bounding_box()  # maybe oriented bbox?
    extent = bbox.get_extent().max()
    if resolution is not None:
        depth = int(np.ceil(np.log2(extent / resolution)))
    else:
        resolution = extent / 2**depth
    logger.info('Using depth=%d for PSR with resolution=%.4f m',
                depth, extent/2**depth)

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth)
        debug['psr_densities'] = densities

    if densities_thresh_resolution is not None:
        vertices_to_remove = densities < np.log2(extent / densities_thresh_resolution)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        debug['psr_removed_by_density'] = vertices_to_remove
    elif densities_thresh_ratio is not None:
        vertices_to_remove = densities < np.quantile(densities, densities_thresh_ratio)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        debug['psr_removed_by_density'] = vertices_to_remove

    return mesh, debug


def advancing_front_meshing(pcd: o3d.geometry.PointCloud,
                            visibility_dir: Optional[Path] = None,
                            simplify: Optional[Callable] = None,
                            num_parallel: int = 5,
                            ) -> o3d.geometry.TriangleMesh:
    if pcdmeshing is None:
        raise ValueError("Could not import package pcdmeshing for advancing_front_meshing. "
                         "Install it from https://github.com/cvg/pcdmeshing")
    all_path = obs_path = None
    if visibility_dir is not None:
        all_path = visibility_dir / "pointcloud-all.ply"
        obs_path = visibility_dir / "pointcloud-obs.ply"
    mesh, simplified = pcdmeshing.run_block_meshing(
        pcd,
        num_parallel=num_parallel,
        voxel_size=5,
        use_visibility=visibility_dir is not None,
        pcd_all_path=all_path,
        pcd_obs_path=obs_path,
        simplify_fn=simplify)
    return mesh, simplified


def simplify_mesh(mesh, factor, max_error):
    target_size = len(mesh.triangles) // factor
    mesh = mesh.simplify_quadric_decimation(target_size, maximum_error=max_error)
    return mesh


def mesh_from_pointcloud(pcd: o3d.geometry.PointCloud,
                         method: str = 'poisson',
                         psr_depth: int = 12,
                         psr_resolution: Optional[int] = None,
                         psr_densities_thresh_ratio: float = 0.01,
                         psr_densities_thresh_res: Optional[float] = None,
                         af_visibility_dir: Optional[Path] = None,
                         af_num_parallel: int = 10,
                         simplify_factor: Optional[int] = None,
                         simplify_error: Optional[float] = 1e-9,
                         ) -> Tuple[o3d.geometry.TriangleMesh, Dict]:
    debug = {}
    simplified = None
    simplify = None
    if simplify_factor is not None:
        simplify = lambda m: simplify_mesh(m, simplify_factor, simplify_error)

    if method == 'poisson':
        mesh, debug_psr = poisson_meshing(
            pcd, psr_depth, psr_resolution,
            psr_densities_thresh_res, psr_densities_thresh_ratio)
        debug = {**debug, **debug_psr}
        if simplify:
            simplified = simplify(mesh)
    elif method == 'advancing_front':
        mesh, simplified = advancing_front_meshing(
            pcd, visibility_dir=af_visibility_dir,
            simplify=simplify, num_parallel=af_num_parallel)
    else:
        raise ValueError(f'Unknown reconstruction method {method} not in '
                         '[poisson, advancing_front].')

    return mesh, simplified, debug
