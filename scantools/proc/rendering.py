import logging
from typing import Tuple
import open3d as o3d
import numpy as np

try:
    import raybender as rb
    import raybender.utils as rbutils
except ImportError as error:
    logging.getLogger(__name__).error(
        'Could not find raybender; install it from https://github.com/cvg/raybender.')
    raise error

from ..capture import Pose, Camera
from ..utils.geometry import to_homogeneous


def compute_rays(T_cam2w: Pose, camera: Camera, stride: int = 1):
    w, h = np.array((camera.width, camera.height)) // stride
    center = T_cam2w.t
    origins = np.tile(center.astype(np.float32)[None], (h*w, 1))

    p2d = np.mgrid[:h, :w].reshape(2, -1)[::-1].T.astype(np.float32)
    p2d += 0.5  # to COLMAP coordinates
    if stride != 1:
        p2d *= stride
    p2d_norm = camera.image2world(p2d)
    R = T_cam2w.R
    # It is much faster to perform the transformation in fp32
    directions = to_homogeneous(p2d_norm.astype(np.float32)) @ R.astype(np.float32).T
    # directions = to_homogeneous(p2d_norm) @ R.T

    # Outputs must be contiguous.
    origins = np.ascontiguousarray(origins, dtype=np.float32)
    directions = np.ascontiguousarray(directions, dtype=np.float32)
    return origins, directions


class Renderer:
    def __init__(self, mesh: o3d.geometry.TriangleMesh):
        vertices = np.asarray(mesh.vertices).astype(np.float32)
        if mesh.has_vertex_colors():
            vertex_colors = np.asarray(mesh.vertex_colors).astype(np.float32)
        else:
            vertex_colors = np.random.rand(vertices.shape[0], 3).astype(np.float32)
        triangles = np.asarray(mesh.triangles).astype(np.int32)

        self.scene = rb.create_scene()
        rb.add_triangle_mesh(self.scene, vertices, triangles)
        self.geom = (triangles, vertices, vertex_colors)

    def render_from_capture(self, T_cam2w: Pose, camera: Camera) -> Tuple[np.ndarray]:
        T_w2cam = T_cam2w.inverse()
        ray_origins, ray_directions = compute_rays(T_cam2w, camera)

        geom_ids, bcoords = rb.ray_scene_intersection(
            self.scene, ray_origins, ray_directions)
        *_, tri_ids, bcoords, valid = rbutils.filter_intersections(geom_ids, bcoords)

        rgb, depth = rbutils.interpolate_rgbd_from_geometry(
            *self.geom, tri_ids, bcoords, valid,
            T_w2cam.R, T_w2cam.t, camera.width, camera.height)

        return rgb, depth

    def compute_intersections(self, rays: Tuple) -> Tuple:
        origins, directions = rays
        origins = np.ascontiguousarray(origins, dtype=np.float32)
        directions = np.ascontiguousarray(directions, dtype=np.float32)

        geom_ids, bcoords = rb.ray_scene_intersection(self.scene, origins, directions)
        *_, tri_ids, bcoords, valid = rbutils.filter_intersections(geom_ids, bcoords)
        locations = rb.barycentric_interpolator(tri_ids, bcoords, *self.geom[:2])
        return locations, valid

    def release(self):
        if self.scene is None:
            raise ValueError('Cannot release twice, create a new object.')
        rb.release_scene(self.scene)
        self.scene = None
