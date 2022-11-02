from pathlib import Path
from typing import Optional, List
from collections import defaultdict
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.cm

from ..capture import Camera, Pose, Session, Trajectories

TRAJECTORY_PLY_HEADER = '''ply
format ascii 1.0
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element edge {num_edges}
property int vertex1
property int vertex2
end_header
'''


class MeshlabProject:
    project_label = 'MeshLabProject'
    meshgroup_label = 'MeshGroup'
    rastergroup_label = 'RasterGroup'
    trajectory_cmap = matplotlib.cm.jet

    def __init__(self, path: Optional[Path] = None):
        if path is None:
            soup = BeautifulSoup(features='xml')
            project = soup.new_tag(self.project_label)
            soup.append(project)
            mgroup = soup.new_tag(self.meshgroup_label)
            project.append(mgroup)
            rgroup = soup.new_tag(self.rastergroup_label)
            project.append(rgroup)
        else:
            assert path.exists()
            with open(path) as fid:
                soup = BeautifulSoup(fid, features='xml')
            project = soup.find(self.project_label)
            assert project is not None
            mgroup = project.find(self.meshgroup_label)
            if mgroup is None:
                mgroup = soup.new_tag(self.meshgroup_label)
                project.append(mgroup)
            rgroup = project.find(self.rastergroup_label)
            if rgroup is None:
                rgroup = soup.new_tag(self.meshgroup_label)
                project.append(rgroup)

        self.soup = soup
        self.mgroup = mgroup
        self.rgroup = rgroup
        self.trajectories = defaultdict(list)
        self.trajectory_cmaps = defaultdict(list)

    def add_mesh(self, label: str, path: Path, T: Optional[np.ndarray] = None):
        tag_args = dict(name='MLMesh', label=label, filename=str(path))
        if self.mgroup.find(**tag_args) is not None:
            self.mgroup.find(**tag_args).extract()  # delete the tag
        mesh = self.soup.new_tag(**tag_args)

        mat_tag = self.soup.new_tag('MLMatrix44')
        if T is None:
            T = np.eye(4, dtype=int)
        mat_str = '\n' + '\n'.join(''.join(map(lambda x: str(x)+' ', row)) for row in T) + '\n'
        mat_tag.string = mat_str
        mesh.append(mat_tag)
        self.mgroup.append(mesh)

    def add_camera(self, label: str, camera: Camera, pose: Pose, camera_scale: float = 1.):
        tag_args = dict(name='MLRaster', label=label)
        if self.rgroup.find(**tag_args) is not None:
            raise ValueError('An identical raster tag already exists.')
        raster = self.soup.new_tag(**tag_args)

        width = camera.width
        height = camera.height
        fx, fy, cx, cy = camera.projection_params
        pixel_size = 0.0001 * camera_scale  # dummy
        f_mm = (fx + fy) / 2 * pixel_size

        tvec = pose.t * np.array([-1, -1, -1])
        R = pose.r.as_matrix().T
        # Bundler format: the y and z camera axes are inverted
        R[1] *= -1
        R[2] *= -1

        tvec = ' '.join(np.r_[tvec, 1].astype(str).tolist())
        rmat = np.eye(4)
        rmat[:3, :3] = R
        rmat_flat = ' '.join(rmat.reshape(-1).astype(str).tolist())

        camera_tag = self.soup.new_tag(
            'VCGCamera',
            TranslationVector=tvec,
            RotationMatrix=rmat_flat,
            LensDistortion='0 0',
            ViewportPx=f'{width} {height}',
            CenterPx=f'{cx} {cy}',
            PixelSizeMm=f'{pixel_size} {pixel_size}',
            FocalMm=f'{f_mm}',
        )
        raster.append(camera_tag)
        self.rgroup.append(raster)

    def add_trajectory_point(self, trajectory: str, pose: Pose):
        self.trajectories[trajectory].append(pose)

    def add_trajectory(self, label: str, poses: Trajectories, session: Session, cmap):
        self.trajectories[label] = []  # empty
        for ts, id_ in sorted(poses.key_pairs()):
            T_cam2w = poses[ts, id_]
            camera = session.sensors.get(id_)
            if camera is None:  # it's a rig! - pick any first camera
                cam_id, T_cam2rig = next(iter(session.rigs[id_].items()))
                camera = session.sensors[cam_id]
                T_cam2w = T_cam2w * T_cam2rig
            # Meshlab has trouble loading when there are too many camera frustums.
            # mlp.add_camera(f'{label}/{ts}/{id_}', camera, T_cam2w, trajectory=label)
            self.add_trajectory_point(label, T_cam2w)
        self.trajectory_cmaps[label] = cmap

    def _write_trajectory(self, name: str, poses: List[Pose], path: Path, cmap):
        if isinstance(cmap, str):
            colors = np.repeat(np.array(matplotlib.colors.to_rgba(cmap))[None], len(poses), 0)
        else:
            colors = cmap(np.linspace(0, 1, len(poses)))
        colors = (colors[:, :3]*255).astype(int)
        ply_text = TRAJECTORY_PLY_HEADER.format(num_points=len(poses), num_edges=len(poses)-1)
        for pose, color in zip(poses, colors):
            ply_text += ' '.join(map(str, pose.t.astype(np.float32).tolist() + color.tolist()))
            ply_text += '\n'
        ply_text += ''.join(f'{i} {i+1}\n' for i in range(len(poses)-1))
        ply_path = Path(str(path) + f'_traj_{name}.ply')
        with open(ply_path, 'w') as f:
            f.write(ply_text)
        self.add_mesh(f'trajectory/{name}', ply_path.name)

    def __repr__(self) -> str:
        # remove the indentation of prettify
        return '\n'.join(x.lstrip() for x in self.soup.prettify().split('\n'))

    def write(self, path: Path):
        for name, poses in self.trajectories.items():
            self._write_trajectory(
                name, poses, path, self.trajectory_cmaps.get(name, self.trajectory_cmap))
        with open(path, 'w') as f:
            f.write(str(self))
