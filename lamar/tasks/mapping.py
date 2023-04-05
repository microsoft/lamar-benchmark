import logging
from pathlib import Path
from copy import deepcopy
import numpy as np

import pycolmap
from hloc import triangulation

from scantools import run_capture_to_empty_colmap
from scantools.proc.alignment.image_matching import get_keypoints
from scantools.utils.geometry import to_homogeneous

from .feature_extraction import FeatureExtraction
from .feature_matching import FeatureMatching
from ..utils.capture import list_images_for_session
from ..utils.misc import same_configs, write_config


logger = logging.getLogger(__name__)


class MappingPaths:
    def __init__(self, root, config, session_id):
        self.root = root
        self.workdir = root / 'mapping' / session_id / config['name'] / config['features']['name']
        if config['name'] == 'triangulation':
            self.workdir /= Path(config['pairs']['name'], config['matches']['name'])
            self.sfm_empty = self.workdir / 'sfm_empty'
        self.sfm = self.workdir / 'sfm'
        self.config = self.workdir / 'configuration.json'


class Mapping:
    methods = {}
    method2class = {}
    method = {}

    def __init_subclass__(cls):
        '''Register the child classes into the parent'''
        name = cls.method['name']
        cls.methods[name] = cls.method
        cls.method2class[name] = cls

    def __new__(cls, config, *_, **__):
        '''Instanciate the object from the child class'''
        return super().__new__(cls.method2class[config['name']])

    def __init__(self, config, outputs, capture, session_id,
                 extraction: FeatureExtraction):
        assert extraction.session_id == session_id
        self.config = config = {
            **deepcopy(config),
            'features': extraction.config,
        }
        self.session_id = session_id
        self.extraction = extraction
        self.paths = MappingPaths(outputs, config, session_id)
        self.paths.workdir.mkdir(parents=True, exist_ok=True)

        overwrite = not same_configs(config, self.paths.config)
        if overwrite:
            logger.info('Mapping session %s via %s of features %s.',
                        session_id, config['name'], config['features']['name'])
            self.run(capture)
            write_config(config, self.paths.config)

    def run(self, capture):
        raise NotImplementedError

    def get_points3D(self, key, point2D_indices):
        raise NotImplementedError


class Triangulation(Mapping):
    method = {
        'name': 'triangulation',
        # some COLMAP parameters and thresholds
    }

    def __init__(self, config, outputs, capture, session_id,
                 extraction: FeatureExtraction, matching: FeatureMatching):
        assert matching.query_id == matching.ref_id == session_id
        config = {
            **deepcopy(config),
            'pairs': matching.pair_selection.config.to_dict(),
            'matches': matching.config,
        }
        self.matching = matching
        super().__init__(config, outputs, capture, session_id, extraction)
        self.reconstruction = pycolmap.Reconstruction(self.paths.sfm)
        keys, names, _ = list_images_for_session(capture, self.session_id)
        self.name2key = dict(zip(names, keys))
        self.key2imageid = {
            self.name2key[image.name]: image.image_id
            for image in self.reconstruction.images.values()
        }

    def run(self, capture):
        run_capture_to_empty_colmap.run(capture, [self.session_id], self.paths.sfm_empty)
        triangulation.main(
            self.paths.sfm,
            self.paths.sfm_empty,
            capture.sessions_path(),
            self.matching.pair_selection.paths.pairs_hloc,
            self.extraction.paths.features,
            self.matching.paths.matches,
        )

    def get_points3D(self, key, point2D_indices):
        image = self.reconstruction.images[self.key2imageid[key]]
        valid = []
        xyz = []
        ids = []
        if len(image.points2D) > 0:
            for idx in point2D_indices:
                p = image.points2D[idx]
                valid.append(p.has_point3D())
                if valid[-1]:
                    ids.append(p.point3D_id)
                    xyz.append(self.reconstruction.points3D[ids[-1]].xyz)
        return np.array(valid, bool), xyz, ids


class MeshLifting(Mapping):
    method = {
        'name': 'mesh_lifting',
        'mesh_id': 'mesh_simplified',
    }

    def __init__(self, config, outputs, capture, session_id,
                 extraction: FeatureExtraction, matching: FeatureMatching = None):
        super().__init__(config, outputs, capture, session_id, extraction)
        from scantools.proc.rendering import Renderer
        from scantools.utils.io import read_mesh
        session = capture.sessions[self.session_id]
        proc_path = capture.proc_path(self.session_id)
        mesh = read_mesh(proc_path / session.proc.meshes[self.config['mesh_id']])
        self.renderer = Renderer(mesh)
        keys, names, _ = list_images_for_session(capture, self.session_id)
        self.name2key = dict(zip(names, keys))
        self.key2name = dict(zip(keys, names))
        self.session = session
        self.cache = {}

    def run(self, capture):
        pass

    def lift_points2D(self, key, p2d):
        camera = self.session.sensors[key[1]]
        T_cam2w = self.session.get_pose(*key)

        origins = np.tile(T_cam2w.t.astype(np.float32)[None], (len(p2d), 1))
        p2d_norm = camera.image2world(p2d.astype(np.float32))
        R = T_cam2w.R
        directions = to_homogeneous(p2d_norm.astype(np.float32)) @ R.astype(np.float32).T
        origins = np.ascontiguousarray(origins, dtype=np.float32)
        directions = np.ascontiguousarray(directions, dtype=np.float32)
        rays = (origins, directions)
        xyz, valid = self.renderer.compute_intersections(rays)
        return np.array(valid, bool), xyz

    def get_points3D(self, key, point2D_indices):
        name = self.key2name[key]
        (p2d,), _ = get_keypoints(self.extraction.paths.features, [name])
        if key in self.cache:
            valid, p2d_idx_to_xyz = self.cache[key]
        else:
            valid, xyz = self.lift_points2D(key, p2d)
            xyz_idx = 0
            p2d_idx_to_xyz = {}
            for idx, is_valid in enumerate(valid):
                if is_valid:
                    p2d_idx_to_xyz[idx] = xyz[xyz_idx]
                    xyz_idx += 1
            assert xyz_idx == len(xyz)
            self.cache[key] = [valid, p2d_idx_to_xyz]
        xyz = [p2d_idx_to_xyz[idx] for idx in point2D_indices if valid[idx]]
        valid = valid[np.array(point2D_indices, int)]
        return valid, list(xyz), [-1]*len(xyz)


class Hybrid(Mapping):
    method = {
        'name': 'hybrid',
        'triangulation': Triangulation.method,
        'lifting': MeshLifting.method,
    }

    def __init__(self, config, outputs, capture, session_id,
                 extraction: FeatureExtraction, matching: FeatureMatching):
        self.triangulation = Mapping(
            config['triangulation'], outputs, capture, session_id,
            extraction, matching)
        self.lifting = Mapping(
            config['lifting'], outputs, capture, session_id, extraction)
        config = {
            **deepcopy(config),
            'pairs': matching.pair_selection.config.to_dict(),
            'matches': matching.config,
        }
        super().__init__(config, outputs, capture, session_id, extraction)
        self.name2key = self.lifting.name2key  # pylint: disable=no-member

    def run(self, capture):
        pass

    def get_points3D(self, key, point2D_indices):
        valid_tri, xyz_tri, ids_tri = self.triangulation.get_points3D(key, point2D_indices)
        indices_lifting = np.array(point2D_indices)[~valid_tri]
        valid_lift, xyz_lift, ids_lift = self.lifting.get_points3D(key, indices_lifting)
        valid = []
        xyz = []
        ids = []
        i = 0
        for v in valid_tri:
            if v:
                valid.append(v)
                xyz.append(xyz_tri.pop(0))
                ids.append(ids_tri.pop(0))
            else:
                v = valid_lift[i]
                i += 1
                valid.append(v)
                if v:
                    xyz.append(xyz_lift.pop(0))
                    ids.append(ids_lift.pop(0))
        assert len(xyz_tri) == len(ids_tri) == len(xyz_lift) == len(ids_lift) == 0
        return np.array(valid, bool), xyz, ids
