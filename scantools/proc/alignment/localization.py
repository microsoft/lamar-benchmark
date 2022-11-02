import logging
from typing import Dict, List, Tuple, Optional
from math import ceil
from collections import defaultdict
import dataclasses
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
import pycolmap

from . import Paths, save_stats, image_matching as imatch
from .scan import lift_points_to_3d
from ...capture import Capture, Pose, Trajectories, Camera
from ...utils.configuration import BaseConf
from ...viz.alignment import plot_pnp_inliers, plot_pnp_inliers_rig, plot_raw_matches
from ...viz.image import save_plot


logger = logging.getLogger(__name__)


class LocalizerConf(BaseConf):
    keyframing: imatch.KeyFramingConf = dataclasses.field(default_factory=imatch.KeyFramingConf)
    min_num_matches: Optional[int] = 10  # discard pairs with few matches
    min_num_inliers: int = 50  # discard localizations with few inliers
    pnp_error_multiplier: float = 1.0  # absolute pose inlier threshold multiplier
    num_visualize: int = 150  # number of queries to visualize
    dump_name: str = 'loc'

class RelocConf(BaseConf):
    do_rematching: bool = True
    Rt_thresh: Tuple[float] = (45., np.inf)  # [degrees, meters]
    pairs_file: str = 'pairs_overlap-pgo_{}.txt'
    dump_name: str = 'reloc'
    min_num_inliers: int = 20


@dataclasses.dataclass
class Batch2d3dMatcher:
    capture: Capture
    matching: imatch.MatchingConf
    conf: LocalizerConf
    paths: Paths
    visualize: bool = False

    # internal variables
    query_id = None
    ref_id = None
    image_root = None
    pairs = None
    impath2key_q = None
    impath2key_r = None
    matches_2d3d = None

    def __post_init__(self):
        self.image_root = self.capture.sessions_path()
        self.query_id = self.paths.query_id
        self.ref_id = self.paths.ref_id

        # Select the query images
        session_q = self.capture.sessions[self.query_id]
        self.impath2key_q = imatch.list_images_for_matching(
            session_q,
            self.capture.data_path(self.query_id).relative_to(self.image_root),
            self.matching,
            keyframing=self.conf.keyframing,
            poses=session_q.trajectories)

        # List the reference images
        self.impath2key_r = {}
        session_r = self.capture.sessions[self.ref_id]
        self.impath2key_r = imatch.list_images_for_matching(
            session_r,
            self.capture.data_path(self.ref_id).relative_to(self.image_root),
            self.matching)

    def do_matching_2d2d(self, from_overlap: bool = False,
                         T_q2w: Optional[Trajectories] = None):
        # Extract the query features
        path_q = self.paths.session(self.query_id)
        imatch.extract_session_features(
            self.impath2key_q.keys(), self.image_root, path_q, self.matching)

        # Extract the features of all images in each reference session
        path_r = self.paths.session(self.ref_id)
        imatch.extract_session_features(
            self.impath2key_r.keys(), self.image_root, path_r, self.matching)

        # Match the queries to all references
        if from_overlap:
            self.pairs = imatch.match_from_overlap(
                self.impath2key_q, self.impath2key_r, self.query_id, self.ref_id, self.capture,
                T_q2w, self.paths.outputs, self.matching, path_q, path_r)
        else:
            self.pairs = imatch.match_from_retrieval(
                self.impath2key_q.keys(), self.impath2key_r.keys(),
                path_q, path_r, self.paths.outputs, self.matching)
        return self.pairs

    def get_matches_2d3d(self, qname: str) -> Dict[str, np.ndarray]:
        (kp_q,), (noise,) = imatch.get_keypoints(self.paths.features(self.query_id)[1], [qname])
        assert noise is not None, 'Update hloc to store keypoint uncertainties'
        refs = self.pairs.get(qname, [])
        if len(refs) == 0:
            logger.info('Could not find any reference image for query %s.', qname)
        kp_r, _ = imatch.get_keypoints(self.paths.features(self.ref_id)[1], refs)
        all_matches = imatch.get_matches(self.paths.matches, zip([qname]*len(refs), refs))

        ret = {
            'kp_q': [np.empty((0, 2))],
            'kp_r': [np.empty((0, 2))],
            'p3d': [np.empty((0, 3))],
            'indices': [np.empty((0,), int)],
            'node_ids_ref': [np.empty((0, 2), object)]
        }
        for i, (ref, kp_r, matches) in enumerate(zip(refs, kp_r, all_matches)):
            if self.conf.min_num_matches and (len(matches) < self.conf.min_num_matches):
                continue
            kp_q_m, kp_r_m = kp_q[matches[:, 0]], kp_r[matches[:, 1]]
            key = self.impath2key_r[ref]
            p3d, valid = lift_points_to_3d(kp_r_m, key,
                                           self.capture.sessions[self.ref_id],
                                           self.capture.data_path(self.ref_id),
                                           align=True)
            if np.sum(valid) > 0:
                ret['kp_q'].append(kp_q_m[valid])
                ret['kp_r'].append(kp_r_m[valid])
                ret['p3d'].append(p3d[valid])
                ret['indices'].append(np.full(np.count_nonzero(valid), i))
                ret['node_ids_ref'].append(np.array(
                    [(key, kp_r_idx) for kp_r_idx in matches[valid, 1]], dtype=object))
        ret = {k: np.concatenate(v, 0) for k, v in ret.items()}
        return {**ret, 'keypoint_noise': noise}

    def compute_all_matches_2d3d(self, parallel: bool = True):
        self.matches_2d3d = {}
        names = list(self.impath2key_q)
        map_ = thread_map if parallel else lambda f, x: map(f, tqdm(x))
        ret = map_(self.get_matches_2d3d, names)
        self.matches_2d3d = {self.impath2key_q[n]: r for n, r in zip(names, ret)}

    def visualize_raw_matches(self):
        names = list(self.impath2key_q)
        for i in np.arange(0, len(names), np.ceil(len(names)/self.conf.num_visualize), int):
            key = self.impath2key_q[names[i]]
            plot_raw_matches(
                names[i], self.pairs[names[i]], self.matches_2d3d[key], self.image_root)
            viz_dir = self.paths.outputs / f'viz_matches_{self.conf.dump_name}'
            viz_path = viz_dir / f'{"_".join(map(str, key))}.png'
            viz_path.parent.mkdir(exist_ok=True, parents=True)
            save_plot(viz_path)
            plt.close()

    def run(self, overwrite: bool = False,
            from_overlap: bool = False,
            query_poses: Optional[Trajectories] = None) -> Dict:
        if self.paths.pairs.exists() and self.paths.matches.exists() and not overwrite:
            self.pairs = imatch.parse_retrieval(self.paths.pairs)
        else:
            self.do_matching_2d2d(from_overlap, query_poses)
            if self.visualize:
                self.visualize_raw_matches()

        path_matches2d3d = self.paths.outputs / f'matches_2d3d_{self.conf.dump_name}.pkl'
        if path_matches2d3d.exists() and not overwrite:
            with open(path_matches2d3d, 'rb') as fid:
                self.matches_2d3d = pickle.load(fid)
        else:
            self.compute_all_matches_2d3d()
            with open(path_matches2d3d, 'wb') as fid:
                pickle.dump(self.matches_2d3d, fid)
        return self.matches_2d3d


@dataclasses.dataclass
class BatchLocalizer:
    capture: Capture
    matching: imatch.MatchingConf
    conf: LocalizerConf
    paths: Paths
    visualize: bool = True

    # internal variables
    query_id = None
    ref_id = None
    matcher = None
    impath2key_q = None

    def __post_init__(self):
        self.matcher = Batch2d3dMatcher(self.capture, self.matching, self.conf, self.paths)
        self.impath2key_q = self.matcher.impath2key_q
        self.query_id = self.paths.query_id
        self.ref_id = self.paths.ref_id

    def _estimate_camera_pose(self, qname: str, camera: Camera) -> Dict:
        data = self.matcher.matches_2d3d[self.impath2key_q[qname]]
        if len(data['p3d']) == 0:
            return {'success': False}
        thresh = self.conf.pnp_error_multiplier * data['keypoint_noise']
        ret = pycolmap.absolute_pose_estimation(
            data['kp_q'], data['p3d'], camera.asdict, thresh, return_covariance=True)
        if ret['success']:
            ret['covariance'] *= data['keypoint_noise']**2
        return {**data, **ret}

    def _estimate_rig_pose(self, qnames: List[str], cameras: List[Camera],
                           T_cams2rig: List[Pose]) -> Dict:
        p3d = []
        p2d = []
        matches = []
        for qname in qnames:
            data = self.matcher.matches_2d3d[self.impath2key_q[qname]]
            p3d.append(data['p3d'])
            p2d.append(data['kp_q'])
            matches.append(data)
        if sum(map(len, p3d)) < 3:
            return {'success': False}
        camera_dicts = [camera.asdict for camera in cameras]
        T_rig2cams = [T.inverse() for T in T_cams2rig]
        qvecs = [p.qvec for p in T_rig2cams]
        tvecs = [p.t for p in T_rig2cams]
        thresh = self.conf.pnp_error_multiplier * data['keypoint_noise']
        ret = pycolmap.rig_absolute_pose_estimation(
            p2d, p3d, camera_dicts, qvecs, tvecs, thresh, return_covariance=True)
        if ret['success']:
            ret['covariance'] *= data['keypoint_noise']**2
        ret['matches'] = matches
        return ret

    def _run(self, parallel: bool = True) -> Tuple[List[int], Trajectories]:
        session = self.capture.sessions[self.query_id]
        cam2impath = {key: impath for impath, key in self.impath2key_q.items()}

        rig2cams = defaultdict(list)
        for ts, cam_id in cam2impath.keys():
            if session.rigs is None:
                rig2cams[ts, cam_id].append(cam_id)
            else:
                for cam_or_rig_id in session.trajectories[ts]:
                    if cam_id in session.rigs.get(cam_or_rig_id, {}):
                        # Here we assume that a sensor is part of at most one rig at a timestamp.
                        rig2cams[ts, cam_or_rig_id].append(cam_id)
                        break
                else:  # not part of any rig -> add as single image
                    rig2cams[ts, cam_id].append(cam_id)
        keys_rig = list(rig2cams.keys())

        poses = Trajectories()
        ninls = []

        def _worker_fn(idx: int):
            ts, rig_id = keys_rig[idx]
            cam_ids = rig2cams[ts, rig_id]
            qnames = [cam2impath[ts, i] for i in cam_ids]
            cameras = [session.sensors[i] for i in cam_ids]
            if len(cam_ids) == 1:
                (qname,), (camera,) = qnames, cameras
                ret = self._estimate_camera_pose(qname, camera)
            else:
                ret = self._estimate_rig_pose(
                    qnames, cameras, [session.rigs[rig_id, i] for i in cam_ids])
            if not ret['success']:
                return
            ninl = ret['num_inliers']
            ninls.append(ninl)
            if ninl < self.conf.min_num_inliers:
                return
            # the covariance returned by pycolmap is on the left side,
            # which is the right side of the inverse.
            T_cam2w = Pose(*Pose(ret['qvec'], ret['tvec']).inv.qt, ret['covariance'])
            if len(cam_ids) == 1 and rig_id in (session.rigs or {}):
                # rig_id corresponds to a rig and not to a single camera
                T_cam2w = T_cam2w * session.rigs[rig_id, cam_ids[0]].inverse() # T_w_cam * T_cam_rig
            poses[ts, rig_id] = T_cam2w

            if self.visualize and idx % ceil(len(keys_rig)/self.conf.num_visualize) == 0:
                viz_path = self.paths.outputs/f'viz_inliers_{self.conf.dump_name}/{ts}_{rig_id}.png'
                viz_path.parent.mkdir(exist_ok=True, parents=True)
                if len(cam_ids) == 1:
                    plot_pnp_inliers(qname, self.matcher.pairs[qname], ret, self.matcher.image_root)
                else:
                    plot_pnp_inliers_rig(qnames, self.matcher.pairs, ret, self.matcher.image_root)
                save_plot(viz_path)
                plt.close()

        map_ = thread_map if parallel else lambda f, x: map(f, tqdm(x))
        map_(_worker_fn, range(len(keys_rig)))
        return ninls, poses

    def run(self, overwrite: bool = False, **match_args) -> Trajectories:
        self.matcher.run(overwrite, **match_args)  # 2D-2D matching and 2D-3D lifting
        path = self.paths.outputs / f'trajectory_{self.conf.dump_name}.txt'
        if path.exists() and not overwrite:
            return Trajectories.load(path)

        ninls, poses = self._run()
        poses.save(path)
        if len(ninls) == 0:
            return poses
        success = np.asarray(ninls) >= self.conf.min_num_inliers
        logger.info('PnP+RANSAC inlier statistics: %.1f/%.1f/%.1f/%.1f mean/med/q.1/q.9 [%%]'
                    ', success: #%d/%.1f%%',
                    np.mean(ninls), np.median(ninls),
                    np.percentile(ninls, 10), np.percentile(ninls, 90),
                    success.sum(), 100*success.mean())
        stats = {
            'ratio_success': success.mean(),
            'num_success': success.sum(),
            'num_inliers': ninls,
        }
        save_stats(self.paths.stats(self.conf.dump_name), stats)
        return poses
