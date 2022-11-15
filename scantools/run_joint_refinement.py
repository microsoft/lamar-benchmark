from typing import List, Union, Dict, Tuple, Set
from pathlib import Path
import os
import pickle
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pycolmap
import matplotlib.pyplot as plt

from hloc.pairs_from_retrieval import pairs_from_score_matrix

from . import logger
from .capture import Capture, Proc, Trajectories, Pose, KeyType
from .proc import alignment
from .proc.alignment import image_matching as imatch
from .proc.alignment.refinement import (
    Triangulator, GlobalRefiner, qt_dict_from_trajectories, RefinementConf, QTDictType)
from .proc.alignment.sequence import compute_pose_covariances, BAConf
from .proc.alignment.image_matching import MatchingConf, parse_retrieval, match_multi_sequences
from .proc.overlap import OverlapTracer, Renderer, compute_rays
from .utils.frustum import frustum_intersection_multisessions
from .utils.geometry import compute_epipolar_error
from .viz.meshlab import MeshlabProject
from .viz.alignment import plot_sequence_trajectories, colmap_reconstruction_to_ply
from .viz.qualitymap import ColMapPlotter

SFM_OUTPUT = 'refinement/colmap-refined'


def compute_pairs(capture: Capture,
                  session_ids: List[str],
                  sessionid2name2key,
                  sessionid2poses: Dict[str, Trajectories],
                  conf: MatchingConf, tracer: OverlapTracer):
    sessionid2names_keys = {i: list(zip(*sessionid2name2key[i].items())) for i in session_ids}
    all_names = [(i, n) for i in session_ids for n in sessionid2names_keys[i][0]]
    all_keys = [(i, k) for i in session_ids for k in sessionid2names_keys[i][1]]
    pose_list = [capture.sessions[i].get_pose(*k, sessionid2poses[i]) for i, k in all_keys]

    logger.info('Computing frustum intersections.')
    frustum_intersect = frustum_intersection_multisessions(
        capture, all_keys, sessionid2poses, num_threads=24)
    frustum_intersect[range(len(all_keys)), range(len(all_keys))] = False

    dR, dt = imatch.get_pairwise_distances(pose_list, pose_list)
    invalid = ~frustum_intersect | (dR > conf.Rt_thresh[0]) | (dt > conf.Rt_thresh[1])
    selected_ij_list = pairs_from_score_matrix(-dt, invalid, conf.num_pairs*2)
    selected_ij = defaultdict(list)
    for i, j in selected_ij_list:
        selected_ij[i].append(j)

    logger.info('Computing view overlaps via ray tracing.')
    pairs_ij = {}
    for i, js in tqdm(selected_ij.items()):
        sid_q, (ts_q, id_q) = all_keys[i]
        session_q = capture.sessions[sid_q]
        T_q = session_q.get_pose(ts_q, id_q, sessionid2poses[sid_q])
        cam_q = session_q.sensors[id_q]
        rays_q = compute_rays(T_q, cam_q, stride=tracer.get_stride(cam_q))
        intersections = tracer.renderer.compute_intersections(rays_q)
        js_ovs = []
        for j in js:
            sid_r, (ts_r, id_r) = all_keys[j]
            session_r = capture.sessions[sid_r]
            T_r = session_r.get_pose(ts_r, id_r, sessionid2poses[sid_r])
            cam_r = session_r.sensors[id_r]
            ov = tracer.compute_overlap_from_rays(rays_q, *intersections, T_r, cam_q, cam_r)
            if ov is not None:
                ov = np.mean(ov)
                if ov > 0.05:  # at least 5% visible
                    js_ovs.append((j, ov))
        if len(js_ovs) > 0:
            js, ovs = zip(*js_ovs)
            pairs_ij[i] = [js[idx] for idx in np.argsort(ovs)[::-1]]
    pairs_ij = [(i, j) for i, js in pairs_ij.items() for j in js]

    pairs_per_session = defaultdict(set)
    for i, j in pairs_ij:
        sid_i, name_i = all_names[i]
        sid_j, name_j = all_names[j]
        pair = (name_i, name_j)
        if sid_i <= sid_j:
            pairs_per_session[sid_i, sid_j].add(pair)
        else:
            pairs_per_session[sid_j, sid_i].add(pair[::-1])
    pairs_per_session = {k: list(v) for k, v in pairs_per_session.items()}
    return pairs_per_session


def scale_camera(scale: Union[float, np.ndarray], camera: pycolmap.Camera) -> Tuple[float]:
    if not isinstance(scale, np.ndarray):
        scale = np.array([scale, scale])
    w = int(round(camera.width * scale[0]))
    h = int(round(camera.height * scale[1]))
    scale_x = w / camera.width
    scale_y = h / camera.height
    camera.width = w
    camera.height = h
    if len(camera.focal_length_idxs()) == 1:
        camera.focal_length = camera.focal_length * (scale_x + scale_y) / 2
    else:
        camera.focal_length_x = camera.focal_length_x * scale_x
        camera.focal_length_y = camera.focal_length_y * scale_y
    camera.principal_point_x = camera.principal_point_x * scale_x
    camera.principal_point_y = camera.principal_point_y * scale_y
    return np.array((scale_x, scale_y))


def add_session_to_reconstruction(reconstruction, session, name2key, poses, features: Path,
                                  scale_cameras_to_unit_noise: bool = True):
    sensor_id_to_colmap_camera_id = {}
    camera_id_to_scale = {}
    key_to_colmap_image_id = {}
    for name, key in name2key.items():
        _, sensor_id = key
        T_w2c = session.get_pose(*key, poses).inverse()
        (kps,), (uncertainty,) = imatch.get_keypoints(features, [name])

        camera_id = sensor_id_to_colmap_camera_id.get(sensor_id)
        if camera_id is None:
            camera = pycolmap.Camera(session.sensors[sensor_id].asdict)
            camera_id = camera.camera_id = len(reconstruction.cameras)
            if scale_cameras_to_unit_noise:
                camera_id_to_scale[camera_id] = scale_camera(1/uncertainty, camera)
            else:
                camera_id_to_scale[camera_id] = np.ones(2)
            reconstruction.add_camera(camera)
            sensor_id_to_colmap_camera_id[sensor_id] = camera_id

        kps *= camera_id_to_scale[camera_id]
        image = pycolmap.Image(
            name, kps, T_w2c.t, T_w2c.qvec, camera_id, len(reconstruction.images)
        )
        image.registered = True
        reconstruction.add_image(image)
        key_to_colmap_image_id[key] = image.image_id

    return sensor_id_to_colmap_camera_id, key_to_colmap_image_id, camera_id_to_scale


def add_tracks_to_reconstruction(reconstruction, key_to_colmap_image_id, p3d_dict, roots_to_nodes):
    for p3d_key, xyz in tqdm(p3d_dict.items()):
        track = pycolmap.Track()
        for im_key, p2d_idx in roots_to_nodes[p3d_key]:
            image_id = key_to_colmap_image_id[im_key]
            track.add_element(image_id, p2d_idx)
        p3d_id = reconstruction.add_point3D(xyz, track)
        for el in track.elements:
            image = reconstruction.images[el.image_id]
            image.set_point3D_for_point2D(el.point2D_idx, p3d_id)


def scale_reconstruction(reconstruction, camera_id_to_scale):
    for camera in reconstruction.cameras.values():
        scale_camera(1/camera_id_to_scale[camera.camera_id], camera)
    for image in reconstruction.images.values():
        scale = camera_id_to_scale[image.camera_id]
        p2ds = image.points2D
        for i, _ in enumerate(p2ds):
            p2ds[i].xy = p2ds[i].xy / scale  # modified in place


def epipolar_match_filtering(name2key, pairs, matches_path, features_path, session, max_error):
    filtered_matches = {}
    for name_i in tqdm(pairs):
        key_i = name2key[name_i]
        (kps_i,), (noise_i,) = imatch.get_keypoints(features_path, [name_i])
        T_i2w = session.get_pose(*key_i, session.trajectories)
        cam_i = session.sensors[key_i[1]]

        for name_j in pairs[name_i]:
            key_j = name2key[name_j]
            (kps_j,), (noise_j,) = imatch.get_keypoints(features_path, [name_j])
            T_j2w = session.get_pose(*key_j, session.trajectories)
            cam_j = session.sensors[key_j[1]]
            matches = imatch.get_matches(matches_path, [[name_i, name_j]])[0]

            if matches.shape[0] == 0:
                filtered_matches[name_i, name_j] = np.zeros([0, 2])
                continue

            errors = compute_epipolar_error(
                T_i2w, T_j2w, cam_i, cam_j, kps_i[matches[:, 0]], kps_j[matches[:, 1]])
            valid_matches = (errors <= (max_error * (noise_i + noise_j) / 2))
            filtered_matches[name_i, name_j] = matches[valid_matches]

    return filtered_matches


def add_to_correspondence_graph(graph, pairs, colmap_images, all_matches: Union[Dict, Path]):
    name2id = {im.name: im.image_id for im in colmap_images.values()}
    pairs = [(i, j) for i in pairs for j in pairs[i]]
    processed = set()
    for name_i, name_j in pairs:
        id_i = name2id[name_i]
        id_j = name2id[name_j]
        assert graph.exists_image(id_i)
        assert graph.exists_image(id_j)

        if isinstance(all_matches, dict):
            matches = all_matches[name_i, name_j]
        else:
            matches = imatch.get_matches(all_matches, [[name_i, name_j]])[0]
        if matches.shape[0] == 0:
            continue

        if (name_j, name_i) in processed:
            continue
        processed.add((name_i, name_j))

        graph.add_correspondences(id_i, id_j, matches.astype(np.uint32))


def compute_keypoint_noise(rec: pycolmap.Reconstruction) -> float:
    nums = 0
    sos = 0
    for image_id in tqdm(rec.images):
        im = rec.images[image_id]
        p2ds = im.get_valid_point2D_ids()
        if len(p2ds) == 0:
            continue
        p3ds = [rec.points3D[im.points2D[i].point3D_id] for i in p2ds]
        proj = rec.cameras[im.camera_id].world_to_image(im.project(p3ds))
        diffs = np.stack(proj) - np.stack([im.points2D[i].xy for i in p2ds])
        sos += np.linalg.norm(diffs.reshape(-1))**2
        nums += len(diffs)
    noise = np.sqrt(sos / (2*nums-1))
    return noise


def recover_keyrig_keys(pose_keys, image_keys, session):
    '''Hack to recover which poses were used for matching.'''
    # TODO: track the keyrigs at keyframe selection
    keypose_keys = set()
    for ts, cam_id in pose_keys:
        if cam_id in session.sensors:
            if (ts, cam_id) in image_keys:
                keypose_keys.add((ts, cam_id))
        else: # rig
            rig_id = cam_id
            for cam_id_ in session.rigs[rig_id]:
                if (ts, cam_id_) in image_keys:
                    keypose_keys.add((ts, rig_id))
                    break
    return keypose_keys


def compute_covariances_per_sequence(capture: Capture,
                                     reconstruction: pycolmap.Reconstruction,
                                     sequence_ids: List[str],
                                     qts: Dict[str, QTDictType],
                                     key2imageid: Dict[str, Dict[KeyType, int]],
                                     conf: RefinementConf,
                                     observed_keypoint_noise: float = 1.0):
    covariances = defaultdict(dict)
    for sid in tqdm(sequence_ids):
        session = capture.sessions[sid]
        refiner = GlobalRefiner(reconstruction, key2imageid, conf)
        refiner.add_session_to_problem(session, qts[sid], session.trajectories, is_reference=False)
        refiner.make_cameras_constant()
        refiner.make_points_constant()
    #     refiner.solve()  # do a few steps of LM with frozen 3D points

        covar = compute_pose_covariances(refiner.problem, qts[sid], BAConf())
        if covar is None:
            logger.info('Cannot compute covariance for sequence %s.', sid)
            continue
        covariances[sid] = {}
        for k in qts[sid]:
            covariances[sid][k] = covar[k]*observed_keypoint_noise**2
    return dict(covariances)


def log_pose_uncertainties(output_dir: Path,
                           covariances: Dict[KeyType, np.ndarray],
                           qts: QTDictType,
                           rec: pycolmap.Reconstruction,
                           keyrig_keys: Dict[str, Set[KeyType]]):
    uncertainty = {}
    uncs = {}
    centers = {}
    for sid in covariances:
        keys = sorted(covariances[sid])
        cov = np.stack([covariances[sid][k] for k in keys])
        unc = np.sqrt(np.linalg.eig(cov)[0].max(1))
        if np.iscomplexobj(unc):  # numerical issues with symmetric matrix
            unc = unc.real
        uncertainty[sid] = dict(zip(keys, unc))
        is_keyrig = np.array([k in keyrig_keys[sid] for k in keys])
        centers[sid] = np.stack([Pose(*qts[sid][k]).inv.t for k in keys])[is_keyrig]
        uncs[sid] = unc[is_keyrig]

    all_unc = np.concatenate(list(uncs.values()), 0)
    cutoff = 0.1/3
    ratio_valid = np.mean(all_unc <= cutoff)*100
    logger.info('Translation uncertainty: %.3f%% < %.1fcm', ratio_valid, cutoff*100)

    max_ = 0.1
    bins = np.geomspace(0.001, max_, 30)
    plt.hist(all_unc[all_unc < max_], bins=bins, ec="k", color=(0.8,)*3)
    plt.xscale('log')
    plt.axvline(cutoff, color='r')
    plt.xlabel(f'translation uncertainty [m] - {ratio_valid:.2f}% < {cutoff*100:.1f}cm')
    plt.savefig(output_dir / 'uncertainty_t_hist_all.pdf')
    plt.close()

    plotter = ColMapPlotter(rec, centers=np.concatenate(list(centers.values())))
    plotter.plot_uncertainties(centers, uncs, max_unc_cm=5)
    plt.savefig(output_dir / 'uncertainty_t_keyrigs.pdf')
    plt.close()

    return uncertainty


def run(capture: Capture, ref_id: str, sequence_ids: List[str], conf: RefinementConf):
    workdir = capture.registration_path()
    sequence_ids = sorted(i for q in sequence_ids for i in capture.sessions if i.startswith(q))
    assert len(sequence_ids) > 0
    output_dir = workdir / 'refinement/'
    output_dir.mkdir(parents=True, exist_ok=True)

    # List images for the optimization
    image_root = capture.sessions_path()
    poses_init = {}
    image_names = {}
    for i in sequence_ids:
        session = capture.sessions[i]
        poses_init[i] = Trajectories.load(workdir / i / ref_id / 'trajectory_ba.txt')
        keys = poses_init[i].key_pairs()
        assert keys == session.trajectories.key_pairs()
        prefix = capture.data_path(i).relative_to(image_root)
        image_names[i] = imatch.list_images_for_matching(
            session, prefix, conf.matching,
            keyframing=conf.keyframings[session.device],
            poses=session.trajectories)
    image_names[ref_id] = imatch.list_images_for_matching(
        capture.sessions[ref_id], capture.data_path(ref_id).relative_to(image_root), conf.matching)

    # Check that local features are extracted
    local_features = {}
    for i in image_names:
        local_features[i] = workdir / i / conf.matching.lfeats_file
        assert local_features[i].exists()
    #     extract_features.main(
    #         conf.matching.local_features, image_root,
    #         feature_path=local_features[i], image_list=image_names[i].keys())

    logger.info('Computing image pairs between sequences.')
    all_pairs_path = output_dir / f'pairs_seq_seq_{conf.matching_seq_seq.num_pairs}.pkl'
    if all_pairs_path.exists():
        with open(all_pairs_path, 'rb') as f:
            all_pairs = pickle.load(f)
    else:
        proc = capture.sessions[ref_id].proc
        mesh_path = proc.meshes.get('mesh_simplified', proc.meshes.get('mesh'))
        tracer = OverlapTracer(Renderer(capture.proc_path(ref_id) / mesh_path), num_rays=60)
        all_pairs = compute_pairs(
            capture, sequence_ids, image_names, poses_init, conf.matching_seq_seq, tracer)
        with open(all_pairs_path, 'wb') as fid:
            pickle.dump(all_pairs, fid)
        del tracer

     # Hacky: remove duplicates. Should instead be done in Compute pairs.
    all_pairs = {ij: list({(n0, n1) if n0 <= n1 else (n1, n0) for n0, n1 in ps})
                 for ij, ps in all_pairs.items()}

    logger.info('Starting the sequence-to-sequence matching.')
    pairwise_paths = {}
    for (qid, rid), pairs in tqdm(all_pairs.items()):
        paths = alignment.Paths(workdir, conf.matching_seq_seq, qid, rid)
        pairwise_paths[qid, rid] = paths
        paths.outputs.mkdir(exist_ok=True)
        with open(paths.pairs, 'w') as fid:
            fid.write('\n'.join(' '.join(p) for p in pairs))
    match_multi_sequences(conf.matching_seq_seq.matcher, pairwise_paths, local_features)
    for i in sequence_ids:
        paths = alignment.Paths(workdir, conf.matching_seq_ref, i, ref_id)
        assert paths.matches.exists()
        pairwise_paths[i, ref_id] = paths

    logger.info('Creating an empty COLMAP reconstruction.')
    capture2colmap_camid = {}
    camid2scale = {}
    key2imageid = {}
    reconstruction = pycolmap.Reconstruction()
    for i in tqdm([ref_id]+sequence_ids):
        session = capture.sessions[i]
        capture2colmap_camid[i], key2imageid[i], scales = add_session_to_reconstruction(
            reconstruction, session, image_names[i],
            session.trajectories if i == ref_id else poses_init[i], local_features[i]
        )
        camid2scale.update(scales)
    logger.info('The reconstruction has %d images and %d cameras.',
                len(reconstruction.images), len(reconstruction.cameras))

    logger.info('Importing ref-ref tracks and matches.')
    with open(capture.session_path(ref_id) / 'tracks.pkl', 'rb') as f:
        tracks_ref = pickle.load(f)
    add_tracks_to_reconstruction(
        reconstruction, key2imageid[ref_id],
        tracks_ref['track_p3d'], tracks_ref['root_to_nodes_mapping'])
    paths_ref = alignment.Paths(workdir, conf.matching, ref_id, ref_id)
    pairs_ref = parse_retrieval(paths_ref.pairs)
    matches_ref_ref = epipolar_match_filtering(
        image_names[ref_id], pairs_ref, paths_ref.matches, paths_ref.features(ref_id)[1],
        capture.sessions[ref_id], conf.max_reprojection_error)

    logger.info('Building the correspondence graph.')
    graph = pycolmap.CorrespondenceGraph()
    for image in reconstruction.images.values():
        graph.add_image(image.image_id, len(image.points2D))
    add_to_correspondence_graph(graph, pairs_ref, reconstruction.images, matches_ref_ref)
    for paths in tqdm(pairwise_paths.values()):
        pairs = parse_retrieval(paths.pairs)
        add_to_correspondence_graph(graph, pairs, reconstruction.images, paths.matches)

    logger.info('Triangulating all images.')
    triangulator = Triangulator(graph, reconstruction, conf.max_reprojection_error)
    triangulator.triangulate_all_images()
    logger.info('Statistics after triangulation:\n%s', reconstruction.summary())
    tri_path = output_dir / "colmap-triangulated"
    tri_path.mkdir(parents=True, exist_ok=True)
    reconstruction.write(tri_path)

    poses_opt = {ref_id: qt_dict_from_trajectories(capture.sessions[ref_id].trajectories)}
    for i in poses_init:
        poses_opt[i] = qt_dict_from_trajectories(poses_init[i])

    logger.info('Starting the global refinement.')
    for i in range(conf.num_refinement_iterations):
        logger.info('Iteration #%d of the refinement...', i)

        n_filter = reconstruction.filter_observations_with_negative_depth()
        logger.info('Filtered out %s observations with negative depth.', n_filter)

        refiner = GlobalRefiner(reconstruction, key2imageid, conf)
        for sid in tqdm([ref_id]+sequence_ids):
            session = capture.sessions[sid]
            refiner.add_session_to_problem(
                session, poses_opt[sid], session.trajectories, is_reference=(sid == ref_id))
        refiner.make_cameras_constant()
        logger.info('Solver report:\n%s', refiner.solve().BriefReport())

        refiner.update_reconstruction_poses(poses_opt)

        triangulator.recursive_complete_and_merge_tracks()
        n_filter = triangulator.filter_points()
        logger.info('Filtered out %s observations with high reprojection error.', n_filter)

    logger.info('Statistics after refinement:\n%s', reconstruction.summary())
    output_path = workdir / (SFM_OUTPUT + '-normed')
    output_path.mkdir(parents=True, exist_ok=True)
    reconstruction.write(output_path)

    logger.info('Computing keypoint noise and pose covariances.')
    keypoint_noise = compute_keypoint_noise(reconstruction)
    logger.info('Observed keypoint noise: %.3f', keypoint_noise)
    covariances = compute_covariances_per_sequence(
        capture, reconstruction, sequence_ids, poses_opt, key2imageid, conf, keypoint_noise)

    keyrigs = {i: recover_keyrig_keys(poses_opt[i], key2imageid[i].keys(), capture.sessions[i])
               for i in sequence_ids}
    log_pose_uncertainties(output_dir, covariances, poses_opt, reconstruction, keyrigs)

    scale_reconstruction(reconstruction, camid2scale)
    reconstruction.extract_colors_for_all_images(str(image_root))

    output_path = workdir / SFM_OUTPUT
    output_path.mkdir(parents=True, exist_ok=True)
    reconstruction.write(output_path)

    poses_out = {}
    for sid, poses in poses_opt.items():
        T_cam2w = Trajectories()
        for key, qt in poses.items():
            T_cam2w[key] = Pose(*qt).inverse()
            if sid in covariances:
                T_cam2w[key] = Pose(*T_cam2w[key].qt, covariances[sid][key])
        T_cam2w.save(workdir / sid / 'trajectory_refined.txt')
        poses_out[sid] = T_cam2w

        session = capture.sessions[sid]
        if session.proc is None:
            session.proc = Proc()
        session.proc.alignment_trajectories = T_cam2w
        session.proc.save(capture.proc_path(sid))

    mlp_path = output_dir / 'alignment_refined.mlp'
    mlp = MeshlabProject()
    mesh_path = capture.proc_path(ref_id) / capture.sessions[ref_id].proc.meshes['mesh_simplified']
    mlp.add_mesh(f'{ref_id}/mesh', os.path.relpath(mesh_path, mlp_path.parent))
    plot_sequence_trajectories(mlp, capture, poses_out)
    sparse_ply_path = mlp_path.parent / 'sparse.ply'
    colmap_reconstruction_to_ply(sparse_ply_path, reconstruction)
    mlp.add_mesh('sparse', os.path.relpath(sparse_ply_path, mlp_path.parent))
    mlp.write(mlp_path)
