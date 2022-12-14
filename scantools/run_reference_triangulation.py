import argparse
import collections
from pathlib import Path
from queue import Queue
import pickle
import numpy as np
from tqdm import tqdm
import pyceres
import pycolmap

from . import logger
from .capture import Capture
from .proc.alignment import Paths, image_matching as imatch
from .proc.alignment.localization import Batch2d3dMatcher, LocalizerConf
from .proc.alignment.scan import lift_points_to_3d
from .utils.geometry import compute_epipolar_error


default_matching_conf = imatch.MatchingConf.from_dict(
    dict(
        global_features='netvlad',
        local_features='superpoint_aachen',
        matcher='superglue'))
default_matching_conf.global_features['preprocessing']['resize_max'] = 1024


def track_element_to_tuple(obs, colmap_image_id_to_imkey_dict):
    return (colmap_image_id_to_imkey_dict[obs.image_id], obs.point2D_idx)


def initialize_colmap_reconstruction(session, keypoints, keys):
    images = session.images
    sensors = session.sensors
    trajectory = session.trajectories

    # COLMAP reconstruction.
    reconstruction = pycolmap.Reconstruction()

    # Cameras.
    sensor_id_to_colmap_camera_dict = {}
    num_cameras = 0
    for sensor_id in sensors:
        if sensors[sensor_id].sensor_type == 'camera':
            camera = pycolmap.Camera(sensors[sensor_id].asdict)
            camera.camera_id = num_cameras
            num_cameras += 1
            reconstruction.add_camera(camera)
            sensor_id_to_colmap_camera_dict[sensor_id] = camera

    # Images.
    # TODO: decide if we want to skip top camera.
    imkey_to_colmap_image_dict = {}
    num_images = 0
    for imkey in keys:
        _, sensor_id = imkey
        pose = trajectory[imkey].inverse()  # COLMAP uses world-to-camera
        camera = sensor_id_to_colmap_camera_dict[sensor_id]
        kps = keypoints[imkey][0]
        image = pycolmap.Image(
            images[imkey], kps, pose.t, pose.qvec, camera.camera_id, num_images
        )
        image.registered = True
        num_images += 1
        reconstruction.add_image(image)
        imkey_to_colmap_image_dict[imkey] = image

    # Tracks (single observation).
    # tracks = {}
    # for imkey in keypoints:
    #     image = imkey_to_colmap_image_dict[imkey]
    #     image_id = image.image_id
    #     _, p3ds, valid = keypoints[imkey]
    #     for kp_idx, (p3d, is_valid) in enumerate(zip(p3ds, valid)):
    #         if is_valid:
    #             p3d_id = reconstruction.add_point3D(
    #                 p3d, pycolmap.Track([pycolmap.TrackElement(image_id, kp_idx)]),
    #                 np.array([0, 0, 0]))
    #             tracks[imkey, kp_idx] = p3d_id
    #             image.set_point3D_for_point2D(kp_idx, p3d_id)

    return reconstruction, sensor_id_to_colmap_camera_dict, imkey_to_colmap_image_dict


def build_correspondence_graph(pairs, matcher, imkey_to_colmap_image_dict, all_matches=None):
    paths = matcher.paths

    corr_graph = pycolmap.CorrespondenceGraph()
    processed_pairs = set()
    for ref in pairs:
        key_r = matcher.impath2key_r[ref]
        image_r = imkey_to_colmap_image_dict[key_r]

        if not corr_graph.exists_image(image_r.image_id):
            corr_graph.add_image(image_r.image_id, len(image_r.points2D))

        for tar in pairs[ref]:
            key_t = matcher.impath2key_r[tar]
            image_t = imkey_to_colmap_image_dict[key_t]

            if not corr_graph.exists_image(image_t.image_id):
                corr_graph.add_image(image_t.image_id, len(image_t.points2D))

            if all_matches is None:
                matches = imatch.get_matches(paths.matches, [[ref, tar]])[0]
            else:
                matches = all_matches[key_r][key_t]
            if matches.shape[0] == 0:
                continue

            # Check if not already processed
            if (key_t, key_r) in processed_pairs:
                continue
            processed_pairs.add((key_r, key_t))

            corr_graph.add_correspondences(
                image_r.image_id, image_t.image_id, matches.astype(np.uint32))

    return corr_graph


def epipolar_match_filtering(pairs, matcher, session, keypoints,
                             max_epipolar_error, all_matches=None):
    paths = matcher.paths
    sensors = session.sensors
    trajectory = session.trajectories

    filtered_matches = {}
    inlier_ratios = []
    for ref in tqdm(pairs):
        key_r = matcher.impath2key_r[ref]
        kp_r, _, _ = keypoints[key_r]
        T_q2w_r = trajectory[key_r]
        filtered_matches[key_r] = {}

        for tar in pairs[ref]:
            key_t = matcher.impath2key_r[tar]
            kp_t, _, _ = keypoints[key_t]
            T_q2w_t = trajectory[key_t]

            if all_matches is None:
                matches = imatch.get_matches(paths.matches, [[ref, tar]])[0]
            else:
                matches = all_matches[key_r][key_t]
            if matches.shape[0] == 0:
                filtered_matches[key_r][key_t] = np.zeros([0, 2])
                continue

            errors = compute_epipolar_error(T_q2w_r, T_q2w_t,
                                            sensors[key_r[1]], sensors[key_t[1]],
                                            kp_r[matches[:, 0]], kp_t[matches[:, 1]])
            valid_matches = (errors <= max_epipolar_error)

            filtered_matches[key_r][key_t] = matches[valid_matches]
            inlier_ratios.append(np.mean(valid_matches))
    logger.info('mean/med/min/max valid matches %.2f/%.2f/%.2f/%.2f%%.',
                np.mean(inlier_ratios) * 100, np.median(inlier_ratios) * 100,
                np.min(inlier_ratios) * 100, np.max(inlier_ratios) * 100)

    return filtered_matches


def mesh_match_filtering(pairs, matcher, session, keypoints,
                         max_reproj_error, max_distance=np.inf,
                         all_matches=None):
    paths = matcher.paths
    trajectory = session.trajectories
    sensors = session.sensors

    filtered_matches = {}
    inlier_ratios = []
    for ref in tqdm(pairs):
        key_r = matcher.impath2key_r[ref]
        kp_r, p3d_r, valid_r = keypoints[key_r]
        T_q2w_r = trajectory[key_r]
        filtered_matches[key_r] = {}

        for tar in pairs[ref]:
            key_t = matcher.impath2key_r[tar]
            kp_t, p3d_t, valid_t = keypoints[key_t]
            T_q2w_t = trajectory[key_t]

            if all_matches is None:
                matches = imatch.get_matches(paths.matches, [[ref, tar]])[0]
            else:
                matches = all_matches[key_r][key_t]
            if matches.shape[0] == 0:
                filtered_matches[key_r][key_t] = np.zeros([0, 2])
                continue

            valid_matches = valid_r[matches[:, 0]] & valid_t[matches[:, 1]]
            # 3D check.
            valid_matches &= (
                np.linalg.norm(p3d_r[matches[:, 0]] - p3d_t[matches[:, 1]], axis=1) < max_distance)
            # 2D reference -> target check.
            p3d_r2t = T_q2w_t.inverse().transform_points(p3d_r[matches[:, 0]])
            p2d_r2t = sensors[key_t[1]].world2image(p3d_r2t[:, : -1] / p3d_r2t[:, -1 :])
            valid_matches &= (p3d_r2t[:, -1] >= 0)
            valid_matches &= (
                np.linalg.norm(p2d_r2t - kp_t[matches[:, 1]], axis=1) <= max_reproj_error)
            # 2D target -> reference check.
            p3d_t2r = T_q2w_r.inverse().transform_points(p3d_t[matches[:, 1]])
            p2d_t2r = sensors[key_r[1]].world2image(p3d_t2r[:, : -1] / p3d_t2r[:, -1 :])
            valid_matches &= (p3d_t2r[:, -1] >= 0)
            valid_matches &= (
                np.linalg.norm(p2d_t2r - kp_r[matches[:, 0]], axis=1) <= max_reproj_error)

            filtered_matches[key_r][key_t] = matches[valid_matches]
            inlier_ratios.append(np.mean(valid_matches))
    logger.info('mean/med/min/max valid matches %.2f/%.2f/%.2f/%.2f%%.',
                np.mean(inlier_ratios) * 100, np.median(inlier_ratios) * 100,
                np.min(inlier_ratios) * 100, np.max(inlier_ratios) * 100)

    return filtered_matches


def complete_and_merge_tracks(triangulator, options):
    num_completed_obs = triangulator.complete_all_tracks(options)
    num_merged_obs = triangulator.merge_all_tracks(options)
    return num_completed_obs + num_merged_obs


def recursive_complete_and_merge_tracks(triangulator, options):
    while complete_and_merge_tracks(triangulator, options):
        pass


def triangulate_points(corr_graph, reconstruction, max_reproj_error,
                       max_angle_error=2.0, min_tri_angle=1.5, completion_freq=100):
    triangulator = pycolmap.IncrementalTriangulator(corr_graph, reconstruction)
    options = pycolmap.IncrementalTriangulatorOptions()
    options.create_max_angle_error = max_angle_error
    options.continue_max_angle_error = max_angle_error
    options.merge_max_reproj_error = max_reproj_error
    options.complete_max_reproj_error = max_reproj_error
    # Triangulate images in decreasing order of num correspondences.
    images = [
        (image.image_id, corr_graph.num_correspondences_for_image(image.image_id))
        for image in reconstruction.images.values()]
    images = sorted(images, key=lambda x: -x[1])
    for idx, (image_id, _) in enumerate(tqdm(images)):
        triangulator.triangulate_image(options, image_id)
        if (idx + 1) % completion_freq == 0:
            complete_and_merge_tracks(triangulator, options)
    logger.info(reconstruction.summary())
    # Retriangulation.
    recursive_complete_and_merge_tracks(triangulator, options)
    # First BA.
    logger.info('First robust BA...')
    bundle_adjustment(
        reconstruction, max_num_iterations=50, max_num_linear_solver_iterations=100,
        max_reproj_error=max_reproj_error)
    recursive_complete_and_merge_tracks(triangulator, options)
    logger.info('Filtered out %d observations with high reprojection error.',
                reconstruction.filter_all_points3D(max_reproj_error, min_tri_angle))
    # Second BA.
    logger.info('Second non-robust BA...')
    bundle_adjustment(
        reconstruction, max_num_iterations=50, max_num_linear_solver_iterations=100)
    recursive_complete_and_merge_tracks(triangulator, options)
    logger.info('Filtered out %d observations with high reprojection error.',
                reconstruction.filter_all_points3D(max_reproj_error, min_tri_angle))
    # Dump stats.
    logger.info(reconstruction.summary())
    return triangulator, options


def bundle_adjustment(rec, max_num_iterations=100, max_num_linear_solver_iterations=200, max_reproj_error=None):
    logger.info('Filtered out %d observations with negative depth.',
                rec.filter_observations_with_negative_depth())
    prob = pyceres.Problem()
    if max_reproj_error:
        loss = pyceres.LossFunction({"name": "soft_l1", "params": [max_reproj_error]})
    else:
        loss = pyceres.TrivialLoss()
    for im in rec.images.values():
        cam = rec.cameras[im.camera_id]
        for p in im.points2D:
            if p.has_point3D():
                cost = pyceres.factors.BundleAdjustmentCost(cam.model_id, p.xy, im.qvec, im.tvec)
                prob.add_residual_block(cost, loss, [rec.points3D[p.point3D_id].xyz, cam.params])
    for cam in rec.cameras.values():
        if prob.has_parameter_block(cam.params):
            prob.set_parameter_block_constant(cam.params)
    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.ITERATIVE_SCHUR
    options.preconditioner_type = pyceres.PreconditionerType.SCHUR_JACOBI
    options.max_num_iterations = max_num_iterations
    options.max_linear_solver_iterations = max_num_linear_solver_iterations
    options.minimizer_progress_to_stdout = False
    options.function_tolerance = 0
    options.gradient_tolerance = 0
    options.parameter_tolerance = 0
    options.num_threads = -1
    summary = pyceres.SolverSummary()
    pyceres.solve(options, prob, summary)
    print(summary.BriefReport())


def run(capture: Capture, ref_id: str, matching_conf: imatch.MatchingConf = default_matching_conf,
        overwrite: bool = False, max_epipolar_error: float = 8, max_reproj_error: float = 8,
        max_angle_error: float = 2.0, min_tri_angle: float = 1.5, visualize: bool = True):
    # Save the paths to the working directories
    paths = Paths(capture.registration_path(), matching_conf, ref_id, ref_id)
    paths.outputs.mkdir(exist_ok=True, parents=True)

    session = capture.sessions[ref_id]

    logger.info('2D-2D matching.')
    matcher = Batch2d3dMatcher(capture, matching_conf, LocalizerConf(), paths, False)
    if paths.matches.exists() and not overwrite:
        pairs = imatch.parse_retrieval(paths.pairs)
    else:
        pairs = matcher.do_matching_2d2d(
            from_overlap=True, T_q2w=session.trajectories)

    logger.info('Lifting all keypoints to 3D using rendered depth-maps.')
    keypoints = {}
    for ref in tqdm(matcher.impath2key_r):
        key = matcher.impath2key_r[ref]
        kps, _ = imatch.get_keypoints(paths.features(ref_id)[1], [ref])
        kps = kps[0]
        p3ds, valid = lift_points_to_3d(kps, key,
                                        session,
                                        capture.data_path(ref_id),
                                        align=True)
        keypoints[key] = [kps, p3ds, valid]

    logger.info('Filtering matches based on epipolar geometry.')
    epi_filtered_matches = epipolar_match_filtering(
        pairs, matcher, session, keypoints,
        max_epipolar_error)

    logger.info('Filtering remaining matches based on mesh.')
    filtered_matches = mesh_match_filtering(
        pairs, matcher, session, keypoints,
        max_reproj_error, all_matches=epi_filtered_matches)

    logger.info('Creating the COLMAP reconstruction.')
    reconstruction, _, imkey_to_colmap_image_dict = (
        initialize_colmap_reconstruction(session, keypoints, matcher.impath2key_r.values()))

    # High-confidence track triangulation.
    logger.info('Building correspondence graph for mesh-filtered matches.')
    corr_graph = build_correspondence_graph(
        pairs, matcher, imkey_to_colmap_image_dict,
        all_matches=filtered_matches)

    logger.info('Triangulating high-confidence tracks.')
    triangulate_points(
        corr_graph, reconstruction, max_reproj_error, max_angle_error, min_tri_angle)
    if visualize:
        output_path = capture.session_path(ref_id) / 'high-confidence-sparse'
        output_path.mkdir(exist_ok=True)
        reconstruction.write(output_path)

    # Additional track triangulation.
    logger.info('Building correspondence graph for epipolar-filtered matches.')
    corr_graph = build_correspondence_graph(
        pairs, matcher, imkey_to_colmap_image_dict,
        all_matches=epi_filtered_matches)

    logger.info('Triangulating additional tracks.')
    triangulator, options = triangulate_points(
        corr_graph, reconstruction, max_reproj_error, max_angle_error, min_tri_angle)
    if visualize:
        output_path = capture.session_path(ref_id) / 'additional-sparse'
        output_path.mkdir(exist_ok=True)
        reconstruction.write(output_path)

    # Bundle structure.
    logger.info('Running final bundle adjustment.')
    bundle_adjustment(reconstruction)
    recursive_complete_and_merge_tracks(triangulator, options)
    logger.info('Filtered out %d observations with high reprojection error.',
                reconstruction.filter_all_points3D(max_reproj_error, min_tri_angle))
    logger.info(reconstruction.summary())
    if visualize:
        output_path = capture.session_path(ref_id) / 'final-sparse'
        output_path.mkdir(exist_ok=True)
        reconstruction.write(output_path)

    # Track extraction from COLMAP reconstruction.
    logger.info('Extracting tracks from reconstruction.')
    colmap_image_id_to_imkey_dict = {
        image.image_id : imkey for imkey, image in imkey_to_colmap_image_dict.items()}
    track_p3d = {}
    root_to_nodes_mapping = {}
    node_to_root_mapping = {}
    for point3d_idx in reconstruction.points3D:
        point3d = reconstruction.points3D[point3d_idx]
        xyz = point3d.xyz
        observations = point3d.track.elements
        root = track_element_to_tuple(observations[0], colmap_image_id_to_imkey_dict)
        track_p3d[root] = xyz
        root_to_nodes_mapping[root] = []
        for obs in observations:
            node = track_element_to_tuple(obs, colmap_image_id_to_imkey_dict)
            node_to_root_mapping[node] = root
            root_to_nodes_mapping[root].append(node)

    # Statistics with respect to mesh points.
    errors = []
    for point3d_idx in reconstruction.points3D:
        point3d = reconstruction.points3D[point3d_idx]
        xyz = point3d.xyz
        observations = point3d.track.elements
        p3ds = []
        for obs in observations:
            node = track_element_to_tuple(obs, colmap_image_id_to_imkey_dict)
            p3d = keypoints[node[0]][1][node[1]]
            valid = keypoints[node[0]][2][node[1]]
            if valid:
                p3ds.append(p3d)
        if len(p3ds) > 0:
            errors.append(np.linalg.norm(np.mean(p3ds, axis=0) - xyz))
    errors = np.array(errors) * 100
    logger.info('Error w.r.t. mesh avg.: mean/med/q1/q7/q9 %.2f/%.2f/%.2f/%.2f/%.2f cm.',
                np.mean(errors), np.median(errors),
                np.percentile(errors, 10),
                np.percentile(errors, 70),
                np.percentile(errors, 90))

    # Add single-observation tracks.
    logger.info('Adding single-observation tracks.')
    for imkey in keypoints:
        _, p3ds, valid = keypoints[imkey]
        for kp_idx, (p3d, is_valid) in enumerate(zip(p3ds, valid)):
            node = (imkey, kp_idx)
            if is_valid and node not in node_to_root_mapping:
                # If the point can be lifted using mesh and is not triangulated yet.
                track_p3d[node] = p3d
                root_to_nodes_mapping[node] = [node]
                node_to_root_mapping[node] = node

    with open(capture.session_path(ref_id) / 'tracks.pkl', 'wb') as fid:
        pickle.dump(
            {'track_p3d': track_p3d,
             'root_to_nodes_mapping': root_to_nodes_mapping,
             'node_to_root_mapping': node_to_root_mapping}, fid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     argument_default=argparse.SUPPRESS)
    parser.add_argument('--capture_path', type=Path, required=True)
    parser.add_argument('--ref_id', type=str, required=True)
    args = parser.parse_args().__dict__
    args['capture'] = Capture.load(args.pop('capture_path'))

    run(**args)
