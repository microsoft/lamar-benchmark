import argparse
from pathlib import Path
from typing import List

from scantools import logger, run_combine_navvis_sessions
from scantools.capture import Capture
from scantools import (
    run_navvis_to_capture, run_meshing, run_rendering, to_meshlab_visualization,
    run_scan_aligner, run_pose_graph_optimizer)


conf_matcher = {'output': 'matches-superglue',
                'model': {'name': 'superglue', 'weights': 'outdoor', 'sinkhorn_iterations': 5}}

align_conf = run_scan_aligner.Conf.from_dict(dict(
    matching=dict(
        global_features='netvlad',
        local_features='superpoint_aachen',
        matcher=conf_matcher),
    keyframing=dict(num=500),
))
align_conf.matching.global_features['preprocessing']['resize_max'] = 1024


def main(capture_path: Path, session_ids: List[str], navvis_dir: Path):
    if capture_path.exists():
        capture = Capture.load(capture_path)
    else:
        capture = Capture(sessions={}, path=capture_path)

    tiles_format = 'center'
    mesh_id = 'mesh'
    downsample_max_edge = 1920
    meshing_method = 'advancing_front'

    for session in session_ids:
        if session not in capture.sessions:
            logger.info('Exporting NavVis session %s.', session)
            run_navvis_to_capture.run(
                navvis_dir / session, capture, tiles_format, session,
                downsample_max_edge=downsample_max_edge)

        if (not capture.sessions[session].proc
                or mesh_id not in capture.sessions[session].proc.meshes):
            logger.info('Meshing session %s.', session)
            run_meshing.run(capture, session, 'point_cloud_final', mesh_id, method=meshing_method)

        if not capture.sessions[session].depths:
            logger.info('Rendering session %s.', session)
            run_rendering.run(capture, session, mesh_id=mesh_id+'_simplified')

        to_meshlab_visualization.run(
            capture, session, f'trajectory_{session}', export_mesh=True, export_poses=True,
            mesh_id=mesh_id)

    for i, ref_id in enumerate(session_ids):
        for query_id in session_ids[i+1:]:
            if ('icp', ref_id) in capture.sessions[query_id].proc.alignment_global:
                logger.info('Skipping scan pair (%s, %s).', query_id, ref_id)
                continue
            logger.info('Aligning %s to %s.', query_id, ref_id)
            run_scan_aligner.run(capture, ref_id, query_id, align_conf)

    run_pose_graph_optimizer.run(capture, session_ids)

    if len(session_ids) > 1:
        logger.info('Merging sessions: %s.', ', '.join(session_ids))
        session_id = run_combine_navvis_sessions.run(
            capture, session_ids, export_combined_pointcloud=True,
            export_depths=True, export_meshes=True)
        logger.info('Meshing combined session %s.', session_id)
        run_meshing.run(
            capture, session_id, 'point_cloud_combined', 'mesh', method=meshing_method)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture_path', type=Path, required=True)
    parser.add_argument('--input_path', type=Path, required=True)
    parser.add_argument('--sessions', nargs='+', type=str, required=True)
    args = parser.parse_args()

    main(args.capture_path, args.sessions, args.input_path)
