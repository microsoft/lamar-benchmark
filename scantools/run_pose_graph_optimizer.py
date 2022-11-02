import argparse
from pathlib import Path
from typing import List
import os
import numpy as np
import open3d as o3d

from . import logger
from .capture import Capture, Pose
from .viz.meshlab import MeshlabProject

o3dreg = o3d.pipelines.registration

def run(capture: Capture, session_ids: List[str], ref_index: int = 0,
        pairwise_label: str = 'icp', absolute_label='pose_graph_optimized'):

    assert 0 <= ref_index < len(session_ids)
    session2index = dict(zip(session_ids, range(len(session_ids))))

    pose_graph = o3dreg.PoseGraph()
    for session in session_ids:
        alignments = capture.sessions[session].proc.alignment_global
        for (label, target), pose_info in alignments.items():
            if label != pairwise_label or target not in session2index:
                continue
            pose, info = pose_info
            info = np.array(info, float).reshape(6, 6)
            edge = o3dreg.PoseGraphEdge(
                session2index[session], session2index[target], pose.to_4x4mat(), info,
                uncertain=False)
            pose_graph.edges.append(edge)

        # Initialize with the relative pose w.r.t the fixed reference session.
        pose_info_init = alignments.get((pairwise_label, session_ids[ref_index]))
        if pose_info_init is None:
            T = np.eye(4)
        else:
            T = pose_info_init[0].to_4x4mat()
        pose_graph.nodes.append(o3dreg.PoseGraphNode(T))

    num_edges = [0 for _ in session_ids]
    for edge in pose_graph.edges:
        num_edges[edge.source_node_id] += 1
        num_edges[edge.target_node_id] += 1
    for session, num in zip(session_ids, num_edges):
        if num == 0:
            logger.warning(f'Session {session} does not have any pairwise constraint.')

    option = o3dreg.GlobalOptimizationOption(
        max_correspondence_distance=5e-2,  # for line process weight update
        edge_prune_threshold=0.25,
        reference_node=ref_index)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        o3dreg.global_optimization(
            pose_graph,
            o3dreg.GlobalOptimizationLevenbergMarquardt(),
            o3dreg.GlobalOptimizationConvergenceCriteria(),
            option)

    o3d.io.write_pose_graph(str(capture.registration_path() / 'pose_graph.json'), pose_graph)

    logger.info('Writing absolute scan transforms.')
    mlp = MeshlabProject()
    mlp_path = capture.viz_path() / 'all_align.mlp'
    for session, node in zip(session_ids, pose_graph.nodes):
        T = node.pose.copy()
        proc = capture.sessions[session].proc
        proc.alignment_global[absolute_label, proc.alignment_global.no_ref] = (
            Pose(r=T[:3, :3], t=T[:3, 3]), [])
        proc.save(capture.proc_path(session))

        mesh_path = capture.proc_path(session) / capture.sessions[session].proc.meshes['mesh']
        mlp.add_mesh(
            f'{session}/mesh', os.path.relpath(mesh_path, mlp_path.parent), T=T)
    mlp.write(mlp_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--capture_path', type=Path, required=True)
    parser.add_argument('--session_ids', nargs='+', type=str, required=True)
    args = parser.parse_args().__dict__
    args['capture'] = Capture.load(args.pop('capture_path'))

    run(**args)
