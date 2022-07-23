import argparse
from pathlib import Path
from typing import Optional, Dict

from scantools.capture import Capture

from .tasks import (
    FeatureExtraction, PairSelection, FeatureMatching, Mapping, PoseEstimation, ChunkAlignment)
from .tasks.chunk_alignment import keys_from_chunks
from .utils.capture import read_query_list, build_chunks, avoid_duplicate_keys_in_chunks


def run(outputs: Path,
        capture: Capture,
        ref_id: str,
        query_id: str,
        retrieval: str,
        feature: str,
        matcher: str,
        use_radios: bool = False,
        sequence_length_seconds: Optional[int] = None,
        num_pairs_loc: int = 10,
        num_pairs_map: int = 10,
        retrieval_mapping: Optional[str] = None,
        filter_pairs_mapping: Optional[Dict] = None,
        query_filename: str = 'queries.txt'):

    is_rig = 'hololens' in query_id
    is_sequential = sequence_length_seconds is not None
    if filter_pairs_mapping is None:
        filter_pairs_mapping = {
            'filter_frustum': {'do': True},
            'filter_pose': {'do': True, 'num_pairs_filter': 250},
        }

    configs = {
        'extraction': FeatureExtraction.methods[feature],
        'pairs_map': {
            'method': PairSelection.methods[retrieval_mapping or retrieval],
            'num_pairs': num_pairs_map,
            **filter_pairs_mapping,
        },
        'matching': FeatureMatching.methods[matcher],
        'mapping': Mapping.methods['triangulation'],
        'pairs_loc': {
            'method': PairSelection.methods[retrieval],
            'num_pairs': num_pairs_loc,
        },
        'poses': PoseEstimation.methods['rig' if is_rig else 'single_image'],
        # for multi-frame localization
        'extra_pairs_reloc': {
            'filter_frustum': {'do': True},
            'filter_pose': {'do': True, 'num_pairs_filter': 100}
        },
        'chunks': ChunkAlignment.methods['rig' if is_rig else 'single_image'],
    }
    if use_radios:
        configs['pairs_loc']['filter_radio'] = {'do': True, 'num_pairs_filter': 1000}
    if retrieval == 'overlap':  # add pose filtering to speed up the overlap
        configs['pairs_loc'].update({
            'filter_frustum': {'do': True},
            'filter_pose': {'do': True, 'num_pairs_filter': 100},
        })

    query_list_path = capture.session_path(query_id) / query_filename
    query_list = image_keys = read_query_list(query_list_path)
    if is_sequential:
        query_list, query_chunks = build_chunks(
            capture, query_id, query_list, sequence_length_seconds)
        image_keys = keys_from_chunks(query_chunks)

    extraction_map = FeatureExtraction(outputs, capture, ref_id, configs['extraction'])
    pairs_map = PairSelection(outputs, capture, ref_id, ref_id, configs['pairs_map'])
    matching_map = FeatureMatching(
        outputs, capture, ref_id, ref_id, configs['matching'], pairs_map, extraction_map)

    mapping = Mapping(
        configs['mapping'], outputs, capture, ref_id, extraction_map, matching_map)

    extraction_query = FeatureExtraction(
        outputs, capture, query_id, configs['extraction'], image_keys)

    if is_sequential:
        query_list, query_chunks = avoid_duplicate_keys_in_chunks(
            capture.sessions[query_id], query_list, query_chunks)
        chunk_alignment = ChunkAlignment(
            configs, outputs, capture, query_id, extraction_query, mapping, query_chunks,
            sequence_length_seconds)
        results = chunk_alignment.evaluate(
            capture.sessions[query_id].proc.alignment_trajectories, query_list)
    else:
        pairs_loc = PairSelection(
            outputs, capture, query_id, ref_id, configs['pairs_loc'], query_list)
        matching_query = FeatureMatching(
            outputs, capture, query_id, ref_id, configs['matching'],
            pairs_loc, extraction_query, extraction_map)
        pose_estimation = PoseEstimation(
            configs['poses'], outputs, capture, query_id,
            extraction_query, matching_query, mapping, query_list)
        results = pose_estimation.evaluate(
            capture.sessions[query_id].proc.alignment_trajectories)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--scene', type=str, required=True)
    parser.add_argument('--ref_id', type=str, required=True)
    parser.add_argument('--query_id', type=str, required=True)
    parser.add_argument(
        '--captures', type=Path, default=Path("./data/"), help="Path to captures directory")
    parser.add_argument(
        '--outputs', type=Path, default=Path("./outputs/"), help="Path to outputs directory")
    parser.add_argument(
        '--retrieval', type=str, required=True, choices=list(PairSelection.methods))
    parser.add_argument(
        '--feature', type=str, required=True, choices=list(FeatureExtraction.methods))
    parser.add_argument(
        '--matcher', type=str, required=True, choices=list(FeatureMatching.methods))
    parser.add_argument('--use_radios', action='store_true')
    parser.add_argument('--sequence_length_seconds', type=int)
    args = parser.parse_args().__dict__
    scene = args.pop("scene")
    args['capture'] = Capture.load(args.pop('captures') / scene)
    args['outputs'] = args['outputs'] / scene
    run(**args)
