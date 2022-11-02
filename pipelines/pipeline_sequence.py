from typing import List, Optional
from pathlib import Path
import argparse
import shutil

from scantools import logger
from scantools.capture import Capture, Session
from scantools import (
    run_phone_to_capture,
    run_sequence_aligner,
    run_joint_refinement,
    run_combine_sequences,
    run_map_query_split,
    run_radio_transfer,
)
from scantools.run_joint_refinement import MatchingConf, RefinementConf


conf_matcher = {'output': 'matches-superglue',
                'model': {'name': 'superglue', 'weights': 'outdoor', 'sinkhorn_iterations': 5}}
conf_matching = MatchingConf('netvlad', 'superpoint_aachen', conf_matcher)

conf_align = {
    'ios': run_sequence_aligner.Conf.from_dict(dict(
        **run_sequence_aligner.conf_ios, matching=conf_matching.to_dict())),
    'hl': run_sequence_aligner.Conf.from_dict(dict(
        **run_sequence_aligner.conf_hololens, matching=conf_matching.to_dict())),
}
conf_align['ios'].matching.local_features['model']['max_keypoints'] = 2048
conf_align['hl'].matching.local_features['model']['max_keypoints'] = 1024

conf_refine = RefinementConf(
    conf_matching,
    keyframings={
        Session.Device.PHONE: conf_align['ios'].localizer.keyframing,
        Session.Device.HOLOLENS: conf_align['hl'].localizer.keyframing,
    },
)

eval_keyframing = run_combine_sequences.KeyFramingConf()
map_keyframing = run_combine_sequences.KeyFramingConf(max_distance=0.5, max_elapsed=0.4)


def read_sequence_list(path) -> List[str]:
    sequences = []
    with open(path, 'r') as fid:
        for line in fid.read().split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            sequences.append(line.split('#')[0].strip())
    return sequences

sequence_list_dir = Path(__file__).parent / 'sequences'


def process_sequence(capture, ref_id, input_path, conf, kind):
    sequence_id = f'{kind}_{input_path.name}'
    logger.info('Working on %s.', sequence_id)

    chunk_ids = sorted(filter(lambda i: i.startswith(sequence_id), capture.sessions))
    if len(chunk_ids) == 0:
        if kind == 'ios':
            chunk_ids = run_phone_to_capture.run(input_path, capture, sequence_id)
            # run_image_anonymization.run(capture, sequence_id, use_gpu=True, visualize=True)
        elif kind.startswith('hl'):
            shutil.copytree(input_path, capture.session_path(sequence_id))
            capture.sessions[sequence_id] = Session.load(capture.sessions_path() / sequence_id)
            chunk_ids = [sequence_id]
        else:
            raise ValueError(kind)

    logger.info('Found %d chunks for sequence %s', len(chunk_ids), sequence_id)
    chunk_ids_aligned = []
    num_failed = 0
    for session_id in chunk_ids:
        path_trajectory = capture.registration_path() / session_id / ref_id / 'trajectory_ba.txt'
        if not path_trajectory.exists():
            logger.info('Aligning session %s.', session_id)
            success = run_sequence_aligner.run(
                capture, ref_id, session_id, conf,
                overwrite=False,
                visualize_diff=False,
                vis_mesh_id='mesh_simplified')
            if not success:
                num_failed += 1
                continue
        chunk_ids_aligned.append(session_id)
    if num_failed > 0:
        logger.warning('Could not align %d/%d chunks for session %s.',
                       num_failed, len(chunk_ids), sequence_id)

    return chunk_ids_aligned


def run(capture_path: Path,
        ref_id: str,
        phone_dir: Optional[Path] = None,
        hololens_dir: Optional[Path] = None,
        phone_sequences: List[str] = ('*',),
        hololens_sequences: List[str] = ('*',),
        ):

    capture = Capture.load(capture_path, wireless=False)

    select_path = capture.path / 'sequences_select.txt'
    if select_path.exists():
        sequence_ids = read_sequence_list(select_path)
        for i in sequence_ids:
            if i not in capture.sessions:
                raise ValueError(i, list(capture.sessions.keys()))
        logger.info('Read %d sequences from %s', len(sequence_ids), select_path)
    else:
        sequence_ids = []
        if phone_dir is not None:
            phone_paths = [p for g in phone_sequences for p in phone_dir.glob(g)]
            for path in phone_paths:
                sequence_ids += process_sequence(capture, ref_id, path, conf_align['ios'], 'ios')
        if hololens_dir is not None:
            hololens_paths = [hololens_dir / g for g in hololens_sequences]
            for path in hololens_paths:
                sequence_ids += process_sequence(capture, ref_id, path, conf_align['hl'], 'hl')
        assert len(sequence_ids) > 0
        logger.info('Found %d sequences', len(sequence_ids))
        with open(capture.path / 'sequences.txt', 'w') as fid:
            fid.write("\n".join(sequence_ids))

    if not all((capture.registration_path()/i/'trajectory_refined.txt').exists()
               for i in sequence_ids):
        logger.info('Running the joint refinement.')
        run_joint_refinement.run(capture, ref_id, sequence_ids, conf_refine)

    logger.info('Splitting sequences into maps and queries.')
    map_ids, query_ids = run_map_query_split.run(capture, sequence_ids, ref_id=ref_id)
    query_ids_phone = list(filter(lambda i: i.startswith('ios'), query_ids))
    query_ids_hololens = list(filter(lambda i: i.startswith('hl'), query_ids))

    map_id = 'map'
    query_id_phone = 'query_phone'
    query_id_hololens = 'query_hololens'
    logger.info('Writing map and query sessions')
    run_combine_sequences.run(
        capture, map_ids, map_id, overwrite_poses=True,
        keyframing=map_keyframing, reference_id=ref_id)
    for i, ids in [[query_id_phone, query_ids_phone], [query_id_hololens, query_ids_hololens]]:
        run_combine_sequences.run(
            capture, ids, i, overwrite_poses=False, keyframing=eval_keyframing)

    run_radio_transfer.run(capture, [map_id, query_id_phone, query_id_hololens])


def get_data_CAB():
    ref_id = '2022-06-21_09.28.22+2022-06-25_11.14.36'
    phone_sequences = read_sequence_list(sequence_list_dir / 'CAB_phone.txt')
    hololens_sequences = read_sequence_list(sequence_list_dir / 'CAB_hololens.txt')
    return ref_id, phone_sequences, hololens_sequences

def get_data_HGE():
    ref_id = '2022-02-06_12.55.11+2022-02-26_16.21.10'
    phone_sequences = read_sequence_list(sequence_list_dir / 'HGE_phone.txt')
    hololens_sequences = read_sequence_list(sequence_list_dir / 'HGE_hololens.txt')
    return ref_id, phone_sequences, hololens_sequences

def get_data_LIN():
    ref_id = '2022-07-03_08.30.21'
    phone_sequences = read_sequence_list(sequence_list_dir / 'LIN_phone.txt')
    hololens_sequences = read_sequence_list(sequence_list_dir / 'LIN_hololens.txt')
    return ref_id, phone_sequences, hololens_sequences


def main(args):
    scene = args.scene
    ref_id, phone_sequences, hololens_sequences = eval('get_data_'+scene)()
    if args.skip_phone:
        phone_sequences = []
    if args.skip_hololens:
        hololens_sequences = []
    logger.info('Found %d phone and %d HoloLens sequences in lists.',
                len(phone_sequences), len(hololens_sequences))
    run(args.capture_root/scene, ref_id, args.phone_dir/scene, args.hololens_dir/scene,
        phone_sequences, hololens_sequences)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture_root', type=Path, default=Path('data/captures/'))
    parser.add_argument('--scene', type=str, required=True, choices=['CAB', 'HGE', 'LIN'])
    parser.add_argument('--phone_dir', type=Path, default=Path('/media/HD8TB/ios_rec/'))
    parser.add_argument('--hololens_dir', type=Path,
                        default=Path('/media/SSD2/hololens_proc/all-fix'))
    parser.add_argument('--skip_phone', action='store_true')
    parser.add_argument('--skip_hololens', action='store_true')
    main(parser.parse_args())
