import logging
from typing import Optional
from copy import deepcopy

from hloc import match_features, match_dense

from .feature_extraction import FeatureExtraction
from .pair_selection import PairSelection
from ..utils.misc import same_configs, write_config

logger = logging.getLogger(__name__)


class FeatureMatchingPaths:
    def __init__(self, root, config, query_id, ref_id):
        self.root = root
        feature_name = config['features']['name']
        matches_name = config['name']
        self.workdir = root / 'matching' / query_id / ref_id / feature_name / matches_name
        self.matches = self.workdir / 'matches.h5'
        self.config = self.workdir / 'configuration.json'


class FeatureMatching:
    methods = {
        'loftr': {
            'name': 'loftr',
            'hloc': {
                'model': {
                    'name': 'loftr',
                    'weights': 'outdoor'
                },
                'preprocessing': {
                    'grayscale': True,
                    'resize_max': 1024,
                    'dfactor': 8
                },
                'max_error': 2,  # max error for assigned keypoints (in px)
                'cell_size': 8,  # size of quantization patch (max 1 kp/patch)
            }
        },
        'superglue': {
            'name': 'superglue',
            'hloc': {
                'model': {
                    'name': 'superglue',
                    'weights': 'outdoor',
                    'sinkhorn_iterations': 5,
                },
            },
        },
        'mnn': {
            'name': 'mnn',
            'hloc': {
                'model': {
                    'name': 'nearest_neighbor',
                    'do_mutual_check': True,
                },
            }
        },
        'ratio_mnn_0_9': {
            'name': 'ratio_mnn',
            'hloc': {
                'model': {
                    'name': 'nearest_neighbor',
                    'do_mutual_check': True,
                    'ratio_threshold': 0.9,
                },
            }
        },
        'ratio_mnn_0_8': {
            'name': 'ratio_mnn',
            'hloc': {
                'model': {
                    'name': 'nearest_neighbor',
                    'do_mutual_check': True,
                    'ratio_threshold': 0.8,
                },
            }
        },
        'adalam': {
            'name': 'adalam',
            'hloc': {
                'model': {
                    'name': 'adalam'
                },
            }
        }
    }

    def __init__(self, outputs, capture, query_id, ref_id, config,
                 pair_selection: PairSelection,
                 overwrite=False):
        self.config = config = deepcopy(config)
        self.query_id = query_id
        self.ref_id = ref_id
        self.pair_selection = pair_selection
        self.extraction = FeatureExtraction(outputs, capture, query_id, config['extraction'])
        self.extraction_ref = FeatureExtraction(outputs, capture, ref_id, config['extraction'])
        self.paths = FeatureMatchingPaths(outputs, {**config['matching'], 'features': self.extraction.config}, query_id, ref_id)
        self.paths.workdir.mkdir(parents=True, exist_ok=True)

        if 'hloc' in config['extraction']:
            logger.info('Matching local features with %s for sessions (%s, %s).',
                        config['matching']['name'], query_id, ref_id)
            if not same_configs(config, self.paths.config):
                logger.warning('Existing matches will be overwritten.')
                overwrite = True
            match_features.main(
                config['matching']['hloc'],
                pair_selection.paths.pairs_hloc,
                self.extraction.paths.features,
                matches=self.paths.matches,
                features_ref=self.extraction_ref.paths.features,
                overwrite=overwrite,
            )
            write_config(config, self.paths.config)
        else:
            logger.info('Matching dense features with %s for sessions (%s, %s).',
                        config['matching']['name'], query_id, ref_id)
            if not same_configs(config, self.paths.config):
                logger.warning('Existing matches will be overwritten.')
                overwrite = True
                print("IIIIII")
            if query_id == ref_id:
                return
                match_dense.main(
                    config['matching']['hloc'],
                    pair_selection.paths.pairs_hloc,
                    self.extraction.image_root,
                    matches=self.paths.matches,
                    features=self.extraction.paths.features,
                    max_kps=8192,
                    overwrite=overwrite,
                )
            else:
                match_dense.main(
                    config['matching']['hloc'],
                    pair_selection.paths.pairs_hloc,
                    self.extraction.image_root,
                    matches=self.paths.matches,
                    features=self.extraction.paths.features,
                    features_ref=self.extraction_ref.paths.features,
                    max_kps=None,
                    overwrite=overwrite,
                )
            write_config(config, self.paths.config)

