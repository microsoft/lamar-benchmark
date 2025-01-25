import logging
from typing import Optional
from copy import deepcopy

from hloc import match_dense

from .feature_extraction import FeatureExtraction, FeatureExtractionPaths
from .feature_matching import FeatureMatching, FeatureMatchingPaths
from .pair_selection import PairSelection
from ..utils.misc import same_configs, write_config

logger = logging.getLogger(__name__)


class FeatureExtractionAndMatching:
    def __init__(self, outputs, capture, query_id, ref_id, config,
                 pair_selection: PairSelection,
                 overwrite=False):
        self.config = config = deepcopy(config)
        self.query_id = query_id
        self.ref_id = ref_id
        self.pair_selection = pair_selection

        if config.extraction:
            self.extraction = FeatureExtraction(outputs, capture, query_id, config.extraction)
            self.extraction_ref = FeatureExtraction(outputs, capture, ref_id, config.extraction)
            self.paths = FeatureMatchingPaths(outputs, config, query_id, ref_id)
            self.paths.workdir.mkdir(parents=True, exist_ok=True)

            logger.info('Matching local features with %s for sessions (%s, %s).',
                        config.matching['name'], query_id, ref_id)
            if not same_configs(config, self.paths.config):
                logger.warning('Existing matches will be overwritten.')
                overwrite = True
            match_features.main(
                config['hloc'],
                pair_selection.paths.pairs_hloc,
                self.extraction.paths.features,
                matches=self.paths.matches,
                features_ref=self.extraction_ref.paths.features,
                overwrite=overwrite,
            )

            write_config(config, self.paths.config)
        else:
            self.extraction = FeatureExtraction(outputs, capture, query_id, config.extraction)
            self.extraction_ref = FeatureExtraction(outputs, capture, ref_id, config.extraction)
            self.paths = FeatureMatchingPaths(outputs, config, query_id, ref_id)
            self.paths.workdir.mkdir(parents=True, exist_ok=True)

            logger.info('Matching local features with %s for sessions (%s, %s).',
                        config.matching['name'], query_id, ref_id)
            if not same_configs(config, self.paths.config):
                logger.warning('Existing matches will be overwritten.')
                overwrite = True
            if query_id == ref_id:
                match_dense.main(
                    config.matching['hloc'],
                    pair_selection.paths.pairs_hloc,
                    self.extraction.image_root,
                    matches=self.paths.matches,
                    features=self.extraction.paths.features,
                    max_kps=8192,
                    overwrite=overwrite,
                )
            else:
                match_dense.main(
                    config.matching['hloc'],
                    pair_selection.paths.pairs_hloc,
                    self.extraction.image_root,
                    matches=self.paths.matches,
                    features=self.extraction.paths.features,
                    features_ref=self.extraction_ref.paths.features,
                    max_kps=None,
                    overwrite=overwrite,
                )

            write_config(config, self.paths.config)
