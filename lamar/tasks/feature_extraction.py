import logging
from copy import deepcopy
import numpy as np

from hloc import extract_features

from ..utils.capture import list_images_for_session
from ..utils.misc import same_configs, write_config

logger = logging.getLogger(__name__)


class FeatureExtractionPaths:
    def __init__(self, root, config, session_id):
        self.root = root
        self.workdir = root / 'extraction' / session_id / config['name']
        self.features = self.workdir / 'features.h5'
        self.config = self.workdir / 'configuration.json'


class FeatureExtraction:
    methods = {
        'superpoint': {
            'name': 'superpoint',
            'hloc': {
                'model': {
                    'name': 'superpoint',
                    'nms_radius': 3,
                    'max_keypoints': 2048,
                },
                'preprocessing': {
                    'grayscale': True,
                    'resize_max': 1024,
                },
            },
        },
        'dumbpoint': {
            'name': 'dumbpoint',
            'hloc': {
                'model': {
                    'name': 'dumbpoint',
                },
                'preprocessing': {
                    'grayscale': True,
                    'resize_max': 1024,
                },
            },
        },
        'r2d2': {
            'name': 'r2d2',
            'hloc': {
                'model': {
                    'name': 'r2d2',
                    'max_keypoints': 5000,
                },
                'preprocessing': {
                    'grayscale': False,
                    'resize_max': 1024,
                }
            }
        },
        'd2net': {
            'name': 'd2net',
            'hloc': {
                'model': {
                    'name': 'd2net',
                    'multiscale': False,
                },
                'preprocessing': {
                    'grayscale': False,
                    'resize_max': 1600,
                }
            }
        },
        'd2net-ms': {
            'name': 'd2net-ms',
            'hloc': {
                'model': {
                    'name': 'd2net',
                    'multiscale': True,
                },
                'preprocessing': {
                    'grayscale': False,
                    'resize_max': 1600,
                }
            }
        },
        'sift': {
            'name': 'sift',
            'hloc': {
                'model': {
                    'name': 'dog',
                    'options': {
                        'first_octave': -1,
                        'upright': True
                    }
                },
                'preprocessing': {
                    'grayscale': True,
                    'resize_max': 1600,
                },
            },
        },
        'sosnet': {
            'name': 'sosnet',
            'hloc': {
                'model': {
                    'name': 'dog',
                    'descriptor': 'sosnet',
                    'options': {
                        'first_octave': -1,
                        'upright': True
                    }
                },
                'preprocessing': {
                    'grayscale': True,
                    'resize_max': 1600,
                },
            }
        },
    }

    def __init__(self, outputs, capture, session_id, config, query_keys=None, overwrite=False):
        self.config = config = deepcopy(config)
        self.session_id = session_id
        self.paths = FeatureExtractionPaths(outputs, config, session_id)
        self.paths.workdir.mkdir(parents=True, exist_ok=True)
        if not same_configs(config, self.paths.config):
            overwrite = True

        logger.info('Extraction local features %s for session %s.', config['name'], session_id)
        _, names, image_root = list_images_for_session(capture, session_id, query_keys)
        names = np.unique(names)
        extract_features.main(
            config['hloc'],
            image_root,
            feature_path=self.paths.features,
            image_list=names,
            as_half=True,
            overwrite=overwrite,
        )

        write_config(config, self.paths.config)


class RetrievalFeatureExtraction(FeatureExtraction):
    methods = {
        'netvlad': {
            'name': 'netvlad',
            'hloc': {
                'model': {'name': 'netvlad'},
                'preprocessing': {'resize_max': 640},
            },
        },
        'ap-gem': {
            'name': 'ap-gem',
            'hloc': {
                'model': {'name': 'dir'},
                'preprocessing': {'resize_max': 640},
            }
        },
        'cosplace': {
            'name': 'cosplace',
            'hloc': {
                'model': {'name': 'cosplace'},
                'preprocessing': {'resize_max': 640},
            }
        },
        'openibl': {
            'name': 'openibl',
            'hloc': {
                'model': {'name': 'openibl'},
                'preprocessing': {'resize_max': 640},
            }
        },
        'salad': {
            'name': 'salad',
            'hloc': {
                'model': {'name': 'salad'},
                'preprocessing': {'resize_max': 640},
            }
        }
    }
