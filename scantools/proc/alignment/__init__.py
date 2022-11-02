from pathlib import Path
import dataclasses
from typing import Union, List, Dict
import json
import numpy as np

from .image_matching import MatchingConf


@dataclasses.dataclass
class Paths:
    registration_root: Path
    conf: MatchingConf
    query_id: str
    ref_id: str

    outputs = None

    def __post_init__(self):
        ref_id = self.ref_id
        self.outputs = self.registration_root / self.query_id / ref_id

    def session(self, session_id: str):
        return self.registration_root / session_id

    def features(self, session_id: str):
        dir_ = self.session(session_id)
        return dir_ / self.conf.gfeats_file, dir_ / self.conf.lfeats_file

    @property
    def pairs(self):
        return self.outputs / self.conf.pairs_file

    @property
    def matches(self):
        return self.outputs / self.conf.matches_file

    def stats(self, name: str):
        return self.outputs / f'log_{name}.json'


def save_stats(path: Path, stats: Dict):

    class NumpyEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.bool_):
                return bool(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return json.JSONEncoder.default(self, o)

    with open(path, 'w') as fid:
        json.dump(stats, fid, indent=4, cls=NumpyEncoder)
