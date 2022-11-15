from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Iterator

from .. import logger
from .session import Session


# TODO: make dataclasses frozen or generate getter+setter properties with typecheck
# TODO: enforce that no session/rig/sensor id has any comma or starts/ends with /
@dataclass
class Capture:
    sessions: Dict[str, Session] = None
    path: Optional[Path] = None

    sessions_dirname = 'sessions'
    align_dirname = 'alignment'

    @classmethod
    def load(cls, path: Path, session_ids: Optional[Iterator[str]] = None, **kwargs) -> 'Capture':
        if not isinstance(path, Path):
            path = Path(path)
        if not path.exists():
            raise IOError(f'Capture directory does not exists: {path}')
        sessions_path = path / cls.sessions_dirname
        if not sessions_path.exists():
            raise IOError(f'Sessions directory does not exists: {sessions_path}')

        logger.info('Loading Capture from %s.', path.resolve())
        if session_ids is None:
            session_ids = [p.name for p in sessions_path.iterdir() if p.is_dir()]
        sessions = {s: Session.load(sessions_path / s, **kwargs) for s in session_ids}
        return cls(sessions, path)

    def save(self, path: Optional[Path] = None, session_ids: Optional[Iterator[str]] = None):
        self.path = path = path or self.path
        if path is None:
            raise ValueError('No valid path found as argument or attribute.')

        if session_ids is None:
            session_ids = self.sessions.keys()
        else:
            assert len(set(session_ids) - self.sessions.keys()) == 0

        logger.info('Writing Capture to %s.', path.resolve())
        sessions_path = path / self.sessions_dirname
        sessions_path.mkdir(exist_ok=True, parents=True)
        for session_id in session_ids:
            self.sessions[session_id].save(sessions_path / session_id)
        self.path = path

    def __repr__(self) -> str:
        representation = '\n'.join(
            f'[{session_id:5}] = {session}' for session_id, session in self.sessions.items()
        )
        return representation

    def sessions_path(self) -> Path:
        return self.path / self.sessions_dirname

    def session_path(self, session_id: str) -> Path:
        return self.sessions_path() / session_id

    def data_path(self, session_id: str) -> Path:
        return self.session_path(session_id) / Session.data_dirname

    def proc_path(self, session_id: str) -> Path:
        return self.session_path(session_id) / Session.proc_dirname

    def viz_path(self) -> Path:
        return self.path / 'visualization'

    def registration_path(self) -> Path:
        return self.path / 'registration'

    def extra_path(self) -> Path:
        return self.path / 'extra'
