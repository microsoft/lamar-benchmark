from typing import List, Tuple
from functools import cached_property
import numpy as np
from scipy.spatial.transform import Rotation


class Pose:
    def __init__(self, r=None, t=None, covar=None):
        # rotation
        if r is None:
            r = [1, 0, 0, 0]
        if isinstance(r, (list, np.ndarray, np.generic)):
            if isinstance(r, list):
                r = np.array(r, dtype=float)
            if r.shape == (4,):
                qvec_scipy = r[[1, 2, 3, 0]]
                r = Rotation.from_quat(qvec_scipy)
            elif r.shape == (3, 3):
                r = Rotation.from_matrix(r)
            else:
                raise ValueError(f'Invalid rotation: {r}')
        elif not isinstance(r, Rotation):
            raise ValueError(f'Unknown rotation format: {r}')

        # translation
        if t is None:
            t = [0, 0, 0]
        if isinstance(t, list):
            t = np.array(t, dtype=float)
        elif not isinstance(t, (np.ndarray, np.generic)):
            raise ValueError(f'Unknown translation format: {t}')
        if t.shape != (3,) or not np.all(np.isfinite(t)):
            raise ValueError(f'Invalid translation: {t}')

        if covar is not None and covar.shape != (6, 6):
            raise ValueError(f'Invalid covariance: {covar}')

        self._r = r
        self._t = t
        self._covar = covar

    def to_list(self) -> List[str]:
        data = [self.qvec, self.t]
        if self.covar is not None:
            data.append(self.covar.flatten())
        return np.concatenate(data).astype(str).tolist()

    @classmethod
    def from_list(cls, qt: List[str]) -> 'Pose':
        qw, qx, qy, qz, tx, ty, tz, *covar = qt
        if len(covar) == 0:
            covar = None
        elif len(covar) == 36:
            covar = np.reshape(np.array(covar, float), (6, 6))
        else:
            raise ValueError(
                'Invalid format. Expected: [qw, qx, qy, qz, tx, ty, tz] '
                f'or [qw, qx, qy, qz, tx, ty, tz, covar_6x6]; '
                f'Obtained: {qt}')
        return Pose([qw, qx, qy, qz], [tx, ty, tz], covar=covar)

    @classmethod
    def from_4x4mat(cls, T) -> 'Pose':
        if isinstance(T, list):
            T = np.array(T, dtype=float)
        elif not isinstance(T, (np.ndarray, np.generic)):
            raise ValueError(f'Unknown type for 4x4 transformation matrix: {T}')
        if T.shape != (4, 4):
            raise ValueError(f'Invalid 4x4 transformation matrix: {T}')
        return Pose(T[:3, :3], T[:3, 3])

    @property
    def r(self) -> Rotation:
        return self._r

    @cached_property
    def R(self) -> np.ndarray:
        return self.r.as_matrix()

    @property
    def qvec(self) -> np.ndarray:
        qvec_scipy = self._r.as_quat()
        qvec = qvec_scipy[[3, 0, 1, 2]]
        return qvec

    @property
    def t(self) -> np.ndarray:
        return self._t

    @property
    def covar(self) -> np.ndarray:
        return self._covar

    @property
    def qt(self) -> Tuple[np.ndarray]:
        return (self.qvec, self.t)

    @cached_property
    def adjoint(self) -> np.ndarray:
        tx, ty, tz = self.t
        skew_t = np.array([[0, -tz, ty],
                           [tz, 0, -tx],
                           [-ty, tx, 0]])
        return np.block([[self.R, np.zeros((3, 3))], [skew_t@self.R, self.R]])

    @cached_property
    def adjoint_inv(self) -> np.ndarray:
        R_inv = self.R.transpose()
        tx, ty, tz = self.t
        skew_t = np.array([[0, -tz, ty],
                           [tz, 0, -tx],
                           [-ty, tx, 0]])
        return np.block([[R_inv, np.zeros((3, 3))], [-R_inv @ skew_t, R_inv]])

    def to_4x4mat(self) -> np.ndarray:
        T = np.hstack((self.R, self.t[:, None]))
        T = np.vstack((T, (0, 0, 0, 1)))
        return T

    @cached_property
    def inv(self) -> 'Pose':
        r_inv = self.r.inv()
        rotmat_inv = r_inv.as_matrix()
        t_inv = -1.0 * (rotmat_inv @ self.t)
        covar = self.covar
        if covar is not None:
            # here we assume that the noise is applied on the right
            covar = self.adjoint_inv @ covar @ self.adjoint_inv.T
        return Pose(r_inv, t_inv, covar)

    def inverse(self) -> 'Pose':
        return self.inv

    def __mul__(self, other) -> 'Pose':
        if not isinstance(other, self.__class__):
            return NotImplemented
        r_new = self.r * other.r
        t_new = self.t + self.R @ other.t
        covar = other.covar
        if self.covar is not None:
            adj = other.inv.adjoint
            covar = (0 if covar is None else covar) + adj @ self.covar @ adj.T
        return Pose(r_new, t_new, covar)

    def transform_points(self, points3d: np.ndarray) -> np.ndarray:
        if points3d.shape[-1] != 3:
            raise ValueError(f'Points must be in shape (..., 3): {points3d.shape}')
        return points3d @ self.R.transpose() + self.t

    def magnitude(self) -> Tuple[float, float]:
        dr = np.rad2deg(self.r.magnitude())
        dt = np.linalg.norm(self.t)
        return dr, dt

    def __repr__(self) -> str:
        return 'q:{},  t:{}'.format(self.qvec, self.t)
