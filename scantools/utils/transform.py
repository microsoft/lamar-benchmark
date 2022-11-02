""" Transformation utility functions """
import numpy as np


def check_transformation_matrix(tmat):
    """ Checks if argument is of type numpy.ndarray and shape (4, 4)

    Parameters
    ----------
    tmat : numpy.ndarray
        Transformation Matrix of shape (4, 4)

    Raises
    ------
    TypeError
        If matrix not of type numpy.ndarray
    ValueError
        Invalid shape or numpy.nan encountered
    """
    if not isinstance(tmat, np.ndarray):
        raise TypeError("Invalid trans_init type")
    if len(tmat.shape) != 2 or tmat.shape[0] != 4 or tmat.shape[1] != 4:
        raise ValueError("Invalid trans_init dimensions:", tmat.shape)
    if np.isnan(tmat).any():
        raise ValueError("Invalid value numpy.nan encountered:", tmat)


def check_qvec_valid(qvec):
    if isinstance(qvec, np.ndarray):

        if (len(qvec.shape) != 1) or (qvec.shape[0] != 4):
            raise ValueError("Invalid qvec dimensions:", qvec.shape,
                             "instead of (4,).")
        # Check no Nones
        if np.equal(qvec, None).any():
            raise ValueError("Invalid value encountered in qvec: None")

        # Check dtype
        if qvec.dtype != np.float64 and \
           qvec.dtype != np.float32 and \
           qvec.dtype != np.int64 and \
           qvec.dtype != np.int32:
            raise ValueError("Invalid data type: ", qvec.dtype,
                             "instead of {float64, float32, int64, int32}.")

        # Check no nans
        if np.isnan(qvec).any():
            raise ValueError("Invalid value encountered in qvec: nan")

    else:
        raise TypeError("Invalid qvec type:", type(qvec),
                        "instead of numpy.ndarray.")


def check_tvec_valid(tvec):
    if isinstance(tvec, np.ndarray):

        # Check correct dim
        if (len(tvec.shape) != 1) or (tvec.shape[0] != 3):
            raise ValueError("Invalid tvec dimensions:", tvec.shape,
                             "instead of (3,).")

        # Check no Nones
        if np.equal(tvec, None).any():
            raise ValueError("Invalid value encountered in tvec: None")

        # Check dtype
        if tvec.dtype != np.float64 and \
           tvec.dtype != np.float32 and \
           tvec.dtype != np.int64 and \
           tvec.dtype != np.int32:
            raise ValueError("Invalid data type:", tvec.dtype,
                             "instead of {float64, float32, int64, int32}.")

        # Check no nans
        if np.isnan(tvec).any():
            raise ValueError("Invalid value encountered in tvec: nan")

    else:
        raise TypeError("Invalid qvec type:", type(tvec),
                        "instead of numpy.ndarray or list.")


def check_rotmat_valid(R):
    if not isinstance(R, np.ndarray):
        raise TypeError("Invalid R type:", type(R),
                        "instead of numpy.ndarray or list.")

    if (len(R.shape) != 2) or (R.shape[0] != 3) or (R.shape[1] != 3):
        raise ValueError("Invalid R dimensions:", R.shape,
                         "instead of (3, 3).")
    if np.isnan(R).any():
        raise ValueError("Invalid value encountered in R: nan")


def qvec2rotmat(qvec):
    check_qvec_valid(qvec)
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def rotmat2qvec(R):
    check_rotmat_valid(R)
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def tvec2center(R, tvec):
    check_rotmat_valid(R)
    check_tvec_valid(tvec)
    return (-1) * R.T @ tvec


def invert_pose(qvec, tvec):
    R = qvec2rotmat(qvec)
    qvec_inv = rotmat2qvec(R.T)
    tvec_inv = tvec2center(R, tvec)
    return qvec_inv, tvec_inv


def create_transform_4x4(R, t):
    check_rotmat_valid(R)
    check_tvec_valid(t)

    Rt = np.column_stack((R, t))
    Rt = np.vstack((Rt, (0, 0, 0, 1)))
    return Rt


# angle to rotation in x-axis
def Rx(theta_x):
    co = np.cos(theta_x)
    si = np.sin(theta_x)

    #pylint: disable=bad-whitespace
    return np.array([[1,   0,   0],
                     [0,  co, -si],
                     [0,  si,  co]])


# angle to rotation in y-axis
def Ry(theta_y):
    co = np.cos(theta_y)
    si = np.sin(theta_y)

    #pylint: disable=bad-whitespace
    return np.array([[co,  0, -si],
                     [0,   1,   0],
                     [si,  0,  co]])


# angle to rotation in y-axis
def Rz(theta_z):
    co = np.cos(theta_z)
    si = np.sin(theta_z)

    #pylint: disable=bad-whitespace
    return np.array([[co, -si,  0],
                     [si,  co,  0],
                     [0,    0,  1]])


def apply(T, points):
    """
    Apply transformation to points.

    This can be used for T = [R t; 0 1] or K (calibration matrix).
    For example:
        - K * points2D
        - [R t; 0 1] * points3D

    Parameters
    ----------
    T : matrix
    points : points
    """
    # convert to homogeneous coord., apply transform, convert back to non-homog.
    func = lambda x: nonhomogen(T @ homogen(x))

    # vectorize function
    vfunc = np.vectorize(func, signature='(n)->(m)')

    return vfunc(points)


def homogen(xs):
    func = lambda x: np.concatenate((x, [1]))
    vfunc = np.vectorize(func, signature='(n)->(m)')
    return vfunc(xs)


def nonhomogen(xs):
    func = lambda x: x[:-1]/x[-1]
    vfunc = np.vectorize(func, signature='(n)->(m)')
    return vfunc(xs)


def get_point3D_from_depth(points2D, depth_map, intrinsic_matrix):
    """
    Get 3D points from 2D point measurements having the depth map and intrinsic
    matrix (K).

    Parameters
    ----------
    points2D : numpy.ndarray (2xN points)
        2D point measurements
    depth_map : numpy.ndarray
        Depth map
    intrinsic_matrix : numpy.ndarray
        calibration matrix

    Returns
    -------
    numpy.ndarray
        3D points (3xN points)
    """
    depth_value = lambda point2D: depth_map[int(point2D[1]), int(point2D[0])]

    K = intrinsic_matrix
    K_inv = np.linalg.inv(K)

    # point2D to norm. coord., then multiply by the depth to get the 3D point
    func = lambda point2D: (K_inv @ homogen(point2D)) * depth_value(point2D)

    # vectorize
    vfunc = np.vectorize(func, signature='(n)->(m)')

    return vfunc(points2D)
