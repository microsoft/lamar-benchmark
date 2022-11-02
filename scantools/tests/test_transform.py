""" Tests for transform.py """
import pytest
import numpy as np

from ..utils.transform import (
    check_qvec_valid,
    check_tvec_valid,
    create_transform_4x4,
    check_transformation_matrix,
)


@pytest.mark.parametrize(
    "tmat",
    [
        np.identity(4),
        np.zeros((4, 4)),
        np.ones((4, 4)),
        np.array([[0, 0, 0.5, -10],
                  [-1, 5, 9, 1],
                  [5, 4, 1, 0],
                  [0, 2, -5.9, 28.4]]),
    ])
def test_check_transformation_matrix_valid(tmat):
    """ Checks if check_transformation_matrix returns True for numpy array of shape (4, 4)

    Parameters
    ----------
    tmat : np.ndarray
        Matrix of shape (4, 4)
    """
    check_transformation_matrix(tmat)

@pytest.mark.parametrize(
    "tmat_invalid",
    [
        np.zeros((4,)),
        np.zeros((4, 5)),
        np.ones((2, 2)),
        np.array([[0, 0, 0.5, -10],
                  [-1, np.nan, 9, 1],
                  [5, 4, 1, 0],
                  [0, 2, -5.9, np.nan]]),
    ])
def test_check_transformation_matrix_invalid(tmat_invalid):
    """ Checks check_transformation_matrix returns raises a ValueError
        for numpy array of shape different than (4, 4) or array containing nan

    Parameters
    ----------
    tmat : np.ndarray
        Matrix of shape any shape different than (4, 4)
    """
    with pytest.raises(ValueError):
        check_transformation_matrix(tmat_invalid)

@pytest.mark.parametrize(
    "tmat_type_invalid",
    [
        None,
        [],
        [1, 2, 3, 4],
        [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
    ])
def test_check_transformation_matrix_type_invalid(tmat_type_invalid):
    """ Checks if check_transformation_matrix raises TypeError
        for input of type different than numpy.ndarray

    Parameters
    ----------
    tmat : np.ndarray
        Matrix of shape (4, 4)
    """
    with pytest.raises(TypeError):
        check_transformation_matrix(tmat_type_invalid)

@pytest.mark.parametrize(
    "qvec",
    [
        np.array([0, 0, 0, 0]),
        np.array([1, 2, 3, 4]),
    ])
def test_check_qvec_valid(qvec):
    """ Checks check_qvec_valid function returns true for valid qvec

    Parameters
    ----------
    qvec : numpy.ndarray
        quaternion, vector of shape (4,)
    """
    check_qvec_valid(qvec)


@pytest.mark.parametrize(
    "qvec",
    [
        None,
        [],
        [0, 0, 0, 0],
        [1.0, 2.0, 3, 4],
    ])
def test_check_qvec_valid_type_error(qvec):
    """ Checks check_qvec_valid function raises TypeError
        for qvec of different type than numpy.ndarray

    Parameters
    ----------
    qvec : any except numpy.ndarray
        Invalid quaternion
    """
    with pytest.raises(TypeError):
        check_qvec_valid(qvec)


@pytest.mark.parametrize(
    "qvec",
    [
        # incorrect dims of numpy array
        np.array([]),
        np.array([[], [], [], []]),  # (4, 0)
        np.array([[], [], [], []]).T,  # (0, 4)
        np.array([[1, 2, 3, 4]]),  # (1, 4)
        np.array([[1, 2, 3, 4]]).T,  # (4, 1)
        np.array([1, None, 3, 4]), # contains None, dtype object
        np.array([1, 2, np.nan, 4]), # contains nan
        np.array([1, 2, 3, float('nan')]), # contains nan
        np.array([1, 2, [], 4], dtype=object), # incorrect dtype
    ])
def test_check_qvec_valid_val_error(qvec):
    """ Checks check_qvec_valid function raises ValueError
        for invalid shapes or values

    Parameters
    ----------
    qvec : numpy.ndarray
        any shape except (4,) or contains numpy.nan
    """
    with pytest.raises(ValueError):
        check_qvec_valid(qvec)


@pytest.mark.parametrize(
    "tvec",
    [
        np.array([0, 0, 0]),
        np.array([1, 2, 3])
    ])
def test_check_tvec_valid(tvec):
    """ Checks check_tvec_valid function returns True
        for tvec of type numpy.ndarray and shape (3,)

    Parameters
    ----------
    tvec : numpy.ndarray
        Translation vector, array of shape (3,)
    """
    check_tvec_valid(tvec)

@pytest.mark.parametrize(
    "tvec",
    [
        None,
        [],
        [0, 0, 0],
        [1.0, 2.0, 3],
    ])
def test_check_tvec_valid_type_error(tvec):
    with pytest.raises(TypeError):
        check_tvec_valid(tvec)


@pytest.mark.parametrize(
    "tvec",
    [
        # incorrect dims of numpy array
        np.array([]),
        np.array([[], [], []]),  # (3, 0)
        np.array([[], [], []]).T,  # (0, 3)
        np.array([[1, 2, 3]]),  # (1, 3)
        np.array([[1, 2, 3]]).T,  # (3, 1)
        np.array([1, None, 3]), # contains None, dtype object
        np.array([1, np.nan, 3]), # contains nan
        np.array([1, float('nan'), 3]), # contains nan
        np.array([1, [], 3], dtype=object), # incorrect dtype
    ])
def test_check_tvec_valid_val_error(tvec):
    """ Checks check_tvec_valid function raises ValueError
        for arrays with shape different than (3,) or arrays containing nan

    Parameters
    ----------
    tvec : numpy.ndarray
        Translation vector, array of shape (3,)
    """
    with pytest.raises(ValueError):
        check_tvec_valid(tvec)


def test_create_transform_4x4():
    """ Tests that create_transform_4x4 returns expected value """
    theta_z = np.deg2rad(68)
    diag = np.cos(theta_z)
    off_diag = np.sin(theta_z)

    #pylint: disable=bad-whitespace
    expected = np.array([[diag,     -off_diag,  0,   0.5],
                         [off_diag,      diag,  0, -12.123],
                         [0,                 0, 1,   0],
                         [0,                 0, 0,   1]])

    R = expected[0:3, 0:3]
    t = expected[0:3, 3]

    res = create_transform_4x4(R, t)

    assert np.array_equal(res, expected)
