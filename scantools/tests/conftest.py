""" Directory-scoped pytest fixtures """
from enum import Enum
import numpy as np

import pytest

from . import ocam_models
from .. import utils


class Axis(Enum):
    """ Coordinate axis

    Parameters
    ----------
    Enum : X, Y or Z
    """
    X = 0
    Y = 1
    Z = 2


@pytest.fixture(params=[Axis.X, Axis.Y, Axis.Z])
def rotmat(request):
    """ Fixture for generating a rotation matrix

    Parameters
    ----------
    request : Enum
        Axis = {X, Y Z}

    Returns
    -------
    numpy.ndarray
        (3, 3) rotation matrix

    Raises
    ------
    ValueError
        If rotation axis not X, Y or Z
    """
    def get_rotmat(axis, theta):

        if axis == Axis.X:
            return utils.transform.Rx(theta)

        if axis == Axis.Y:
            return utils.transform.Ry(theta)

        if axis == Axis.Z:
            return utils.transform.Rz(theta)

        raise ValueError("Axis type not recognized.")

    theta = np.radians(165)
    return get_rotmat(request.param, theta)


@pytest.fixture
def ocam_model():
    """ Fixture wrapper of a test OCam model

    Returns
    -------
    dict
        OCam test model
    """

    return ocam_models.ocam_test_model()

@pytest.fixture
def ocam_model_navvis():
    """ Fixture wrapper for a test OCam model in NavVis format
    Returns
    -------
    dict
        OCam test model from a sample navvis dataset
    """

    return ocam_models.navvis_ocam_test_model()
