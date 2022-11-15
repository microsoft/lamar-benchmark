""" Omni-directional camera models for testing purposes """

def ocam_test_model():
    """ Ocam model used from open-source repository:
        https://github.com/zhangzichao/omni_cam/blob/master/test/ocam_param.txt

    Returns
    -------
    dict
        Dictionary with args = {length_pol, pol, length_invpol, invpol,
                                xc, yc, c, d, e, height, width}
    """

    model = {
        'length_pol': 5,
        'pol': [-69.6915, 0.0, 0.00054772, 2.1371e-05, -8.7523e-09],
        'length_invpol': 12,
        'invpol': [142.7468, 104.8486, 7.3973, 17.4581, 12.6308, -4.3751,
                   6.9093, 10.9703, -0.6053, -3.9119, -1.0675, 0.0],
        'xc': 320.0,
        'yc': 240.0,
        'c': 1.0,
        'd': 0.0,
        'e': 0.0,
        'height': 640,
        'width': 480,
        'upright': False
    }

    return model


def navvis_ocam_test_model():
    """ Ocam model values from a sample navvis dataset

    Returns
    -------
    dict
        Dictionary with args = {length_pol, pol, length_invpol, invpol,
                                xc, yc, c, d, e, height, width}
    """

    model = {
        'pol': [-1981.58, 0.0, 0.0002158074, -3.908571e-08, 1.59411e-11],
        'length_pol': 5,
        'invpol': [2895.828621, 1595.461529, -72.247443,
                   252.687914, 115.754995, -9.500948,
                   60.304292, 23.368384, -16.294188,
                   13.037419, 3.317868, 0.947768,
                   12.619066, -1.13914, -8.247052,
                   -0.10145, 2.628608, 0.716341],
        'length_invpol': 18,
        'xc': 1721.800356,
        'yc': 2277.523923,
        'c': 1.00042,
        'd': 0.001043,
        'e': -0.00091,
        'height': 3448,
        'width': 4592,
        'upright': False
    }

    return model
