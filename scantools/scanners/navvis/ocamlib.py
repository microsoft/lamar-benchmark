import os
import errno
import logging
import pickle
from pathlib import Path

import numpy as np
import cv2

from ... import utils

logger = logging.getLogger(__name__)


def cam2world(points2D, ocam_model):

    if not isinstance(points2D, np.ndarray):
        raise TypeError("Invalid points2D type:", type(points2D),
                        "instead of numpy.ndarray.")

    # only one point
    if len(points2D.shape) == 1 and points2D.shape[0] == 2:
        points2D = points2D[np.newaxis, :]

    # check dimensions
    if len(points2D.shape) != 2 \
       or points2D.shape[0] < 1 \
       or points2D.shape[1] != 2 \
       or points2D.dtype == object:
        raise ValueError("Invalid points2D dimensions:",
                         points2D.shape, " instead of (-1, 2).")

    # Compute inverse determinant of matrix A = [c d; e 1]
    # where c, d, e are the affine parameters of the camera model
    invdet = 1.0 / (ocam_model['c'] - ocam_model['d'] * ocam_model['e'])

    xp = invdet * ((points2D[:, 0] - ocam_model['xc']) -
                    ocam_model['d'] * (points2D[:, 1] - ocam_model['yc']))
    yp = invdet * (-ocam_model['e'] * (points2D[:, 0] - ocam_model['xc']) +
                    ocam_model['c'] * (points2D[:, 1] - ocam_model['yc']))

    N_pol = ocam_model['length_pol']

    r = np.linalg.norm([xp, yp], axis=0)
    # r_vec[i] = r^i; dim (N, N_pol)
    r_vec = np.tile(r, (N_pol, 1)).T ** np.arange(N_pol)
    zp = np.sum(r_vec * ocam_model['pol'], axis=1)  # dim (N, N_pol)

    invnorm = 1.0 / np.linalg.norm([xp, yp, zp], axis=0)  # dim (N_points,)
    points3D = np.vstack((xp, yp, zp)).T * invnorm[:, np.newaxis]

    # only one point, remove extra dimensions
    N_points = points2D.shape[0]
    if N_points == 1:
        points3D = np.squeeze(points3D)

    return points3D


def world2cam(points3D, ocam_model):

    if not isinstance(points3D, np.ndarray):
        raise TypeError("Invalid points3D type:", type(points3D),
                        "instead of numpy.ndarray.")

    # only one point
    if len(points3D.shape) == 1 and points3D.shape[0] == 3:
        points3D = points3D[np.newaxis, :]

    # check dimensions
    if len(points3D.shape) != 2 \
       or points3D.shape[0] < 1 \
       or points3D.shape[1] != 3 \
       or points3D.dtype == object:
        raise ValueError("Invalid points3D dimensions:", points3D.shape,
                         " instead of (-1, 3).")

    N_points = points3D.shape[0]
    points2D = np.tile([ocam_model['xc'], ocam_model['yc']], (N_points, 1))

    norm = np.linalg.norm(points3D[:, :2], axis=1)
    idxs = np.nonzero(norm)[0] # indices of 3D points with non-zero norm

    if idxs.shape[0] > 0:
        theta = np.arctan(points3D[idxs, 2] / norm[idxs])
        invnorm = 1.0 / norm[idxs]

        N_invpol = ocam_model['length_invpol']

        # t_vec[i] = theta^i
        t_vec = np.tile(theta, (N_invpol, 1)).T ** np.arange(N_invpol)
        rho = np.sum(t_vec * ocam_model['invpol'], axis=1)

        x = points3D[idxs, 0] * invnorm * rho
        y = points3D[idxs, 1] * invnorm * rho

        points2D[idxs, 0] = x * ocam_model['c'] + y * ocam_model['d'] + ocam_model['xc']
        points2D[idxs, 1] = x * ocam_model['e'] + y + ocam_model['yc']

    # only one point, remove extra dimensions
    if N_points == 1:
        points2D = np.squeeze(points2D)

    return points2D

#
# ocam_model : camera model used for world2cam(...)
# width      : output width. Final image will have this width.
# height     : output height. Final image will have this height.
# zoom_factor: original 'sf' value that controls the cropping of the mapping
# angles     : angles to rotate the mirror in the ocam model, this moves the
#              image plane allowing the generation of novel views.
#              Default value: [0,0,0]
#
def create_undistortion_LUT(ocam_model,
                            width,
                            height,
                            zoom_factor,
                            angles=None):

    # list as default argument is dangerous
    if angles is None:
        angles = [0, 0, 0]

    Nxc = (height - 1.0) / 2.0
    Nyc = (width - 1.0) / 2.0
    f = height if ocam_model['upright'] else width
    Nz = (-1.0) * f / zoom_factor

    # angles to rotation
    Rx = utils.transform.Rx(angles[0])
    Ry = utils.transform.Ry(angles[1])
    Rz = utils.transform.Rz(angles[2])
    R = Rz @ Ry @ Rx

    M_dim = width * height
    M = np.zeros((M_dim, 3))
    M[:, 0] = np.repeat(np.arange(height), width) - Nxc
    M[:, 1] = np.tile(np.arange(width), height) - Nyc
    M[:, 2] = Nz

    if ocam_model['upright']:
        # rotate coordinates by pi / 2 clockwise
        R = R @ upright_fix()

    # rotate points in unit sphere
    M = R @ M.T
    M = M.T

    m = world2cam(M, ocam_model)
    mapx = m[:, 1].reshape((height, width))
    mapy = m[:, 0].reshape((height, width))

    # convert to float32
    mapx = mapx.astype('float32')
    mapy = mapy.astype('float32')

    return mapx, mapy


# remap and save image, if output path is provided
def save_remaped_image(src, mapx, mapy, path):
    # remap image using LUT maps
    dst = cv2.remap(src, mapx, mapy, cv2.INTER_LINEAR)

    # save result
    cv2.imwrite(path, dst)

    return dst


def load_cam_LUT(path):
    # load from file
    with open(path, 'rb') as cam_LUT_file:
        ocam_model = pickle.load(cam_LUT_file)

    # extract data
    mapx = ocam_model['mapx']
    mapy = ocam_model['mapy']

    return mapx, mapy


def save_cam_LUT(mapx, mapy, path):
    # pack data
    ocam_model = {'mapx': mapx, 'mapy': mapy}

    # save to file
    with open(path, 'wb') as ocam_binary_file:
        pickle.dump(ocam_model, ocam_binary_file)


def undistort(ocam_model,
              input_image_path,
              output_image_path,
              cam_LUT_path=None,
              output_width=None,
              output_height=None,
              zoom_factor=4,
              angles=None):

    # list as default argument is dangerous
    if angles is None:
        angles = [0, 0, 0]

    # check if input image exists
    if not Path(input_image_path).is_file():
        raise FileNotFoundError(errno.ENOENT,
                                os.strerror(errno.ENOENT),
                                str(input_image_path))

    # skip already computed images
    if Path(output_image_path).is_file():
        logger.debug("Image is already computed: %d", output_image_path)
        return

    # use input image width size as default
    if not output_width:
        output_width = ocam_model['width']

    # use input image height size as default
    if not output_height:
        output_height = ocam_model['height']

    # display info
    logger.debug(
        "=========== ocam.undistort =============\n"
        "  input_image_path:      \t '%s'\n"
        "  output_image_path:     \t '%s'\n"
        "  cam_LUT_path:          \t '%s'\n"
        "  output_width:          \t '%s'\n"
        "  output_height:         \t '%s'\n"
        "  zoom_factor:           \t '%s'\n"
        "  angles:                \t '%s'\n",
        input_image_path, output_image_path, cam_LUT_path,
        output_width, output_height, zoom_factor, angles)

    # load pre-computed LUT from file
    if cam_LUT_path and Path(cam_LUT_path).is_file():
        mapx, mapy = load_cam_LUT(cam_LUT_path)

    # otherwise compute LUT
    else:
        logger.debug("Computing camera LUT...")
        mapx, mapy = create_undistortion_LUT(ocam_model,
                                             output_width,
                                             output_height,
                                             zoom_factor,
                                             angles)

        #  save computed LUT, if file didn't exit but a path was provide
        if cam_LUT_path:
            logger.info("Save pre-computed LUT: %s", cam_LUT_path)
            save_cam_LUT(mapx, mapy, cam_LUT_path)

    # load image
    src = cv2.imread(input_image_path, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
    height, width = src.shape[:2]
    assert width > height, (
        f"Error: Width ({width}) should be greater than height ({height}). "
        f"Found in file: {input_image_path}"
    )
    # remap and save image
    save_remaped_image(src, mapx, mapy, output_image_path)


def undistort_point(point2D,
                    ocam_model,
                    output_width=None,
                    output_height=None,
                    zoom_factor=4,
                    angles=None):
    # use input image width size as default
    if output_width is None:
        output_width = ocam_model['width']

    # use input image height size as default
    if output_height is None:
        output_height = ocam_model['height']

    # tile angles
    if angles is None:
        angles = [0, 0, 0]

    xc = (output_height - 1.0) / 2.0
    yc = (output_width - 1.0) / 2.0
    f = output_height if ocam_model['upright'] else output_width
    z = (-1) * f / zoom_factor

    # input of cam2world is swapped
    point2D_dist = np.zeros(2)
    point2D_dist[0] = point2D[1]
    point2D_dist[1] = point2D[0]

    M = cam2world(point2D_dist, ocam_model)
    M = (M / M[2]) * z

    # angles to rotation
    Rx = utils.transform.Rx(angles[0])
    Ry = utils.transform.Ry(angles[1])
    Rz = utils.transform.Rz(angles[2])
    R = Rz @ Ry @ Rx

    if ocam_model['upright']:
        # rotate coordinates by pi / 2 clockwise
        R = R @ upright_fix()

    # rotate point in unit sphere
    M = np.linalg.inv(R) @ M.T
    M = M.T

    point2D_undist = np.zeros(2)
    point2D_undist[0] = M[1] + yc
    point2D_undist[1] = M[0] + xc

    return point2D_undist


def distort_point(point2D,
                  ocam_model,
                  output_width=None,
                  output_height=None,
                  zoom_factor=4,
                  angles=None):
    # use input image width size as default
    if output_width is None:
        output_width = ocam_model['width']

    # use input image height size as default
    if output_height is None:
        output_height = ocam_model['height']

    # tile angles
    if angles is None:
        angles = [0, 0, 0]

    xc = (output_height - 1.0) / 2.0
    yc = (output_width - 1.0) / 2.0
    f = output_height if ocam_model['upright'] else output_width
    z = (-1) * f / zoom_factor

    # point in 3D
    M = np.zeros(3)
    M[0] = point2D[1] - xc
    M[1] = point2D[0] - yc
    M[2] = z

    # angles to rotation
    Rx = utils.transform.Rx(angles[0])
    Ry = utils.transform.Ry(angles[1])
    Rz = utils.transform.Rz(angles[2])
    R = Rz @ Ry @ Rx

    if ocam_model['upright']:
        # rotate coordinates by pi / 2 clockwise
        R = R @ upright_fix()

    # rotate point in unit sphere
    M = R @ M.T
    M = M.T

    # world to camera
    point2D_dist = world2cam(M, ocam_model)

    # output of world2cam is swapped
    point2D_dist[0], point2D_dist[1] = point2D_dist[1], point2D_dist[0]

    return point2D_dist


def upright_fix():
    # pi / 2 clockwise rotation around z axis
    # sin(pi / 2) = 1, cos(pi / 2) = 0
    R = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    return R
