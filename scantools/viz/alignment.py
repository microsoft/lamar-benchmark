from pathlib import Path
from typing import List, Dict
from functools import partial
from copy import deepcopy
from matplotlib import cm
import numpy as np

from .meshlab import MeshlabProject
from .image import plot_images, plot_matches, cm_RdGn, cm_normals
from ..utils.io import read_image
from ..capture import Capture, Camera, Pose, Trajectories


def plot_pnp_inliers(qname: str, refs: List[str], ret: Dict, data_root: Path, num_pairs: int = 2):
    masks = [ret['indices'] == i for i in range(len(refs))]
    ninls = [np.array(ret['inliers'])[m].sum() for m in masks]
    idxs = np.argsort(ninls)[::-1][:num_pairs]

    qim = read_image(data_root / qname)
    ims = []
    titles = []
    for i in idxs:
        ref = refs[i]
        rim = read_image(data_root / ref)
        ims.extend([qim, rim])
        titles.extend([f'{ninls[i]}/{masks[i].sum()}', Path(ref).name])
    plot_images(ims, titles)

    for i, idx in enumerate(idxs):
        mask = masks[idx]
        color = cm_RdGn(np.array(ret['inliers'])[mask])
        plot_matches(
            ret['kp_q'][mask], ret['kp_r'][mask],
            color=color.tolist(), indices=(i*2, i*2+1), a=0.1)


def plot_raw_matches(qname: str, refs: List[str], match_data: Dict, data_root: Path, **args):
    match_data = {**match_data, 'inliers': np.full(len(match_data['indices']), True, bool)}
    plot_pnp_inliers(qname, refs, match_data, data_root, **args)


def plot_pnp_inliers_rig(qnames: str, retrieval: Dict[str, List[str]], ret: Dict,
                         data_root: Path):
    masks = []
    inlier_masks = []
    offset = 0
    ims = []
    titles = []
    for i, qname in enumerate(qnames):
        if qname not in retrieval or len(retrieval[qname]) == 0:
            continue
        refs = retrieval[qname]
        indices = ret['matches'][i]['indices']
        inl_masks = []
        for j, ref in enumerate(refs):
            mask = indices == j
            inlier_mask = np.array(ret['inliers'][offset:offset+len(mask)])[mask]
            inl_masks.append(inlier_mask)
        idx = np.argmax([m.sum() for m in inl_masks])
        inlier_masks.append(inl_masks[idx])
        masks.append(indices == idx)

        ref = refs[idx]
        ims.extend([read_image(data_root / qname), read_image(data_root / ref)])
        titles.extend([Path(qname).name, Path(ref).name])
        offset += len(indices)
    plot_images(ims, titles)

    for i, (mask, inlier_mask) in enumerate(zip(masks, inlier_masks)):
        color = cm_RdGn(inlier_mask)
        plot_matches(
            ret['matches'][i]['kp_q'][mask], ret['matches'][i]['kp_r'][mask],
            color=color.tolist(), indices=(i*2, i*2+1), a=0.1)


def plot_rendering_diff(renderer, path: Path, camera: Camera, pose: Pose):
    im = read_image(path)
    render, depth = renderer.render_from_capture(pose, camera)
    if im.ndim == 2:  # gray
        render = render.mean(-1)
    mask = depth > 0
    render *= np.mean(im[mask])/255. / np.mean(render[mask])
    render = (np.clip(render, 0, 1)*255).astype(np.uint8)
    ov = np.abs(render.astype(int) - im)
    tiled = np.concatenate([render, im, np.clip(ov, 0, 255).astype(np.uint8)], 1)
    return tiled


def plot_normal_overlay(renderer, path: Path, camera: Camera, pose: Pose, a: float = 0.2):
    im = read_image(path)
    if im.ndim == 2:  # gray
        im = np.repeat(im[:, :, None], 3, 2)
    render, depth, normals = renderer.render_from_capture(pose, camera, with_normals=True)
    normals = cm_normals(normals)
    invalid = depth <= 0
    render[invalid] = 1
    normals[invalid] = 1
    ov = np.clip(normals*255*a + im*(1-a), 0, 255).astype(np.uint8)
    render = (np.clip(render, 0, 1)*255).astype(np.uint8)
    tiled = np.concatenate([im, ov, render], 1)
    return tiled


def plot_sequence_trajectories(mlp: MeshlabProject,
                               capture: Capture,
                               sessionid2poses: Dict[str, Trajectories]):
    ids = list(sessionid2poses)
    t = np.linspace(0, 1, len(ids), endpoint=False)  # equidistant on hue ring
    np.random.RandomState(0).shuffle(t)  # shuffle to spread the colors
    colors = cm.hsv(t)[:, :3]
    for idx, session_id in enumerate(ids):
        mlp.add_trajectory(
            session_id,
            sessionid2poses[session_id],
            capture.sessions[session_id],
            partial(lambda x, i: colors[i][None].repeat(len(x), 0), i=idx))


def colmap_reconstruction_to_ply(path: Path,
                                 reconstruction,
                                 min_track_length: int = 3,
                                 max_reprojection_error: float = 1.0):
    rec = deepcopy(reconstruction)
    to_remove = []
    for i, p in rec.points3D.items():
        if p.track.length() < min_track_length or p.error > max_reprojection_error:
            to_remove.append(i)
    for i in to_remove:
        del rec.points3D[i]
    rec.export_PLY(str(path))
