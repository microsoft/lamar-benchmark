from typing import Dict, Optional, Union
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.widgets import PolygonSelector
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pycolmap
import open3d as o3d

from ..capture import Capture, Trajectories
from ..utils.io import read_pointcloud


def rasterize(xy, cameras=None, resolution=0.2, margin=10, max_count=10):
    xy = xy[:, :2]  # pick first 2 dimensions if 3D
    if not isinstance(margin, np.ndarray):
        margin = np.array(((margin,)*2,)*2)
    if cameras is None:  # use bounds of 3D points instead of cameras
        cameras = xy
    cameras = cameras[:, :2]
    min_ = cameras.min(0) - margin[0]
    max_ = cameras.max(0) + margin[1]
    raster_size = ((max_ - min_)[::-1] // resolution).astype(int)+1

    def to_ij(xy):
        uv = (xy - min_) / (max_ - min_)
        valid = np.all((uv >= 0) & (uv < 1), -1)
        ij = np.stack([1 - uv[..., 1], uv[..., 0]], -1)
        ij = np.floor(ij*np.array(raster_size)).astype(int)
        return ij, valid

    ij, valid = to_ij(xy)
    ij_select, counts = np.unique(ij[valid], axis=0, return_counts=True)
    raster = np.zeros(raster_size)
    raster[tuple(ij_select.T)] = np.clip(counts/max_count, 0, 1)
    extent = [min_[0], max_[0], min_[1], max_[1]]
    return raster, extent


def plot_legend(ax, ps=6.0, lw=1.0, label_order=None, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if label_order is not None:
        by_label = {k: by_label[k] for k in label_order if k in by_label}
    lgnd = ax.legend(by_label.values(), by_label.keys(), **kwargs)
    for handle in lgnd.legendHandles:
        if isinstance(handle, mpl.collections.PathCollection):
            handle.set_sizes([ps])
        elif isinstance(handle, mpl.lines.Line2D):
            handle.set_linewidth(lw)


def set_plot_limits(ax, centers, margin=5):
    if isinstance(margin, int):
        margin = (margin,)*4
    min_, max_ = np.percentile(centers, [0, 100], 0)
    ax.set_xlim(min_[0]-margin[0], max_[0]+margin[1])
    ax.set_ylim(min_[1]-margin[2], max_[1]+margin[3])


class MapPlotter:
    def __init__(self, xyz, centers=None, masks=((0, 1), (2, 1), (0, 2)), max_count=(10,)*3,
                 **kwargs):
        self.masks = list(map(list, masks))
        self.rasters = []
        self.extents = []
        for i, m in enumerate(self.masks):
            c = centers[:, m] if centers is not None else None
            r, e = rasterize(xyz[:, m], c, max_count=max_count[i], **kwargs)
            self.rasters.append(r)
            self.extents.append(e)

    def plot_2d(self, index=0, dpi=200, cmap='gray', alpha=1.0, rotation=None, **kwargs):
        _, ax = plt.subplots(1, 1, dpi=dpi, **kwargs)
        im = ax.imshow(1-self.rasters[index], extent=self.extents[index], cmap=cmap, alpha=alpha)
        ax.set_aspect('equal')
        if rotation is None:
            return ax
        tfm = mpl.transforms.Affine2D().rotate_deg_around(0, 0, rotation) + ax.transData
        im.set_transform(tfm)
        return ax, tfm

    def plot_3d(self, dpi=200, cmap='gray', alpha=1.0, hspace=0.02, wspace=-0.29, **kwargs):
        fig = plt.figure(dpi=dpi, **kwargs)
        gs_kw = {
            'width_ratios': [np.diff(self.extents[0][:2]), np.diff(self.extents[1][:2])],
            'height_ratios': [np.diff(self.extents[0][2:]), np.diff(self.extents[2][2:])]
        }
        gs = fig.add_gridspec(2, 2, hspace=hspace, wspace=wspace, **gs_kw)

        ax_xy = fig.add_subplot(gs[0, 0])
        ax_zy = fig.add_subplot(gs[0, 1])
        ax_xz = fig.add_subplot(gs[1, 0])
        axs = [ax_xy, ax_zy, ax_xz]

        for ax, r, ex in zip(axs, self.rasters, self.extents):
            ax.imshow(1-r, extent=ex, cmap=cmap, alpha=alpha)
        ax_xz.sharex(ax_xy)
        ax_zy.sharey(ax_xy)
        ax_xy.set_aspect('equal')
        ax_zy.invert_xaxis()

        ax_xy.xaxis.tick_top()
        ax_zy.xaxis.tick_top()
        ax_zy.yaxis.tick_right()
        ax_xy.xaxis.set_label_position('top')
        ax_zy.xaxis.set_label_position('top')
        ax_zy.yaxis.set_label_position('right')

        ax_xy.set_ylabel('y')
        ax_xy.set_xlabel('x')
        ax_xz.set_ylabel('z')
        ax_zy.set_xlabel('z')

        for ax in axs:
            ax.tick_params(axis='both', which='major', labelsize=6)
        ax_zy.tick_params(axis='y', labelright=False)
        return axs

    def plot_uncertainties(self, sid2centers: Dict[str, np.ndarray],
                           sid2uncs: Dict[str, float], max_unc_cm: float,
                           index: int = 0, as_lines: bool = False, **kwargs):
        ax = self.plot_2d(index, **kwargs)
        cmap = dict(cmap='turbo', norm=plt.Normalize(0, max_unc_cm))
        if as_lines:
            for i in sid2centers:
                centers = sid2centers[i][:, self.masks[index]]
                uncs = sid2uncs[i]
                segments = np.stack([centers[:-1], centers[1:]], 1)
                lc = LineCollection(segments, **cmap)
                lc.set_array((uncs[:-1]+uncs[1:])/2*100)
                lc.set_linewidth(0.5)
                obj = ax.add_collection(lc)
        else:
            all_centers = np.concatenate(list(sid2centers.values()), 0)[:, self.masks[index]]
            all_uncs = np.concatenate(list(sid2uncs.values()), 0)
            indices = np.argsort(all_uncs)[::-1]
            obj = ax.scatter(
                *all_centers[indices].T, s=2, linewidth=0, c=all_uncs[indices]*100, **cmap)
        cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.1)
        ax.get_figure().colorbar(obj, cax=cax, format='%.0fcm')
        return ax

    def plot_trajectory(self, ax, traj: Union[Trajectories, Path],
                        color='r',
                        markers: Optional[str] = None,
                        max_unc_cm: Optional[float] = None,
                        index: int = 0,
                        label: str = None):
        if not isinstance(traj, Trajectories):
            if label is None:
                label = traj.stem
            traj = Trajectories.load(traj)
        keys = sorted(traj.key_pairs())
        centers = np.stack([traj[k].t for k in keys])
        mask = self.masks[index]
        ax.plot(*centers.T[mask], c=color, linewidth=0.4 if markers is None else 0.1, label=label)
        if markers is not None:
            if max_unc_cm and traj[keys[0]].covar is not None:
                covar = np.stack([traj[k].covar[3:, 3:] for k in keys])
                unc = np.sqrt(np.linalg.eig(covar)[0].max(1))
                color = mpl.cm.jet(np.clip(unc/max_unc_cm, 0, 1))
            ax.scatter(*centers.T[mask], c=color, s=3, linewidth=0)

    def plot_alignment_trajectories(self, ax, sequence_path: Path,
                                    max_unc_cm: Optional[float] = None,
                                    keys=('init', 'loc', 'pgo', 'reloc', 'pgo2', 'ba')):
        traj = Trajectories.load(sequence_path/'trajectory_ba.txt')
        for k in keys:
            if k == 'loc':
                self.plot_trajectory(ax, sequence_path/'trajectory_loc.txt', 'k', '.', max_unc_cm)
            elif k == 'init':
                self.plot_trajectory(ax, sequence_path/'trajectory_pgo_init.txt', 'lime')
            elif k == 'pgo':
                self.plot_trajectory(ax, sequence_path/'trajectory_pgo.txt', 'r')
            elif k == 'reloc':
                self.plot_trajectory(
                    ax, sequence_path/'trajectory_reloc.txt', 'magenta', '.', max_unc_cm)
            elif k == 'pgo2':
                self.plot_trajectory(ax, sequence_path/'trajectory_pgo2.txt', 'orange')
            elif k == 'ba':
                self.plot_trajectory(ax, traj, 'b', label='BA')
        centers = np.stack([traj[k].t for k in sorted(traj.key_pairs())])
        set_plot_limits(ax, centers)


class ColMapPlotter(MapPlotter):
    def __init__(self, rec, centers=None, filter_points=True,
                 min_track_length=4, max_error=1.0, **kwargs):
        if not isinstance(rec, pycolmap.Reconstruction):
            rec = pycolmap.Reconstruction(rec)
        points = rec.points3D.values()
        if filter_points:
            points = [
                p for p in points if p.track.length() >= min_track_length and p.error <= max_error]
        xyz = np.stack([p.xyz.astype(np.float16) for p in points])
        if centers is None:
            centers = np.stack([im.projection_center() for im in rec.images.values()])
        del rec
        super().__init__(xyz, centers, **kwargs)


class LidarMapPlotter(MapPlotter):
    def __init__(self, pcd, centers=None, masks=((0, 1),), max_count=(100,),
                 downsample_resolution=0.1, filter_with_normals: bool = True, **kwargs):
        if not isinstance(pcd, o3d.geometry.PointCloud):
            pcd = read_pointcloud(pcd)
        if downsample_resolution is not None:
            pcd = pcd.voxel_down_sample(downsample_resolution)
        if filter_with_normals:
            normals = np.asarray(pcd.normals)
            nangle = np.arcsin(np.abs(normals[:, 2]))
            select = np.where(nangle < np.deg2rad(5))[0]
            pcd = pcd.select_by_index(select)
        xyz = np.asarray(pcd.points).astype(np.float16)
        self.xyz = xyz
        del pcd
        super().__init__(xyz, centers, masks=masks, max_count=max_count, **kwargs)

    @classmethod
    def from_scan_session(cls, capture: Capture, ref_id: str, *args, **kwargs):
        pcds = capture.sessions[ref_id].pointclouds[0]
        pcd = pcds.get('point_cloud_combined', pcds.get('point_cloud_final'))
        assert pcd is not None, pcds.keys()
        return cls(capture.data_path(ref_id) / pcd, *args, **kwargs)


class PolygonAnnotator:
    def __init__(self, ax, scatter=None):
        ax.figure.canvas.mpl_connect('key_press_event', self.on_press)
        self.poly = PolygonSelector(ax, self.on_select, props=dict(linewidth=1))

        self.ax = ax
        self.current = None
        self.polygons = []
        self.scatter = scatter
        if scatter is not None:
            self.is_selected = np.full(len(scatter.get_offsets()), False)

    def on_select(self, vertices):
        self.current = vertices

    def on_press(self, event):
        if event.key == 'v' and self.current is not None:
            self.polygons.append(self.current)
            poly = mpl.patches.Polygon(self.current, ec='r', fill=False)
            self.ax.add_patch(poly)

            if self.scatter is not None:
                mask = mpl.path.Path(self.current).contains_points(self.scatter.get_offsets())
                self.is_selected[mask] = True
                color = self.scatter.get_facecolors()[..., :3]
                color = np.where(self.is_selected[:, None], [1, 0, 0], color)
                self.scatter.set_facecolors(color)
                self.ax.figure.canvas.draw_idle()


class SequenceSelector:
    def __init__(self, fig, color, color_last=None):
        self.fig = fig
        self.color = color
        self.color_last = color_last
        self.selected = []
        self.selected_artists = []
        fig.canvas.mpl_connect('pick_event', self.on_pick)

    def on_pick(self, event):
        artist = event.artist
        self.selected.append(artist.get_label())
        if self.color_last is not None:
            artist.set_color(self.color_last)
            for a in self.selected_artists:
                a.set_color(self.color)
        else:
            artist.set_color(self.color)
        artist.set_zorder(5)
        self.selected_artists.append(artist)
        event.canvas.draw()
