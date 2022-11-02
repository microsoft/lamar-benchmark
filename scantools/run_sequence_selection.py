from typing import Tuple, Set, Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt

from . import logger
from .capture import Capture, Trajectories
from .capture.session import Device
from .utils.tagging import is_session_night, get_session_date
from .viz.qualitymap import LidarMapPlotter, MapPlotter, plot_legend


def select_sessions(pool: Set[str], min_duration_second: float,
                    sid2duration: Dict[str, float], seed: int = 1) -> Tuple[float, Set[str]]:
    ids_select = set()
    total = 0
    state = np.random.RandomState(seed)
    while total < min_duration_second and len(pool) > 0:
        i = state.choice(list(pool))
        ids_select.add(i)
        total += sid2duration[i]
        pool.remove(i)
    return total, ids_select


def plot_failures(plotter: MapPlotter, sid2centers, fail_ids: Set[str]):
    ax = plotter.plot_2d()
    for i, centers in sid2centers.items():
        fail = i in fail_ids
        color = 'r' if fail else 'b'
        zorder = 3 if fail else 2
        label = 'fail' if fail else 'success'
        ax.plot(*centers.T[plotter.masks[0]], c=color, label=label, linewidth=0.5, zorder=zorder)
    plot_legend(ax, loc='best')


def plot_selection(plotter: MapPlotter, sid2centers, fail_ids: Set[str], select_ids: Set[str]):
    ax = plotter.plot_2d()
    for i, centers in sid2centers.items():
        if i in fail_ids:
            continue
        select = i in select_ids
        color = 'r' if select else 'b'
        zorder = 3 if select else 2
        label = 'select' if select else 'discard'
        ax.plot(*centers.T[plotter.masks[0]], c=color, linewidth=0.5, label=label, zorder=zorder)
    plot_legend(ax, loc='best')


def run(capture: Capture, sequence_ids: List[str], ref_id: str, visualize: bool = True,
        duration_total: float = 3*3600, duration_night: float = 3600,
        fail_thresh: Optional[Tuple[float]] = None):
    scene = capture.path.name

    ids_night = set(filter(is_session_night, sequence_ids))
    if scene in ['CVG_v2', 'ETH_HG_v3']:
        probably_day = {i for i in ids_night if not (5 < get_session_date(i).hour < 23)
                        and Device.from_id(i) == Device.HOLOLENS}
        ids_night -= probably_day

    # Gather per-sequence and per-image pose uncertainty
    sid2uncs = {}
    sid2centers = {}
    for i in sequence_ids:
        # dirty: we don't know whether alignment_trajectories stores the BA or refined traj
        # traj = capture.sessions[i].proc.alignment_trajectories
        traj = Trajectories.load(capture.registration_path() / i / ref_id / 'trajectory_ba.txt')
        keys = traj.key_pairs()
        sid2centers[i] = np.array([traj[k].t for k in keys])
        covars = np.stack([traj[k].covar[3:, 3:] for k in keys])
        sid2uncs[i] = uncs = np.sqrt(np.linalg.eig(covars)[0].max(1))

    if visualize:
        plotter = LidarMapPlotter(
            capture.data_path(ref_id)/capture.sessions[ref_id].pointclouds[0, 'point_cloud_final'],
            centers=np.concatenate(list(sid2centers.values())))
        max_unc = {'Lindenhof_v2': 30, 'CVG_v2': 20, 'ETH_HG_v3': 20}[scene]
        plotter.plot_uncertainties(sid2centers, sid2uncs, max_unc)
        plt.savefig(f'ba_uncertainties_t_{scene}.pdf')
        plt.close()

    # automatically detect failed registrations based on high uncertainties
    if fail_thresh is None:
        fail_thresh = {'Lindenhof_v2': (0.3, 20),
                       'CVG_v2': (0.2, 20),
                       'ETH_HG_v3': (0.2, 20)}[scene]
    fail_ids = set()
    for i, uncs in sid2uncs.items():
        timestamps = np.array(list(capture.sessions[i].trajectories.keys()))
        durations = np.diff(timestamps / 1e6)
        durations = np.r_[durations, durations[-1]]
        fail = np.sum(durations[uncs > fail_thresh[0]]) > fail_thresh[1]
        if fail:
            fail_ids.add(i)
    if visualize:
        plot_failures(plotter, sid2centers, fail_ids)
        plt.savefig(f'failed_registrations_{scene}.pdf')
        plt.close()

    # select sequences to meet a target recording duration
    sid2duration = {}
    for i in sequence_ids:
        timestamps = np.array(list(capture.sessions[i].trajectories.keys()))
        sid2duration[i] = (max(timestamps) - min(timestamps)) / 1e6

    hl_ids = {i for i in sequence_ids if Device.from_id(i) == Device.HOLOLENS}
    phone_ids = {i for i in sequence_ids if Device.from_id(i) == Device.PHONE}
    logger.info('From %d sequences (%d HL / %d phone), found %d failed registrations.',
                len(sequence_ids), len(hl_ids), len(phone_ids), len(fail_ids))

    def count_images_sessions(sids):
        return sum(len(capture.sessions[i].images.key_pairs()) for i in sids)

    ids_select = {}
    for label, pool in [['hololens', hl_ids], ['phone', phone_ids]]:
        pool_night = (pool & ids_night) - fail_ids
        total, ids_select_night = select_sessions(pool_night, duration_night, sid2duration)
        logger.info('%s night: %d sequences, %.1fs, %d images.',
                    label, len(ids_select_night), total, count_images_sessions(ids_select_night))

        pool_day = pool - ids_night - fail_ids
        total, ids_select_day = select_sessions(pool_day, duration_total-total, sid2duration)
        logger.info('%s day: %d sequences, %.1fs, %d images.',
                    label, len(ids_select_day), total, count_images_sessions(ids_select_day))

        ids_select |= ids_select_night | ids_select_day

    if visualize:
        plot_selection(plotter, sid2centers, fail_ids, ids_select)
        plt.savefig(f'selected_sequences_{scene}.pdf')
        plt.close()

    with open(capture.path / 'sequences_select.txt', 'w') as fid:
        fid.write("\n".join(ids_select))
