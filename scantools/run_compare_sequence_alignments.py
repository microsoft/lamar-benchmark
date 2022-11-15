import argparse
from pathlib import Path
from typing import List
import numpy as np
import matplotlib.pyplot as plt

from .capture import Capture


def autolabel(ax, rects, labels, ylim=1):
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2.,
            height + 0.01 * ylim,
            label,
            ha='center', va='bottom', rotation=0
        )


def run(capture: Capture, session_ids: List[str]):
    session_ids = sorted(session_ids)
    locations = {}
    for session_id in session_ids:
        session = capture.sessions[session_id]
        assert(session.proc.alignment_trajectories is not None)
        trajectory = session.proc.alignment_trajectories
        for ts, sensor in trajectory.key_pairs():
            if (ts, sensor) not in locations:
                locations[ts, sensor] = []
            locations[ts, sensor].append(trajectory[ts, sensor].t)
    
    err_t = []
    for ts, sensor in locations:
        locs = np.array(locations[ts, sensor])
        assert(len(locs) == len(session_ids))
        # Average error w.r.t. average position.
        mean = np.mean(locs, axis=0)
        err_t.append(np.mean(np.linalg.norm(locs - mean[np.newaxis], axis=1)))
    err_t = np.array(err_t)

    lim = 0.10
    n_bins = 10
    err_t[err_t > lim] = lim
    counts, bins = np.histogram(err_t, range=[0, lim], bins=n_bins)
    fig, ax = plt.subplots()
    bars = plt.bar((bins[: -1] + bins[1 :]) / 2 * 100, counts / len(err_t), width=1.0)
    autolabel(ax, bars, counts, ylim=1)
    plt.ylabel('% of images')
    plt.xlabel('difference (cm)')
    plt.xticks(np.linspace(0, lim, n_bins + 1) * 100)
    plt.ylim([0, 1])
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--capture_path', type=Path, required=True)
    parser.add_argument('--session_ids', type=str, nargs='+', required=True)
    args = parser.parse_args().__dict__
    args['capture'] = Capture.load(args.pop('capture_path'))
    run(**args)
