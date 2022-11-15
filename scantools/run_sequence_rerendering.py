import argparse
import copy
from pathlib import Path
from typing import List
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from . import logger
from .capture import Capture, Trajectories, Proc, Pose
from .utils.misc import add_bool_arg

from .proc.rendering import Renderer
from .utils.io import read_mesh, read_image
from .viz.image import save_plot


# Button handler object
class ButtonHandler:
    def __init__(self, fig, trajectory, session, data_path, renderer, skip=5):
        self.idx = 0
        self.render = False

        self.fig = fig
        self.keys = list(sorted(trajectory.key_pairs()))
        self.trajectory = trajectory
        self.session = session
        self.data_path = data_path
        self.renderer = renderer
        self.skip = skip

    # This function is called when bswitch is clicked
    def switch(self, event):
        self.render = not self.render
        if self.render:
            ts, cam_id = self.keys[self.idx]
            image = render_image(
                cam_id, self.trajectory[ts, cam_id], self.session.images[ts], self.session.sensors,
                self.renderer, rig=(self.session.rigs[cam_id] if cam_id in self.session.rigs else None)
            )
        else:
            ts, cam_id = self.keys[self.idx]
            image = read_raw_image(cam_id, self.data_path, self.session.images[ts])
        self.fig.set_data(image)
        plt.draw()
    
    # This function is called when bprev is clicked
    def previous(self, event):
        self.idx = (self.idx - self.skip + len(self.keys)) % len(self.keys)
        self.render = False
        ts, cam_id = self.keys[self.idx]
        image = read_raw_image(cam_id, self.data_path, self.session.images[ts])
        self.fig.set_data(image)
        plt.draw()

    # This function is called when bnext is clicked
    def next(self, event):
        self.idx = (self.idx + self.skip) % len(self.keys)
        self.render = False
        ts, cam_id = self.keys[self.idx]
        image = read_raw_image(cam_id, self.data_path, self.session.images[ts])
        self.fig.set_data(image)
        plt.draw()

    # This function is called when bquit is clicked
    def quit(self, event):
        plt.close()


def read_raw_image(cam_id, data_path, images):
    if cam_id not in images:
        # It's a rig. Pick first camera.
        cam_id = list(sorted(images.keys()))[0]
    return read_image(data_path / images[cam_id])


def render_image(cam_id, T, images, cameras, renderer, rig=None):
    T = copy.deepcopy(T)
    if cam_id not in images:
        # It's a rig. Pick first camera.
        cam_id = list(sorted(images.keys()))[0]
        T_cam2rig = rig[cam_id]
        T = T * T_cam2rig
    camera = cameras[cam_id]
    render, _ = renderer.render_from_capture(T, camera)
    render = (np.clip(render, 0, 1) * 255).astype(np.uint8)
    return render


def run(capture: Capture, ref_id: str, query_id: str, skip: int):
    # TODO: add support for multi reference in renderer.
    session_ref = capture.sessions[ref_id]
    T_mesh2global = session_ref.proc.alignment_global.get_abs_pose('pose_graph_optimized')
    session_q = capture.sessions[query_id]

    logger.info('Generating interactive visualization diffs by rendering.')
    mesh = read_mesh(capture.proc_path(ref_id) / session_ref.proc.meshes['mesh'])
    renderer = Renderer(mesh)
    trajectory = session_q.proc.alignment_trajectories
    if T_mesh2global is not None:
        trajectory = T_mesh2global.inv() * trajectory

    # Set up plot.
    # Adjust bottom to make room for Buttons
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    # Plot Graph 1 and set axes and title
    ts, cam_id = sorted(trajectory.key_pairs())[0]
    fig = plt.imshow(read_raw_image(cam_id, capture.data_path(query_id), session_q.images[ts]))
    ax.axis('off')

    # Initialize Button handler object
    callback = ButtonHandler(fig, trajectory, session_q, capture.data_path(query_id), renderer, skip)

    # Connect to a "switch" Button, setting its left, top, width, and height
    axswitch = plt.axes([0.40, 0.13, 0.2, 0.05])
    bswitch = Button(axswitch, 'Switch')
    bswitch.on_clicked(callback.switch)

    # Connect to a "prev" Button, setting its left, top, width, and height
    axprev = plt.axes([0.25, 0.07, 0.2, 0.05])
    bprev = Button(axprev, 'Prev')
    bprev.on_clicked(callback.previous)

    # Connect to a "next" Button, setting its left, top, width, and height
    axnext = plt.axes([0.55, 0.07, 0.2, 0.05])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next)

    # Connect to a "quit" Button, setting its left, top, width, and height
    axquit = plt.axes([0.40, 0.01, 0.2, 0.05])
    bquit = Button(axquit, 'Quit')
    bquit.on_clicked(callback.quit)

    # Show
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     argument_default=argparse.SUPPRESS)
    parser.add_argument('--capture_path', type=Path, required=True)
    parser.add_argument('--query_id', type=str, required=True)
    parser.add_argument('--ref_id', type=str, required=True)
    parser.add_argument('--skip', type=int, default=5)
    args = parser.parse_args().__dict__
    args['capture'] = Capture.load(args.pop('capture_path'))

    run(**args)
