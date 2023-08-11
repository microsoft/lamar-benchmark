import argparse
from bisect import bisect_left
from pathlib import Path
import json
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.backend_bases import MouseButton

from . import logger
from .capture import Capture
from .utils.io import read_image, write_image
from .proc.anonymization import blur_detections


class ButtonHandler:
    def __init__(self, fig, ax, capture, session_id, keys, anon_keys):
        self.fig = fig
        self.ax = ax

        self.capture = capture
        self.session = capture.sessions[session_id]
        self.session_id = session_id

        self.idx = 0
        self.anon_idx = -1

        self.keys = keys
        self.anon_keys = anon_keys

        self.artists = []
        self.annotations = []
        self.current_annotation = []

    # This function is called when banonymize is clicked
    def anonymize(self, _):
        cam_id, ts = self.keys[self.idx]
        file_path = self.capture.data_path(self.session_id) / self.session.images[ts, cam_id]
        image = read_image(file_path)
        if self.annotations:
            blurred, _ = blur_detections(image, self.annotations)
            self.fig.set_data(blurred)
            plt.draw()
            json_path = self.capture.path / 'anonymization' / self.session_id / 'manual.json'
            ann = {'key': (ts, cam_id), 'bounding_boxes': self.annotations}
            with open(json_path, 'a') as fid:
                json.dump(ann, fid)
                fid.write('\n')
            write_image(file_path, blurred)

    # This function is called when bprev is clicked
    def previous(self, _):
        num_keys = len(self.keys)
        self.idx = max(self.idx - 1, 0)
        cam_id, ts = self.keys[self.idx]
        file_path = self.capture.data_path(self.session_id) / self.session.images[ts, cam_id]
        image = read_image(file_path)
        self.fig.set_data(image)
        self.ax.set_title(f'{self.idx + 1:04d} / {num_keys:04d}\n{ts}, {cam_id.split("/")[1]}')
        while self.artists:
            self.artists.pop().remove()
        self.annotations = []
        self.current_annotation = []
        plt.draw()

    # This function is called when bnext is clicked
    def next(self, _):
        num_keys = len(self.keys)
        self.idx = min(self.idx + 1, num_keys - 1)
        cam_id, ts = self.keys[self.idx]
        file_path = self.capture.data_path(self.session_id) / self.session.images[ts, cam_id]
        image = read_image(file_path)
        self.fig.set_data(image)
        self.ax.set_title(f'{self.idx + 1:04d} / {num_keys:04d}\n{ts}, {cam_id.split("/")[1]}')
        while self.artists:
            self.artists.pop().remove()
        self.annotations = []
        self.current_annotation = []
        plt.draw()

    def fast_backward(self, _):
        if self.anon_keys:
            self.anon_idx = max(self.anon_idx - 1, 0)
            self.idx = bisect_left(self.keys, self.anon_keys[self.anon_idx]) - 1
            self.next(None)

    def fast_forward(self, _):
        if self.anon_keys:
            self.anon_idx = min(self.anon_idx + 1, len(self.anon_keys) - 1)
            self.idx = bisect_left(self.keys, self.anon_keys[self.anon_idx]) - 1
            self.next(None)

    def on_key_press(self, event):
        if event.key == 'x':
            self.previous(None)
        elif event.key == 'c':
            self.next(None)
        elif event.key == 'a':
            self.anonymize(None)
        elif event.key == 'd':
            self.fast_backward(None)
        elif event.key == 'f':
            self.fast_forward(None)

    def on_click(self, event):
        if event.inaxes == self.ax:
            if event.button is MouseButton.RIGHT:
                if self.annotations:
                    self.annotations.pop()
                    self.artists[-1].remove()
                    self.artists.pop()
                    plt.draw()
            else:
                self.current_annotation.extend([event.xdata, event.ydata])

    def on_release(self, event):
        if event.button is MouseButton.LEFT:
            if event.inaxes == self.ax:
                self.current_annotation.extend([event.xdata, event.ydata])
                self.annotations.append(self.current_annotation)
                self.current_annotation = []
                mx, my, Mx, My = self.annotations[-1]
                bbox, = self.ax.plot([mx, mx, Mx, Mx, mx], [my, My, My, my, my])
                self.artists.append(bbox)
                plt.draw()
            else:
                # Reset latest annotation.
                self.current_annotation = []


def run(capture, session_id):
    logger.info('Controls:')
    logger.info('x - previous frame')
    logger.info('c - next frame')
    logger.info('d - jump to previous detection')
    logger.info('f - jump to next detection')
    logger.info('a - anonymize frame')
    logger.info('left click hold - draw bounding box')
    logger.info('right click - undo last drawn bounding box')

    session = capture.sessions[session_id]
    anon_path = capture.path / 'anonymization' / session_id

    for subsession_id in tqdm(session.proc.subsessions):
        s_anon_path = anon_path / subsession_id
        labels_path = s_anon_path / 'labels.json'
        list_path = s_anon_path / 'list.txt'
        anon_files = set()
        if labels_path.exists():
            with open(labels_path, 'r') as f:
                labels = json.load(f)
            with open(list_path, 'r') as f:
                files = f.readlines()
                files = ['/'.join(line.split("'")[1].split('/')[8 :]) for line in files]
            for frame in labels['frames']:
                frame_idx = frame['index'] - 1
                file_path = capture.path / files[frame_idx]
                if len(frame['faces']) or len(frame['license_plates']):
                    anon_files.add('/'.join(files[frame_idx].split('/')[3:]))
        else:
            logger.warning('No BrighterAI data for subsession %s', subsession_id)

        anon_keys = []
        keys = []
        for key in sorted(session.images.key_pairs()):
            if key[1].startswith(subsession_id):
                keys.append((key[1], key[0]))
                if session.images[key] in anon_files:
                    anon_keys.append((key[1], key[0]))
        
        # Group cameras together.
        keys = sorted(keys)
        anon_keys = sorted(anon_keys)
        logger.info(
            'Processing subsession %s with %d images '
            '(%d images with detections from BrighterAI).',
            subsession_id, len(keys), len(anon_keys))

        # Set up plot.
        # Adjust bottom to make room for Buttons
        _, ax = plt.subplots(dpi=150)
        plt.subplots_adjust(bottom=0.15)

        # Plot Graph 1 and set axes and title
        cam_id, ts = keys[0]
        file_path = capture.data_path(session_id) / session.images[ts, cam_id]
        img = read_image(file_path)
        fig = plt.imshow(img)
        ax.set_title(f'{1:04d} / {len(keys):04d}\n{ts}, {cam_id.split("/")[1]}')
        ax.axis('off')

        # Initialize Button handler object
        callback = ButtonHandler(fig, ax, capture, session_id, keys, anon_keys)

        # Connect to a "Anonymize" Button, setting its left, top, width, and height
        axanonymize = plt.axes([0.40, 0.07, 0.2, 0.05])
        banonymize = Button(axanonymize, 'Anonymize')
        banonymize.on_clicked(callback.anonymize)

        # Connect to a "prev" Button, setting its left, top, width, and height
        axprev = plt.axes([0.29, 0.01, 0.2, 0.05])
        bprev = Button(axprev, 'Prev')
        bprev.on_clicked(callback.previous)

        # Connect to a "next" Button, setting its left, top, width, and height
        axnext = plt.axes([0.51, 0.01, 0.2, 0.05])
        bnext = Button(axnext, 'Next')
        bnext.on_clicked(callback.next)

        plt.connect('button_press_event', callback.on_click)
        plt.connect('button_release_event', callback.on_release)
        plt.connect('key_press_event', callback.on_key_press)

        # Show
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--capture_path', type=Path, required=True)
    parser.add_argument('--session_id', type=str, required=True)
    args = parser.parse_args().__dict__
    args['capture'] = Capture.load(args.pop('capture_path'))
    run(**args)
