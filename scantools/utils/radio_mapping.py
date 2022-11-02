from functools import cached_property

import numpy as np


MIN_SIGNAL_STRENGTH_DBM = -127


class RadioDescriptor:
    def __init__(self):
        self.descriptor = {}

    @cached_property
    def radio_ids(self):
        return self.descriptor.keys()

    def strength(self, radio_id):
        return self.descriptor[radio_id]

    def add_measurement(self, radio_id, rssi_dbm):
        if radio_id not in self.descriptor:
            self.descriptor[radio_id] = []
        self.descriptor[radio_id].append(rssi_dbm)

    def average(self):
        for radio_id in self.descriptor:
            self.descriptor[radio_id] = np.mean(self.descriptor[radio_id])
    
    def convert_to_numpy(self, radio_id_to_idx):
        vect = np.full(len(radio_id_to_idx), MIN_SIGNAL_STRENGTH_DBM)
        for radio_id, rssi_dbm in self.descriptor.items():
            if radio_id not in radio_id_to_idx:
                continue
            vect[radio_id_to_idx[radio_id]] = rssi_dbm
        return vect

class RadioMap:
    def __init__(self, corner, grid_size=1):
        # Save corner.
        self.corner = corner

        # Sparse 3D grid.
        self.radio_map = {}
        self.radio_observations = {}
        self.bin_idx_to_scankeys = {}
        self.bin_idx_to_imkeys = {}
        self.imkey_to_bin_idx = {}

        # Grid size.
        self.grid_size = grid_size

    def tvec_to_bin(self, tvec):
        return tuple(np.floor((tvec - self.corner) / self.grid_size).astype(int))

    def _create_bin(self, bin_idx):
        if bin_idx not in self.radio_map:
            self.radio_map[bin_idx] = RadioDescriptor()
            self.bin_idx_to_scankeys[bin_idx] = set()
            self.bin_idx_to_imkeys[bin_idx] = set()

    def _add_measurement(self, bin_idx, radio_id, rssi_dbm):
        self.radio_map[bin_idx].add_measurement(radio_id, rssi_dbm)
        if radio_id not in self.radio_observations:
            self.radio_observations[radio_id] = set()
        self.radio_observations[radio_id].add(bin_idx)

    def add_measurements(self, bin_idx, scankey, measurements):
        self._create_bin(bin_idx)
        if scankey in self.bin_idx_to_scankeys[bin_idx]:
            # Do not average the same measurement multiple times.
            return
        for radio_id in measurements:
            self._add_measurement(bin_idx, radio_id, measurements[radio_id].rssi_dbm)
        self.bin_idx_to_scankeys[bin_idx].add(scankey)

    def add_image(self, bin_idx, imkey):
        self._create_bin(bin_idx)
        self.bin_idx_to_imkeys[bin_idx].add(imkey)
        self.imkey_to_bin_idx[imkey] = bin_idx

    def finalize(self):
        # Average signal strengths.
        for bin_idx in self.radio_map:
            self.radio_map[bin_idx].average()
        # Convert to numpy.
        self.convert_to_numpy()
        # Remove old data.
        del self.radio_map
        del self.radio_observations
        del self.bin_idx_to_scankeys
        del self.imkey_to_bin_idx
    
    def convert_to_numpy(self):
        bin_indices = sorted(self.radio_map.keys())
        num_bins = len(bin_indices)
        self.idx_to_bin_idx = {
            idx: bidx for idx, bidx in enumerate(bin_indices)
        }
        radio_ids = sorted(self.radio_observations.keys())
        num_radios = len(radio_ids)
        self.radio_id_to_idx = {
            rid: idx for idx, rid in enumerate(radio_ids)
        }
        self.matrix = np.empty((num_bins, num_radios))
        for idx in range(num_bins):
            self.matrix[idx] = (
                self.radio_map[bin_indices[idx]].convert_to_numpy(
                    self.radio_id_to_idx))


def recover_measurements_for_timestamp(query_ts, session_radio, sensor_id,
                                       max_delay_us=10_000_000, past=False):
    if session_radio is None:
        return {}
    timestamps = session_radio.sorted_unique_timestamps
    start_idx = np.searchsorted(timestamps, query_ts - max_delay_us, side='left')
    measurements = {}
    for idx in range(start_idx, len(timestamps)):
        if np.abs(timestamps[idx] - query_ts) > max_delay_us:
            break
        if past and timestamps[idx] > query_ts:
            break
        scan_key = (int(timestamps[idx]), sensor_id)
        if scan_key in session_radio:
            measurements[scan_key] = session_radio[scan_key]
    return measurements


def build_radio_map(session, max_delay_us=10_000_000):
    trajectory = session.trajectories
    if session.proc.subsessions:
        prefixes = [f'{subsession}/' for subsession in session.proc.subsessions]
    else:
        prefixes = ['']
    tvecs = [trajectory[key].t for key in trajectory.key_pairs()]
    radio_map = RadioMap(np.floor(np.min(tvecs, axis=0)).astype(int))
    no_radio_imkeys = []
    valid_bins = []
    for prefix in prefixes:
        for imkey in trajectory.key_pairs():
            if not imkey[1].startswith(prefix):
                continue
            ts = imkey[0]
            bin_idx = radio_map.tvec_to_bin(trajectory[imkey].t)
            wifi_measurements = recover_measurements_for_timestamp(
                ts, session.wifi, f'{prefix}wifi_sensor', max_delay_us=max_delay_us)
            bt_measurements = recover_measurements_for_timestamp(
                ts, session.bt, f'{prefix}bt_sensor', max_delay_us=max_delay_us)
            if len(wifi_measurements) + len(bt_measurements) == 0:
                no_radio_imkeys.append(imkey)
                continue
            for scan_key, measurements in wifi_measurements.items():
                radio_map.add_measurements(bin_idx, scan_key, measurements)
            for scan_key, measurements in bt_measurements.items():
                radio_map.add_measurements(bin_idx, scan_key, measurements)
            if session.rigs and imkey[1] in session.rigs:
                for cam_id in session.rigs[imkey[1]]:
                    radio_map.add_image(bin_idx, (imkey[0], cam_id))
            else:
                radio_map.add_image(bin_idx, imkey)
            valid_bins.append(bin_idx)
    # Associate images without radios to nearest valid bin.
    valid_bins = np.array(valid_bins)
    for imkey in no_radio_imkeys:
        bin_idx = radio_map.tvec_to_bin(trajectory[imkey].t)
        index = np.argmin(np.linalg.norm(
            valid_bins - np.array(bin_idx)[np.newaxis, :], axis=-1))
        bin_idx = tuple(valid_bins[index])
        if session.rigs and imkey[1] in session.rigs:
            for cam_id in session.rigs[imkey[1]]:
                radio_map.add_image(bin_idx, (imkey[0], cam_id))
        else:
            radio_map.add_image(bin_idx, imkey)
    radio_map.finalize()
    return radio_map


def build_query_descriptor(imkey, session, max_delay_us=10_000_000):
    if '/' in imkey[1]:
        subsession = imkey[1].split('/')[0]
        prefix = f'{subsession}/'
    else:
        prefix = ''
    descriptor = RadioDescriptor()
    wifi_measurements = recover_measurements_for_timestamp(
        imkey[0], session.wifi, f'{prefix}wifi_sensor',
        max_delay_us=max_delay_us, past=True)
    for measurements in wifi_measurements.values():
        for radio_id in measurements:
            descriptor.add_measurement(radio_id, measurements[radio_id].rssi_dbm)
    bt_measurements = recover_measurements_for_timestamp(
        imkey[0], session.bt, f'{prefix}bt_sensor',
        max_delay_us=max_delay_us, past=True)
    for measurements in bt_measurements.values():
        for radio_id in measurements:
            descriptor.add_measurement(radio_id, measurements[radio_id].rssi_dbm)
    descriptor.average()
    return descriptor


def retrieve_relevant_map_images(descriptor, radio_map, num_images=250):
    # Example usage:
    # radio_map = build_radio_map(session_ref)
    # descriptor_q = build_query_descriptor(imkey, session_q)
    # images, dists = retrieve_relevant_map_images(descriptor_q, radio_map)

    vect = descriptor.convert_to_numpy(radio_map.radio_id_to_idx)
    valid_coords = np.where(vect > MIN_SIGNAL_STRENGTH_DBM)[0]
    if len(valid_coords) == 0:
        return [], []
    distances = np.linalg.norm(
        vect[valid_coords][np.newaxis] - radio_map.matrix[:, valid_coords],
        axis=1)
    max_distance = np.linalg.norm(
        np.full(len(valid_coords), MIN_SIGNAL_STRENGTH_DBM) - vect[valid_coords])
    distances = [(dist, idx) for idx, dist in enumerate(distances)]

    # Sort distances and build final retrieval list.
    distances = sorted(distances)
    images = []
    dists = []
    last_distance = np.inf
    for distance, idx in distances:
        if distance > max_distance - 1e-6:
            # Don't retrieve images with no radio overlap.
            break
        bin_idx = radio_map.idx_to_bin_idx[idx]
        if distance > last_distance + 1e-6:
            # Stop retrieving once we surpasses the last distance.
            break
        imkeys = radio_map.bin_idx_to_imkeys[bin_idx]
        images.extend(imkeys)
        dists.extend([distance] * len(imkeys))
        if len(images) > num_images:
            last_distance = distance

    return images, dists
