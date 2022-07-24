from functools import cached_property

import numpy as np


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


def radio_descriptor_distance(descriptor_q, descriptor_ref, min_signal_strength=-127):
    if len(descriptor_q.radio_ids) == 0:
        return np.inf
    score = 0
    for signal_id in descriptor_q.radio_ids:
        if signal_id not in descriptor_ref.radio_ids:
            score += (min_signal_strength - descriptor_q.strength(signal_id)) ** 2
        else:
            score += (descriptor_ref.strength(signal_id) - descriptor_q.strength(signal_id)) ** 2
    return np.sqrt(score)


def retrieve_relevant_map_images(descriptor, radio_map, num_images=250):
    # Example usage:
    # radio_map = build_radio_map(session_ref)
    # descriptor_q = build_query_descriptor(imkey, session_q)
    # images, dists = retrieve_relevant_map_images(descriptor_q, radio_map)

    # Shortlist of bins with common signals with query.
    bin_idx_shortlist = set()
    for radio_id in descriptor.radio_ids:
        if radio_id not in radio_map.radio_observations:
            continue
        for bin_idx in radio_map.radio_observations[radio_id]:
            bin_idx_shortlist.add(bin_idx)

    # Compute distances w.r.t. shortlisted bins.
    distances = []
    for bin_idx in bin_idx_shortlist:
        distances.append(
            (radio_descriptor_distance(descriptor, radio_map.radio_map[bin_idx]), bin_idx))

    # Sort distances and build final retrieval list.
    distances = sorted(distances)
    images = []
    dists = []
    bins = []
    last_distance = np.inf
    for distance, bin_idx in distances:
        bins.append(bin_idx)
        if distance > last_distance + 1e-6:
            break
        for imkey in radio_map.bin_idx_to_imkeys[bin_idx]:
            images.append(imkey)
            dists.append(distance)
        if len(images) > num_images:
            last_distance = distance

    return images, dists
