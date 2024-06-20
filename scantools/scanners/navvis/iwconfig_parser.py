import collections

from pytijo import parser
import re

IWConfigData = collections.namedtuple('IWConfigData',
                                      ['mac_address',
                                       'signal_strength_dbm',
                                       'frequency_khz',
                                       'time_offset_ms',
                                       'ssid'])

WifiMeasurement = collections.namedtuple(
    'WifiMeasurement', [
        'timestamp_s',
        'mac_address',
        'signal_strength_dbm',
        'center_channel_freq_khz',
        'time_offset_ms',
        'ssid'])

def frequency_string_to_khz(frequency_str):

    data = frequency_str.split()
    assert 1 <= len(data) <= 2

    frequency = float(data[0])
    assert frequency >= 0

    if len(data) == 2:
        unit = data[1]
        assert unit in ('GHz', 'MHz', 'kHz')

        if unit == 'GHz':
            frequency *= 1000_000.0
        elif unit == 'MHz':
            frequency *= 1000.0
    else:
        # New format report MHz without units
        frequency *= 1000.0

    return int(frequency)


def parse_iwconfig(data):

    structures = [
        {
            'wifi samples': [{
                '#id': r'(Cell \d{1,3})',
                'mac_address': r'Address:\s((?:[a-fA-F0-9]{2}[:|\-]?){6})',
                'signal_level': r'Signal level=(-?\d{1,2}) dBm',
                'frequency': r'Frequency:(\d{1,2}.\d{1,6}\s([ G|M|k]Hz))',
                'time_offset': r'Extra: Last beacon: (\d{1,6})ms',
                'ssid': r'ESSID:"(.{1,32})"'
            }]
        }, {
            'wifi samples': [{
                '#id': r'(BSS (?:[a-fA-F0-9]{2}[:|\-]?){6})',
                'mac_address': r'\s((?:[a-fA-F0-9]{2}[:|\-]?){6})',
                'signal_level': r'signal: (-?\d{1,2}.\d{1,2}) dBm',
                'frequency': r'freq: (\d{1,4})',
                'time_offset': r'last seen: (\d{1,6}) ms',
                'ssid': r'SSID: (.{1,32})'
            }]
        }]

    parsed_samples = []
    for structure in structures:
        parsed = parser.parse(data, structure)
        if parsed['wifi samples'] is not None:
            break
    if parsed['wifi samples'] is None:
        print('Warning - no wifi data found')
        return parsed_samples

    for wifi_sample in parsed['wifi samples']:
        assert wifi_sample['mac_address'] is not None
        assert wifi_sample['signal_level'] is not None
        assert -127 <= float(wifi_sample['signal_level']) <= 127
        assert wifi_sample['frequency'] is not None
        assert wifi_sample['time_offset'] is not None
        assert wifi_sample['ssid'] is not None

        wifi_sample = IWConfigData(
            wifi_sample['mac_address'],
            float(wifi_sample['signal_level']),
            frequency_string_to_khz(wifi_sample['frequency']),
            wifi_sample['time_offset'],
            wifi_sample['ssid'])

        parsed_samples.append(wifi_sample)

    return parsed_samples
