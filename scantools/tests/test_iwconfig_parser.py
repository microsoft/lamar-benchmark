""" Tests for iwconfig_parser.py """
import pytest

from ..scanners.navvis import iwconfig_parser


@pytest.mark.parametrize("data, expected", [
    ('1.234 GHz', 1234_000),
    ('1.234 MHz', 1234),
    ('1.234 kHz', 1),  # we store it as integers
    ('1.234', 1234)
])
def test_frequency_string_to_khz(data, expected):
    res = iwconfig_parser.frequency_string_to_khz(data)
    assert res == expected


@pytest.mark.parametrize("data", [
    ('1.234 gHz'),
    ('1.234 mHz'),
    ('1.234 KHz'),
    ('1.234 Hz')
])
def test_frequency_string_to_khz_incorrect_units(data):
    with pytest.raises(Exception):
        iwconfig_parser.frequency_string_to_khz(data)


def test_parse_iwconfig():
    # Old format.
    data_path = './test_data/navvis_files/wifi/00000-wifi.log'

    expected = [
        iwconfig_parser.IWConfigData(mac_address='82:2A:A8:9B:DA:E7',
                                     signal_strength_dbm=-70.0,
                                     frequency_khz=5180000,
                                     time_offset_ms='24'),
        iwconfig_parser.IWConfigData(mac_address='82:2A:DA:E7:A8:9B',
                                     signal_strength_dbm=-56.0,
                                     frequency_khz=2437000,
                                     time_offset_ms='0'),
        iwconfig_parser.IWConfigData(mac_address='82:2A:A8:9A:DA:E7',
                                     signal_strength_dbm=56.0,
                                     frequency_khz=2437000,
                                     time_offset_ms='5160')
    ]

    with open(data_path, 'r') as f:
        data = f.readlines()
        results = iwconfig_parser.parse_iwconfig(data)
        assert results == expected

    # New format.
    data_path = './test_data/navvis_files/wifi/00001-wifi.log'

    expected = [
        iwconfig_parser.IWConfigData(mac_address='5e:fb:3a:c4:1b:77',
                                     signal_strength_dbm=-72.0,
                                     frequency_khz=2437000,
                                     time_offset_ms='0'),
        iwconfig_parser.IWConfigData(mac_address='e0:28:6d:c3:d9:cb',
                                     signal_strength_dbm=-58.0,
                                     frequency_khz=2437000,
                                     time_offset_ms='10023'),
        iwconfig_parser.IWConfigData(mac_address='ec:b1:d7:d5:f8:81',
                                     signal_strength_dbm=48.0,
                                     frequency_khz=5180000,
                                     time_offset_ms='488')
    ]

    with open(data_path, 'r') as f:
        data = f.readlines()
        results = iwconfig_parser.parse_iwconfig(data)
        assert results == expected


def test_parse_iwconfig_no_data():
    res = iwconfig_parser.parse_iwconfig('')
    assert len(res) == 0


@pytest.mark.parametrize("input_file", [
    ('incorrect_mac.log'),
    ('incorrect_frequency.log'),
    ('incorrect_signal_level.log'),
    ('incorrect_last_beacon.log')
])
def test_parse_iwconfig_incorrect_input(input_file):
    with pytest.raises(Exception):
        data_path = './test_data/navvis_files/wifi/' / input_file

        with open(data_path, 'r') as f:
            data = f.readlines()
            iwconfig_parser.parse_iwconfig(data)
