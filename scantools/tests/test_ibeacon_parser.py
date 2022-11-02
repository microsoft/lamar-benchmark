""" Tests for ibeacon_parse.py """
import pytest

from ..scanners.navvis import ibeacon_parser


def test_parse_navvis_ibeacon_packet():
    data = '4c000215f7826da64fa24e988024bc5b71e0893efaf1af95b8'
    res = ibeacon_parser.parse_navvis_ibeacon_packet(data)
    expected = ibeacon_parser.IBeaconData(
        uuid='f7826da6-4fa2-4e98-8024-bc5b71e0893e',
        major_version=64241,
        minor_version=44949,
        broadcasting_power_dbm=-72)
    assert res == expected


@pytest.mark.parametrize("data", [
    (''),
    ('4c000215f7826da64fa24e988024bc5b71e0893efaf1af95b'),
    ('4c000215f7826da64fa24e988024bc5b71e0893efaf1af95b83c4f')])
def test_parse_navvis_ibeacon_packet_correct_len(data):
    with pytest.raises(Exception):
        ibeacon_parser.parse_navvis_ibeacon_packet(data)


@pytest.mark.parametrize("data", [
    (' 4c000215f7826da64fa24e988024bc5b71e0893efaf1af95b8'),
    ('4c000215f7826da64fa24e988024bc5b71e0893efaf1af95b8 '),
    (' 4c000215f7826da64fa24e988024bc5b71e0893efaf1af95b8 ')])
def test_parse_navvis_ibeacon_packet_empty_chars(data):
    res = ibeacon_parser.parse_navvis_ibeacon_packet(data)
    expected = ibeacon_parser.IBeaconData(
        uuid='f7826da6-4fa2-4e98-8024-bc5b71e0893e',
        major_version=64241,
        minor_version=44949,
        broadcasting_power_dbm=-72)
    assert res == expected
