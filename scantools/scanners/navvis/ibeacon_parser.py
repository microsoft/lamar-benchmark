from binascii import hexlify, unhexlify
import collections
import struct


IBeaconData = collections.namedtuple('IBeaconData', ['uuid',
                                                     'major_version',
                                                     'minor_version',
                                                     'broadcasting_power_dbm'])

BluetoothMeasurement = collections.namedtuple(
    'BluetoothData', [
        'timestamp_s',
        'signal_strength_dbm',
        'guid',
        'major_version',
        'minor_version',
        'broadcasting_power_dbm'])


def hexstring_to_binary_data(data):
    assert (len(data) % 2) == 0
    # numBytes = int(len(data) / 2)
    # return [int(data[2 * k:2 * k + 2], 16) for k in range(0, numBytes)]
    return unhexlify(data)


def binary_data_to_hexstring(data):
    return hexlify(data).decode('ascii')


def as_uint16_be(data):
    """ Converts to big-endian 16-bit unsigned integer """
    assert len(data) == 2
    return (data[0] << 8) | data[1]


def binary_data_to_uuid(data):
    """ Array of binary data to uuid format. """
    hexstring = binary_data_to_hexstring(data)
    return hexstring[0:8] + '-' + \
        hexstring[8:12] + '-' + \
        hexstring[12:16] + '-' + \
        hexstring[16:20] + '-' + \
        hexstring[20:32]


def bin_to_int(bytestring):
    """One element byte string to signed int."""
    return struct.unpack("b", bytes([bytestring]))[0]


def get_manufacturer_data(m_payload):
    assert len(m_payload) == 25

    # first 2 bytes are the manufacturer company id
    return m_payload[2:]


def get_manufacturer_company_id(data):
    return (data[1] << 8) | data[0]


def parse_ibeacon_data(data):

    assert len(data) == 21
    uuid = binary_data_to_uuid(data[:16])
    major = as_uint16_be(data[16:18])
    minor = as_uint16_be(data[18:20])
    broadcasting_power_dbm = bin_to_int(data[20])

    return IBeaconData(uuid, major, minor, broadcasting_power_dbm)


def parse_navvis_ibeacon_packet(ibeacon_packet):
    """ Parses NavVis version of iBeacon packet (provides only last 25 bytes).

    iBeacon packet format:
    https://support.kontakt.io/hc/en-gb/articles/201492492-iBeacon-advertising-packet-structure

    NavVis documentation:
    https://docs.navvis.com/mapping/v2.5/en/html/dataset_description_m6.html?highlight=bluetooth#beacons
    """

    binary_data = hexstring_to_binary_data(ibeacon_packet.strip())

    # iBeacon packet format consists of 30 bytes
    # NavVis does not provide first 5 bytes
    assert len(binary_data) == 25

    manufacturer_data = get_manufacturer_data(binary_data)
    manufacturer_id = get_manufacturer_company_id(binary_data)
    assert manufacturer_id == 0x004C

    beacon_type = manufacturer_data[0]
    assert beacon_type == 2

    data_len = manufacturer_data[1]
    assert data_len == 21  # 16 bytes uuid + 2 major + 2 minor + 1 signal power

    payload = manufacturer_data[2:]
    parsed_data = parse_ibeacon_data(payload)

    return parsed_data
