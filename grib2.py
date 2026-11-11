"""Minimal pure-Python (numpy) GRIB2 reader for regular lat/lon fields.

Supports what NOAA's GFS pressure-level files use: grid template 3.0 (regular latitude/longitude), data
representation templates 5.0 (simple packing) and 5.3 (complex packing with spatial differencing), and an
optional bitmap.  Other templates raise `NotImplementedError` instead of returning wrong numbers.
No C library (eccodes) is needed, so it installs the same way on Linux and Windows.

    for message in read_messages(open("x.grb2", "rb").read()):
        message.values        # (n_lat, n_lon) array, NaN where the bitmap says "no value"
        message.lat0, message.lon0, message.dlat, message.dlon   # first point and signed steps (degrees)
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np


@dataclass
class Grib2Field:
    values: np.ndarray   # (n_lat, n_lon)
    lat0: float          # latitude of the first row
    lon0: float          # longitude of the first column
    dlat: float          # signed latitude step between rows (negative when rows run north to south)
    dlon: float          # longitude step between columns


def _sign_magnitude(raw: int, nbytes: int) -> int:
    """GRIB stores signed integers as a sign bit plus magnitude, not two's complement."""
    sign_bit = 1 << (8 * nbytes - 1)
    return -(raw & (sign_bit - 1)) if raw & sign_bit else raw


def _unpack_bits(buf: np.ndarray, bit_offsets: np.ndarray, widths) -> np.ndarray:
    """Read big-endian unsigned integers of `widths` bits (0..32) starting at each bit offset.

    A 6-byte window always covers the 32 bits plus the 7 bits of misalignment."""
    offsets = np.asarray(bit_offsets, dtype=np.int64)
    widths = np.broadcast_to(np.asarray(widths, dtype=np.int64), offsets.shape)
    byte = offsets >> 3
    window = np.zeros(offsets.shape, dtype=np.uint64)
    for k in range(6):
        window = (window << np.uint64(8)) | buf[byte + k].astype(np.uint64)
    shift = (48 - (offsets & 7) - widths).astype(np.uint64)
    mask = (np.uint64(1) << widths.astype(np.uint64)) - np.uint64(1)
    return ((window >> shift) & mask).astype(np.int64)


def _pad_bytes(data: bytes) -> np.ndarray:
    return np.frombuffer(data + b"\x00" * 8, dtype=np.uint8)


def _unpack_data(sec5: bytes, sec7: bytes) -> np.ndarray:
    """Decode a data section into float values (one per present grid point)."""
    npoints, template = struct.unpack(">IH", sec5[5:11])
    reference = struct.unpack(">f", sec5[11:15])[0]
    binary_scale = _sign_magnitude(struct.unpack(">H", sec5[15:17])[0], 2)
    decimal_scale = _sign_magnitude(struct.unpack(">H", sec5[17:19])[0], 2)
    nbits = sec5[19]
    data = sec7[5:]
    buf = _pad_bytes(data)

    if template == 0:  # simple packing
        if nbits == 0:
            ints = np.zeros(npoints, dtype=np.int64)
        else:
            ints = _unpack_bits(buf, np.arange(npoints, dtype=np.int64) * nbits, nbits)
    elif template == 3:  # complex packing + spatial differencing
        (split_method, missing_mode, _primary, _secondary, n_groups, width_ref, width_bits, length_ref,
         length_inc, last_length, length_bits, order, extra) = struct.unpack(">BBIIIBBIBIBBB", sec5[21:49])
        if missing_mode != 0:
            raise NotImplementedError("GRIB2 complex packing with embedded missing values is not supported")
        if order not in (1, 2):
            raise NotImplementedError(f"spatial differencing of order {order}")
        pos = 0
        first = []
        for _ in range(order):
            first.append(_sign_magnitude(int.from_bytes(data[pos:pos + extra], "big"), extra))
            pos += extra
        minimum = _sign_magnitude(int.from_bytes(data[pos:pos + extra], "big"), extra)
        pos += extra

        def read_array(count, width):
            nonlocal pos
            if width == 0:
                return np.zeros(count, dtype=np.int64)
            values = _unpack_bits(buf, pos * 8 + np.arange(count, dtype=np.int64) * width, width)
            pos += -(-count * width // 8)   # each array is padded to a whole byte
            return values

        group_ref = read_array(n_groups, nbits)
        group_width = read_array(n_groups, width_bits) + width_ref
        group_length = read_array(n_groups, length_bits) * length_inc + length_ref
        group_length[-1] = last_length
        if int(group_length.sum()) != npoints:
            raise ValueError("corrupt GRIB2 message: group lengths do not add up to the number of points")

        # every value's bit position: its group's start + index within the group * the group's width
        group_bits = group_length * group_width
        group_start = pos * 8 + np.concatenate(([0], np.cumsum(group_bits)[:-1]))
        group_of = np.repeat(np.arange(n_groups), group_length)
        first_of_group = np.concatenate(([0], np.cumsum(group_length)[:-1]))
        index_in_group = np.arange(npoints) - first_of_group[group_of]
        widths = group_width[group_of]
        ints = _unpack_bits(buf, group_start[group_of] + index_in_group * widths, widths) + group_ref[group_of]

        ints = ints.astype(np.int64)
        if order == 1:
            ints[0] = first[0]
            ints[1:] += minimum
            ints = np.cumsum(ints)
        else:
            ints[0], ints[1] = 0, first[1] - first[0]
            ints[2:] += minimum
            ints = first[0] + np.cumsum(np.cumsum(ints))
    else:
        raise NotImplementedError(f"GRIB2 data representation template 5.{template}")

    return (reference + ints.astype(np.float64) * 2.0 ** binary_scale) / 10.0 ** decimal_scale


def read_messages(blob: bytes):
    """Yield a `Grib2Field` for every GRIB2 message in `blob`."""
    pos = 0
    while True:
        pos = blob.find(b"GRIB", pos)
        if pos < 0:
            return
        if blob[pos + 7] != 2:
            raise NotImplementedError("only GRIB edition 2 is supported")
        total = struct.unpack(">Q", blob[pos + 8:pos + 16])[0]
        yield _decode_message(blob[pos:pos + total])
        pos += total


def _decode_message(message: bytes) -> Grib2Field:
    grid = sec5 = sec7 = None
    bitmap = None
    pos = 16
    while message[pos:pos + 4] != b"7777":
        length, number = struct.unpack(">IB", message[pos:pos + 5])
        section = message[pos:pos + length]
        if number == 3:
            grid = section
        elif number == 5:
            sec5 = section
        elif number == 6:
            indicator = section[5]
            if indicator == 0:
                bitmap = np.unpackbits(np.frombuffer(section[6:], dtype=np.uint8)).astype(bool)
            elif indicator != 255:
                raise NotImplementedError(f"GRIB2 bitmap indicator {indicator}")
        elif number == 7:
            sec7 = section
        pos += length
    if grid is None or sec5 is None or sec7 is None:
        raise ValueError("incomplete GRIB2 message")

    template = struct.unpack(">H", grid[12:14])[0]
    if template != 0:
        raise NotImplementedError(f"GRIB2 grid template 3.{template} (only regular lat/lon 3.0)")
    ni, nj = struct.unpack(">II", grid[30:38])
    la1, lo1, _flags, _la2, _lo2, di, dj = struct.unpack(">IIBIIII", grid[46:71])
    scanning = grid[71]
    la1, lo1 = _sign_magnitude(la1, 4), _sign_magnitude(lo1, 4)
    if scanning & 0b00100000:
        raise NotImplementedError("GRIB2 grid with column-major scanning")
    lat0, lon0 = la1 / 1e6, lo1 / 1e6
    dlat = (dj / 1e6) * (1 if scanning & 0b01000000 else -1)
    dlon = (di / 1e6) * (-1 if scanning & 0b10000000 else 1)

    present = _unpack_data(sec5, sec7)
    if bitmap is None:
        values = present
    else:
        values = np.full(ni * nj, np.nan)
        values[bitmap[:ni * nj]] = present
    return Grib2Field(values.reshape(nj, ni), lat0, lon0, dlat, dlon)
