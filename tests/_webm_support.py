"""Write a small WebM file for tests, as a browser recorder or a muxer lays one out.

The file holds the EBML elements the length is read from, and no real audio:
see https://www.matroska.org/technical/elements.html for the IDs.
"""

import struct
from collections.abc import Sequence

UNKNOWN_SIZE = b"\x01\xff\xff\xff\xff\xff\xff\xff"


def _size(length: int) -> bytes:
    width = 1
    while length >= (1 << (7 * width)) - 1:
        width += 1
    return ((1 << (7 * width)) | length).to_bytes(width, "big")


def element(element_id: int, body: bytes, *, unknown_size: bool = False) -> bytes:
    """One EBML element: its ID, its size (or the unknown size) and its body."""
    id_bytes = element_id.to_bytes((element_id.bit_length() + 7) // 8, "big")
    return id_bytes + (UNKNOWN_SIZE if unknown_size else _size(len(body))) + body


def _uint(value: int) -> bytes:
    return value.to_bytes(max(1, (value.bit_length() + 7) // 8), "big")


def simple_block(offset_ms: int) -> bytes:
    """A SimpleBlock on track 1, ``offset_ms`` after its cluster's timestamp."""
    return element(0xA3, b"\x81" + struct.pack(">h", offset_ms) + b"\x80" + b"\x00" * 40)


def block_group(offset_ms: int, duration_ms: int, *, duration_first: bool = False) -> bytes:
    """A BlockGroup holding one Block on track 1 and the Block's duration, after the Block unless ``duration_first``."""
    block = element(0xA1, b"\x81" + struct.pack(">h", offset_ms) + b"\x00" + b"\x00" * 40)
    duration = element(0x9B, _uint(duration_ms))
    return element(0xA0, duration + block if duration_first else block + duration)


def webm(
    block_ms: Sequence[int] = (),
    *,
    duration: float | None = None,
    timestamp_scale: int | None = None,
    cluster_ms: int = 1000,
    unknown_sizes: bool = False,
    last_block_duration_ms: int | None = None,
    duration_first: bool = False,
    timestamp_last: bool = False,
) -> bytes:
    """A WebM file with an Opus track and a block at each of ``block_ms``.

    ``duration`` is written to Segment Info in ``timestamp_scale`` units, as a
    muxer that knows the length writes it; leave it out for the file a browser
    recorder writes. ``unknown_sizes`` writes the Segment and each Cluster with
    the unknown size, as a recorder that streams the file does.
    ``last_block_duration_ms`` writes the last block as a BlockGroup that
    states its own duration, written before its Block when ``duration_first``.
    ``timestamp_last`` writes each Cluster's Timestamp after its blocks.
    """
    info = b""
    if timestamp_scale is not None:
        info += element(0x2AD7B1, _uint(timestamp_scale))
    if duration is not None:
        info += element(0x4489, struct.pack(">d", duration))
    track = element(0xD7, b"\x01") + element(0x86, b"A_OPUS")
    parts = [element(0x1549A966, info), element(0x1654AE6B, element(0xAE, track))]

    clusters: dict[int, list[bytes]] = {}
    for index, at in enumerate(block_ms):
        start = at - at % cluster_ms
        last = index == len(block_ms) - 1
        if last and last_block_duration_ms is not None:
            block = block_group(at - start, last_block_duration_ms, duration_first=duration_first)
        else:
            block = simple_block(at - start)
        clusters.setdefault(start, []).append(block)
    for start, blocks in clusters.items():
        timestamp, body = element(0xE7, _uint(start)), b"".join(blocks)
        cluster = body + timestamp if timestamp_last else timestamp + body
        parts.append(element(0x1F43B675, cluster, unknown_size=unknown_sizes))

    header = element(0x1A45DFA3, element(0x4282, b"webm"))
    return header + element(0x18538067, b"".join(parts), unknown_size=unknown_sizes)
