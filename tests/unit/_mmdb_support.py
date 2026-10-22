"""Write a small IPv4 MaxMind DB file mapping networks to countries, for tests.

maxminddb only reads the format, so this writes the subset of it a country
file uses: a 24-bit search tree, maps, strings and unsigned integers. See
https://maxmind.github.io/MaxMind-DB/ for the format.
"""

import ipaddress
import struct
from pathlib import Path
from typing import Any, NamedTuple

_METADATA_MARKER = b"\xab\xcd\xefMaxMind.com"


class _Unsigned(NamedTuple):
    """An unsigned integer of a named width: libmaxminddb checks each metadata field's type."""

    type_number: int
    value: int


def _uint16(value: int) -> _Unsigned:
    return _Unsigned(5, value)


def _uint32(value: int) -> _Unsigned:
    return _Unsigned(6, value)


def _uint64(value: int) -> _Unsigned:
    return _Unsigned(9, value)


def _control(type_number: int, size: int) -> bytes:
    if size >= 29:
        raise ValueError("only short values are supported")
    if type_number <= 7:
        return bytes([(type_number << 5) | size])
    return bytes([size, type_number - 7])


def _encode(value: Any) -> bytes:
    if isinstance(value, _Unsigned):
        raw = value.value.to_bytes((value.value.bit_length() + 7) // 8, "big")
        return _control(value.type_number, len(raw)) + raw
    if isinstance(value, str):
        raw = value.encode()
        return _control(2, len(raw)) + raw
    if isinstance(value, dict):
        body = b"".join(_encode(key) + _encode(item) for key, item in value.items())
        return _control(7, len(value)) + body
    if isinstance(value, list):
        return _control(11, len(value)) + b"".join(_encode(item) for item in value)
    raise TypeError(type(value))


def write_country_database(path: Path, countries: dict[str, str]) -> Path:
    """Write ``path`` mapping each IPv4 network (``"192.0.2.0/24"``) to the country code given for it."""
    # A binary trie over the address bits: a node is a two-item list, a leaf a data offset.
    root: list[Any] = [None, None]
    data = b""
    for network, country in countries.items():
        parsed = ipaddress.IPv4Network(network)
        offset = len(data)
        data += _encode({"country": {"iso_code": country}})
        node = root
        bits = int(parsed.network_address)
        for depth in range(parsed.prefixlen):
            bit = (bits >> (31 - depth)) & 1
            if depth == parsed.prefixlen - 1:
                node[bit] = offset
            else:
                if not isinstance(node[bit], list):
                    node[bit] = [None, None]
                node = node[bit]

    nodes: list[list[Any]] = []

    def number(node: list[Any]) -> int:
        nodes.append(node)
        index = len(nodes) - 1
        for child in node:
            if isinstance(child, list):
                number(child)
        return index

    number(root)
    index_of = {id(node): index for index, node in enumerate(nodes)}
    node_count = len(nodes)

    def record(child: Any) -> int:
        if child is None:
            return node_count
        if isinstance(child, list):
            return index_of[id(child)]
        return node_count + 16 + int(child)

    tree = b"".join(struct.pack(">I", record(left))[1:] + struct.pack(">I", record(right))[1:] for left, right in nodes)
    metadata = {
        "binary_format_major_version": _uint16(2),
        "binary_format_minor_version": _uint16(0),
        "build_epoch": _uint64(1),
        "database_type": "Test-Country",
        "description": {"en": "Test country database"},
        "ip_version": _uint16(4),
        "languages": ["en"],
        "node_count": _uint32(node_count),
        "record_size": _uint16(24),
    }
    path.write_bytes(tree + b"\x00" * 16 + data + _METADATA_MARKER + _encode(metadata))
    return path
