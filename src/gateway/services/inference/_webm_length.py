"""The length of a WebM (Matroska) file, read from its EBML elements.

A muxer that knows the length writes it to Segment Info as ``Duration``. A
browser recorder (MediaRecorder) streams the file and writes no Duration, so
the length is counted from the timestamps of the blocks instead: from the
first block's start to the last block's end, where a block states its
duration, else to its start. See
https://www.matroska.org/technical/elements.html for the elements.
"""

import io
import struct
from decimal import Decimal
from typing import BinaryIO

# The most elements one measurement walks. A 20 ms Opus frame is one element,
# so this covers a two-hour recording; a crafted file of tiny elements stops
# here instead of being walked two bytes at a time.
MAX_ELEMENTS = 400_000

_READ_SIZE = 64 * 1024
# The largest element read whole. Every element read whole is a number or
# Segment Info, which hold a few bytes; any other element is skipped.
_MAX_READ = 64 * 1024
_NANOSECONDS = Decimal(1_000_000_000)
_DEFAULT_TIMESTAMP_SCALE = 1_000_000

_SEGMENT = 0x18538067
_INFO = 0x1549A966
_TIMESTAMP_SCALE = 0x2AD7B1
_DURATION = 0x4489
_CLUSTER = 0x1F43B675
_CLUSTER_TIMESTAMP = 0xE7
_SIMPLE_BLOCK = 0xA3
_BLOCK_GROUP = 0xA0
_BLOCK = 0xA1
_BLOCK_DURATION = 0x9B
# The elements whose children are walked where they stand rather than
# skipped, which is how a Segment or Cluster of unknown size is read.
_WALKED = frozenset({_SEGMENT, _CLUSTER, _BLOCK_GROUP})


class _MalformedError(Exception):
    pass


class _Budget:
    """The elements left to walk, shared by every walk of one file."""

    def __init__(self) -> None:
        self._left = MAX_ELEMENTS

    def spend(self) -> None:
        self._left -= 1
        if self._left < 0:
            raise _MalformedError


class _Cursor:
    """Reads ``audio`` forward in large reads, so each element is not a read of its own."""

    def __init__(self, audio: BinaryIO, budget: _Budget) -> None:
        self._audio = audio
        self.budget = budget
        self._buffer = b""
        self._at = 0

    def take(self, size: int) -> bytes:
        """The next ``size`` bytes, or fewer at the end of the file."""
        if size > _MAX_READ:
            raise _MalformedError
        if len(self._buffer) - self._at < size:
            self._buffer = self._buffer[self._at :] + self._audio.read(max(_READ_SIZE, size))
            self._at = 0
        taken = self._buffer[self._at : self._at + size]
        self._at += len(taken)
        return taken

    def skip(self, size: int) -> None:
        left = len(self._buffer) - self._at
        if size <= left:
            self._at += size
            return
        self._audio.seek(size - left, io.SEEK_CUR)
        self._buffer, self._at = b"", 0

    def vint(self, *, keep_marker: bool) -> tuple[int, int] | None:
        """An EBML variable-size integer and its width in bytes, or ``None`` at the end of the file."""
        first = self.take(1)
        if not first:
            return None
        if first[0] == 0:
            raise _MalformedError
        width = 9 - first[0].bit_length()
        value = first[0] if keep_marker else first[0] & (0xFF >> width)
        rest = self.take(width - 1)
        if len(rest) < width - 1:
            return None
        return int.from_bytes(bytes([value]) + rest, "big"), width

    def header(self) -> tuple[int, int | None] | None:
        """The next element's ID and size, ``None`` for the unknown size, or ``None`` at the end of the file."""
        element_id = self.vint(keep_marker=True)
        if element_id is None:
            return None
        self.budget.spend()
        if element_id[1] > 4:
            raise _MalformedError
        size = self.vint(keep_marker=False)
        if size is None:
            return None
        value, width = size
        return element_id[0], None if value == (1 << (7 * width)) - 1 else value


def _uint(body: bytes) -> int:
    if len(body) > 8:
        raise _MalformedError
    return int.from_bytes(body, "big")


def _info(body: bytes, budget: _Budget) -> tuple[int, Decimal | None]:
    """The timestamp scale and the duration, in scale units, that Segment Info states."""
    cursor = _Cursor(io.BytesIO(body), budget)
    scale, duration = _DEFAULT_TIMESTAMP_SCALE, None
    while (header := cursor.header()) is not None:
        element_id, size = header
        if size is None:
            raise _MalformedError
        if element_id == _TIMESTAMP_SCALE:
            scale = _uint(cursor.take(size)) or _DEFAULT_TIMESTAMP_SCALE
        elif element_id == _DURATION and size in (4, 8):
            body = cursor.take(size)
            if len(body) < size:
                raise _MalformedError
            duration = Decimal(str(struct.unpack(">f" if size == 4 else ">d", body)[0]))
        else:
            cursor.skip(size)
    if duration is not None and not (duration.is_finite() and duration > 0):
        duration = None
    return scale, duration


def webm_seconds(audio: BinaryIO) -> Decimal | None:
    """The length of the WebM file in ``audio``, in seconds, or ``None`` when it states none.

    ``audio`` is read from its current position, which is the file's start.
    """
    try:
        return _walk(_Cursor(audio, _Budget()))
    except _MalformedError:
        return None


class _Span:
    """The earliest start and the latest end of the blocks seen, in ticks."""

    def __init__(self) -> None:
        self.first: int | None = None
        self.end: int | None = None

    def add(self, start: int, end: int) -> None:
        self.first = start if self.first is None else min(self.first, start)
        self.end = end if self.end is None else max(self.end, end)


def _walk(cursor: _Cursor) -> Decimal | None:
    scale = _DEFAULT_TIMESTAMP_SCALE
    file = _Span()
    # A Cluster's blocks are held relative to it until it ends, since its
    # Timestamp may follow them; so is a Block and its duration, in either order.
    cluster, cluster_start = _Span(), 0
    group_offset: int | None = None
    group_duration: int | None = None
    while (header := cursor.header()) is not None:
        element_id, size = header
        if element_id in _WALKED:
            if element_id == _CLUSTER:
                _fold(file, cluster, cluster_start)
                cluster, cluster_start = _Span(), 0
            elif element_id == _BLOCK_GROUP:
                group_offset = group_duration = None
            continue
        if size is None:
            raise _MalformedError
        if element_id == _INFO:
            scale, duration = _info(cursor.take(size), cursor.budget)
            if duration is not None:
                return duration * scale / _NANOSECONDS
        elif element_id == _CLUSTER_TIMESTAMP:
            cluster_start = _uint(cursor.take(size))
        elif element_id in (_SIMPLE_BLOCK, _BLOCK):
            track = cursor.vint(keep_marker=False)
            if track is None:
                break
            # The header is the track number, a 2-byte timestamp and a flags byte.
            if size < track[1] + 3:
                raise _MalformedError
            offset_bytes = cursor.take(2)
            if len(offset_bytes) < 2:
                break
            cursor.skip(size - track[1] - 2)
            offset = int.from_bytes(offset_bytes, "big", signed=True)
            if element_id == _BLOCK:
                group_offset = offset
                cluster.add(offset, offset + (group_duration or 0))
            else:
                cluster.add(offset, offset)
        elif element_id == _BLOCK_DURATION:
            group_duration = _uint(cursor.take(size))
            if group_offset is not None:
                cluster.add(group_offset, group_offset + group_duration)
        else:
            cursor.skip(size)
    _fold(file, cluster, cluster_start)
    if file.first is None or file.end is None:
        return None
    return (file.end - file.first) * scale / _NANOSECONDS


def _fold(file: _Span, cluster: _Span, cluster_start: int) -> None:
    if cluster.first is not None and cluster.end is not None:
        file.add(cluster_start + cluster.first, cluster_start + cluster.end)
