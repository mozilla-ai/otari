"""The length of an uploaded audio file, read from the file itself.

A transcription priced per minute or per second is billed on the length the
provider reports. A model that reports tokens instead states no length, so the
upload is measured here. The format is recognized by the file's own bytes, not
by its name, and a file that cannot be read measures nothing.
"""

from decimal import Decimal
from typing import BinaryIO, cast

from tinytag import TinyTag, TinyTagException

# The most reads and seeks one measurement may make. A real file states its
# length in far fewer; a crafted one that has the parser hunt for a frame
# header byte by byte through the whole upload stops here instead.
MAX_FILE_OPERATIONS = 100_000


class _OverBudgetError(Exception):
    pass


class _BoundedReader:
    """``audio``, refusing a read or a seek past :data:`MAX_FILE_OPERATIONS`."""

    def __init__(self, audio: BinaryIO) -> None:
        self._audio = audio
        self._left = MAX_FILE_OPERATIONS

    def _spend(self) -> None:
        self._left -= 1
        if self._left < 0:
            raise _OverBudgetError

    def read(self, size: int = -1) -> bytes:
        self._spend()
        return self._audio.read(size)

    def seek(self, offset: int, whence: int = 0) -> int:
        self._spend()
        return self._audio.seek(offset, whence)

    def tell(self) -> int:
        return self._audio.tell()


def audio_seconds(audio: BinaryIO) -> Decimal | None:
    """The length of the audio in ``audio``, in seconds, or ``None`` when it cannot be read or is empty.

    Reads only what the format needs to state its length, so a long file is
    not read whole, and gives up past a bounded number of reads. Blocking; run
    it off the event loop.
    """
    reader = cast("BinaryIO", _BoundedReader(audio))
    try:
        # tinytag raises any error its parser meets, the budget's included, as its own.
        duration = TinyTag.get(file_obj=reader, tags=False, duration=True).duration
    except TinyTagException:
        return None
    if duration is None:
        return None
    seconds = Decimal(str(duration))
    return seconds if seconds.is_finite() and seconds > 0 else None
