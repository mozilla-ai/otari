"""The length of an uploaded audio file, read from the file itself."""

import io
import wave
from decimal import Decimal

from gateway.services.inference import audio_seconds
from gateway.services.inference._audio_length import MAX_FILE_OPERATIONS


def _wav(seconds: float, *, rate: int = 8000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(b"\x00\x00" * int(seconds * rate))
    return buffer.getvalue()


def test_a_wav_file_measures_its_length_in_seconds() -> None:
    assert audio_seconds(io.BytesIO(_wav(4.5))) == Decimal("4.5")


def test_a_file_that_is_not_audio_measures_nothing() -> None:
    assert audio_seconds(io.BytesIO(b"fake-audio-content-mp3")) is None


def test_an_empty_file_measures_nothing() -> None:
    assert audio_seconds(io.BytesIO(b"")) is None


def test_a_header_with_no_frames_measures_nothing() -> None:
    assert audio_seconds(io.BytesIO(_wav(0))) is None


class _CountingFile(io.BytesIO):
    reads = 0

    def read(self, size: int | None = -1) -> bytes:
        self.reads += 1
        return super().read(size)


def test_a_file_that_hunts_for_a_frame_header_gives_up_within_its_budget() -> None:
    """Bytes that look like the start of an MP3 frame header at every offset."""
    audio = _CountingFile(b"\xff" * (4 * MAX_FILE_OPERATIONS))

    assert audio_seconds(audio) is None
    assert audio.reads < MAX_FILE_OPERATIONS


def test_a_long_file_is_measured_within_the_budget() -> None:
    assert audio_seconds(io.BytesIO(_wav(600))) == Decimal("600")
