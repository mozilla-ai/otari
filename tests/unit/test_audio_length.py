"""The length of an uploaded audio file, read from the file itself."""

import io
import wave
from decimal import Decimal

import pytest

from _webm_support import element, webm
from gateway.services.inference import _webm_length, audio_seconds
from gateway.services.inference._audio_length import MAX_FILE_OPERATIONS
from gateway.services.inference._webm_length import MAX_ELEMENTS


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


def _recording(seconds: int) -> list[int]:
    """The block timestamps of a recording in 20 ms Opus frames, in milliseconds."""
    return list(range(0, seconds * 1000, 20))


def test_a_webm_file_measures_the_duration_its_header_states() -> None:
    assert audio_seconds(io.BytesIO(webm(_recording(2), duration=4500.0))) == Decimal("4.5")


def test_a_webm_duration_is_read_in_the_files_timestamp_scale() -> None:
    audio = webm(_recording(2), duration=4_500_000.0, timestamp_scale=1000)

    assert audio_seconds(io.BytesIO(audio)) == Decimal("4.5")


def test_a_browser_recording_with_no_duration_measures_its_blocks() -> None:
    """MediaRecorder streams the file: no Duration, and a Segment and Clusters of unknown size."""
    audio = webm(_recording(6), unknown_sizes=True)

    assert audio_seconds(io.BytesIO(audio)) == Decimal("5.98")


def test_a_webm_file_with_sized_clusters_and_no_duration_measures_its_blocks() -> None:
    assert audio_seconds(io.BytesIO(webm(_recording(6)))) == Decimal("5.98")


def test_a_last_block_that_states_its_duration_counts_to_its_end() -> None:
    audio = webm(_recording(6), unknown_sizes=True, last_block_duration_ms=20)

    assert audio_seconds(io.BytesIO(audio)) == Decimal("6")


def test_a_block_duration_written_before_its_block_counts_to_the_blocks_end() -> None:
    audio = webm([0], unknown_sizes=True, last_block_duration_ms=5000, duration_first=True)

    assert audio_seconds(io.BytesIO(audio)) == Decimal("5")


def test_blocks_written_before_their_clusters_timestamp_are_read_from_it() -> None:
    audio = webm(_recording(2), unknown_sizes=True, timestamp_last=True)

    assert audio_seconds(io.BytesIO(audio)) == Decimal("1.98")


def test_block_timestamps_are_read_in_the_files_timestamp_scale() -> None:
    """At a scale of 2 ms a tick, blocks a tick apart are 2 ms apart."""
    audio = webm(range(0, 3000, 10), timestamp_scale=2_000_000, unknown_sizes=True)

    assert audio_seconds(io.BytesIO(audio)) == Decimal("5.98")


def test_a_recording_cut_off_mid_block_measures_the_blocks_it_starts() -> None:
    """The cut takes the last block whole and the end of the one before it."""
    audio = webm(_recording(6), unknown_sizes=True)

    assert audio_seconds(io.BytesIO(audio[:-50])) == Decimal("5.96")


def test_a_webm_file_with_no_blocks_and_no_duration_measures_nothing() -> None:
    assert audio_seconds(io.BytesIO(webm(unknown_sizes=True))) is None


def test_a_webm_file_with_an_unknown_size_where_none_is_allowed_measures_nothing() -> None:
    audio = webm(unknown_sizes=True) + element(0x1654AE6B, b"", unknown_size=True) + webm(_recording(1))

    assert audio_seconds(io.BytesIO(audio)) is None


def test_a_webm_file_of_tiny_elements_gives_up_within_its_budget() -> None:
    """A recording, then Void elements of two bytes each, as many as the upload holds."""
    recording = webm(_recording(1), unknown_sizes=True)
    audio = _CountingFile(recording + element(0xEC, b"") * (2 * MAX_ELEMENTS))

    assert audio_seconds(io.BytesIO(recording)) == Decimal("0.98")
    assert audio_seconds(audio) is None
    assert audio.reads < MAX_FILE_OPERATIONS


@pytest.mark.parametrize(("budget", "expected"), [(56, Decimal("0.98")), (55, None)])
def test_the_element_budget_covers_exactly_the_elements_in_the_file(
    monkeypatch: pytest.MonkeyPatch, budget: int, expected: Decimal | None
) -> None:
    """EBML header, Segment, Info, Tracks, Cluster, Timestamp and 50 blocks: 56 elements."""
    monkeypatch.setattr(_webm_length, "MAX_ELEMENTS", budget)

    assert audio_seconds(io.BytesIO(webm(_recording(1), unknown_sizes=True))) == expected


def test_a_block_too_short_for_its_header_measures_nothing() -> None:
    """A SimpleBlock holding its track number alone, before a Void element it must not read into."""
    short_block = element(0xA3, b"\x81")
    audio = webm(_recording(1), unknown_sizes=True) + short_block + element(0xEC, b"")

    assert audio_seconds(io.BytesIO(audio)) is None


def test_elements_inside_segment_info_count_toward_the_budget() -> None:
    """Segment Info elements, each as full of Void elements as one may be, then a recording."""
    info = element(0x1549A966, element(0xEC, b"") * 32_000)
    infos = info * (MAX_ELEMENTS // 32_000 + 1)
    audio = webm(unknown_sizes=True) + infos + webm(_recording(1), unknown_sizes=True)

    assert audio_seconds(io.BytesIO(audio)) is None


def test_a_long_browser_recording_is_measured_within_the_budget() -> None:
    audio = webm(_recording(90 * 60), unknown_sizes=True)

    assert audio_seconds(io.BytesIO(audio)) == Decimal("5399.98")
