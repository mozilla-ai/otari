"""The endpoint label on a usage-log row is an identifier, not a URL.

These values are written to the database and grouped, filtered and
displayed by string equality. They were copied from route paths when the
routes lived at /v1 and they stay as they are: rewriting them would make new
rows incomparable with old ones. The routes moved; the labels did not.
"""

from gateway.api.routes import (
    audio,
    batches,
    chat,
    embeddings,
    images,
    messages,
    moderations,
    rerank,
    responses,
    videos,
)


def test_labels_keep_their_historical_values() -> None:
    assert chat.USAGE_ENDPOINT == "/v1/chat/completions"
    assert messages.USAGE_ENDPOINT == "/v1/messages"
    assert responses.USAGE_ENDPOINT == "/v1/responses"
    assert embeddings.USAGE_ENDPOINT == "/v1/embeddings"
    assert audio.USAGE_ENDPOINT_TRANSCRIPTIONS == "/v1/audio/transcriptions"
    assert audio.USAGE_ENDPOINT_SPEECH == "/v1/audio/speech"
    assert rerank.USAGE_ENDPOINT == "/v1/rerank"
    assert moderations.USAGE_ENDPOINT == "/v1/moderations"
    assert images.USAGE_ENDPOINT == "/v1/images/generations"
    assert images.USAGE_ENDPOINT_EDITS == "/v1/images/edits"
    assert batches.USAGE_ENDPOINT == "/v1/batches"
    assert batches.USAGE_ENDPOINT_RESULTS == "/v1/batches/results"
    assert videos.USAGE_ENDPOINT == "/v1/videos"
    assert videos.USAGE_ENDPOINT_COMPLETED == "/v1/videos/completed"
