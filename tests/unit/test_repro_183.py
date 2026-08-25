
"""
Regression test for issue #183: image-related provider errors should surface as 400, not generic 502.
"""

from any_llm.exceptions import AnyLLMError
from fastapi import status

from gateway.api.routes._pipeline import classify_provider_error


def test_classify_image_provider_error_returns_400() -> None:
    class ImageError(AnyLLMError):
        pass

    exc = ImageError("image content not supported")
    mapping = classify_provider_error(exc)
    assert mapping is not None
    assert mapping.status_code == status.HTTP_400_BAD_REQUEST
