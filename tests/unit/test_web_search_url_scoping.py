"""Unit tests for `web_search_url_targets_platform`.

Guards the decision to forward the platform token to the web-search backend:
the token may only go to the platform itself, never to a confusable foreign
host. See the [major] review note on raw-prefix matching.
"""

import pytest

from gateway.api.routes._pipeline import web_search_url_targets_platform


@pytest.mark.parametrize(
    ("web_search_url", "platform_base"),
    [
        # Canonical base (with path) -> the platform's own web-search endpoint.
        ("https://api.otari.ai/api/v1/gateway/web-search", "https://api.otari.ai/api/v1"),
        # Internal service addressing (what deployments actually use).
        ("http://backend:8000/api/v1/gateway/web-search", "http://backend:8000/api/v1"),
        # Trailing slash on the base must not matter.
        ("http://backend:8000/api/v1/gateway/web-search", "http://backend:8000/api/v1/"),
        # Path-less base (base_path == "") — any path on the same origin is "under" it.
        ("https://api.otari.ai/gateway/web-search", "https://api.otari.ai"),
        # Explicit default port on one side must normalize against an omitted one.
        ("https://api.otari.ai:443/api/v1/gateway/web-search", "https://api.otari.ai/api/v1"),
        ("http://backend/api/v1/gateway/web-search", "http://backend:80/api/v1"),
    ],
)
def test_accepts_platform_targets(web_search_url: str, platform_base: str) -> None:
    assert web_search_url_targets_platform(web_search_url, platform_base) is True


@pytest.mark.parametrize(
    ("web_search_url", "platform_base"),
    [
        # Different backend entirely (bundled SearXNG / third-party adapter).
        ("http://searxng:8080", "http://backend:8000/api/v1"),
        # Confusable host suffix against a path-less base — the startswith trap.
        ("https://api.otari.ai.evil.com/api/v1/gateway/web-search", "https://api.otari.ai"),
        # Userinfo confusable: real host is evil.com.
        ("https://api.otari.ai@evil.com/gateway/web-search", "https://api.otari.ai"),
        # Same host, but path not under the base path.
        ("http://backend:8000/api/v2/gateway/web-search", "http://backend:8000/api/v1"),
        # Path-prefix collision without a `/` boundary.
        ("http://backend:8000/api/v1beta/web-search", "http://backend:8000/api/v1"),
        # Scheme mismatch.
        ("http://api.otari.ai/api/v1/gateway/web-search", "https://api.otari.ai/api/v1"),
        # No base configured.
        ("http://backend:8000/api/v1/gateway/web-search", None),
        ("http://backend:8000/api/v1/gateway/web-search", ""),
    ],
)
def test_rejects_non_platform_targets(web_search_url: str, platform_base: str | None) -> None:
    assert web_search_url_targets_platform(web_search_url, platform_base) is False
