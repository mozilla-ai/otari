import pytest

from gateway.core.link_urls import normalize_link_url


@pytest.mark.parametrize(
    "url", ["https://example.com/terms", "http://example.com:8443/terms", "https://example.com:8443/terms"]
)
def test_an_absolute_web_address_is_accepted(url: str) -> None:
    assert normalize_link_url(f" {url} ", "terms_url") == url


@pytest.mark.parametrize(
    "url", ["https://:443/terms", "https://example.com:bad/terms", "javascript:alert(1)", "/terms"]
)
def test_a_link_with_no_usable_host_is_refused(url: str) -> None:
    with pytest.raises(ValueError, match="^terms_url must be an absolute http\\(s\\) URL$"):
        normalize_link_url(url, "terms_url")


def test_a_credential_is_named_ahead_of_a_missing_host() -> None:
    """A value wrong in both ways names the credential, which is the one the writer must remove."""
    with pytest.raises(ValueError, match="^terms_url must carry no username or password$"):
        normalize_link_url("https://operator:hunter2@:8080/terms", "terms_url")


@pytest.mark.parametrize("url", ["https://user:hunter2@example.com/terms", "https://hunter2@example.com:bad/terms"])
def test_a_refusal_never_repeats_the_value(url: str) -> None:
    with pytest.raises(ValueError) as refused:
        normalize_link_url(url, "terms_url")

    assert "hunter2" not in str(refused.value)
