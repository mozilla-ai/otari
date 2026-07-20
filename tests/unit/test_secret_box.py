"""Unit tests for encryption at rest of provider credentials."""

import pytest

from gateway.services.secret_box import (
    SecretBoxUnavailableError,
    SecretDecryptionError,
    decrypt_secret,
    encrypt_secret,
    generate_secret_key,
    secret_box_configured,
)


def test_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    token = encrypt_secret("sk-super-secret")
    assert token != "sk-super-secret"
    assert decrypt_secret(token) == "sk-super-secret"


def test_unset_key_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTARI_SECRET_KEY", raising=False)
    monkeypatch.delenv("GATEWAY_SECRET_KEY", raising=False)
    assert secret_box_configured() is False
    with pytest.raises(SecretBoxUnavailableError):
        encrypt_secret("x")


def test_invalid_key_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", "not-a-valid-fernet-key")
    with pytest.raises(SecretBoxUnavailableError):
        encrypt_secret("x")


def test_wrong_key_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    token = encrypt_secret("sk-abc")
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    with pytest.raises(SecretDecryptionError):
        decrypt_secret(token)


def test_decryption_error_never_contains_ciphertext(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    token = encrypt_secret("sk-abc")
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    with pytest.raises(SecretDecryptionError) as excinfo:
        decrypt_secret(token)
    assert token not in str(excinfo.value)


def test_multifernet_rotation(monkeypatch: pytest.MonkeyPatch) -> None:
    old, new = generate_secret_key(), generate_secret_key()

    monkeypatch.setenv("OTARI_SECRET_KEY", old)
    old_token = encrypt_secret("sk-rotate")

    # Prepend the new key: new encrypts, both decrypt, so old ciphertext still reads.
    monkeypatch.setenv("OTARI_SECRET_KEY", f"{new},{old}")
    assert decrypt_secret(old_token) == "sk-rotate"
    fresh_token = encrypt_secret("sk-fresh")

    # Retire the old key: fresh ciphertext still reads, old ciphertext no longer does.
    monkeypatch.setenv("OTARI_SECRET_KEY", new)
    assert decrypt_secret(fresh_token) == "sk-fresh"
    with pytest.raises(SecretDecryptionError):
        decrypt_secret(old_token)


def test_keys_may_be_whitespace_or_comma_separated(monkeypatch: pytest.MonkeyPatch) -> None:
    a, b = generate_secret_key(), generate_secret_key()
    monkeypatch.setenv("OTARI_SECRET_KEY", a)
    token = encrypt_secret("sk-x")
    monkeypatch.setenv("OTARI_SECRET_KEY", f"  {b}\n{a} ")
    assert decrypt_secret(token) == "sk-x"
