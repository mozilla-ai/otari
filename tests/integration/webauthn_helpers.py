"""A software authenticator, so the passkey tests exercise real verification.

The alternative was stubbing ``webauthn_service``'s two verify calls, which
would have left every test asserting that this deployment's own bookkeeping
works while the part that decides whether a passkey is genuine went unexercised.
This builds the payloads a browser and an authenticator actually produce, signs
them with a real P-256 key, and lets py_webauthn verify them, so a test that
passes is a ceremony that would pass in a browser.

What it deliberately does *not* do is model an authenticator's storage. A test
holds the key pair and says which credential answers, because the properties
under test are the server's (does a replayed challenge fail, does a credential
registered under another relying-party ID assert, does one identity's ceremony
complete against another's), and each of those is easier to state when the test
chooses the answer.

Everything here is the WebAuthn wire format at
https://www.w3.org/TR/webauthn-2/, which is why the byte offsets are spelled out
rather than abbreviated: an off-by-one in ``authData`` fails as an opaque
signature mismatch.
"""

import hashlib
import json
import secrets
import struct
from dataclasses import dataclass, field
from typing import Any

import cbor2
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from webauthn.helpers import base64url_to_bytes, bytes_to_base64url

# Authenticator data flag bits (WebAuthn 6.1). UP is "a person was present", UV
# "and they were verified"; BE/BS say the credential is backup-eligible and
# currently backed up, which is what a synced platform passkey reports; AT says
# attested credential data follows, which only a registration carries.
FLAG_UP = 0x01
FLAG_UV = 0x04
FLAG_BE = 0x08
FLAG_BS = 0x10
FLAG_AT = 0x40

# COSE_Key for an EC2 key on P-256 (RFC 8152): kty=2 (EC2), alg=-7 (ES256),
# crv=1 (P-256), then the two 32-byte coordinates.
_COSE_KTY = 1
_COSE_ALG = 3
_COSE_CRV = -1
_COSE_X = -2
_COSE_Y = -3

# All-zero, the value an authenticator that declines to identify its model
# reports. Every platform passkey does exactly this, so it is the honest default.
_AAGUID = b"\x00" * 16


@dataclass
class SoftwareAuthenticator:
    """One authenticator holding one credential, driving both ceremonies."""

    rp_id: str
    origin: str
    credential_id: bytes = field(default_factory=lambda: secrets.token_bytes(32))
    private_key: ec.EllipticCurvePrivateKey = field(default_factory=lambda: ec.generate_private_key(ec.SECP256R1()))
    sign_count: int = 0
    # Reported back in the registration response and stored as a hint. "internal"
    # is what a laptop's own authenticator says.
    transports: tuple[str, ...] = ("internal",)

    @property
    def credential_id_b64(self) -> str:
        return bytes_to_base64url(self.credential_id)

    def _rp_id_hash(self, rp_id: str | None = None) -> bytes:
        return hashlib.sha256((rp_id or self.rp_id).encode()).digest()

    def _cose_key(self) -> bytes:
        numbers = self.private_key.public_key().public_numbers()
        return cbor2.dumps(
            {
                _COSE_KTY: 2,
                _COSE_ALG: -7,
                _COSE_CRV: 1,
                # Fixed 32-byte big-endian coordinates: a short int would encode
                # to fewer bytes and the key would not parse.
                _COSE_X: numbers.x.to_bytes(32, "big"),
                _COSE_Y: numbers.y.to_bytes(32, "big"),
            }
        )

    def _client_data(self, *, ceremony_type: str, challenge: str, origin: str | None = None) -> bytes:
        """The clientDataJSON the browser builds, with the challenge as it sent it.

        ``challenge`` arrives already base64url-encoded, straight out of the
        options payload, because that is the form the browser echoes and the
        form the server compares.
        """
        return json.dumps(
            {
                "type": ceremony_type,
                "challenge": challenge,
                "origin": origin or self.origin,
                "crossOrigin": False,
            }
        ).encode()

    def register(self, challenge: str, *, origin: str | None = None, rp_id: str | None = None) -> dict[str, Any]:
        """The response a browser returns from ``navigator.credentials.create``.

        Attestation format "none", which is what a passkey produces unless the
        relying party asks otherwise, and what ``begin_registration`` requests.
        """
        client_data = self._client_data(ceremony_type="webauthn.create", challenge=challenge, origin=origin)
        cose_key = self._cose_key()
        attested = (
            _AAGUID
            + struct.pack(">H", len(self.credential_id))
            + self.credential_id
            + cose_key
        )
        auth_data = (
            self._rp_id_hash(rp_id)
            + bytes([FLAG_UP | FLAG_UV | FLAG_BE | FLAG_BS | FLAG_AT])
            + struct.pack(">I", self.sign_count)
            + attested
        )
        attestation_object = cbor2.dumps({"fmt": "none", "attStmt": {}, "authData": auth_data})
        return {
            "id": self.credential_id_b64,
            "rawId": self.credential_id_b64,
            "type": "public-key",
            "response": {
                "clientDataJSON": bytes_to_base64url(client_data),
                "attestationObject": bytes_to_base64url(attestation_object),
                "transports": list(self.transports),
            },
            "clientExtensionResults": {},
        }

    def authenticate(
        self,
        challenge: str,
        *,
        origin: str | None = None,
        rp_id: str | None = None,
        sign_count: int | None = None,
    ) -> dict[str, Any]:
        """The response a browser returns from ``navigator.credentials.get``.

        The counter advances by default, which is what a hardware authenticator
        does; ``sign_count`` pins it so a test can replay an old value and
        assert the clone check fires.
        """
        if sign_count is None:
            self.sign_count += 1
        else:
            self.sign_count = sign_count
        client_data = self._client_data(ceremony_type="webauthn.get", challenge=challenge, origin=origin)
        auth_data = (
            self._rp_id_hash(rp_id)
            + bytes([FLAG_UP | FLAG_UV | FLAG_BE | FLAG_BS])
            + struct.pack(">I", self.sign_count)
        )
        # The assertion signs authenticatorData concatenated with the *hash* of
        # the client data, not the client data itself (WebAuthn 6.3.3).
        signed = auth_data + hashlib.sha256(client_data).digest()
        signature = self.private_key.sign(signed, ec.ECDSA(hashes.SHA256()))
        return {
            "id": self.credential_id_b64,
            "rawId": self.credential_id_b64,
            "type": "public-key",
            "response": {
                "clientDataJSON": bytes_to_base64url(client_data),
                "authenticatorData": bytes_to_base64url(auth_data),
                "signature": bytes_to_base64url(signature),
                "userHandle": None,
            },
            "clientExtensionResults": {},
        }


def challenge_of(options: dict[str, Any]) -> str:
    """The base64url challenge out of an options payload, as the browser reads it."""
    challenge: str = options["challenge"]
    # Round-trips through the decoder so a malformed payload fails here, in the
    # helper, rather than as a confusing verification error later.
    base64url_to_bytes(challenge)
    return challenge
