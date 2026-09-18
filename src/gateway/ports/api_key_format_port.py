"""What an API key this deployment mints looks like, and where a presented one belongs.

The seam between key handling and whichever build decides the shape of its
credentials. The core adapter mints the open-source format and treats every
presented key as local, so a standalone gateway behaves as it always has. A
hosted overlay binds a richer format behind the same interface: a product
prefix, a format version, a region tag and a checksum, minted and parsed
without editing any file in this tree (otari-ai#1665).

The methods are plain ``def`` rather than ``async def``, unlike the other ports.
Each is a pure computation over a short string, with no I/O behind it in any
build, and the verify path calls ``route`` before it touches the database
precisely so that a refusal costs no round trip.

Two rules bind every adapter, and the verify path depends on both:

* A key's shape never grants access. ``route`` decides where a key is checked,
  and the hash lookup that follows is what decides whether it is valid. A
  region tag or a checksum is an untrusted routing hint.
* A key an adapter does not recognize is ``Local``, never ``Malformed``. Keys
  minted before this port existed (``gw-``, ``tk_``, ``tk_live.``) still
  authenticate by hash, so an adapter refuses only a key that claims its own
  format and fails it (otari#646 took the shape check off the verify path for
  this reason).
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Local:
    """The key is checked here, against this deployment's own rows."""


@dataclass(frozen=True)
class Misdirected:
    """The key belongs to another deployment, reachable at ``host``.

    Answered as ``421 Misdirected Request`` naming the host, before any lookup.
    """

    host: str


@dataclass(frozen=True)
class Malformed:
    """The key claims this adapter's format and fails it.

    A bad checksum, an unknown region, or a credential of another kind in the
    product's format (a gateway token presented as an API key). Answered as
    ``401`` before any lookup, since no API key row can match it.
    """


KeyRoute = Local | Misdirected | Malformed


class ApiKeyFormatPort(Protocol):
    """The format of the API keys this build mints and recognizes.

    API keys only: the credential an end user presents. The gateway's own token
    to the platform is a different credential with its own checker.
    """

    def mint(self) -> str:
        """Return a new API key in this build's format.

        The whole plaintext, returned once. The caller hashes it for storage
        and never keeps it.
        """
        ...

    def fingerprint(self, api_key: str) -> str:
        """Return the display-only leading characters stored as ``key_prefix``.

        The format's prefix plus seven of the random characters that follow it,
        so an operator can tell keys apart after the one-time reveal. It never
        gates auth and cannot be recovered from the stored hash.
        """
        ...

    def route(self, presented: str) -> KeyRoute:
        """Say where a presented API key is checked, without looking it up.

        ``presented`` is always a credential offered as an API key (``Otari-Key``,
        ``Authorization: Bearer`` or ``x-api-key``); this port sees no other
        header and no other kind of credential. A gateway token is parsed by
        whatever checks gateway tokens, never here, so one that arrives in the
        product's format is ``Malformed`` rather than a kind to route.

        ``Local`` for a key checked here, including every key whose format this
        adapter does not recognize. ``Misdirected`` for a key that names another
        deployment. ``Malformed`` for a key that claims this adapter's format and
        fails it.
        """
        ...


__all__ = ["ApiKeyFormatPort", "KeyRoute", "Local", "Malformed", "Misdirected"]
