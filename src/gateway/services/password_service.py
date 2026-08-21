"""Password hashing for the dashboard sign-in.

Ported from the platform's ``app.core.security`` (``get_password_hash`` /
``verify_password``), with the same algorithm and the same stored format, so the
``user.hashed_password`` column means one thing in both editions: an identity
imported from otari.ai signs in here, and one created here signs in there. That
portability is why the algorithm is not a free choice, and why the column was
left unbounded (`gateway.models.tenancy`) rather than sized for one hash.

**bcrypt directly, not passlib.** The platform reaches bcrypt through
``passlib.context.CryptContext``, which has been unmaintained since 2020 and
whose bcrypt backend breaks against bcrypt 4.1+. The ``$2b$`` string this module
writes is what passlib would have written at the same cost, so dropping the
wrapper costs no compatibility and keeps a dead dependency out of the OSS
edition's runtime.

Everything here is CPU-bound by design (a password hash that is fast is a
password hash that is useless), so the primitives are sync and the ``_async``
wrappers run them on a worker thread. Callers on a request path must use the
wrappers: one hash or verify at cost 12 is on the order of 200ms on ordinary
hardware, and calling it inline would block the event loop, and so every other
in-flight request on that worker, for as long as it runs. The extension releases
the GIL while it hashes, so the thread genuinely runs in parallel rather than
merely off the loop.
"""

import asyncio
import hashlib

import bcrypt

# The cost passlib defaults to, so a hash written here and a hash written by the
# platform are indistinguishable rather than merely compatible.
_BCRYPT_ROUNDS = 12

# bcrypt 5.0 and later refuse a password over 72 bytes outright; 4.x truncated
# it silently, which is why pyproject floors the dependency at 5.0. Refusing is
# the right behavior and is kept, so the ceiling is published here and enforced
# by the caller with a message a person can act on, rather than surfacing as a
# ``ValueError`` from the library.
MAX_PASSWORD_BYTES = 72

# Long enough to be worth hashing, short enough not to be a policy nobody asked
# for. Deliberately the only rule: composition requirements (a digit, a symbol)
# push people toward predictable substitutions, and measurably so.
MIN_PASSWORD_LENGTH = 8

_absent_password_hash: str | None = None


def hash_password(password: str) -> str:
    """Return the bcrypt hash of ``password``.

    The caller is responsible for having checked the length ceiling; a password
    over ``MAX_PASSWORD_BYTES`` raises ``ValueError`` from bcrypt itself.
    """
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=_BCRYPT_ROUNDS)).decode()


def verify_password(password: str, hashed_password: str) -> bool:
    """Whether ``password`` matches ``hashed_password``.

    Returns False rather than raising for anything bcrypt refuses to parse: an
    over-length candidate, and a stored value that is not a bcrypt hash at all
    (a column hand-edited, or written by some other scheme), both mean "this
    does not authenticate", and letting either escape as a ``ValueError`` would
    turn a failed sign-in into a 500.
    """
    try:
        return bcrypt.checkpw(password.encode(), hashed_password.encode())
    except ValueError:
        return False


def verify_absent_password(password: str) -> bool:
    """Burn one verification against a hash nothing matches; always False.

    Called where there is no stored hash to check, so that "no such identity"
    and "wrong password" take the same wall-clock time. Without it the miss
    returns before any hashing happens, and the difference is large enough to
    measure over the network, which turns the sign-in endpoint into an oracle
    for which addresses hold an account.

    The stand-in hash is minted on first use rather than at import: it costs a
    full hashing round, and paying that during module import would add it to
    every startup, including the CLI's, for a value most processes never read.
    """
    global _absent_password_hash  # noqa: PLW0603
    if _absent_password_hash is None:
        _absent_password_hash = hash_password(hashlib.sha256(b"otari:no-such-identity").hexdigest())
    return verify_password(password, _absent_password_hash)


async def hash_password_async(password: str) -> str:
    """``hash_password``, off the event loop."""
    return await asyncio.to_thread(hash_password, password)


async def verify_password_async(password: str, hashed_password: str) -> bool:
    """``verify_password``, off the event loop."""
    return await asyncio.to_thread(verify_password, password, hashed_password)


async def verify_absent_password_async(password: str) -> bool:
    """``verify_absent_password``, off the event loop."""
    return await asyncio.to_thread(verify_absent_password, password)


__all__ = [
    "MAX_PASSWORD_BYTES",
    "MIN_PASSWORD_LENGTH",
    "hash_password",
    "hash_password_async",
    "verify_absent_password",
    "verify_absent_password_async",
    "verify_password",
    "verify_password_async",
]
