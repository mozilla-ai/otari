"""The country a request comes from, for the provider country gate.

A trusted proxy's header (``client_country_header``, such as Cloudflare's
``CF-IPCountry``) is read first, then the client address is looked up in a
MaxMind-format file: ``client_country_database`` when it is set, else the
country file an operator stored in the database. The address is the one
uvicorn resolved behind ``forwarded_allow_ips``.
"""

import contextlib
import io
import ipaddress
import os
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from types import ModuleType
from typing import TYPE_CHECKING, Any

from gateway.core.country_codes import COUNTRY_CODES
from gateway.log_config import logger
from gateway.services.providers import CountryGate

if TYPE_CHECKING:
    from maxminddb.reader import Reader

    from gateway.core.config import GatewayConfig


class CountryDatabaseError(ValueError):
    """A country file cannot be read."""


@dataclass(frozen=True)
class CountryDatabaseFile:
    """What a country file's metadata says it is."""

    database_type: str
    built_at: datetime


def _country_code(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    code = value.strip().upper()
    return code if code in COUNTRY_CODES else None


def _maxminddb(needed_by: str) -> ModuleType:
    try:
        import maxminddb  # noqa: PLC0415 - the geoip extra, needed only when a country file is used
    except ImportError as exc:
        msg = f"{needed_by} needs maxminddb, which is not installed. Install it with: pip install gateway[geoip]"
        raise CountryDatabaseError(msg) from exc
    return maxminddb


def open_country_database(path: str) -> "Reader":
    """Open the file at ``path``, or say why it cannot be used."""
    maxminddb = _maxminddb("client_country_database")
    try:
        reader: Reader = maxminddb.open_database(path)
    except (OSError, maxminddb.InvalidDatabaseError) as exc:
        msg = f"client_country_database {path!r} cannot be read as a MaxMind DB file: {type(exc).__name__}"
        raise CountryDatabaseError(msg) from exc
    return reader


def inspect_country_database(content: bytes) -> CountryDatabaseFile:
    """What ``content`` is, when it is a MaxMind DB country file such as GeoLite2 Country or DB-IP Lite Country.

    Raises:
        CountryDatabaseError: ``content`` is no MaxMind DB file, or one that holds no countries.
    """
    maxminddb = _maxminddb("A country file")
    try:
        metadata = maxminddb.open_database(io.BytesIO(content), maxminddb.MODE_FD).metadata()
        built_at = datetime.fromtimestamp(metadata.build_epoch, UTC)
    # TypeError: metadata that lacks a field, or holds one of the wrong type.
    except (ValueError, TypeError, OverflowError, OSError, maxminddb.InvalidDatabaseError) as exc:
        msg = "The file cannot be read as a MaxMind DB file"
        raise CountryDatabaseError(msg) from exc
    database_type = str(metadata.database_type)
    # A city file holds countries too, at five times the size; the gate reads only the country.
    if "country" not in database_type.lower() or "city" in database_type.lower():
        msg = f"The file is a {database_type!r} database, not a country database"
        raise CountryDatabaseError(msg)
    return CountryDatabaseFile(database_type=database_type, built_at=built_at)


class _Database:
    """The open file, opened again when it is replaced, so an update needs no restart."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._opened: tuple[str, tuple[int, int, int], Reader] | None = None
        # The version that failed to open, so a bad file is not opened and logged on every request.
        self._failed: tuple[str, tuple[int, int, int]] | None = None
        # The path last found missing, logged once until it is back.
        self._missing: str | None = None

    def get(self, path: str) -> "Reader | None":
        try:
            stat = os.stat(path)
        except OSError:
            if self._missing != path:
                self._missing = path
                logger.warning("client_country_database %r cannot be read; every address has no country", path)
            return None
        self._missing = None
        version = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
        opened = self._opened
        if opened is not None and opened[0] == path and opened[1] == version:
            return opened[2]
        if self._failed == (path, version):
            return None
        with self._lock:
            opened = self._opened
            if opened is not None and opened[0] == path and opened[1] == version:
                return opened[2]
            if self._failed == (path, version):
                return None
            try:
                reader = open_country_database(path)
            except CountryDatabaseError as exc:
                self._failed = (path, version)
                logger.warning("%s; every address has no country", exc)
                return None
            self._failed = None
            self._opened = (path, version, reader)
            return reader


_database = _Database()


class _StoredDatabase:
    """The stored country file this process has open, from a temporary file of its own.

    Swapped whole when the stored file changes, so a lookup sees the old file or
    the new one. The old reader is left to close when nothing holds it, since a
    lookup on another thread may still be reading it.
    """

    def __init__(self) -> None:
        self.opened: tuple[str, str, Reader] | None = None
        # The SHA-256 that failed to open, so a bad file is opened and logged once.
        self.failed: str | None = None

    @property
    def sha256(self) -> str | None:
        return self.opened[0] if self.opened is not None else None

    @property
    def reader(self) -> "Reader | None":
        return self.opened[2] if self.opened is not None else None

    def swap(self, opened: "tuple[str, str, Reader] | None") -> None:
        """Put ``opened`` (SHA-256, temporary path, reader) in place of the file open before, and delete that file."""
        previous, self.opened, self.failed = self.opened, opened, None
        if previous is not None:
            # Deleting an open file is safe on POSIX; elsewhere it stays until the temporary directory is cleared.
            with contextlib.suppress(OSError):
                os.unlink(previous[1])


_stored = _StoredDatabase()


def open_stored_country_database(sha256: str, content: bytes) -> "tuple[str, str, Reader]":
    """Write ``content`` to a temporary file and open it, so the C reader can map it rather than copy it.

    Raises:
        CountryDatabaseError: ``content`` cannot be opened.
        OSError: the temporary file cannot be written, which may pass.
    """
    maxminddb = _maxminddb("The stored country file")
    file = tempfile.NamedTemporaryFile(prefix="otari-country-", suffix=".mmdb", delete=False)
    try:
        with file:
            file.write(content)
    except OSError:
        os.unlink(file.name)
        raise
    try:
        reader: Reader = maxminddb.open_database(file.name)
    except (OSError, ValueError, maxminddb.InvalidDatabaseError) as exc:
        os.unlink(file.name)
        msg = f"The stored country file cannot be read as a MaxMind DB file: {type(exc).__name__}"
        raise CountryDatabaseError(msg) from exc
    return sha256, file.name, reader


def stored_country_database_seen(sha256: str) -> bool:
    """Whether ``sha256`` is the stored file this process has open, or the one it failed to open."""
    return sha256 in (_stored.sha256, _stored.failed)


def use_stored_country_database(opened: "tuple[str, str, Reader] | None") -> None:
    """Look addresses up in ``opened`` from now on, or in no stored file."""
    _stored.swap(opened)


def stored_country_database_failed(sha256: str) -> None:
    """Note that the stored file ``sha256`` failed to open, keeping the file open before."""
    _stored.failed = sha256


def reset_stored_country_database() -> None:
    """Close the stored file, so the next load starts clean (startup and tests)."""
    _stored.swap(None)


def _looked_up(database: str | None, client_host: str) -> str | None:
    try:
        address = ipaddress.ip_address(client_host)
    except ValueError:
        return None
    reader = _database.get(database) if database is not None else _stored.reader
    if reader is None:
        return None
    from maxminddb import InvalidDatabaseError  # noqa: PLC0415 - installed, since a reader is open

    try:
        record: Any = reader.get(address)
    except ValueError:
        # An IPv6 address in a file that holds IPv4 only.
        return None
    except InvalidDatabaseError as exc:
        # The file opened, but the record this address points at is damaged.
        source = f"client_country_database {database!r}" if database is not None else "The stored country file"
        logger.warning("%s: %s; the address has no country", source, exc)
        return None
    if not isinstance(record, Mapping):
        return None
    country = record.get("country")
    return _country_code(country.get("iso_code")) if isinstance(country, Mapping) else None


def client_country(
    headers: Mapping[str, str], client_host: str | None, *, header: str | None, database: str | None
) -> str | None:
    """The request's ISO 3166-1 alpha-2 country code, or None when no source knows it.

    ``headers`` is keyed in lower case, as Starlette's are. A header that is
    present decides on its own, so a value that is no country code (Cloudflare's
    ``T1`` for Tor, ``XX`` for unknown) gives None rather than a lookup of the
    proxy's view of the address. With no ``database``, the address is looked
    up in the stored country file, where one is open.
    """
    if header is not None:
        value = headers.get(header.lower())
        if value is not None:
            return _country_code(value)
    if client_host is not None:
        return _looked_up(database, client_host)
    return None


def request_country_gate(config: "GatewayConfig", headers: Mapping[str, str], client_host: str | None) -> CountryGate:
    """The country gate for one request: ``provider_supported_countries``, and a hosted provider's own list."""
    return CountryGate(
        client_country(
            headers, client_host, header=config.client_country_header, database=config.client_country_database
        )
    )
