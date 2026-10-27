"""The country a request comes from, for the provider country gate.

A trusted proxy's header (``client_country_header``, such as Cloudflare's
``CF-IPCountry``) is read first, then the client address is looked up in a
MaxMind-format file (``client_country_database``). The address is the one
uvicorn resolved behind ``forwarded_allow_ips``.
"""

import io
import ipaddress
import os
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


def _looked_up(database: str, client_host: str) -> str | None:
    try:
        address = ipaddress.ip_address(client_host)
    except ValueError:
        return None
    reader = _database.get(database)
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
        logger.warning("client_country_database %r: %s; the address has no country", database, exc)
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
    proxy's view of the address.
    """
    if header is not None:
        value = headers.get(header.lower())
        if value is not None:
            return _country_code(value)
    if database is not None and client_host is not None:
        return _looked_up(database, client_host)
    return None


def request_country_gate(config: "GatewayConfig", headers: Mapping[str, str], client_host: str | None) -> CountryGate:
    """The country gate for one request: ``provider_supported_countries``, and a hosted provider's own list."""
    return CountryGate(
        client_country(
            headers, client_host, header=config.client_country_header, database=config.client_country_database
        )
    )
