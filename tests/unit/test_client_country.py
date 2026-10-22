"""The country a request comes from: a trusted proxy's header, else the address looked up in a local file."""

from pathlib import Path

import pytest
from _mmdb_support import write_country_database
from pydantic import ValidationError

from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.main import _validate_client_country_database
from gateway.services.inference import CountryDatabaseError, _client_country, client_country, open_country_database


@pytest.fixture
def database(tmp_path: Path) -> str:
    return str(write_country_database(tmp_path / "countries.mmdb", {"192.0.2.0/24": "DE", "198.51.100.0/24": "US"}))


def test_the_address_is_looked_up_in_the_database(database: str) -> None:
    assert client_country({}, "192.0.2.7", header=None, database=database) == "DE"


def test_an_address_the_database_does_not_hold_has_no_country(database: str) -> None:
    assert client_country({}, "203.0.113.1", header=None, database=database) is None


def test_a_client_that_is_not_an_address_has_no_country(database: str) -> None:
    assert client_country({}, "testclient", header=None, database=database) is None
    assert client_country({}, None, header=None, database=database) is None


def test_the_trusted_header_wins_over_the_database(database: str) -> None:
    headers = {"cf-ipcountry": "fr"}

    assert client_country(headers, "192.0.2.7", header="CF-IPCountry", database=database) == "FR"


def test_a_missing_header_falls_back_to_the_database(database: str) -> None:
    assert client_country({}, "198.51.100.1", header="CF-IPCountry", database=database) == "US"


@pytest.mark.parametrize("value", ["T1", "XX", "", "DEU", "1"])
def test_a_header_value_that_is_no_country_code_is_no_country(value: str) -> None:
    assert client_country({"cf-ipcountry": value}, "192.0.2.7", header="CF-IPCountry", database=None) is None


def test_a_code_the_database_gives_no_country_is_no_country(tmp_path: Path) -> None:
    # DB-IP gives ZZ to an address it cannot place.
    path = write_country_database(tmp_path / "countries.mmdb", {"192.0.2.0/24": "ZZ"})

    assert client_country({}, "192.0.2.7", header=None, database=str(path)) is None


def test_no_source_gives_no_country() -> None:
    assert client_country({"cf-ipcountry": "DE"}, "192.0.2.7", header=None, database=None) is None


def test_a_replaced_file_is_read_again(tmp_path: Path) -> None:
    path = tmp_path / "countries.mmdb"
    write_country_database(path, {"192.0.2.0/24": "DE"})
    assert client_country({}, "192.0.2.7", header=None, database=str(path)) == "DE"

    write_country_database(tmp_path / "next.mmdb", {"192.0.2.0/24": "NL"}).replace(path)

    assert client_country({}, "192.0.2.7", header=None, database=str(path)) == "NL"


def test_a_file_that_failed_is_not_opened_again_until_it_is_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "countries.mmdb"
    path.write_bytes(b"not a database")
    opened: list[str] = []

    def counting_open(database: str) -> object:
        opened.append(database)
        return open_country_database(database)

    monkeypatch.setattr(_client_country, "open_country_database", counting_open)

    assert client_country({}, "192.0.2.7", header=None, database=str(path)) is None
    assert client_country({}, "192.0.2.7", header=None, database=str(path)) is None
    assert len(opened) == 1

    write_country_database(tmp_path / "next.mmdb", {"192.0.2.0/24": "NL"}).replace(path)

    assert client_country({}, "192.0.2.7", header=None, database=str(path)) == "NL"
    assert len(opened) == 2


def test_a_missing_file_is_logged_once_until_it_is_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    warnings: list[object] = []
    monkeypatch.setattr(logger, "warning", lambda *args: warnings.append(args))
    path = tmp_path / "countries.mmdb"

    assert client_country({}, "192.0.2.7", header=None, database=str(path)) is None
    assert client_country({}, "192.0.2.7", header=None, database=str(path)) is None
    assert len(warnings) == 1

    write_country_database(path, {"192.0.2.0/24": "DE"})
    assert client_country({}, "192.0.2.7", header=None, database=str(path)) == "DE"
    path.unlink()

    assert client_country({}, "192.0.2.7", header=None, database=str(path)) is None
    assert len(warnings) == 2


def test_a_damaged_record_has_no_country(tmp_path: Path) -> None:
    path = write_country_database(tmp_path / "countries.mmdb", {"192.0.2.0/24": "DE"})
    raw = path.read_bytes()
    # The record's first bytes, after the 16-byte separator, become an extended type that does not exist.
    record = raw.index(b"\x00" * 16 + b"\xe1") + 16
    path.write_bytes(raw[:record] + b"\x00\xff" + raw[record + 2 :])

    assert client_country({}, "192.0.2.7", header=None, database=str(path)) is None


def test_opening_a_missing_file_says_which_setting(tmp_path: Path) -> None:
    with pytest.raises(CountryDatabaseError, match="client_country_database"):
        open_country_database(str(tmp_path / "missing.mmdb"))


def test_opening_a_file_that_is_no_database_fails(tmp_path: Path) -> None:
    path = tmp_path / "countries.mmdb"
    path.write_bytes(b"not a database")

    with pytest.raises(CountryDatabaseError, match="client_country_database"):
        open_country_database(str(path))


def test_the_sources_are_off_by_default() -> None:
    config = GatewayConfig()

    assert config.client_country_header is None
    assert config.client_country_database is None


def test_a_blank_source_is_off() -> None:
    config = GatewayConfig(client_country_header="  ", client_country_database="")

    assert config.client_country_header is None
    assert config.client_country_database is None


def test_a_header_that_is_no_header_name_is_refused() -> None:
    with pytest.raises(ValidationError, match="client_country_header"):
        GatewayConfig(client_country_header="CF IPCountry")


def test_startup_refuses_a_database_it_cannot_read(tmp_path: Path) -> None:
    config = GatewayConfig(client_country_database=str(tmp_path / "missing.mmdb"))

    with pytest.raises(ValueError, match="client_country_database"):
        _validate_client_country_database(config)


def test_startup_accepts_a_readable_database(database: str) -> None:
    _validate_client_country_database(GatewayConfig(client_country_database=database))
