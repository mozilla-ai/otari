from pathlib import Path
from typing import Never

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from gateway.core.config import GatewayConfig
from gateway.db import APIKey, User
from gateway.main import create_app
from gateway.services.mail import Mailer


def test_create_app_bootstraps_first_api_key(tmp_path: Path) -> None:
    database_path = tmp_path / "bootstrap.db"
    config = GatewayConfig(database_url=f"sqlite:///{database_path}")
    app = create_app(config)

    with TestClient(app):
        pass

    engine = create_engine(config.database_url)
    with Session(engine) as db:
        keys = db.query(APIKey).all()
        users = db.query(User).all()

    engine.dispose()

    assert len(keys) == 1
    assert keys[0].key_name == "bootstrap"
    assert keys[0].metadata_.get("bootstrap") is True
    # The bootstrap key is the operator's first and most-used key, so it must carry a
    # fingerprint like any other (regression guard for the third mint path).
    assert keys[0].key_prefix is not None
    assert keys[0].key_prefix.startswith("gw-")
    assert len(users) == 1
    # The bootstrap key has no explicit owner, so it lands on the shared "default"
    # user rather than a per-key virtual one.
    assert users[0].user_id == "default"
    assert users[0].user_id == keys[0].user_id


def test_create_app_does_not_create_second_bootstrap_key(tmp_path: Path) -> None:
    database_path = tmp_path / "bootstrap-once.db"
    config = GatewayConfig(database_url=f"sqlite:///{database_path}")

    app = create_app(config)
    with TestClient(app):
        pass
    # Second startup should not create another key
    app_again = create_app(config)
    with TestClient(app_again):
        pass

    engine = create_engine(config.database_url)
    with Session(engine) as db:
        key_count = db.query(APIKey).count()
    engine.dispose()

    assert key_count == 1


def test_create_app_skips_bootstrap_when_disabled(tmp_path: Path) -> None:
    database_path = tmp_path / "no-bootstrap.db"
    config = GatewayConfig(database_url=f"sqlite:///{database_path}", bootstrap_api_key=False)

    app = create_app(config)
    with TestClient(app):
        pass

    engine = create_engine(config.database_url)
    with Session(engine) as db:
        key_count = db.query(APIKey).count()
        user_count = db.query(User).count()
    engine.dispose()

    assert key_count == 0
    assert user_count == 0


def test_first_boot_seeds_a_fresh_database_without_sending_mail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A brand new deployment must complete first-run seeding with no mail configured.

    Startup is the mail-dependent surface that is easiest to miss, because it is
    on nobody's feature list: the obvious ones are invitations and password
    reset. otari.ai's equivalent (``app/initial_data.py``) gets it wrong today,
    reaching a bare ``assert settings.emails_enabled`` inside ``send_email``
    while seeding its first superuser, so a fresh database with no SMTP cannot
    finish seeding at all and a second run only succeeds because the superuser
    already exists by then.

    Otari does not reach mail here today, for a reason that expires: its
    first-run seeding mints a master key and a bootstrap API key, and neither
    carries an address. Per-user sign-in (#649) and signup (#650) are what put
    an address in this path, so this asserts the property now rather than
    leaving it to be rediscovered. Patching ``send`` to raise catches *both*
    wrong answers, since an unconfigured send would otherwise return
    ``delivered=False`` and let seeding continue as if nothing were missing.
    """

    async def _refuse(*_args: object, **_kwargs: object) -> Never:
        raise AssertionError("first-boot seeding must not send mail")

    monkeypatch.setattr(Mailer, "send", _refuse)
    # No mail settings at all: the state every self-hoster who wants no email is in.
    config = GatewayConfig(database_url=f"sqlite:///{tmp_path / 'first-boot.db'}")

    with TestClient(create_app(config)) as client:
        assert client.get("/health").status_code == 200
        # And the deployment reports the truth about itself rather than failing later.
        assert client.get("/v1/bootstrap").json()["mail_ready"] is False

    engine = create_engine(config.database_url)
    with Session(engine) as db:
        assert db.query(APIKey).count() == 1
    engine.dispose()
