"""End-to-end reproduction + fix verification for issue #106.

Drives the *real* FastAPI app (real ASGI lifespan, real ``init_db``, real auth
dependency) against a file-backed SQLite database under genuine concurrency.
Each request opens its own connection (NullPool), so the contention is real.

Issue #106 reports intermittent HTTP 500s against a standalone SQLite gateway:
an unknown key occasionally returning 500 instead of 401, and a cold-start 500
on the first request right after ``POST /v1/keys``. Root cause: SQLite's
default rollback-journal mode makes readers and writers block each other, so a
connection raises "database is locked" and the gateway surfaces it as a 500.

Two scenarios:

  A. Writer blocked by a concurrent reader (deterministic; isolates WAL).
     A read snapshot is held open (any in-flight read: an auth lookup, a key
     listing, the usage reader) while ``POST /v1/keys`` runs. In rollback-journal
     mode the writer cannot take EXCLUSIVE while the reader holds SHARED, so it
     times out -> 500. WAL readers never block the writer -> 200. This is the
     cold-start "500 right after key creation" symptom.

  B. Burst load on the shipped config. 40 concurrent key creations + 60
     concurrent unknown-key auth lookups must yield zero 5xx, and every unknown
     key must get a clean 401.
"""

import asyncio
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncEngine

from gateway.auth.models import generate_api_key
from gateway.core import database
from gateway.core.config import API_KEY_HEADER, GatewayConfig

MASTER_KEY = "e2e-master-key"


def _make_pragma_installer(journal_mode: str | None, busy_timeout_ms: int) -> Any:
    def installer(engine: AsyncEngine) -> None:
        @event.listens_for(engine.sync_engine, "connect")
        def _set(dbapi_connection: Any, _: Any) -> None:
            cur = dbapi_connection.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            if journal_mode is not None:
                cur.execute(f"PRAGMA journal_mode={journal_mode}")
            cur.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
            cur.close()

    return installer


def _new_app(installer: Any | None) -> Any:
    from gateway.main import create_app

    database.reset_db()
    tmp = Path(tempfile.mkdtemp()) / "e2e.db"
    config = GatewayConfig(
        database_url=f"sqlite:///{tmp}",
        master_key=MASTER_KEY,
        auto_migrate=True,
        enable_metrics=False,
        require_pricing=False,
    )
    if installer is not None:
        database._configure_sqlite_pragmas = installer
    return create_app(config)


async def _scenario_a_write_blocked_by_read(journal_label: str, installer: Any | None) -> int:
    """Hold a read snapshot open, then create a key. Returns the POST status."""
    original = database._configure_sqlite_pragmas
    app = _new_app(installer)
    try:
        async with app.router.lifespan_context(app):

            async def hold_read() -> None:
                async with database._engine.connect() as conn:  # type: ignore[union-attr]
                    await conn.execute(text("BEGIN"))
                    await conn.execute(text("SELECT count(*) FROM api_keys"))  # acquire SHARED
                    await asyncio.sleep(0.8)  # hold the snapshot
                    await conn.rollback()

            async def create_key() -> int:
                await asyncio.sleep(0.2)  # ensure the read snapshot is held first
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://e2e") as client:
                    r = await client.post(
                        "/v1/keys",
                        headers={API_KEY_HEADER: f"Bearer {MASTER_KEY}"},
                        json={"key_name": "cold-start"},
                    )
                    return r.status_code

            _, status_code = await asyncio.gather(hold_read(), create_key())
    finally:
        database._configure_sqlite_pragmas = original
        database.reset_db()

    print(f"  journal={journal_label:<6} POST /v1/keys with a read snapshot held -> HTTP {status_code}")
    return status_code


async def _scenario_b_burst_shipped(writers: int = 40, readers: int = 60) -> tuple[Counter[int], Counter[int]]:
    """Burst load on the real shipped pragmas; expect zero 5xx and clean 401s."""
    app = _new_app(installer=None)  # real _configure_sqlite_pragmas (WAL + 5000ms)
    reader_status: Counter[int] = Counter()
    writer_status: Counter[int] = Counter()
    try:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://e2e") as client:

                async def write(i: int) -> None:
                    r = await client.post(
                        "/v1/keys",
                        headers={API_KEY_HEADER: f"Bearer {MASTER_KEY}"},
                        json={"key_name": f"k{i}"},
                    )
                    writer_status[r.status_code] += 1

                async def read_unknown(_: int) -> None:
                    unknown = generate_api_key()  # valid format, not in DB
                    r = await client.get("/v1/models", headers={API_KEY_HEADER: f"Bearer {unknown}"})
                    reader_status[r.status_code] += 1

                await asyncio.gather(*[write(i) for i in range(writers)], *[read_unknown(i) for i in range(readers)])
    finally:
        database.reset_db()

    print(f"  burst: writer={dict(sorted(writer_status.items()))}  reader={dict(sorted(reader_status.items()))}")
    return reader_status, writer_status


async def main() -> int:
    print("\nScenario A - cold-start write blocked by a concurrent read snapshot:")
    delete_status = await _scenario_a_write_blocked_by_read(
        "delete", _make_pragma_installer(journal_mode=None, busy_timeout_ms=200)
    )
    wal_status = await _scenario_a_write_blocked_by_read(
        "WAL", _make_pragma_installer(journal_mode="WAL", busy_timeout_ms=200)
    )
    shipped_status = await _scenario_a_write_blocked_by_read("shipped", None)

    print("\nScenario B - burst load on the shipped config:")
    readers, writers = await _scenario_b_burst_shipped()

    reproduced = delete_status >= 500
    fixed_wal = wal_status < 500
    fixed_shipped = shipped_status < 500
    burst_clean = all(s < 500 for s in readers) and all(s < 500 for s in writers)
    auth_401 = set(readers) == {401}

    print("\n--- verdict ---")
    print(f"  A: pre-fix (rollback journal) reproduces the cold-start 500: {reproduced}  (got {delete_status})")
    print(f"  A: WAL fixes it: {fixed_wal}  (got {wal_status})")
    print(f"  A: shipped config fixes it: {fixed_shipped}  (got {shipped_status})")
    print(f"  B: burst load on shipped config has zero 5xx: {burst_clean}")
    print(f"  B: every unknown key got a clean 401: {auth_401}")

    ok = reproduced and fixed_wal and fixed_shipped and burst_clean and auth_401
    print(f"\n  RESULT: {'PASS - problem reproduced and fix verified end-to-end' if ok else 'INCONCLUSIVE'}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
