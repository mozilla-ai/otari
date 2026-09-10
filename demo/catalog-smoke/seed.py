"""Seed the smoke gateway with three people to sign in as, once.

Run by ``run.sh`` after the gateway answers ``/health``. Standard library
only, so it runs on any host with Python 3. Idempotent by way of the marker
``run.sh`` writes after a successful seed; ``--reset`` clears it with the
database.

What it makes:

- **Platform admin.** The bootstrap operator, claimed with an email and a
  password. Deployment-wide standing: Settings, Model pricing's catalog
  controls, Accounts, every provider. The master key stays the API credential.
- **Org admin.** ``Acme``, a second organization, with an admin who can set
  Acme's rate overrides, provider keys, members and budgets but not the
  deployment's.
- **Member.** A plain member of Acme: reads the catalog at Acme's rates,
  makes their own API keys, sees their own usage.

Both Acme people are added to the roster, then claimed through signup and
verified from the link the console mail transport writes to the gateway log,
which is the path a real invitee walks with real mail.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = os.environ["OTARI_BASE"]
MASTER_KEY = os.environ["OTARI_MASTER_KEY"]
LOG = Path(os.environ["OTARI_LOG"])
PASSWORD = os.environ.get("OTARI_SMOKE_PASSWORD", "smoke-test-1234")

OPERATOR = "operator@otari.local"
ACME_ADMIN = "admin@acme.local"
ACME_MEMBER = "member@acme.local"

_TOKEN = re.compile(r"token=([\w-]+)")


def call(method: str, path: str, body: dict | None = None, *, auth: bool = True) -> tuple[int, dict | list | None]:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(f"{BASE}{path}", data=data, method=method)
    request.add_header("Content-Type", "application/json")
    if auth:
        request.add_header("Authorization", f"Bearer {MASTER_KEY}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
            return response.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            return exc.code, json.loads(raw)
        except ValueError:
            return exc.code, {"detail": raw.decode(errors="replace")}


def must(status: int, payload: object, what: str, *ok: int) -> None:
    if status not in (ok or (200, 201)):
        sys.exit(f"seed: {what} failed with {status}: {payload}")


def claim_operator() -> None:
    status, payload = call("PUT", "/v1/auth/password", {"email": OPERATOR, "new_password": PASSWORD})
    # 409 is the operator already claimed on an earlier run with a kept database.
    must(status, payload, "claiming the operator", 200, 409)


def organization_named(name: str) -> str | None:
    status, payload = call("GET", "/v1/organizations/me/memberships?skip=0&limit=100")
    must(status, payload, "listing organizations")
    rows = payload["data"] if isinstance(payload, dict) and "data" in payload else payload
    for row in rows or []:
        organization = row.get("organization", row)
        if organization.get("name") == name:
            return str(organization["id"])
    return None


def ensure_acme() -> tuple[str, str]:
    """Acme's id and the operator's home organization's id."""
    home = organization_named("Default organization")
    acme = organization_named("Acme")
    if acme is None:
        status, payload = call("POST", "/v1/organizations", {"name": "Acme"})
        must(status, payload, "creating Acme")
        acme = str(payload["id"])
    if home is None:
        sys.exit("seed: the default organization is missing")
    return acme, home


def active_workspaces() -> list[str]:
    """The caller's active organization's workspaces, the default one first."""
    status, payload = call("GET", "/v1/workspaces?skip=0&limit=100")
    must(status, payload, "listing workspaces")
    rows = payload["data"] if isinstance(payload, dict) and "data" in payload else payload
    return [str(row["id"]) for row in rows or []]


def add_member(email: str, role: str, workspaces: list[str]) -> None:
    """Add to the active organization, and into its workspaces at the same rank.

    An organization membership alone leaves a person with no workspace, and a
    workspace is where a key, a usage row and the Models page live.
    """
    body = {
        "email": email,
        "role": role,
        "workspace_assignments": [{"workspace_id": workspace_id, "role": role} for workspace_id in workspaces],
    }
    status, payload = call("POST", "/v1/organizations/me/members", body)
    # 409 is the address already on the roster from an earlier run.
    must(status, payload, f"adding {email}", 201, 409)


def signup_and_verify(email: str) -> None:
    before = LOG.stat().st_size if LOG.exists() else 0
    status, payload = call(
        "POST", "/v1/auth/signup", {"email": email, "password": PASSWORD, "terms_accepted": True}, auth=False
    )
    must(status, payload, f"signing up {email}")
    # The console transport writes the verification mail, link and all, to the
    # gateway log. Read what landed after the request.
    token = None
    for _ in range(50):
        time.sleep(0.2)
        with LOG.open("rb") as handle:
            handle.seek(before)
            tail = handle.read().decode(errors="replace")
        matches = _TOKEN.findall(tail)
        if matches:
            token = matches[-1]
            break
    if token is None:
        # Already claimed on an earlier run: signup answers 200 without writing
        # anything, by design. Sign-in will tell.
        return
    status, payload = call("POST", "/v1/auth/verify-email", {"token": token}, auth=False)
    must(status, payload, f"verifying {email}")


def sign_in_works(email: str) -> bool:
    status, _ = call("POST", "/v1/auth/session", {"email": email, "password": PASSWORD}, auth=False)
    return status == 200


def main() -> None:
    claim_operator()
    acme, home = ensure_acme()
    call("POST", "/v1/organizations/me/switch", {"organization_id": acme})
    workspaces = active_workspaces()
    add_member(ACME_ADMIN, "admin", workspaces)
    add_member(ACME_MEMBER, "member", workspaces)
    call("POST", "/v1/organizations/me/switch", {"organization_id": home})
    for email in (ACME_ADMIN, ACME_MEMBER):
        signup_and_verify(email)
    for email in (OPERATOR, ACME_ADMIN, ACME_MEMBER):
        if not sign_in_works(email):
            sys.exit(f"seed: {email} cannot sign in; see {LOG}")
    print("seed: operator, Acme admin and member can sign in")


if __name__ == "__main__":
    main()
