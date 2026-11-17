"""`otari operator`: deployment administration a running gateway serves over HTTP."""

from typing import Any
from urllib.parse import urlsplit

import click

from otari_agent.usage_import import API_ROOT

# The admin list's largest page. An email search rarely matches more than one
# account, and the exact match is picked out below.
_SEARCH_LIMIT = 1000

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


@click.group(name="operator")
def operator_group() -> None:
    """Administer a deployment's operators."""


@operator_group.command(name="handover")
@click.argument("email")
@click.option("--url", envvar="OTARI_URL", default="http://localhost:8000", help="Base URL of the Otari gateway.")
@click.option(
    "--master-key",
    envvar="OTARI_MASTER_KEY",
    default=None,
    help="The deployment's master key. Prefer setting OTARI_MASTER_KEY, which keeps it out of shell history.",
)
@click.option("--yes", "-y", is_flag=True, help="Do not ask for confirmation.")
def handover(email: str, url: str, master_key: str | None, yes: bool) -> None:
    """Make the operator signed in as EMAIL the deployment's bootstrap operator.

    The bootstrap operator is the account the master key acts as, and the one that
    cannot be deactivated or lose operator access. The account taking it must
    already be an active operator with a verified address and a password. The
    previous bootstrap operator stays an operator.
    """
    import httpx

    if not master_key:
        raise click.UsageError("The master key is required. Set OTARI_MASTER_KEY or pass --master-key.")
    parts = urlsplit(url)
    if parts.scheme != "https" and parts.hostname not in _LOOPBACK_HOSTS:
        raise click.UsageError(f"Refusing to send the master key to {url} without HTTPS. Use an https:// URL.")

    with httpx.Client(
        base_url=f"{url.rstrip('/')}{API_ROOT}",
        headers={"Authorization": f"Bearer {master_key}"},
        timeout=30.0,
        # Plain HTTP is allowed only to this machine, so no proxy from the environment may carry it elsewhere.
        trust_env=parts.scheme == "https",
    ) as client:
        target = _find_account(client, email)
        if not yes:
            click.confirm(
                f"Make {target['email']} the bootstrap operator? The master key will act as their account from now on.",
                abort=True,
            )
        _request(client, "POST", f"/admin/users/{target['id']}/bootstrap-operator")

    click.echo(f"{target['email']} is now the bootstrap operator.")
    click.echo(
        "The master key now acts as their account, including resetting its password. "
        "If anyone who should no longer have it holds the master key, rotate it: from Settings for a key "
        "Otari generated, or by changing OTARI_MASTER_KEY where you set it."
    )


def _find_account(client: Any, email: str) -> dict[str, Any]:
    """Resolve EMAIL to its one account, matching the whole address case-insensitively."""
    body = _request(client, "GET", "/admin/users", params={"search": email, "limit": _SEARCH_LIMIT})
    if body.get("count", 0) > len(body.get("data", [])):
        raise click.ClickException(
            f"More than {_SEARCH_LIMIT} accounts match {email}, so it cannot be told apart here. "
            "Hand over from Accounts instead."
        )
    wanted = email.strip().lower()
    matches = [account for account in body.get("data", []) if (account.get("email") or "").lower() == wanted]
    if not matches:
        raise click.ClickException(f"No account on this deployment signs in as {email}.")
    if len(matches) > 1:
        raise click.ClickException(
            f"More than one account signs in as {email} in different letter cases. Hand over from Accounts instead."
        )
    return dict(matches[0])


def _request(client: Any, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
    """Send one request and turn anything but a success into the server's own reason."""
    import httpx

    try:
        response = client.request(method, path, **kwargs)
    except httpx.HTTPError as exc:
        raise click.ClickException(f"Could not reach the gateway: {exc}") from exc
    if response.status_code in (401, 404) and method == "GET":
        # The admin surface answers 404 to anybody who is not an operator.
        raise click.ClickException("The gateway refused the master key. Check OTARI_MASTER_KEY and --url.")
    if not 200 <= response.status_code < 300:
        raise click.ClickException(f"The gateway refused ({response.status_code}): {_detail(response)}")
    if response.status_code == 204:
        return {}
    try:
        result: dict[str, Any] = response.json()
    except ValueError as exc:
        message = f"The gateway answered with something other than JSON: {response.text[:200]}"
        raise click.ClickException(message) from exc
    return result


def _detail(response: Any) -> str:
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    return str(detail) if detail else response.text[:500]
