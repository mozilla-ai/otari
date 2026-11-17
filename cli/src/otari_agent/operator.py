"""`otari operator`: deployment administration a running gateway serves over HTTP."""

from typing import Any

import click

from otari_agent.usage_import import API_ROOT

# The admin list's largest page. An email search rarely matches more than one
# account, and the exact match is picked out below.
_SEARCH_LIMIT = 1000


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
    help="The deployment's master key. Defaults to OTARI_MASTER_KEY.",
)
@click.option("--yes", "-y", is_flag=True, help="Do not ask for confirmation.")
def handover(email: str, url: str, master_key: str | None, yes: bool) -> None:
    """Make the operator signed in as EMAIL the deployment's bootstrap operator.

    The bootstrap operator is the account the master key acts as, and the one that
    cannot be deactivated or lose operator access. The account taking it must
    already be an active operator with a password. The previous bootstrap operator
    stays an operator.
    """
    import httpx

    if not master_key:
        raise click.UsageError("The master key is required. Pass --master-key or set OTARI_MASTER_KEY.")

    with httpx.Client(
        base_url=f"{url.rstrip('/')}{API_ROOT}",
        headers={"Authorization": f"Bearer {master_key}"},
        timeout=30.0,
    ) as client:
        target = _find_account(client, email)
        if not yes:
            click.confirm(
                f"Make {email} the bootstrap operator? The master key will act as their account from now on.",
                abort=True,
            )
        response = _request(client, "POST", f"/admin/users/{target['id']}/bootstrap-operator")

    click.echo(f"{response.get('email') or email} is now the bootstrap operator.")


def _find_account(client: Any, email: str) -> dict[str, Any]:
    """Resolve EMAIL to its account, matching the whole address case-insensitively."""
    body = _request(client, "GET", "/admin/users", params={"search": email, "limit": _SEARCH_LIMIT})
    wanted = email.strip().lower()
    for account in body.get("data", []):
        if (account.get("email") or "").lower() == wanted:
            return dict(account)
    raise click.ClickException(f"No account on this deployment signs in as {email}.")


def _request(client: Any, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
    """Send one request and turn a refusal into the server's own reason."""
    import httpx

    try:
        response = client.request(method, path, **kwargs)
    except httpx.HTTPError as exc:
        raise click.ClickException(f"Could not reach the gateway: {exc}") from exc
    if response.status_code in (401, 404) and path.startswith("/admin/users") and method == "GET":
        # The admin surface answers 404 to anybody who is not an operator.
        raise click.ClickException("The gateway refused the master key. Check --master-key and --url.")
    if response.status_code >= 400:
        raise click.ClickException(f"The gateway refused ({response.status_code}): {_detail(response)}")
    result: dict[str, Any] = response.json()
    return result


def _detail(response: Any) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    return str(detail) if detail else response.text[:500]
