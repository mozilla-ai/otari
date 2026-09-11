"""Which Apprise destinations Otari accepts, how they are masked, and the SSRF gate.

Three things a destination string has to survive before it is stored: Apprise
has to recognize it (:mod:`otari_alerts.dispatcher`), this module's allowlist
has to classify it, and, when its netloc is an address the operator chose, that
address has to be one this deployment may post to.

The address loop itself is Otari's, reached through
``gateway.services.url_safety.reject_internal_host``: the literal-or-resolve
walk and the reserved-range table behind the MCP and web-search gates are the
same here, and a copy would be a second place to fix a CIDR.
"""

from urllib.parse import urlsplit, urlunsplit

from gateway.services.url_safety import UnsafeURLError, reject_internal_host
from otari_alerts.config import allow_private_hosts

# Which Apprise schemas Otari accepts as an alert destination, split by what
# their netloc means.
#
# The split is the whole SSRF gate, so it is an allowlist in both directions: a
# schema nobody has classified is refused rather than assumed safe. Assuming
# the opposite is wrong in a way that matters: ``mailto://``, ``gotify://``,
# ``ntfy://``, ``matrix://``, ``rocket://`` and ``mmost://`` all dial a host the
# operator wrote, so treating everything outside the webhook schemas as "posts
# to a vendor endpoint" would let them through unchecked.
#
# Adding a schema is one line here.

# The netloc is a host the operator chose, so it is address-checked.
ALERT_SCHEMES_WITH_OPERATOR_HOST = frozenset(
    {
        "apprise",
        "apprises",
        "form",
        "forms",
        "gotify",
        "gotifys",
        "json",
        "jsons",
        "mailto",
        "mailtos",
        "matrix",
        "matrixs",
        "mmost",
        "mmosts",
        "ncloud",
        "nclouds",
        "ntfy",
        "rocket",
        "rockets",
        "xml",
        "xmls",
    }
)

# The netloc is a credential: the plugin posts to an endpoint compiled into it,
# so there is no operator-chosen address to check.
ALERT_SCHEMES_WITH_FIXED_ENDPOINT = frozenset(
    {
        "discord",
        "msteams",
        "opsgenie",
        "pagerduty",
        "pbul",
        "pover",
        "ses",
        "signal",
        "slack",
        "sns",
        "tgram",
        "twilio",
    }
)

SUPPORTED_ALERT_SCHEMES = ALERT_SCHEMES_WITH_OPERATOR_HOST | ALERT_SCHEMES_WITH_FIXED_ENDPOINT


def redact_alert_destination(value: str) -> str:
    """Mask an Apprise destination down to the part that is safe to display.

    Stricter than ``gateway.services.url_safety.redact_url_secrets``, which is
    why this exists: Apprise carries credentials in the URL **path**, not only
    in the userinfo and query. ``discord://webhook_id/webhook_token`` is
    entirely path, so that function would return the token untouched.

    Scheme and host survive, because those are what make a row recognizable in
    a list. Everything that can carry a secret is replaced wholesale: userinfo,
    every path segment, every query value. A schema whose netloc is a token
    rather than a host loses that too, so ``slack://botA/botB`` reads
    ``slack://***``.
    """
    try:
        parts = urlsplit(value)
    except ValueError:
        return "***"
    scheme = parts.scheme.lower()
    if not scheme:
        return "***"

    host = parts.hostname if scheme in ALERT_SCHEMES_WITH_OPERATOR_HOST else None
    netloc = host or "***"
    path = "/***" if parts.path.strip("/") else ""
    query = "***" if parts.query else ""
    return urlunsplit((parts.scheme, netloc, path, query, ""))


async def validate_alert_destination(scheme: str, host: str | None) -> None:
    """Reject an alert destination that points inside the deployment.

    Takes the schema and the address Apprise resolved rather than a URL,
    because only the plugin knows which of the two the netloc was:
    ``slack://tokA/tokB`` parses ``tokA`` into the host slot and it is not one.

    A gate of its own rather than ``validate_outbound_fetch_url``, whose
    override is ``OTARI_WEB_SEARCH_ALLOW_PRIVATE_HOSTS``: an operator turning on
    private alert destinations must not thereby let the web-search backend fetch
    internal hosts. The loop itself is shared through ``reject_internal_host``.

    The check is at write time and Apprise re-resolves when it sends, so this is
    TOCTOU-vulnerable to rebinding, the same as the MCP and web-search write
    paths. It is fail-closed by default with an environment-only override
    (``OTARI_ALERT_ALLOW_PRIVATE_HOSTS``) rather than a dashboard toggle,
    matching what ``gateway.services.runtime_settings_service`` says about SSRF
    gates.

    Raises:
        UnsafeURLError: If the destination names no host it should have, or
            resolves to a private, loopback or reserved address.

    """
    if scheme.lower() in ALERT_SCHEMES_WITH_FIXED_ENDPOINT:
        return
    if not host:
        raise UnsafeURLError(f"alert destination {scheme!r} must name a host")
    if allow_private_hosts():
        return
    await reject_internal_host(host, host_label="alert destination", override_var="OTARI_ALERT_ALLOW_PRIVATE_HOSTS")
