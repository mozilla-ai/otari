"""Text shapes that are credentials, and masking them.

Narrow on purpose: each pattern matches a credential and nothing an operator
reads a log for, so it can run over every log line. The redaction of upstream
messages (``services/upstream_redaction``) starts from these and adds the
broader shapes a client-facing error must not carry.
"""

import re

REDACTION_PLACEHOLDER = "[redacted]"

CREDENTIAL_SHAPES: tuple[re.Pattern[str], ...] = (
    # Authorization-header credentials, whatever scheme. The scheme is a word of
    # its own (not the tail of ``api_token``), and an access log's ``HTTP/1.1``
    # after a query value is not a credential.
    re.compile(r"\b(?:Bearer|Basic|Token)\s+(?!HTTP/)[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE),
    # Provider key formats: a known prefix plus the key body. Covers OpenAI
    # (sk-, sk-proj-), Anthropic (sk-ant-), Groq (gsk_), xAI (xai-), and
    # otari's own keys (tk-, tk_, legacy gw-).
    re.compile(r"\b(?:sk|pk|rk|gsk|xai|tk|gw)[-_][A-Za-z0-9._-]{8,}", re.IGNORECASE),
    # Keys whose prefix is the word itself (Mailgun's ``key-``). Hyphen only and a
    # digit required, so prose such as ``key-rotation-schedule`` stays readable.
    re.compile(r"\bkey-(?=[A-Za-z0-9._-]*\d)[A-Za-z0-9._-]{8,}", re.IGNORECASE),
    # Google API keys, which carry a fixed prefix and no separator.
    re.compile(r"\bAIza[A-Za-z0-9._-]{10,}"),
    # GitHub tokens: OAuth, personal, user-to-server, server-to-server, refresh,
    # and fine-grained personal tokens.
    re.compile(r"\b(?:gh[opusr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{22,})"),
    # AWS access key IDs.
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    # Slack tokens.
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    # A prefixless key of 32 characters or more, as Mistral issues. Upper case, lower
    # case and digits together tell it from a hex id or a commit SHA an operator
    # debugs with; the upstream redactor masks every such run, hex included.
    re.compile(
        r"(?<![A-Za-z0-9_-])(?=[A-Za-z0-9_-]*[a-z])(?=[A-Za-z0-9_-]*[A-Z])(?=[A-Za-z0-9_-]*\d)"
        r"[A-Za-z0-9_-]{32,}(?![A-Za-z0-9_-])"
    ),
    # A password embedded in a URL's userinfo; the host stays readable.
    re.compile(r"(?<=://)[^/\s:@]+:[^/\s@]+(?=@)"),
    # An explicitly named credential, as it appears in a query string or an
    # echoed request body. A ``${VAR}`` reference names where a credential
    # comes from rather than holding one, so it stays.
    re.compile(r"\b(?:api[_-]?key|access[_-]?token|token|secret|password)\s*[=:]\s*(?!\$\{)\S+", re.IGNORECASE),
)


def mask_credentials(text: str, placeholder: str = REDACTION_PLACEHOLDER) -> str:
    """Replace every credential-shaped run in ``text`` with ``placeholder``."""
    for pattern in CREDENTIAL_SHAPES:
        text = pattern.sub(placeholder, text)
    return text
