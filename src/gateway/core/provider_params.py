"""Provider-call parameters a caller, or a default standing in for one, may never set."""

# any-llm's ``*Params`` are provider-call models, so a future any-llm version
# could add a credential / transport / provider-selection field or a raw-body
# extension. Derivation picks up new fields automatically (the point for benign
# params), but exposing one of these as a client-settable request field would let
# a caller override an operator-controlled value: the provider-call merge spreads
# request fields last (e.g. ``{**get_provider_kwargs(...), **request_fields}``),
# so a client value would win. The gateway resolves these itself (from ``config``
# / the platform service), so they must never be derived onto a public request
# schema, and they are also stripped before forwarding (see
# ``_tools._strip_gateway_fields``) to cover schemas that allow extra fields
# (the Responses request uses ``extra="allow"``).
SENSITIVE_PARAM_FIELDS: frozenset[str] = frozenset(
    {
        "api_key",
        "api_base",
        "base_url",
        "provider",
        "organization",
        "api_version",
        "client",
        "credentials",
        "extra_body",
        "aws_access_key_id",
        "aws_secret_access_key",
        # any-llm's acompletion()/aresponses() forward this dict straight to
        # the provider's client constructor (see build_attempt_client_args in
        # _platform.py). A hybrid-mode attempt with no extra_params of its own
        # (the common case for every provider except Bedrock) never sets this
        # key at all, so a caller-supplied client_args smuggled in through a
        # schema that allows extra fields would otherwise reach the provider
        # call unfiltered.
        "client_args",
    }
)

# What an owned endpoint may not carry as a default request field: the fields
# above, and the ones that shape the request itself, because a default travels
# in the body over the value the gateway set.
FORBIDDEN_ENDPOINT_DEFAULTS: frozenset[str] = SENSITIVE_PARAM_FIELDS | {"model", "messages", "input", "stream"}


__all__ = ["FORBIDDEN_ENDPOINT_DEFAULTS", "SENSITIVE_PARAM_FIELDS"]
