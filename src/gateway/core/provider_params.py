"""Provider-call parameters a caller, a default standing in for one, or a tenant's stored key may never set."""

from collections.abc import Mapping
from typing import Any

# any-llm's ``*Params`` are provider-call models, so a future any-llm version
# could add a credential / transport / provider-selection field or a raw-body
# extension. Derivation picks up new fields automatically (the point for benign
# params), but exposing one of these as a client-settable request field would let
# a caller override an operator-controlled value: the provider-call merge spreads
# request fields last (e.g. ``{**get_provider_kwargs(...), **request_fields}``),
# so a client value would win. The gateway resolves these itself (from
# ``config``), so they must never be derived onto a public request
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
        # any-llm hands these to the provider SDK as per-request transport
        # options, so a caller could add headers or query parameters to an
        # upstream call made with the operator's credential.
        "extra_headers",
        "extra_query",
        "aws_access_key_id",
        "aws_secret_access_key",
        # any-llm's acompletion()/aresponses() forward this dict straight to
        # the provider's client constructor, so a caller-supplied client_args
        # smuggled in through a schema that allows extra fields would otherwise
        # reach the provider call unfiltered.
        "client_args",
    }
)

# What an owned endpoint may not carry as a default request field: the fields
# above, and the ones that shape the request itself, because a default travels
# in the body over the value the gateway set.
FORBIDDEN_ENDPOINT_DEFAULTS: frozenset[str] = SENSITIVE_PARAM_FIELDS | {"model", "messages", "input", "stream"}

# The fields the chat completion request declares beyond any-llm's
# ``CompletionParams``. The route parses each one, so none is ever an extra a
# provider instance's ``request_fields`` could forward.
CHAT_GATEWAY_FIELDS: frozenset[str] = frozenset(
    {
        "guardrails",
        "max_tool_iterations",
        "mcp_server_ids",
        "mcp_servers",
        "metadata",
        "model",
        "safety_identifier",
        "session_label",
        "tools_header",
    }
)


# Providers whose SDK clients take an injected HTTP client, so an endpoint a
# tenant chose can be dialed only through the pinned, public-only client
# (``services/providers/_owned_endpoint_network.py``). A provider missing here
# cannot be given a tenant's ``api_base``: its SDK would dial the address itself.
# The first set takes an httpx2 client (the Anthropic SDK, alone or beside
# OpenAI's), the rest an httpx one. ``tests/unit/test_tenant_provider_endpoints.py``
# builds each provider and checks every SDK client it holds uses the one injected.
TENANT_ENDPOINT_HTTPX2_PROVIDERS: frozenset[str] = frozenset({"anthropic", "azureanthropic", "deepseek", "meta"})
TENANT_ENDPOINT_PROVIDERS: frozenset[str] = TENANT_ENDPOINT_HTTPX2_PROVIDERS | {
    "atlascloud",
    "azureopenai",
    "cascadia",
    "cerebras",
    "dashscope",
    "databricks",
    "deepinfra",
    "edenai",
    "fireworks",
    "github",
    "gmi",
    "groq",
    "inception",
    "kenari",
    "llama",
    "llamacpp",
    "llamafile",
    "minimax",
    "moonshot",
    "mzai",
    "nebius",
    "neosantara",
    "openai",
    "openrouter",
    "perplexity",
    "portkey",
    "qiniu",
    "requesty",
    "sambanova",
    "telnyx",
    "together",
    "vllm",
    "zai",
}

# The ``client_args`` an organization's provider key may carry. They reach the
# provider SDK's client constructor, where other names choose the destination
# (``base_url``, ``endpoint_url``, ``server_url``, ``host``, ``http_options``),
# the transport (``http_client``), the headers (``default_headers``) or one of
# the deployment's own credentials (``profile_name``), so a tenant may set only
# the names its provider needs and none of those.
TENANT_CLIENT_ARGS_ANY_PROVIDER: frozenset[str] = frozenset({"timeout", "max_retries"})
TENANT_CLIENT_ARGS: dict[str, frozenset[str]] = {
    "azureanthropic": frozenset({"resource"}),
    "azureopenai": frozenset({"api_version"}),
    "bedrock": frozenset({"region_name", "aws_access_key_id", "aws_secret_access_key", "aws_session_token"}),
    "huggingface": frozenset({"bill_to"}),
    "openai": frozenset({"organization", "project"}),
    "vertexai": frozenset({"project", "location"}),
    "vertexaianthropic": frozenset({"project_id", "region"}),
}


def tenant_client_args_allowed(provider: str) -> frozenset[str]:
    """The ``client_args`` names an organization's key for ``provider`` may carry."""
    return TENANT_CLIENT_ARGS_ANY_PROVIDER | TENANT_CLIENT_ARGS.get(provider, frozenset())


def tenant_client_args(provider: str, client_args: Mapping[str, Any] | None) -> dict[str, Any]:
    """``client_args`` with every name ``provider``'s key may not carry dropped."""
    allowed = tenant_client_args_allowed(provider)
    return {name: value for name, value in (client_args or {}).items() if name in allowed}


__all__ = [
    "CHAT_GATEWAY_FIELDS",
    "FORBIDDEN_ENDPOINT_DEFAULTS",
    "SENSITIVE_PARAM_FIELDS",
    "TENANT_CLIENT_ARGS",
    "TENANT_CLIENT_ARGS_ANY_PROVIDER",
    "TENANT_ENDPOINT_HTTPX2_PROVIDERS",
    "TENANT_ENDPOINT_PROVIDERS",
    "tenant_client_args",
    "tenant_client_args_allowed",
]
