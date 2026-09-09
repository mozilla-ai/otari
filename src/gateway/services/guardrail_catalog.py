"""The guardrail catalog behind the dashboard's mandate form.

A profile is not a name this repository knows. It is a key in the operator's own
``service.yaml`` on the guardrails service, which builds one guardrail per
entry at boot, so the set of profiles a deployment has is that service's to state
and never ours to guess. The catalog is therefore a join of two sources, neither
of them a list written here:

* **Which profiles exist**, from ``GET {guardrails_url}/profiles`` on the running
  service. It answers with each profile's name, the ``any_guardrail`` class it is
  built from, and the model id, if the operator pinned one.
* **What each one takes**, from ``any_guardrail.parameter_registry``, keyed by
  that class name. Upstream generates it from the guardrails' own signatures and
  docstrings and keeps it in a stdlib+pydantic leaf precisely so a consumer can
  render a configuration form without importing a model backend (any-guardrail
  #206). Nothing here constructs a guardrail; only the registry is read.

Only ``validate``-stage parameters are published. The ``create`` stage is the
guardrails service's constructor, fixed by the operator's YAML at boot, so an organization
that could set one would be storing a value nothing sends: ``POST /validate``
takes ``validate_kwargs`` and nothing else. That is the same reason
``extra_kwargs_for_creation`` has no column on an organization guardrail (see
`services/tenancy/organization_guardrail_service.py`).

A deployment whose service is down, unconfigured, or too old to publish
``/profiles`` gets a catalog marked unavailable with a reason rather than an
error. The form falls back to naming a profile by hand, which is the whole of
what it could do before this existed, so a guardrails outage must not also take
away the page that configures guardrails.
"""

from __future__ import annotations

import json
from typing import Any, Literal

import httpx
from any_guardrail.base import GuardrailName
from any_guardrail.parameter_registry import get_parameter_schema
from pydantic import BaseModel, Field

from gateway.log_config import logger
from gateway.services.url_safety import redact_url_secrets

# Short, because this runs while an operator watches a settings page load. The
# guardrails service answers `/profiles` out of memory (it holds its built guardrails), so a
# slow answer means the host is struggling rather than the work being large.
_CATALOG_TIMEOUT_S = 5.0

# The timeout bounds how long the answer may take and not how large it may be, so
# a fast oversized body would be read into a worker whole and then turned into
# one model per row. A real deployment configures a handful of profiles; these
# are two orders of magnitude above that, and an answer past either is a service
# that is not the one this expects rather than a catalog worth truncating.
_MAX_CATALOG_BYTES = 1024 * 1024
_MAX_PROFILES = 500

ParameterType = Literal["string", "integer", "number", "boolean", "enum", "json"]

# Declared here rather than reused from `any_guardrail.parameters.ParameterType`,
# so the published contract is this API's own and a new upstream member cannot
# silently widen it. An unrecognized type degrades to "json", which is already
# upstream's "not flat-form-able, use a raw editor" signal, so the parameter
# stays configurable instead of vanishing from the form.
_KNOWN_TYPES: frozenset[str] = frozenset({"string", "integer", "number", "boolean", "enum", "json"})


class GuardrailParameterSpec(BaseModel):
    """One ``validate_kwargs`` key a profile accepts, typed for a form control."""

    name: str = Field(description="The keyword argument's name, as it is sent in validate_kwargs")
    type: ParameterType = Field(description="Value shape, so a form can render the matching control")
    required: bool = Field(
        description=(
            "Whether a value must be supplied for the guardrail to run. Folds together the "
            "signature having no default and upstream's effectively-required flag, which covers a "
            "parameter that defaults to a value the guardrail then refuses to run without"
        )
    )
    default: Any = Field(default=None, description="The signature default, or null when there is none")
    choices: list[str] | None = Field(default=None, description="Allowed values for an enum parameter")
    secret: bool = Field(
        default=False,
        description="Whether the value is a credential, so a form masks it and never echoes it back",
    )
    description: str | None = Field(default=None, description="One-line help text from the guardrail's docstring")


class GuardrailProfileSpec(BaseModel):
    """One profile the operator's guardrails service has built."""

    profile: str = Field(description="The name a guardrail entry puts in its profile field")
    guardrail: str = Field(description="The any-guardrail class the profile is built from")
    model_id: str | None = Field(default=None, description="The model the operator pinned, when they pinned one")
    parameters: list[GuardrailParameterSpec] = Field(
        default_factory=list, description="The validate_kwargs this profile accepts"
    )
    parameters_known: bool = Field(
        description=(
            "False when this gateway's any-guardrail is older than the service's and has no schema "
            "for that class. The profile is still selectable; only its typed fields are missing"
        )
    )


class GuardrailCatalog(BaseModel):
    """The profiles a guardrail entry may name, or why they could not be listed."""

    available: bool = Field(description="Whether the guardrails service answered with its profiles")
    reason: str | None = Field(
        default=None, description="Why the catalog is unavailable, in terms a tenant can act on"
    )
    profiles: list[GuardrailProfileSpec] = Field(default_factory=list)


# Deliberately free of the endpoint. This catalog is readable by any dashboard
# session, and a URL names infrastructure that `GET /v1/tool-settings` already
# withholds from a non-operator; the address is logged instead. See the module
# docstring of `api/routes/tool_settings.py`.
_NOT_CONFIGURED = "No guardrails service is configured, so its profiles cannot be listed."
_UNREACHABLE = "The guardrails service could not be reached, so its profiles cannot be listed."
_UNSUPPORTED = "The guardrails service does not publish a profile catalog. It may predate the /profiles endpoint."
_MALFORMED = "The guardrails service answered its profile catalog in a shape this gateway does not understand."
_TOO_LARGE = "The guardrails service answered with more profiles than this gateway will list."


class _CatalogTooLargeError(Exception):
    """The answer went past a size this gateway is willing to hold."""


def _parameter_specs(guardrail: str) -> tuple[list[GuardrailParameterSpec], bool]:
    """The validate-stage parameters of one any-guardrail class, and whether they are known.

    A class name the installed registry has never heard of is reported rather
    than raised: the guardrails service may run a newer any-guardrail than this gateway, and
    a profile whose fields cannot be typed is still a profile an organization can
    mandate and configure through the raw editor.
    """
    try:
        name = GuardrailName(guardrail)
    except ValueError:
        logger.info("Guardrail class %r is not in this gateway's any-guardrail registry", guardrail)
        return [], False

    specs = [
        GuardrailParameterSpec(
            name=spec.name,
            type=spec.type.value if spec.type.value in _KNOWN_TYPES else "json",
            # `required` alone is a property of the signature, not of what the
            # guardrail needs, so a parameter that reads its value from an
            # environment variable would render optional and then fail at
            # validate time. Upstream carries that distinction; fold it in here.
            required=spec.required or spec.effectively_required,
            default=spec.default,
            choices=list(spec.choices) if spec.choices is not None else None,
            secret=spec.secret,
            description=spec.description,
        )
        for spec in get_parameter_schema(name)
        if spec.stage.value == "validate"
    ]
    return specs, True


def _profile_spec(entry: object) -> GuardrailProfileSpec | None:
    """One `/profiles` row, or None when it is not shaped like one."""
    if not isinstance(entry, dict):
        return None
    profile = entry.get("name")
    guardrail = entry.get("guardrail_name")
    if not isinstance(profile, str) or not isinstance(guardrail, str):
        return None
    model_id = entry.get("model_id")
    parameters, known = _parameter_specs(guardrail)
    return GuardrailProfileSpec(
        profile=profile,
        guardrail=guardrail,
        model_id=model_id if isinstance(model_id, str) else None,
        parameters=parameters,
        parameters_known=known,
    )


async def _read_capped(response: httpx.Response) -> bytes:
    """The response body, or :class:`_CatalogTooLargeError` once it passes the cap."""
    body = bytearray()
    async for chunk in response.aiter_bytes():
        body.extend(chunk)
        if len(body) > _MAX_CATALOG_BYTES:
            raise _CatalogTooLargeError
    return bytes(body)


async def fetch_guardrail_catalog(base_url: str | None) -> GuardrailCatalog:
    """List the profiles the guardrails service at ``base_url`` has built.

    Never raises for an unreachable or unhelpful service. The dashboard asks for
    this to populate a picker, so every failure resolves to
    ``available=False`` with a reason the form can show beside the free-text
    field it falls back to.
    """
    url = (base_url or "").strip().rstrip("/")
    if not url:
        return GuardrailCatalog(available=False, reason=_NOT_CONFIGURED)
    # `guardrails_url` may carry userinfo, so the address is masked before it
    # reaches a log line, the way the settings endpoints mask it before it
    # reaches a response. The exception messages below can carry the URL too,
    # which is why each one names the redacted host rather than being logged
    # whole.
    shown = redact_url_secrets(url)

    try:
        async with httpx.AsyncClient(timeout=_CATALOG_TIMEOUT_S) as client:
            # Streamed rather than read whole, so the cap is applied to what
            # arrives instead of after a worker has already held it.
            async with client.stream("GET", f"{url}/profiles") as response:
                if response.status_code == httpx.codes.NOT_FOUND:
                    logger.info("Guardrails service at %s serves no /profiles endpoint", shown)
                    return GuardrailCatalog(available=False, reason=_UNSUPPORTED)
                response.raise_for_status()
                body = json.loads(await _read_capped(response))
    except _CatalogTooLargeError:
        logger.warning("Guardrail catalog from %s exceeded %d bytes", shown, _MAX_CATALOG_BYTES)
        return GuardrailCatalog(available=False, reason=_TOO_LARGE)
    except (httpx.HTTPError, httpx.InvalidURL) as exc:
        # The address goes to the log and not to the response, for the reason
        # `services/guardrails.py` keeps it out of a 502 body.
        #
        # `InvalidURL` alongside, because it is not an `HTTPError`: this reads
        # `guardrails_url` as configured, and only the dashboard's PATCH runs
        # that field through `validate_url`, so a malformed env or YAML value
        # arrives here whole. The docstring promises this never raises, and a
        # mistyped setting is exactly the case an operator opens the page to fix.
        logger.warning("Guardrail catalog unavailable from %s: %s", shown, exc.__class__.__name__)
        return GuardrailCatalog(available=False, reason=_UNREACHABLE)
    except ValueError:
        logger.warning("Guardrail catalog from %s was not JSON", shown)
        return GuardrailCatalog(available=False, reason=_MALFORMED)

    if not isinstance(body, list):
        logger.warning("Guardrail catalog from %s was not a list", shown)
        return GuardrailCatalog(available=False, reason=_MALFORMED)

    if len(body) > _MAX_PROFILES:
        logger.warning("Guardrail catalog from %s held %d profiles", shown, len(body))
        return GuardrailCatalog(available=False, reason=_TOO_LARGE)

    # A row this gateway cannot read is dropped rather than failing the whole
    # catalog: one unrecognized entry must not cost the operator the picker.
    profiles = [spec for spec in (_profile_spec(entry) for entry in body) if spec is not None]
    if len(profiles) != len(body):
        logger.warning(
            "Guardrail catalog from %s held %d rows this gateway could not read", shown, len(body) - len(profiles)
        )
    return GuardrailCatalog(available=True, profiles=sorted(profiles, key=lambda spec: spec.profile))
