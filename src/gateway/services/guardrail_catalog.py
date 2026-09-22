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

Of those profiles only ``validate``-stage parameters are published. The ``create``
stage is the guardrails service's constructor, fixed by the operator's YAML at boot,
so an organization that could set one would be storing a value nothing sends:
``POST /validate`` takes ``validate_kwargs`` and nothing else. That is the same reason
``extra_kwargs_for_creation`` has no column on an organization guardrail (see
`services/tenancy/organization_guardrail_service.py`).

A deployment whose service is down, unconfigured, or too old to publish
``/profiles`` gets a catalog marked unavailable with a reason rather than an
error. The form falls back to naming a profile by hand, which is the whole of
what it could do before this existed, so a guardrails outage must not also take
away the page that configures guardrails.

The built-in catalog
--------------------

Beside that sits a second, local catalog: the guardrails this gateway can run
itself, read straight from ``any_guardrail``'s import-free registry. Nothing is
joined and nothing is fetched, so there is no unavailable state to report. It
carries **both** stages, because a guardrail this gateway constructs itself has no
operator YAML fixing its constructor, and the create stage is where a vendor API
key lives. It carries the one-of requirement groups beside them, because a
constraint satisfied by any of several parameters is one no parameter's own
``required`` flag can state.

It is not every guardrail the library ships. A guardrail that works by holding
model weights in the process running it is not one this gateway builds, so it is
not one this catalog may offer; those belong in the guardrails service above, and
the two catalogs divide on exactly that line. The rule is upstream's own backend
taxonomy rather than a list kept here: a guardrail's ``backend``, which is the
one this gateway would get, and not its ``alternate_backends``, which name paths
nothing storable can select.
"""

from __future__ import annotations

import json
from typing import Any, Literal, get_args

import httpx
from any_guardrail.base import GuardrailName
from any_guardrail.parameter_registry import get_parameter_schema, get_requirement_groups
from any_guardrail.parameters import RequirementGroup
from any_guardrail.registry import GUARDRAIL_METADATA
from any_guardrail.taxonomy import BackendType, GuardrailMetadata
from pydantic import BaseModel, ConfigDict, Field

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

# The one taxonomy this module does declare rather than import from
# any-guardrail, because a type it has never seen has somewhere sensible to go:
# "json" is already upstream's "not flat-form-able, use a raw editor" signal, so
# an unrecognized type still renders a working field. It also types the sidecar
# catalog, whose guardrails service may run a newer any-guardrail than this
# gateway, which is a source the enum import below cannot speak for.
_KNOWN_TYPES: frozenset[str] = frozenset(get_args(ParameterType))


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
    storable: bool = Field(
        default=True,
        description=(
            "Whether a saved value can stand in for this parameter. False for a secret whose type is "
            "json, which upstream uses for a live object (an authenticated SDK client or session) that "
            "cannot be written down. A form offers no field for one"
        ),
    )
    # The variable's name only. Whether it is set on this host is deliberately not
    # answered: the catalog is readable by any dashboard session, and that is the
    # kind of infrastructure detail this module withholds elsewhere.
    env_var: str | None = Field(
        default=None,
        description=(
            "The environment variable that supplies this parameter when no value is stored, so a form "
            "can offer that instead of demanding a credential the deployment already has"
        ),
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
    reason: str | None = Field(default=None, description="Why the catalog is unavailable, in terms a tenant can act on")
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


def _specs_for_stage(name: GuardrailName, stage: str) -> list[GuardrailParameterSpec]:
    """The parameters one any-guardrail class takes at ``stage``, typed for a form."""
    return [
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
            storable=not (spec.secret and spec.type.value == "json"),
            env_var=spec.env_var,
            description=spec.description,
        )
        for spec in get_parameter_schema(name)
        if spec.stage.value == stage
    ]


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

    return _specs_for_stage(name, "validate"), True


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


# ---------------------------------------------------------------------------
# The built-in catalog: the guardrails a hosted API can reach.
# ---------------------------------------------------------------------------

# The taxonomy a guardrail is described by is upstream's own enum, imported rather
# than re-spelled. `any_guardrail.taxonomy` is a stdlib+pydantic leaf that loads no
# model backend, and every value published here is read from `GUARDRAIL_METADATA`,
# so a list written out again in this file could only ever drift from the one set of
# values it exists to accept. A member upstream adds therefore reaches this contract
# instead of degrading to a fallback, and the drift checks over the generated
# artifacts are what report it; see AGENTS.md, "Generated Artifacts".


class BuiltInGuardrailSpec(GuardrailMetadata):
    """One guardrail this gateway can construct and run itself.

    Upstream's own metadata model, extended rather than copied, so a field it adds
    is carried instead of waiting on an edit here. The four taxonomy enums document
    themselves in the published schema, which is why almost nothing below restates
    what a field name and its type already say; the descriptions that remain are on
    the answers only this gateway can give.

    Inheriting also takes upstream's field serializers, which sort every set-valued
    field on the way out, so the JSON is stable across calls without sorting anything
    here.
    """

    model_config = ConfigDict(frozen=True)

    guardrail_name: str = Field(description="The any-guardrail class, and the name a stored guardrail selects")
    # Redeclared for its description alone: the name reads as "accepts a list",
    # and upstream spends a paragraph warning that it does not mean that.
    supports_batch: bool = Field(
        default=False, description="Whether several inputs run as one real batched call, not a per-item loop"
    )
    create_parameters: list[GuardrailParameterSpec] = Field(
        default_factory=list,
        description="Constructor arguments, which is where a vendor API key and an endpoint live",
    )
    validate_parameters: list[GuardrailParameterSpec] = Field(
        default_factory=list, description="Per-call arguments, sent with the text on every check"
    )
    requirement_groups: list[RequirementGroup] = Field(
        default_factory=list,
        description=(
            "One-of constraints that no single parameter's required flag can express. At least one "
            "member of each group must be supplied, or one of the environment variables that satisfies it"
        ),
    )


class BuiltInGuardrailCatalog(BaseModel):
    """The guardrails this gateway can build and call itself."""

    guardrails: list[BuiltInGuardrailSpec] = Field(default_factory=list)


def _reachable_over_a_hosted_api(name: GuardrailName) -> bool:
    """Whether ``name`` runs as a call to a service rather than as a local model.

    ``backend`` alone. A guardrail that defaults to a local model and lists a
    hosted API among its ``alternate_backends`` reaches that second path through
    ``AnyGuardrail.create``'s own ``provider`` argument, which is not in the
    parameter registry and takes a constructed ``Provider`` rather than a name.
    Stored configuration therefore has no field in which to ask for it, and the
    row would build the local model this catalog exists to exclude.
    """
    return GUARDRAIL_METADATA[name].backend is BackendType.HOSTED_API


def _builtin_spec(name: GuardrailName) -> BuiltInGuardrailSpec:
    """One guardrail's row, built from the import-free registry alone."""
    return BuiltInGuardrailSpec(
        **GUARDRAIL_METADATA[name].model_dump(),
        guardrail_name=name.value,
        create_parameters=_specs_for_stage(name, "create"),
        validate_parameters=_specs_for_stage(name, "validate"),
        requirement_groups=get_requirement_groups(name),
    )


def build_builtin_guardrail_catalog() -> BuiltInGuardrailCatalog:
    """Every guardrail any-guardrail reaches over a hosted API, typed for the form that defines one.

    Does no I/O and reaches no service, so unlike `fetch_guardrail_catalog` it has
    no unavailable state: the answer is a property of the installed library. Both
    parameter stages are published, because a guardrail this gateway constructs
    has no operator YAML fixing its constructor.

    The filter belongs here and not in `_specs_for_stage`, which the sidecar half
    shares: an operator's own guardrails service may well run a local model, and
    typing its parameters is what `fetch_guardrail_catalog` exists to do.
    """
    return BuiltInGuardrailCatalog(
        guardrails=sorted(
            (_builtin_spec(name) for name in GuardrailName if _reachable_over_a_hosted_api(name)),
            key=lambda spec: spec.display_name.casefold(),
        )
    )
