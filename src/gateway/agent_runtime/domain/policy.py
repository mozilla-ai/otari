"""Parse a submitted ``.otari-gates.yml`` body.

Pure: the caller (an agent hook, eventually the native dispatcher) reads its
own repo's policy file and Git evidence and submits both in one request, per
the production plan's hook-protocol design. This module never touches a
filesystem, a network, or a clock; it only turns already-received YAML text
into a validated :class:`PolicySpec`.
"""

from __future__ import annotations

from typing import Any

import yaml

from gateway.agent_runtime.domain.types import ChangedPathGate, Enforcement, GateSpec, PolicySpec

# A policy body is a developer-edited text file, not a data export; this bounds
# a pathological input (and an accidental binary) before it ever reaches the
# YAML parser. Production sizing for included/packed policies is AG-005.
# Exported (not module-private) so a caller enforcing its own request-size
# limit, such as the Hook Server route, shares this one number rather than
# duplicating it.
MAX_POLICY_BYTES = 256 * 1024

_SUPPORTED_SCHEMA_VERSIONS = {"1.0"}
_SUPPORTED_GATE_TYPES = {"changed_path"}
_SUPPORTED_ENFORCEMENTS = {"required", "advisory"}

_TOP_LEVEL_FIELDS = {"schema_version", "policy", "gates"}
_POLICY_FIELDS = {"id", "description"}
_GATE_FIELDS = {"id", "type", "enforcement", "forbidden", "message"}


class PolicyError(Exception):
    """A submitted policy body is malformed or fails schema validation.

    Always raised rather than returning a partial policy: a policy the parser
    could not fully understand must never silently evaluate as "no gates".
    """


class _DuplicateKeyLoader(yaml.SafeLoader):
    """A SafeLoader that rejects a mapping with a repeated key instead of keeping the last one."""


def _construct_mapping(loader: yaml.SafeLoader, node: yaml.MappingNode) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        # PyYAML's bundled typeshed stub leaves construct_object untyped.
        key = loader.construct_object(key_node, deep=True)  # type: ignore[no-untyped-call]
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=True)  # type: ignore[no-untyped-call]
    return mapping


_DuplicateKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def _require_fields(document: dict[str, Any], known: set[str], where: str) -> None:
    unknown = set(document) - known
    if unknown:
        raise PolicyError(f"Unknown field(s) in {where}: {', '.join(sorted(unknown))}.")


def _parse_gate(raw: Any) -> GateSpec:
    if not isinstance(raw, dict):
        raise PolicyError(f"Each entry under 'gates' must be a mapping, got {type(raw).__name__}.")
    _require_fields(raw, _GATE_FIELDS, f"gate {raw.get('id', '<missing id>')!r}")

    gate_id = raw.get("id")
    if not isinstance(gate_id, str) or not gate_id:
        raise PolicyError(f"Gate is missing a non-empty 'id': {raw!r}")

    gate_type = raw.get("type")
    if gate_type not in _SUPPORTED_GATE_TYPES:
        raise PolicyError(
            f"Gate {gate_id!r} has unsupported type {gate_type!r}. "
            f"Supported in this build: {', '.join(sorted(_SUPPORTED_GATE_TYPES))}."
        )

    enforcement = raw.get("enforcement")
    if enforcement not in _SUPPORTED_ENFORCEMENTS:
        raise PolicyError(
            f"Gate {gate_id!r} has invalid enforcement {enforcement!r}. "
            f"Must be one of: {', '.join(sorted(_SUPPORTED_ENFORCEMENTS))}."
        )
    enforcement_value: Enforcement = enforcement

    message = raw.get("message")
    if not isinstance(message, str) or not message:
        raise PolicyError(f"Gate {gate_id!r} is missing a non-empty 'message'.")

    forbidden = raw.get("forbidden")
    if (
        not isinstance(forbidden, list)
        or not forbidden
        or not all(isinstance(item, str) and item for item in forbidden)
    ):
        raise PolicyError(f"Gate {gate_id!r} (type 'changed_path') needs a non-empty list of 'forbidden' globs.")

    return ChangedPathGate(
        id=gate_id,
        enforcement=enforcement_value,
        forbidden=tuple(forbidden),
        message=message,
    )


def parse_policy(raw_yaml: str, *, source: str) -> PolicySpec:
    """Parse and validate a policy body already read by the caller."""
    if len(raw_yaml.encode("utf-8")) > MAX_POLICY_BYTES:
        raise PolicyError(f"{source} is larger than {MAX_POLICY_BYTES} bytes.")

    try:
        document = yaml.load(raw_yaml, Loader=_DuplicateKeyLoader)
    except yaml.YAMLError as exc:
        raise PolicyError(f"{source} is not valid YAML: {exc}") from exc

    if not isinstance(document, dict):
        raise PolicyError(f"{source} must contain a YAML mapping at the top level.")
    _require_fields(document, _TOP_LEVEL_FIELDS, source)

    schema_version = document.get("schema_version")
    if schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise PolicyError(
            f"{source} has unsupported schema_version {schema_version!r}. "
            f"Supported: {', '.join(sorted(_SUPPORTED_SCHEMA_VERSIONS))}."
        )

    policy_block = document.get("policy")
    if not isinstance(policy_block, dict):
        raise PolicyError(f"{source} is missing a 'policy' mapping.")
    _require_fields(policy_block, _POLICY_FIELDS, f"{source} 'policy' block")
    policy_id = policy_block.get("id")
    if not isinstance(policy_id, str) or not policy_id:
        raise PolicyError(f"{source} 'policy.id' must be a non-empty string.")

    raw_gates = document.get("gates")
    if not isinstance(raw_gates, list) or not raw_gates:
        raise PolicyError(f"{source} must declare at least one gate under 'gates'.")

    gates = [_parse_gate(raw_gate) for raw_gate in raw_gates]
    seen_ids = set()
    for gate in gates:
        if gate.id in seen_ids:
            raise PolicyError(f"{source} declares gate id {gate.id!r} more than once.")
        seen_ids.add(gate.id)

    return PolicySpec(schema_version=schema_version, policy_id=policy_id, gates=tuple(gates))
