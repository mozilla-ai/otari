"""A composed guardrail as `otari hook` reports it to an Otari server: its gates, and its fingerprint.

The fingerprint is computed from the gates themselves, never from the files
they came from, so the server can recompute it from what it was sent and refuse
a definition that does not match the fingerprint it claims.
"""

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any

# The whole sha256: a server keeps the first definition it is sent for a
# fingerprint, so one must not be found to collide with another.
DIGEST_LENGTH = 64
# What a server stores of one definition. The client keeps to them before it
# fingerprints a definition, so the server always takes what the client sends.
MAX_DEFINITION_GATES = 500
MAX_GATE_MESSAGE_LENGTH = 4_096
MAX_GATE_RUNS = 16


@dataclass(frozen=True)
class GateDefinition:
    gate_id: str
    gate_type: str
    enforcement: str
    origin: str
    runs: tuple[str, ...]
    message: str


def definition_digest(gates: Iterable[GateDefinition]) -> str:
    """The fingerprint of a guardrail: the same gates give the same digest, in any order."""
    canonical = sorted((asdict(gate) for gate in gates), key=lambda gate: gate["gate_id"])
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def bounded(gates: Iterable[GateDefinition]) -> tuple[GateDefinition, ...]:
    """The gates as a server keeps them: each message cut to what it stores, at most its number of gates.

    Empty past the cap rather than cut short, since a part of a guardrail would
    describe a version that never existed.
    """
    gates = tuple(gates)
    if len(gates) > MAX_DEFINITION_GATES or any(len(gate.runs) > MAX_GATE_RUNS for gate in gates):
        return ()
    return tuple(replace(gate, message=gate.message[:MAX_GATE_MESSAGE_LENGTH]) for gate in gates)


def definition_body(gates: Iterable[GateDefinition]) -> dict[str, Any]:
    gates = bounded(gates)
    return {"digest": definition_digest(gates), "gates": [asdict(gate) for gate in gates]}


def gates_from_body(gates: Iterable[Mapping[str, Any]]) -> tuple[GateDefinition, ...]:
    """Gate definitions from a body :func:`definition_body` wrote; raises on a malformed one."""
    return tuple(
        GateDefinition(
            gate_id=str(gate["gate_id"]),
            gate_type=str(gate["gate_type"]),
            enforcement=str(gate["enforcement"]),
            origin=str(gate["origin"]),
            runs=tuple(str(moment) for moment in gate["runs"]),
            message=str(gate["message"]),
        )
        for gate in gates
    )
