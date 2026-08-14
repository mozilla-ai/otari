"""Keep the code-execution contract's three faces in agreement.

The contract is published in three places, each load-bearing for a different
reader: `docs/code-execution-protocol.md` is prose for someone building a
backend, `docs/public/code-execution-openapi.yaml` is the IDL they generate from,
and `gateway.types.code_execution` is what this gateway actually parses. Nothing
but a test stops those drifting apart, and drift is exactly what publishing the
contract was meant to end.

So: every operation, path, status code, and example in the doc has to be in the
spec too, the spec's examples have to validate against its own schemas, and the
gateway's client models have to parse them.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from jsonschema import Draft202012Validator

from gateway.types.code_execution import ExecResponse, SessionHandle

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOC_PATH = _REPO_ROOT / "docs" / "code-execution-protocol.md"
_SPEC_PATH = _REPO_ROOT / "docs" / "public" / "code-execution-openapi.yaml"

_CORE_OPERATIONS = {"CreateSession", "Execute", "DestroySession"}
_FILE_OPERATIONS = {"ListFiles", "GetFile", "PutFile"}

_HTTP_METHODS = ("get", "put", "post", "delete", "patch", "head", "options", "trace")

_SCHEMA_PREFIX = "#/components/schemas/"

_CODE_SPAN = re.compile(r"`([^`]+)`")
_TABLE_ROW = re.compile(r"^\|(?P<cells>.+)\|\s*$")


@pytest.fixture(scope="module")
def doc() -> str:
    return _DOC_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def spec() -> dict[str, Any]:
    with _SPEC_PATH.open(encoding="utf-8") as handle:
        loaded: dict[str, Any] = yaml.safe_load(handle)
    return loaded


def _table_rows(doc: str, *, after: str) -> list[list[str]]:
    """The rows of the first markdown table following a heading or line."""
    assert after in doc, f"{_DOC_PATH.name} no longer contains {after!r}, so its table cannot be checked"
    body = doc.split(after, 1)[1]
    rows: list[list[str]] = []
    for line in body.splitlines():
        match = _TABLE_ROW.match(line)
        if match is None:
            if rows:
                break
            continue
        cells = [cell.strip() for cell in match.group("cells").split("|")]
        if all(set(cell) <= {"-", ":"} for cell in cells):  # the header separator
            continue
        rows.append(cells)
    assert rows, f"no table found after {after!r} in {_DOC_PATH.name}"
    return rows[1:]  # drop the header


def _operations(spec: dict[str, Any]) -> dict[str, tuple[str, str]]:
    """operationId -> (METHOD, path), for every operation in the spec."""
    found: dict[str, tuple[str, str]] = {}
    for path, item in spec["paths"].items():
        for method in _HTTP_METHODS:
            operation = item.get(method)
            if operation is None:
                continue
            found[operation["operationId"]] = (method.upper(), path)
    return found


def _validator(spec: dict[str, Any], schema_name: str) -> Draft202012Validator:
    return Draft202012Validator({"$ref": f"#/components/schemas/{schema_name}", "components": spec["components"]})


def _walk(node: Any) -> Iterator[Any]:
    """Every mapping and sequence in the document, depth first."""
    yield node
    if isinstance(node, dict):
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def _refs_in(node: Any) -> set[str]:
    """Every `$ref` value at or below `node`."""
    return {found["$ref"] for found in _walk(node) if isinstance(found, dict) and isinstance(found.get("$ref"), str)}


def _resolve_ref(spec: dict[str, Any], ref: str) -> Any:
    """What a local JSON pointer points at, or None when it points nowhere."""
    target: Any = spec
    for part in ref.removeprefix("#/").split("/"):
        if not isinstance(target, dict) or part not in target:
            return None
        target = target[part]
    return target


def test_spec_is_openapi_31(spec: dict[str, Any]) -> None:
    # 3.1 and not 3.0 for a concrete reason: its schemas are JSON Schema
    # 2020-12, which is what lets this file's other tests, and the conformance
    # script, validate payloads with a plain JSON Schema validator.
    assert spec["openapi"] == "3.1.0"
    assert spec["components"]["schemas"], "the spec declares no schemas"


def test_spec_version_is_the_contract_version(doc: str, spec: dict[str, Any]) -> None:
    """`info.version` is the contract version, not a release number."""
    match = re.search(r"\*\*Contract version: (\d+)\.\*\*", doc)
    assert match is not None, "the doc no longer states a contract version"
    assert spec["info"]["version"] == match.group(1)


def test_documented_operations_are_the_spec_operations(doc: str, spec: dict[str, Any]) -> None:
    documented = {
        span.group(1) for row in _table_rows(doc, after="## Operations") if (span := _CODE_SPAN.search(row[0]))
    }
    assert documented == _CORE_OPERATIONS | _FILE_OPERATIONS
    assert set(_operations(spec)) == documented


def test_documented_binding_matches_the_spec_paths(doc: str, spec: dict[str, Any]) -> None:
    """The doc's binding table and the spec must name the same method and path.

    Parsing the doc is the point: a path edited in one place and not the other is
    the drift that leaves an implementer building against a URL nothing serves.
    """
    operations = _operations(spec)
    for row in _table_rows(doc, after="## HTTP/JSON binding"):
        operation = _CODE_SPAN.search(row[0])
        binding = _CODE_SPAN.search(row[1])
        assert operation is not None and binding is not None, f"unparsable binding row: {row}"
        method, _, path = binding.group(1).partition(" ")
        # The doc annotates two cells with a query string and a note; the spec
        # carries those as parameters and a request body.
        assert operations[operation.group(1)] == (method, path.split("?", 1)[0])


def test_file_operations_are_marked_optional_in_the_spec(spec: dict[str, Any]) -> None:
    """A backend implementing only the three session operations still conforms.

    The conformance script reads this marker to decide whether a missing
    operation is a skip or a failure, so it is contract data, not a comment.
    """
    optional = set()
    for path, item in spec["paths"].items():
        for method in _HTTP_METHODS:
            operation = item.get(method)
            if operation is not None and operation.get("x-otari-optional"):
                optional.add(operation["operationId"])
                assert "Optional." in operation["description"], (
                    f"{path} {method} is marked optional but does not say so"
                )
    assert optional == _FILE_OPERATIONS


def test_documented_status_codes_are_the_spec_status_codes(doc: str, spec: dict[str, Any]) -> None:
    documented = {
        code for row in _table_rows(doc, after="Status codes:") for code in re.findall(r"\b(\d{3})\b", row[1])
    }
    declared = {
        status
        for path_item in spec["paths"].values()
        for method in _HTTP_METHODS
        if (operation := path_item.get(method)) is not None
        for status in operation["responses"]
    }
    assert documented == declared


def test_every_operation_declares_the_unauthorized_response(spec: dict[str, Any]) -> None:
    """The credential is a deployment property, so any operation can answer 401.

    Declaring it on one operation only would leave a generated client treating
    the same answer elsewhere as an unrecognized failure.
    """
    missing = [
        operation["operationId"]
        for path_item in spec["paths"].values()
        for method in _HTTP_METHODS
        if (operation := path_item.get(method)) is not None
        and operation["responses"].get("401", {}).get("$ref") != "#/components/responses/Unauthorized"
    ]
    assert not missing, f"operation(s) with no 401 declared: {missing}"


def test_documented_example_is_the_spec_example(doc: str, spec: dict[str, Any]) -> None:
    """The doc's worked example and the spec's `ExecResponse` example are one payload.

    Two examples that disagree are worse than one: an implementer copies
    whichever they read first.
    """
    block = re.search(r"```json\n(.*?)```", doc, re.DOTALL)
    assert block is not None, "the doc no longer carries a worked example"
    assert json.loads(block.group(1)) == spec["components"]["schemas"]["ExecResponse"]["examples"][0]


@pytest.mark.parametrize("schema_name", ["SessionHandle", "ExecResponse", "CodeExecutionRequest"])
def test_spec_examples_validate_against_their_schemas(spec: dict[str, Any], schema_name: str) -> None:
    validator = _validator(spec, schema_name)
    examples = spec["components"]["schemas"][schema_name]["examples"]
    assert examples, f"{schema_name} carries no example"
    for example in examples:
        assert not list(validator.iter_errors(example)), f"{schema_name} example violates its own schema: {example}"


def test_gateway_client_parses_the_spec_examples(spec: dict[str, Any]) -> None:
    """The client this gateway ships must accept what the spec advertises."""
    schemas = spec["components"]["schemas"]
    assert SessionHandle.model_validate(schemas["SessionHandle"]["examples"][0]).session_id

    parsed = ExecResponse.model_validate(schemas["ExecResponse"]["examples"][0])
    assert parsed.result_block.type == "code_execution_tool_result"
    assert [ref.filename for ref in parsed.result_block.content.content] == ["chart.png"]


def test_result_block_type_is_not_a_closed_enum(spec: dict[str, Any]) -> None:
    """A tool kind added later must not make an older client reject the response.

    The client keeps `type` an opaque string for this reason; an enum here would
    hand a generated client the opposite rule.
    """
    block_type = spec["components"]["schemas"]["ResultBlock"]["properties"]["type"]
    assert "enum" not in block_type and "const" not in block_type
    assert len(block_type["examples"]) == 3


def test_response_schemas_tolerate_unknown_fields(spec: dict[str, Any]) -> None:
    """The extension policy, spelled out where a generator can read it."""
    offenders = [
        name
        for name, schema in spec["components"]["schemas"].items()
        if schema.get("type") == "object" and schema.get("additionalProperties") is not True
    ]
    assert not offenders, f"schemas that would reject an additive field: {offenders}"


def test_every_ref_resolves(spec: dict[str, Any]) -> None:
    broken = []
    for ref in _refs_in(spec):
        assert ref.startswith("#/"), f"the spec must stay self-contained, found {ref}"
        if _resolve_ref(spec, ref) is None:
            broken.append(ref)
    assert not broken, f"unresolvable reference(s): {broken}"


def test_no_orphan_schemas(spec: dict[str, Any]) -> None:
    """Every schema is reachable from an operation, so none is dead spec.

    Reachability is traced outward from `paths`, not read off the set of `$ref`
    values anywhere in the document: a schema that references itself, or an
    unused pair that reference each other, would otherwise vouch for itself and
    a dead branch of the contract would keep passing as live.
    """
    frontier = _refs_in(spec["paths"])
    reached: set[str] = set()
    while frontier:
        ref = frontier.pop()
        if ref in reached:
            continue
        reached.add(ref)
        target = _resolve_ref(spec, ref)
        # An unresolvable ref is test_every_ref_resolves' business, not this one.
        if target is not None:
            frontier |= _refs_in(target)

    reachable = {ref.removeprefix(_SCHEMA_PREFIX) for ref in reached if ref.startswith(_SCHEMA_PREFIX)}
    # A traversal that reaches nothing would pass this test for the wrong reason.
    assert reachable, "no schema is reachable from paths; the traversal, not the spec, is broken"
    orphans = set(spec["components"]["schemas"]) - reachable
    assert not orphans, f"schema(s) no operation reaches: {orphans}"
