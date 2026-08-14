"""Check a running code-execution backend against the published contract.

    uv run python scripts/check_code_execution_conformance.py --base-url http://localhost:8080

The contract is `docs/code-execution-protocol.md`, with the schemas in
`docs/public/code-execution-openapi.yaml`; this drives a live backend through it.
Two implementations exist (`otari-sandbox-container` as the OSS reference, a
pool-backed one behind otari.ai), and Otari cannot tell which one answered, so
this script is what turns that claim into something checkable rather than
asserted: any third-party backend runs it to show it is interchangeable.

The three session operations are required of a backend, so a failure there means
non-conforming. The file operations are optional, and a backend that does not
serve them reports as skipped.

Requests are validated on the way out as well as responses on the way in, which
keeps this script from drifting from the spec it checks against.

`jsonschema` is a dev dependency of this repo rather than a runtime one, since
the gateway parses the contract with its own Pydantic models. Outside a synced
checkout, run with `uv run --with jsonschema,pyyaml,httpx python scripts/...`.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml
from jsonschema import Draft202012Validator

SPEC_PATH = Path(__file__).resolve().parent.parent / "docs" / "public" / "code-execution-openapi.yaml"

# Echoed by the executed code so a check confirms it ran, and not merely that the
# backend answered with a well-shaped envelope.
_MARKER = "otari-conformance-ok"
_DEFAULT_TIMEOUT_SECONDS = 30
# The contract's rule for a client's own budget: allow more wall-clock than the
# execution budget granted, since transport and the backend's teardown come on
# top of it. See "Execute" in docs/code-execution-protocol.md.
_TIMEOUT_BUFFER_SECONDS = 10.0
# A file operation the backend does not serve answers on the route, not on the
# session: the session was just used successfully, so these statuses mean "not
# implemented here" rather than "gone".
_UNSERVED_STATUSES = (404, 405, 501)

PASS = "pass"
FAIL = "fail"
SKIP = "skip"

_UPLOADED_FILE = "conformance.txt"
_EDITED_FILE = "conformance_editor.txt"


@dataclass(frozen=True)
class Check:
    """One contract requirement, and how the backend answered it."""

    name: str
    status: str
    detail: str = ""


def load_spec(path: Path = SPEC_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        spec: dict[str, Any] = yaml.safe_load(handle)
    return spec


def validator_for(spec: dict[str, Any], schema_name: str) -> Draft202012Validator:
    """A validator for one named schema, with the spec's internal refs resolvable.

    OpenAPI 3.1 schemas are JSON Schema 2020-12, so they validate directly. The
    document's own `components` travel along inside the schema so that its
    `#/components/schemas/...` references resolve with no registry to set up.
    """
    return Draft202012Validator(
        {
            "$ref": f"#/components/schemas/{schema_name}",
            "components": spec["components"],
        }
    )


def schema_errors(spec: dict[str, Any], schema_name: str, payload: Any) -> list[str]:
    validator = validator_for(spec, schema_name)
    return [
        f"{'.'.join(str(part) for part in error.absolute_path) or '(root)'}: {error.message}"
        for error in validator.iter_errors(payload)
    ]


def _exec_payload(spec: dict[str, Any], tool: str, tool_input: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
    payload = {
        "tool": tool,
        "input": tool_input,
        "timeout_seconds": timeout_seconds,
        "tool_use_id": f"srvtoolu_{_MARKER}",
    }
    errors = schema_errors(spec, "ExecRequest", payload)
    if errors:  # pragma: no cover - a bug in this script, not in a backend
        raise AssertionError(f"conformance script built a non-conforming ExecRequest: {errors}")
    return payload


def _checked(
    name: str,
    response: httpx.Response,
    *,
    expected_status: int,
    spec: dict[str, Any],
    schema_name: str | None,
) -> tuple[Check, Any]:
    """Status and schema together, since every operation is checked on both."""
    if response.status_code != expected_status:
        return Check(name, FAIL, f"expected {expected_status}, got {response.status_code}"), None
    if schema_name is None:
        return Check(name, PASS), None
    try:
        body = response.json()
    except ValueError as exc:
        return Check(name, FAIL, f"body of the {expected_status} is not JSON: {exc}"), None
    errors = schema_errors(spec, schema_name, body)
    if errors:
        return Check(name, FAIL, f"response does not match {schema_name}: {'; '.join(errors)}"), None
    return Check(name, PASS), body


def _check_create_session(
    client: httpx.Client, spec: dict[str, Any], *, idle_hint: int
) -> tuple[list[Check], str | None]:
    check, handle = _checked(
        "CreateSession",
        client.post("/sessions", json={"idle_timeout_seconds": idle_hint}),
        expected_status=201,
        spec=spec,
        schema_name="SessionHandle",
    )
    if handle is None:
        return [check], None

    checks = [check]
    # A backend may accept the hint or clamp it to a ceiling of its own, so the
    # reported value may be lower. Higher would break the client's only guarantee
    # about how long the session lives.
    reported = handle.get("idle_timeout_seconds")
    if isinstance(reported, int) and reported > idle_hint:
        checks.append(
            Check(
                "CreateSession honours lifetime hints",
                FAIL,
                f"asked for an idle timeout of {idle_hint}s, handle reports {reported}s",
            )
        )
    else:
        checks.append(Check("CreateSession honours lifetime hints", PASS))
    session_id: str = handle["session_id"]
    return checks, session_id


def _check_exec(
    client: httpx.Client,
    spec: dict[str, Any],
    session_id: str,
    *,
    name: str,
    tool: str,
    tool_input: dict[str, Any],
    timeout_seconds: int,
    expect_marker: bool = True,
) -> Check:
    """One Execute call, checked as an envelope and (usually) as a real run.

    `expect_marker` is off for a call whose output the contract says is not
    echoed back, such as a text-editor `create`: there the envelope and the exit
    status are all there is to check.
    """
    payload = _exec_payload(spec, tool, tool_input, timeout_seconds)
    check, body = _checked(
        name,
        client.post(
            f"/sessions/{session_id}/exec",
            json=payload,
            timeout=timeout_seconds + _TIMEOUT_BUFFER_SECONDS,
        ),
        expected_status=200,
        spec=spec,
        schema_name="ExecResponse",
    )
    if body is None:
        return check

    if body["tool_use_id"] != payload["tool_use_id"]:
        return Check(name, FAIL, "tool_use_id was not echoed from the request")
    content = body["result_block"]["content"]
    return_code = content.get("return_code")
    stdout = content.get("stdout") or ""
    if return_code not in (None, 0):
        return Check(name, FAIL, f"return_code {return_code}, stderr: {content.get('stderr') or '(empty)'}")
    if expect_marker and _MARKER not in stdout:
        return Check(name, FAIL, f"stdout does not carry the executed marker: {stdout or '(empty)'}")
    return Check(name, PASS)


def _check_unknown_tool_refused(client: httpx.Client, session_id: str, *, timeout_seconds: int) -> Check:
    name = "Execute refuses an unknown tool kind"
    # Deliberately not built through _exec_payload: the point is a payload the
    # contract does not describe.
    response = client.post(
        f"/sessions/{session_id}/exec",
        json={"tool": "no_such_tool_kind", "input": {}, "timeout_seconds": timeout_seconds},
        timeout=timeout_seconds + _TIMEOUT_BUFFER_SECONDS,
    )
    if response.status_code in (400, 422):
        return Check(name, PASS)
    if response.is_success:
        return Check(name, FAIL, f"ran an unknown tool kind and answered {response.status_code}")
    return Check(name, PASS, f"refused with {response.status_code}; the contract documents 400 or 422")


def _check_files(client: httpx.Client, spec: dict[str, Any], session_id: str) -> list[Check]:
    """The optional file operations, skipped as a group when none are served."""
    body = f"{_MARKER}\n".encode()
    upload = client.post(
        f"/sessions/{session_id}/files",
        files={"file": (_UPLOADED_FILE, body, "text/plain")},
        data={"path": _UPLOADED_FILE},
    )
    if upload.status_code in _UNSERVED_STATUSES:
        reason = "backend does not serve the optional file operations"
        return [Check(name, SKIP, reason) for name in ("PutFile", "ListFiles", "GetFile")]

    put_check, _ = _checked("PutFile", upload, expected_status=201, spec=spec, schema_name="FileUploadResult")
    checks = [put_check]

    list_check, files = _checked(
        "ListFiles",
        client.get(f"/sessions/{session_id}/files/list", params={"path": "."}),
        expected_status=200,
        spec=spec,
        schema_name="FileList",
    )
    if files is not None:
        paths = [entry["path"] for entry in files["files"]]
        if not any(path.endswith(_UPLOADED_FILE) for path in paths):
            list_check = Check("ListFiles", FAIL, f"the file just written is absent from the listing: {paths}")
    checks.append(list_check)

    download = client.get(f"/sessions/{session_id}/files", params={"path": _UPLOADED_FILE})
    if not download.is_success:
        checks.append(Check("GetFile", FAIL, f"expected 200, got {download.status_code}"))
    elif download.content != body:
        checks.append(Check("GetFile", FAIL, "returned bytes differ from the ones written"))
    else:
        checks.append(Check("GetFile", PASS))

    checks.append(_check_workspace_confinement(client, session_id))
    return checks


def _check_workspace_confinement(client: httpx.Client, session_id: str) -> Check:
    name = "GetFile confines access to the session workspace"
    escape = client.get(f"/sessions/{session_id}/files", params={"path": "../../etc/passwd"})
    if escape.is_success:
        return Check(name, FAIL, "served a path outside the session workspace")
    if escape.status_code == 403:
        return Check(name, PASS)
    return Check(name, PASS, f"refused with {escape.status_code}; the contract documents 403")


def _check_destroy_session(client: httpx.Client, spec: dict[str, Any], session_id: str) -> list[Check]:
    check, _ = _checked(
        "DestroySession",
        client.delete(f"/sessions/{session_id}"),
        expected_status=204,
        spec=spec,
        schema_name=None,
    )
    checks = [check]

    # Releasing a session that is already gone is not an error worth
    # distinguishing: it is in the desired state either way. A server error is.
    name = "DestroySession is idempotent"
    repeat = client.delete(f"/sessions/{session_id}")
    if repeat.status_code >= 500:
        checks.append(Check(name, FAIL, f"releasing an already-released session answered {repeat.status_code}"))
    else:
        checks.append(Check(name, PASS))
    return checks


def run_checks(
    client: httpx.Client,
    spec: dict[str, Any],
    *,
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
) -> list[Check]:
    """Drive one backend through the contract.

    A backend that answers badly produces a failing check, not an exception; a
    backend that stops answering at all produces one too, so the checks that did
    run are still reported.
    """
    try:
        checks, session_id = _check_create_session(client, spec, idle_hint=300)
    except httpx.HTTPError as exc:
        return [Check("CreateSession", FAIL, f"backend not reachable: {exc}")]
    if session_id is None:
        return checks

    try:
        checks.append(
            _check_exec(
                client,
                spec,
                session_id,
                name="Execute code_execution",
                tool="code_execution",
                tool_input={"code": f'print("{_MARKER}")'},
                timeout_seconds=timeout_seconds,
            )
        )
        checks.append(
            _check_exec(
                client,
                spec,
                session_id,
                name="Execute bash_code_execution",
                tool="bash_code_execution",
                tool_input={"command": f"echo {_MARKER}"},
                timeout_seconds=timeout_seconds,
            )
        )
        # `create` then `view`: a create does not echo the file's content back, so
        # the marker has to come back from the view.
        checks.append(
            _check_exec(
                client,
                spec,
                session_id,
                name="Execute text_editor_code_execution (create)",
                tool="text_editor_code_execution",
                tool_input={"command": "create", "path": _EDITED_FILE, "file_text": f"{_MARKER}\n"},
                timeout_seconds=timeout_seconds,
                expect_marker=False,
            )
        )
        checks.append(
            _check_exec(
                client,
                spec,
                session_id,
                name="Execute text_editor_code_execution (view)",
                tool="text_editor_code_execution",
                tool_input={"command": "view", "path": _EDITED_FILE},
                timeout_seconds=timeout_seconds,
            )
        )
        checks.append(_check_unknown_tool_refused(client, session_id, timeout_seconds=timeout_seconds))
        checks.extend(_check_files(client, spec, session_id))
    except httpx.HTTPError as exc:
        checks.append(Check("Execute", FAIL, f"backend stopped answering: {exc}"))
    finally:
        try:
            checks.extend(_check_destroy_session(client, spec, session_id))
        except httpx.HTTPError as exc:
            checks.append(Check("DestroySession", FAIL, f"backend not reachable: {exc}"))
    return checks


def report(checks: list[Check]) -> int:
    """Print one line per check and return the process exit code."""
    labels = {PASS: "PASS", FAIL: "FAIL", SKIP: "SKIP"}
    for check in checks:
        detail = f"  ({check.detail})" if check.detail else ""
        print(f"[{labels[check.status]}] {check.name}{detail}")

    failed = sum(1 for check in checks if check.status == FAIL)
    skipped = sum(1 for check in checks if check.status == SKIP)
    passed = len(checks) - failed - skipped
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    if failed:
        print("This backend does not conform to code-execution contract version 1.")
        return 1
    print("This backend conforms to code-execution contract version 1.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", required=True, help="Base URL of the backend, e.g. http://localhost:8080")
    parser.add_argument("--auth-token", default=None, help="Bearer credential, where the deployment requires one.")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=_DEFAULT_TIMEOUT_SECONDS,
        help=f"Execution budget granted per call (default: {_DEFAULT_TIMEOUT_SECONDS}).",
    )
    parser.add_argument("--spec", type=Path, default=SPEC_PATH, help="Path to the contract's OpenAPI document.")
    args = parser.parse_args(argv)

    spec = load_spec(args.spec)
    headers = {"Authorization": f"Bearer {args.auth_token}"} if args.auth_token else {}
    with httpx.Client(
        base_url=args.base_url.rstrip("/"),
        headers=headers,
        timeout=args.timeout_seconds + _TIMEOUT_BUFFER_SECONDS,
    ) as client:
        checks = run_checks(client, spec, timeout_seconds=args.timeout_seconds)
    return report(checks)


if __name__ == "__main__":
    sys.exit(main())
