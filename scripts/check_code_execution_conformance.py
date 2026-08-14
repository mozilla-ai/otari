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

JSON requests are validated on the way out as well as responses on the way in,
which keeps this script from drifting from the spec it checks against. The one
request that is not JSON, the multipart file upload, is checked by its response.

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
# The budget to fall back to when a backend refuses the requested one as over its
# own cap. Low enough that any backend should accept it, high enough to run the
# trivial snippets these checks send.
_MODEST_BUDGET_SECONDS = 10
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


def _validated_request(spec: dict[str, Any], schema_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Refuse to send a request the contract does not describe.

    A checker that sends a malformed request reports the backend's rejection of
    it as a backend fault, which is the one verdict it must never get wrong.
    """
    errors = schema_errors(spec, schema_name, payload)
    if errors:  # pragma: no cover - a bug in this script, not in a backend
        raise AssertionError(f"conformance script built a non-conforming {schema_name}: {errors}")
    return payload


def _exec_payload(spec: dict[str, Any], tool: str, tool_input: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
    return _validated_request(
        spec,
        "ExecRequest",
        {
            "tool": tool,
            "input": tool_input,
            "timeout_seconds": timeout_seconds,
            "tool_use_id": f"srvtoolu_{_MARKER}",
        },
    )


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
        client.post(
            "/sessions",
            json=_validated_request(spec, "CreateSessionRequest", {"idle_timeout_seconds": idle_hint}),
        ),
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
    #
    # Compared as a number, not as an `int`: JSON has one numeric type, so a
    # backend serializing the timeout as `99999.0` is still conforming (JSON
    # Schema counts a zero-fraction float as an integer) and arrives here as a
    # Python float. An `isinstance(reported, int)` guard would skip it, and skip
    # it precisely in the case this check exists to catch. `bool` is excluded
    # because it is an `int` subclass that no comparison here should reach.
    reported = handle.get("idle_timeout_seconds")
    reported_a_number = isinstance(reported, int | float) and not isinstance(reported, bool)
    if reported_a_number and reported > idle_hint:
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
    response = _post_exec(client, session_id, payload, timeout_seconds=timeout_seconds)

    note = ""
    if response.status_code in (400, 422) and timeout_seconds > _MODEST_BUDGET_SECONDS:
        # The contract lets a backend cap `timeout_seconds` and refuse a larger
        # value rather than clamping it, so this refusal is its ceiling talking,
        # not a broken Execute. Retry inside a budget any backend should accept:
        # without this, `--timeout-seconds 300` reports the reference backend
        # (which caps at 120) as non-conforming to the contract it defines.
        #
        # The retry is what evidences the diagnosis, which is why it is not keyed
        # on the error body: the contract says a client MUST NOT depend on that
        # body, and the two requests differ in nothing but the budget, so a retry
        # that runs proves the budget is what the backend objected to. A refusal
        # for any other reason is refused again and still fails the check.
        note = (
            f"refused a {timeout_seconds}s budget and accepted {_MODEST_BUDGET_SECONDS}s, "
            "the same call at a lower budget; the ceiling is the backend's own"
        )
        payload = _exec_payload(spec, tool, tool_input, _MODEST_BUDGET_SECONDS)
        response = _post_exec(client, session_id, payload, timeout_seconds=_MODEST_BUDGET_SECONDS)

    check, body = _checked(name, response, expected_status=200, spec=spec, schema_name="ExecResponse")
    if body is None:
        return check if not note else Check(check.name, check.status, f"{note}; {check.detail}")

    if body["tool_use_id"] != payload["tool_use_id"]:
        return Check(name, FAIL, "tool_use_id was not echoed from the request")
    content = body["result_block"]["content"]
    return_code = content.get("return_code")
    stdout = content.get("stdout") or ""
    if return_code not in (None, 0):
        return Check(name, FAIL, f"return_code {return_code}, stderr: {content.get('stderr') or '(empty)'}")
    if expect_marker and _MARKER not in stdout:
        return Check(name, FAIL, f"stdout does not carry the executed marker: {stdout or '(empty)'}")
    return Check(name, PASS, note)


def _post_exec(
    client: httpx.Client, session_id: str, payload: dict[str, Any], *, timeout_seconds: int
) -> httpx.Response:
    return client.post(
        f"/sessions/{session_id}/exec",
        json=payload,
        timeout=timeout_seconds + _TIMEOUT_BUFFER_SECONDS,
    )


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
    if response.status_code >= 500:
        # A server error is not a refusal: the backend fell over on input it was
        # supposed to reject.
        return Check(name, FAIL, f"failed with {response.status_code} instead of refusing the call")
    return Check(name, PASS, f"refused with {response.status_code}; the contract documents 400 or 422")


def _session_survives(client: httpx.Client, spec: dict[str, Any], session_id: str, *, timeout_seconds: int) -> bool:
    """Whether the session is still usable, which a 404 alone does not say."""
    check = _check_exec(
        client,
        spec,
        session_id,
        name="session probe",
        tool="code_execution",
        tool_input={"code": f'print("{_MARKER}")'},
        timeout_seconds=timeout_seconds,
    )
    return check.status == PASS


def _check_files(client: httpx.Client, spec: dict[str, Any], session_id: str, *, timeout_seconds: int) -> list[Check]:
    """The optional file operations, each probed on its own.

    The three are independently optional, so they are reported independently. A
    read-only backend that serves `ListFiles` and `GetFile` but not `PutFile` is
    conforming, and skipping its served operations as a group would take the
    workspace-confinement check down with them: the one check here that says
    something about safety rather than shape.
    """
    body = f"{_MARKER}\n".encode()
    upload = client.post(
        f"/sessions/{session_id}/files",
        files={"file": (_UPLOADED_FILE, body, "text/plain")},
        data={"path": _UPLOADED_FILE},
    )
    if upload.status_code in _UNSERVED_STATUSES:
        # A 404 on a file route is ambiguous: no such route, or no such session.
        # Reporting a reclaimed session as "these operations are optional and
        # absent" would pass a backend that dropped the session mid-run, so ask
        # the session before reading any of these statuses as "not served".
        if not _session_survives(client, spec, session_id, timeout_seconds=timeout_seconds):
            return [Check("File operations", FAIL, "the session was gone before they could be checked")]
        uploaded = False
        checks = [Check("PutFile", SKIP, "backend does not serve this optional operation")]
    else:
        put_check, _ = _checked("PutFile", upload, expected_status=201, spec=spec, schema_name="FileUploadResult")
        uploaded = put_check.status == PASS
        checks = [put_check]

    writes_served = upload.status_code not in _UNSERVED_STATUSES
    listing = client.get(f"/sessions/{session_id}/files/list", params={"path": "."})
    known_file: str | None = _UPLOADED_FILE if uploaded else None
    if listing.status_code in _UNSERVED_STATUSES:
        checks.append(Check("ListFiles", SKIP, "backend does not serve this optional operation"))
    else:
        list_check, files = _checked("ListFiles", listing, expected_status=200, spec=spec, schema_name="FileList")
        if files is not None:
            paths = [entry["path"] for entry in files["files"]]
            if uploaded and not any(path.endswith(_UPLOADED_FILE) for path in paths):
                list_check = Check("ListFiles", FAIL, f"the file just written is absent from the listing: {paths}")
            # Without PutFile there is nothing known to read, so borrow a path
            # from the listing to give GetFile something to fetch.
            if known_file is None and paths:
                known_file = paths[0]
        checks.append(list_check)

    checks.extend(
        _check_reads(
            client,
            session_id,
            known_file=known_file,
            expected=body if uploaded else None,
            file_routes_served=writes_served or listing.status_code not in _UNSERVED_STATUSES,
        )
    )
    return checks


def _check_reads(
    client: httpx.Client,
    session_id: str,
    *,
    known_file: str | None,
    expected: bytes | None,
    file_routes_served: bool,
) -> list[Check]:
    """`GetFile`, and the confinement it must enforce.

    Reading a known file comes first, because whether it worked is what tells the
    confinement probe apart from an absent route. Asking the traversal attempt to
    answer both questions cannot work: a backend that sanitizes an escaping path
    and answers `404` is refusing it, and reading that as "no read route here"
    would skip the confinement check on a backend that does serve reads, which is
    the one place a skip is more dangerous than a failure.
    """
    name = "GetFile"
    if known_file is None:
        # Nothing to read. If some other file route answered, the file surface
        # exists and the probe can interpret its own answer; if none did, there is
        # no read route to confine and saying so beats a pass for a check that
        # could not run.
        return [
            Check(name, SKIP, "no file available to read: PutFile is not served and the workspace is empty"),
            _check_workspace_confinement(client, session_id, reads_served=None if file_routes_served else False),
        ]

    download = client.get(f"/sessions/{session_id}/files", params={"path": known_file})
    if download.status_code in _UNSERVED_STATUSES:
        return [
            Check(name, SKIP, "backend does not serve this optional operation"),
            _check_workspace_confinement(client, session_id, reads_served=False),
        ]

    confinement = _check_workspace_confinement(client, session_id, reads_served=True)
    if not download.is_success:
        return [Check(name, FAIL, f"expected 200 for {known_file}, got {download.status_code}"), confinement]
    if expected is not None and download.content != expected:
        return [Check(name, FAIL, "returned bytes differ from the ones written"), confinement]
    detail = "" if expected is not None else f"read {known_file} from the listing; contents unknown to this run"
    return [Check(name, PASS, detail), confinement]


def _check_workspace_confinement(client: httpx.Client, session_id: str, *, reads_served: bool | None) -> Check:
    """Whether an escaping path is refused, given what the read probe already learned.

    `reads_served` is the caller's verdict from reading a known file: `False`
    skips, because there is no read route to confine, and `True` or `None` means
    the probe is worth making. A `404` here is then read as a refusal rather than
    as an absent route: a backend that rejects `../../etc/passwd` by declaring it
    not found has refused it, and the contract's requirement is that it not be
    served.
    """
    name = "GetFile confines access to the session workspace"
    if reads_served is False:
        return Check(name, SKIP, "backend does not serve reads, so there is nothing to confine")

    escape = client.get(f"/sessions/{session_id}/files", params={"path": "../../etc/passwd"})
    if escape.is_success:
        return Check(name, FAIL, "served a path outside the session workspace")
    if escape.status_code == 403:
        return Check(name, PASS)
    if escape.status_code >= 500:
        return Check(name, FAIL, f"failed with {escape.status_code} instead of refusing the path")
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
        checks.extend(_check_files(client, spec, session_id, timeout_seconds=timeout_seconds))
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


def _execution_budget(value: str) -> int:
    """An execution budget the contract allows, rejected at the CLI if not.

    Without this, a zero or negative value only fails when the first request is
    validated, after a session has already been leased, and reports as this
    script being at fault rather than the invocation.
    """
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1 second, as the contract requires")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", required=True, help="Base URL of the backend, e.g. http://localhost:8080")
    parser.add_argument("--auth-token", default=None, help="Bearer credential, where the deployment requires one.")
    parser.add_argument(
        "--timeout-seconds",
        type=_execution_budget,
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
