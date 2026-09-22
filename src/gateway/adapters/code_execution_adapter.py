"""Core adapter for ``CodeExecutionPort``: the published protocol over HTTP.

Satisfies :class:`gateway.ports.code_execution_port.CodeExecutionPort` by
speaking `docs/code-execution-protocol.md` to the backend at ``sandbox_url``,
which is what Otari has always done. The calls here moved out of
``services/sandbox_backend.py`` unchanged, so a deployment running the
reference container behaves exactly as it did before the port existed.

Also the composition root for the capability: :func:`build_code_execution_port`
picks the adapter a deployment's ``sandbox_provider`` names. It is called from
the request path rather than bound in ``container.py``, because a container
factory receives only a session and this choice is made from config; binding
it is the step an overlay that wants to supply its own adapter would add.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from functools import cached_property
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import ValidationError

from gateway.core.env import otari_env
from gateway.ports.code_execution_port import (
    CodeExecutionPort,
    CodeExecutionSession,
    OutputOverBudget,
    SandboxFileEntry,
    SandboxNotReachableError,
    SandboxSessionGoneError,
    SandboxUnavailableError,
)
from gateway.services.url_safety import redact_url_secrets
from gateway.types.code_execution import ExecResponse, ResultBlock, SessionHandle

if TYPE_CHECKING:
    from gateway.core.config import GatewayConfig

logger = logging.getLogger(__name__)

# The tool kind Otari asks a backend to run. The contract names two more
# (``bash_code_execution``, ``text_editor_code_execution``); nothing above this
# port asks for them, so no adapter has to serve them.
CODE_EXECUTION_TOOL = "code_execution"
# Extra wall-clock the HTTP call gets over the exec budget, so the backend
# always answers before the client read timeout fires.
_EXEC_TIMEOUT_BUFFER_S = 10.0


def _contract_violation(exc: ValidationError) -> str:
    """Name the fields that failed validation, without echoing their values.

    A backend's response can carry model output, so the summary is built from
    the error's own locations rather than rendered from ``exc``.
    """
    fields = ", ".join(".".join(str(part) for part in error["loc"]) or "body" for error in exc.errors()) or "body"
    return f"response does not match the code-execution contract ({fields})"


class ProtocolCodeExecutionAdapter:
    """A backend of the operator's own, reached over the published contract."""

    def __init__(self, sandbox_url: str) -> None:
        self._sandbox_url = sandbox_url.rstrip("/")

    @property
    def label(self) -> str:
        # The port's label reaches a span, and a backend URL is the one label
        # that can carry a credential, so it is redacted rather than trusted to
        # have none. The same redaction is what every message below names the
        # backend by, since the pipeline logs these exceptions.
        return self._safe_url

    @cached_property
    def _safe_url(self) -> str:
        return redact_url_secrets(self._sandbox_url)

    @asynccontextmanager
    async def open_session(
        self,
        *,
        image: str | None = None,
        timeout_s: float,
        session_ttl_s: float,
        auth_token: str | None = None,
        resume: str | None = None,
        keep_alive_s: float | None = None,
    ) -> AsyncIterator[CodeExecutionSession]:
        # A contract session lives until the DELETE below, unless it is kept: then
        # the backend reclaims it on the idle timeout it was told at creation,
        # which is the one lifetime hint the contract has.
        del session_ttl_s
        stack = AsyncExitStack()
        headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else None
        client = await stack.enter_async_context(httpx.AsyncClient(timeout=timeout_s, headers=headers))
        try:
            if resume is not None:
                session_id = resume
                await self._probe_session(client, session_id)
                # A session that survived an earlier request is proof the backend
                # holds them, and its idle clock is the one it was created with.
                holds = keep_alive_s is not None
            else:
                session_id, accepted_idle_s = await self._create_session(client, image=image, keep_alive_s=keep_alive_s)
                # ``idle_timeout_seconds`` is a hint the backend MAY ignore, and
                # the handle reports what it actually kept. No value means no
                # idle reclaim will come, so holding the session would leak it:
                # the DELETE below is what ends it, exactly as before reuse.
                holds = keep_alive_s is not None and accepted_idle_s is not None
                if keep_alive_s is not None and not holds:
                    logger.info(
                        "sandbox backend did not accept an idle timeout; session %s will not be held for reuse",
                        session_id,
                    )
        except BaseException:
            await stack.aclose()
            raise

        session = _ProtocolSession(client, self._sandbox_url, session_id, timeout_s, holds_across_requests=holds)
        try:
            yield session
        finally:
            if not session.holds_across_requests:
                try:
                    await client.delete(f"{self._sandbox_url}/sessions/{session_id}")
                except httpx.HTTPError:
                    logger.warning("sandbox session %s cleanup failed", session_id, exc_info=True)
            await stack.aclose()

    async def _create_session(
        self, client: httpx.AsyncClient, *, image: str | None, keep_alive_s: float | None
    ) -> tuple[str, int | None]:
        """Create a session: its id, and the idle timeout the backend says it kept."""
        payload: dict[str, Any] = {"image": image} if image else {}
        if keep_alive_s is not None:
            # Sent only for a session that will be kept, so a deployment without
            # reuse sends the body it always sent.
            payload["idle_timeout_seconds"] = max(1, int(keep_alive_s))
        try:
            response = await client.post(f"{self._sandbox_url}/sessions", json=payload)
            if response.status_code == 503:
                raise SandboxUnavailableError(response.headers.get("Retry-After"))
            response.raise_for_status()
            handle = SessionHandle.model_validate(response.json())
            return handle.session_id, handle.idle_timeout_seconds
        except SandboxUnavailableError:
            raise
        except ValidationError as exc:
            raise SandboxNotReachableError(
                f"failed to create sandbox session at {self._safe_url}: {_contract_violation(exc)}"
            ) from None
        except (httpx.HTTPError, ValueError) as exc:
            raise SandboxNotReachableError(f"failed to create sandbox session at {self._safe_url}: {exc}") from exc

    async def _probe_session(self, client: httpx.AsyncClient, session_id: str) -> None:
        """Confirm a session to resume still exists, through the one read the contract offers.

        ``ListFiles`` is optional in the contract, so a backend without it answers
        404 here too and cannot have its sessions resumed; the caller is told the
        container is gone, which for that backend is the truth of every resume.
        """
        try:
            response = await client.get(f"{self._sandbox_url}/sessions/{session_id}/files/list")
        except httpx.HTTPError as exc:
            raise SandboxNotReachableError(f"failed to reach sandbox session at {self._safe_url}: {exc}") from exc
        if response.status_code == 404:
            raise SandboxSessionGoneError(f"sandbox session {session_id} is gone")
        if response.status_code == 503:
            raise SandboxUnavailableError(response.headers.get("Retry-After"))
        try:
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise SandboxNotReachableError(f"sandbox session {session_id} could not be resumed: {exc}") from exc


class _ProtocolSession:
    """The six operations, against one session of a contract-speaking backend."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        sandbox_url: str,
        session_id: str,
        timeout_s: float,
        *,
        holds_across_requests: bool = False,
    ) -> None:
        self._client = client
        self._sandbox_url = sandbox_url
        self._session_id = session_id
        self._timeout_s = timeout_s
        self._holds = holds_across_requests

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def holds_across_requests(self) -> bool:
        return self._holds

    def discard(self) -> None:
        self._holds = False

    @property
    def _base(self) -> str:
        return f"{self._sandbox_url}/sessions/{self._session_id}"

    async def execute(self, code: str, *, timeout_s: float) -> ResultBlock:
        payload = {
            "tool": CODE_EXECUTION_TOOL,
            "input": {"code": code},
            "timeout_seconds": int(timeout_s),
        }
        try:
            response = await self._client.post(
                f"{self._base}/exec",
                json=payload,
                # Override the client default (which equals the exec budget) so the
                # sandbox always gets to answer before the client read timeout fires.
                timeout=timeout_s + _EXEC_TIMEOUT_BUFFER_S,
            )
            if response.status_code == 503:
                raise SandboxUnavailableError(response.headers.get("Retry-After"))
            response.raise_for_status()
            # A malformed body is a contract violation, indistinguishable to the
            # caller from an unreachable backend: both mean this exec produced no
            # usable result, so they raise the same error.
            return ExecResponse.model_validate(response.json()).result_block
        except SandboxUnavailableError:
            raise
        except ValidationError as exc:
            # Raised `from None`, and the summary is built rather than rendered,
            # so neither the message nor the chained traceback carries the
            # payload into the span. See _contract_violation.
            raise SandboxNotReachableError(f"sandbox exec failed: {_contract_violation(exc)}") from None
        except (httpx.HTTPError, ValueError) as exc:
            raise SandboxNotReachableError(f"sandbox exec failed: {exc}") from exc

    async def put_file(self, path: str, data: bytes, *, mime_type: str) -> None:
        try:
            response = await self._client.post(
                f"{self._base}/files",
                files={"file": (path, data, mime_type)},
                data={"path": path},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise SandboxNotReachableError(f"sandbox refused {path!r}: {exc}") from exc

    async def list_files(self) -> list[SandboxFileEntry]:
        try:
            response = await self._client.get(f"{self._base}/files/list")
            response.raise_for_status()
            entries = response.json().get("files")
        except (httpx.HTTPError, ValueError, AttributeError) as exc:
            # ``ListFiles`` is optional in the contract, so a backend without it
            # simply reports nothing rather than failing the run.
            logger.debug("sandbox session %s workspace not listable: %s", self._session_id, exc)
            return []
        listed: list[SandboxFileEntry] = []
        for entry in entries if isinstance(entries, list) else []:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or entry.get("is_dir"):
                continue
            size = entry.get("size_bytes")
            modified = entry.get("modified_at")
            listed.append(
                SandboxFileEntry(
                    path=entry["path"],
                    size_bytes=size if isinstance(size, int) else -1,
                    modified_at=float(modified) if isinstance(modified, int | float) else None,
                )
            )
        return listed

    async def read_file(self, path: str, *, budget_bytes: int) -> AsyncGenerator[bytes, None]:
        async with self._client.stream("GET", f"{self._base}/files", params={"path": path}) as response:
            response.raise_for_status()
            declared = response.headers.get("content-length", "")
            if declared.isdigit() and int(declared) > budget_bytes:
                raise OutputOverBudget
            async for chunk in response.aiter_bytes():
                yield chunk


def _selected_provider(config: GatewayConfig) -> str:
    """The deployment's ``sandbox_provider``, normalized. One spelling, two readers."""
    return (config.sandbox_provider or "protocol").strip().lower() or "protocol"


def build_code_execution_port(config: GatewayConfig) -> CodeExecutionPort:
    """The adapter this deployment's ``sandbox_provider`` names.

    ``protocol`` (the default) reaches the backend at ``sandbox_url``, which is
    every deployment that existed before the port. Anything else is a hosted
    provider driven in this process and needs no URL.
    """
    if _selected_provider(config) == "e2b":
        from gateway.adapters.e2b_code_execution_adapter import E2BCodeExecutionAdapter

        return E2BCodeExecutionAdapter()
    sandbox_url = config.sandbox_url or otari_env("SANDBOX_URL") or None
    if sandbox_url is None:
        msg = "sandbox_url is required when sandbox_provider is 'protocol'"
        raise ValueError(msg)
    return ProtocolCodeExecutionAdapter(sandbox_url)


def verify_code_execution_ready(config: GatewayConfig) -> None:
    """Fail startup where the deployment names a sandbox this process cannot lease.

    Selecting a hosted provider is itself what makes ``sandbox_configured()``
    true, which is what publishes the tool on ``/v1/tools``, in the playground
    menu and to the pricing warning. A missing extra or credential would
    otherwise be found once per request, as a 502 on work the caller was told
    would run, so it is settled here instead: a deployment that asked for a
    sandbox it cannot reach does not start, the way one that named a bootstrap
    it cannot load does not.
    """
    if not config.sandbox_configured():
        return
    try:
        build_code_execution_port(config)
        if _selected_provider(config) == "e2b":
            from gateway.adapters.e2b_code_execution_adapter import verify_ready

            verify_ready()
    except SandboxNotReachableError as exc:
        # A transport error class for what is a configuration mistake: restate
        # it as one, since nothing is being reached yet.
        raise ValueError(str(exc)) from exc
