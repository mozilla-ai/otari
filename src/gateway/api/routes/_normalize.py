"""Normalizing a request's attachments, and deciding who runs its code.

Two rules govern what leaves here, and they pull in opposite directions.

An attachment the model reads never fails the request.
The payload is forwarded unchanged instead, because a request refused here would
lose a budget reservation already taken and a caller gains nothing from the
refusal.

An attachment the provider's own code execution reads does fail the request when
it cannot be given.
The caller asked for code to be run over that file, so an answer written without
it is worse than no answer, and a block Otari cannot resolve must not reach the
provider carrying a file ID of the caller's choosing.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from typing import Any

from any_llm import AnyLLM, LLMProvider
from fastapi import HTTPException

from gateway.api.routes._attempts import CandidateCannotServe, PrepareKwargs
from gateway.api.routes._tools import (
    _extract_code_execution_tool,
    decide_code_executor,
    first_provider_code_execution_tool,
    parse_code_execution_header,
    provider_runs_code_natively,
    resolve_code_executor_preference,
)
from gateway.core.config import GatewayConfig
from gateway.exceptions import TenancyError
from gateway.exceptions.files_exceptions import (
    AttachedFilesNotReadError,
    NoCandidateHoldsCopiesError,
    ProviderAttachmentError,
    ProviderAttachmentRefusedError,
    ProviderUploadFailedError,
)
from gateway.log_config import logger
from gateway.models.tools import CodeExecutor
from gateway.services.content_normalizer import (
    NormalizationStats,
    WireFormat,
    has_container_blocks,
    name_container_copies,
    normalize_messages,
)
from gateway.services.files import FileService, StagedFile, provider_holds_copies
from gateway.services.model_capabilities import resolve_capabilities
from gateway.services.provider_kwargs import ProviderAccounts, effective_credential
from gateway.services.tools import Dialect


def _executor_preference(
    config: GatewayConfig, code_execution_header: str | None, workspace_executor: CodeExecutor | None
) -> CodeExecutor | None:
    """The three layers admission uses, composed into one preference.

    The workspace pin, the header and the deployment default.
    None for a header outside the vocabulary, which admission refuses, so
    nothing here decides on it.
    """
    try:
        requested = parse_code_execution_header(code_execution_header)
    except ValueError:
        return None
    preference, _ = resolve_code_executor_preference(
        requested=requested, workspace=workspace_executor, deployment=config.effective_code_executor()
    )
    return preference


def provider_container_requested(
    tools: list[dict[str, Any]] | None,
    *,
    config: GatewayConfig,
    provider: LLMProvider | None,
    dialect: Dialect,
    code_execution_header: str | None,
    workspace_executor: CodeExecutor | None = None,
) -> bool:
    """Whether this request's code will run in the dispatched provider's own container.

    The other side of ``sandbox_requested``: a provider's own declaration, in the
    wire format that declaration is native to, which the executor leaves with the
    provider. A file the request attaches then has to be one that provider holds,
    so it is copied there rather than read for the model.
    """
    explicit, remaining = _extract_code_execution_tool(tools)
    if explicit is not None:
        return False
    keyword = first_provider_code_execution_tool(remaining)
    if keyword is None or not provider_runs_code_natively(
        keyword, provider=provider.value if provider is not None else None, dialect=dialect
    ):
        return False
    preference = _executor_preference(config, code_execution_header, workspace_executor)
    if preference is None:
        return False
    decided = decide_code_executor(preference, sandbox_configured=config.sandbox_configured(), native_available=True)
    return decided is CodeExecutor.PROVIDER


def sandbox_requested(
    tools: list[dict[str, Any]] | None,
    *,
    config: GatewayConfig,
    provider: LLMProvider | None,
    dialect: Dialect,
    code_execution_header: str | None,
    workspace_executor: CodeExecutor | None = None,
) -> bool:
    """Whether this request's code will run on the gateway's sandbox.

    True for the explicit ``otari_code_execution`` type, and for a provider's own
    declaration the executor would bring here (see ``_tools.decide_code_executor``):
    a file the request attaches is then staged for the sandbox rather than only
    shown to the model. Decided from the same three layers admission uses, the
    workspace pin (``workspace_executor``, read once in the preamble), the header
    and the deployment default, against the dispatched provider. A header outside
    the vocabulary answers false here and is refused at admission.
    """
    explicit, remaining = _extract_code_execution_tool(tools)
    if explicit is not None:
        return True
    keyword = first_provider_code_execution_tool(remaining)
    if keyword is None or not config.sandbox_configured():
        return False
    preference = _executor_preference(config, code_execution_header, workspace_executor)
    if preference is None:
        return False
    native = provider_runs_code_natively(
        keyword, provider=provider.value if provider is not None else None, dialect=dialect
    )
    return decide_code_executor(preference, sandbox_configured=True, native_available=native) is CodeExecutor.OTARI


async def normalize_request_messages(
    messages: list[dict[str, Any]],
    *,
    fmt: WireFormat,
    config: GatewayConfig,
    provider: LLMProvider | None,
    model: str,
    files: FileService | None,
    user_id: str | None,
    instance: str | None = None,
    workspace_id: uuid.UUID | None = None,
    sandbox_requested: bool = False,
    provider_container: bool = False,
) -> tuple[list[dict[str, Any]], NormalizationStats]:
    """Normalize ``messages`` for the resolved ``provider/model``.

    ``sandbox_requested`` is whether the request declared the gateway's
    code-execution tool; the normalizer then records referenced uploads on the
    stats for the sandbox backend to seed (see ``NormalizationStats.sandbox_inputs``).

    ``provider_container`` is set where the provider runs the code in a container
    of its own. The uploads its ``container_upload`` blocks name are then
    recorded on ``NormalizationStats.container_inputs``, and a block naming none
    refuses the request.

    An attachment the model reads never fails the request, because this runs
    after the budget reservation and a refusal here would lose it.
    The input is returned untouched instead, which is also the answer where file
    understanding is off or the provider could not be parsed.

    Raises:
        ProviderAttachmentError: only where ``provider_container`` is set.
    """
    if provider_container and not config.file_understanding_enabled and has_container_blocks(messages, fmt):
        # Nothing below would examine the blocks, so a `container_upload` would
        # reach the provider naming a file of the caller's choosing.
        raise AttachedFilesNotReadError
    if provider is None or not config.file_understanding_enabled:
        return messages, NormalizationStats()
    try:
        caps = resolve_capabilities(config, provider, model, instance=instance)
        return await normalize_messages(
            messages,
            config=config,
            caps=caps,
            fmt=fmt,
            files=files,
            user_id=user_id,
            workspace_id=workspace_id,
            sandbox_requested=sandbox_requested,
            provider_container=provider_container,
        )
    except ProviderAttachmentError:
        raise
    except Exception as exc:  # noqa: BLE001 - never fail the request or leak the reservation
        if provider_container:
            # An aborted pass leaves the payload as it arrived, container blocks
            # and all, so forwarding it would hand the provider the caller's own
            # file ids. That is what this path exists to stop.
            logger.warning("content normalization failed for a provider container: %s", exc)
            raise ProviderUploadFailedError from exc
        logger.warning("content normalization failed; forwarding messages unchanged: %s", exc)
        return messages, NormalizationStats()


def container_copies_step(
    *,
    files: FileService,
    inputs: Sequence[StagedFile],
    accounts: ProviderAccounts,
    render: Callable[[TenancyError], HTTPException],
) -> PrepareKwargs:
    """The step that sends each candidate the IDs of copies in its own provider account.

    A candidate whose provider cannot hold a copy cannot serve the request.
    A refusal is rendered with ``render``, the dialect's own envelope.
    A copy that could not be made raises :class:`ProviderUploadFailedError`,
    which is that candidate's own failure rather than a refusal of the request.
    """

    async def _prepare(instance: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        provider, _ = AnyLLM.split_model_provider(kwargs["model"])
        if not isinstance(provider, LLMProvider) or not provider_holds_copies(provider):
            raise CandidateCannotServe(render(NoCandidateHoldsCopiesError()))
        try:
            credential = effective_credential(provider, kwargs)
        except LookupError as exc:
            logger.warning("No credential to copy attached files to %s: %s", instance, exc)
            raise ProviderUploadFailedError from exc
        account = accounts.name(provider, instance, credential)
        try:
            copies = await files.provider_file_ids(inputs, account, credential)
        except ProviderAttachmentRefusedError as exc:
            raise render(exc) from exc
        return {**kwargs, "messages": name_container_copies(kwargs["messages"], copies)}

    return _prepare
