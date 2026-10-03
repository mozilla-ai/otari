"""The step that sends each candidate the copies an attached file has in its own account."""

from __future__ import annotations

import uuid
from typing import Any, cast

import pytest
from fastapi import HTTPException

from gateway.api.routes._attempts import CandidateCannotServe
from gateway.api.routes._normalize import container_copies_step
from gateway.exceptions import TenancyError
from gateway.exceptions.files_exceptions import ProviderUploadDisabledError, ProviderUploadFailedError
from gateway.services.files import FileService, StagedFile
from gateway.services.provider_kwargs import ProviderAccounts
from gateway.types.provider_account import ProviderAccount, ResolvedCredential

_STAGED = StagedFile(file_id="file-1", filename="report.csv", mime_type="text/csv", storage_ref="ref-1")
_WORKSPACE = uuid.uuid4()


class _Files:
    def __init__(self, error: Exception | None = None) -> None:
        self.asked: list[tuple[list[str], ProviderAccount]] = []
        self._error = error

    async def provider_file_ids(
        self, files: list[StagedFile], account: ProviderAccount, credential: ResolvedCredential
    ) -> dict[str, str]:
        self.asked.append(([staged.file_id for staged in files], account))
        if self._error is not None:
            raise self._error
        return {staged.file_id: f"copy-of-{staged.file_id}-in-{account.instance}" for staged in files}


def _render(exc: TenancyError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=exc.message)


def _step(files: _Files) -> Any:
    return container_copies_step(
        files=cast(FileService, files),
        inputs=[_STAGED],
        accounts=ProviderAccounts(pepper="p" * 32, workspace_id=_WORKSPACE),
        render=_render,
    )


def _kwargs(model: str = "anthropic:claude-sonnet-4-5", api_key: str = "sk-one") -> dict[str, Any]:
    return {
        "model": model,
        "api_key": api_key,
        "messages": [{"role": "user", "content": [{"type": "container_upload", "file_id": "file-1"}]}],
    }


@pytest.mark.asyncio
async def test_the_candidate_is_sent_the_copy_in_its_own_account() -> None:
    files = _Files()
    kwargs = _kwargs()

    sent = await _step(files)("claude", kwargs)

    assert sent["messages"][0]["content"][0]["file_id"] == "copy-of-file-1-in-claude"
    assert kwargs["messages"][0]["content"][0]["file_id"] == "file-1"
    [(_, account)] = files.asked
    assert (account.instance, account.workspace_id) == ("claude", _WORKSPACE)


@pytest.mark.asyncio
async def test_two_keys_reach_two_accounts() -> None:
    files = _Files()
    step = _step(files)

    await step("a", _kwargs(api_key="sk-one"))
    await step("b", _kwargs(api_key="sk-two"))

    first, second = (account for _, account in files.asked)
    assert first.identity != second.identity


@pytest.mark.asyncio
async def test_a_candidate_whose_provider_holds_no_copies_cannot_serve() -> None:
    files = _Files()

    with pytest.raises(CandidateCannotServe) as exc_info:
        await _step(files)("openai", _kwargs(model="openai:gpt-5"))

    assert exc_info.value.refusal.status_code == 400
    assert files.asked == []


@pytest.mark.asyncio
async def test_a_refusal_is_answered_in_the_dialects_envelope() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await _step(_Files(error=ProviderUploadDisabledError()))("claude", _kwargs())

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_a_copy_that_could_not_be_made_is_the_candidates_failure() -> None:
    with pytest.raises(ProviderUploadFailedError):
        await _step(_Files(error=ProviderUploadFailedError()))("claude", _kwargs())


@pytest.mark.asyncio
async def test_a_candidate_with_no_key_anywhere_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ProviderUploadFailedError):
        await _step(_Files())("claude", {**_kwargs(), "api_key": None})
