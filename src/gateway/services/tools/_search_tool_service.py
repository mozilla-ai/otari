"""The search and fetch instances an operator stores through the dashboard.

Each use case reads or writes the ``search_tool_credentials`` rows in a block
of its own, then hands what it read to this worker's overlay
(``search_tool_store_service``). The request path reads the config's instance maps
synchronously, so a write takes effect at once on the worker that served it,
and on the others within the refresh interval.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from gateway.core.config import GatewayConfig
from gateway.core.database import DATABASE_ERRORS
from gateway.core.settings.tools import BUILTIN_FETCH, ToolKind
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.tools_exceptions import (
    SearchToolAlreadyStoredError,
    SearchToolChangedError,
    SearchToolKeyUnstorableError,
    StoredSearchToolNotFoundError,
)
from gateway.log_config import logger
from gateway.models.secret_fields import restore_redacted_values
from gateway.repositories.tools import SearchToolConflict, SearchToolRepository
from gateway.schemas.tools import (
    CreatedSearchToolSchema,
    CreateSearchToolRequest,
    ReencryptSearchToolsResponse,
    StoredSearchToolSchema,
    UpdateSearchToolRequest,
)
from gateway.services.runtime_settings_service import SettingValue
from gateway.services.search_tool_store_service import (
    apply_stored_rows,
    apply_stored_tools,
    config_file_search_tools,
    config_file_tools,
    stored_tool_names,
)
from gateway.services.secret_box import (
    SecretBoxUnavailableError,
    SecretDecryptionError,
    decrypt_secret,
    encrypt_secret,
)
from gateway.services.tool_settings_service import WEB_SEARCH_DEFAULT_TOOL, apply_override
from gateway.services.tools._search_tool_writes import (
    check_kind_unchanged,
    check_tool_entry,
    check_tool_name_is_free,
    check_tool_write,
    default_to_pin,
)

# The runtime tool settings as stored, every valid one.
ReadToolSettings = Callable[[], Awaitable[dict[str, SettingValue]]]
# Validate one runtime tool setting and stage it in the caller's transaction; returns the normalized value.
StageToolSetting = Callable[[str, SettingValue], Awaitable[SettingValue]]


def _encrypted_key(api_key: str | None) -> tuple[str | None, str | None]:
    """The ciphertext and last four characters to store for ``api_key``, or two Nones to clear it.

    Raises:
        SearchToolKeyUnstorableError: ``OTARI_SECRET_KEY`` is missing or invalid.
    """
    if not api_key:
        return None, None
    try:
        return encrypt_secret(api_key), api_key[-4:]
    except SecretBoxUnavailableError as exc:
        raise SearchToolKeyUnstorableError(str(exc)) from None


class SearchToolService:
    """Configure the deployment's stored search and fetch instances, and keep this worker's overlay of them."""

    def __init__(
        self,
        uow: UnitOfWork,
        *,
        config: GatewayConfig,
        tools: SearchToolRepository,
        read_settings: ReadToolSettings,
        stage_setting: StageToolSetting,
    ) -> None:
        """Bind the unit of work and the collaborators this service composes.

        ``read_settings`` and ``stage_setting`` reach the runtime tool settings,
        whose module takes the session: the builder binds them to the session
        ``uow`` runs on, so they read and write inside this service's blocks.
        """
        self.uow = uow
        self.config = config
        self.tools = tools
        self.read_settings = read_settings
        self.stage_setting = stage_setting

    async def create(self, request: CreateSearchToolRequest) -> CreatedSearchToolSchema:
        """Store a search or fetch instance, pinning the in-loop default first when it adds a second one.

        Raises:
            SearchToolRefusedError: the instance rules refuse the instance.
            SearchToolAlreadyStoredError: a stored instance has the name, or another create won a race.
            SearchToolKeyUnstorableError: a key came without a usable ``OTARI_SECRET_KEY``.
        """
        name = request.name.strip()
        kind = request.kind
        entry = request.to_entry()
        check_tool_entry(name, entry, kind=kind, inherited_api_base=self.config.web_search_url)
        async with self.uow:
            # Another replica may have stored an instance this one would clash with or
            # name, or the default setting the pin below reads.
            await self._reload_stored_tools()
            check_tool_write(
                self.config,
                name,
                entry,
                kind=kind,
                sets_name=True,
                sets_options=request.options is not None,
                sets_fetch_tool=True,
            )
            check_tool_name_is_free(self.config, name, kind)
            conflict = f"A stored {kind} tool '{name}' already exists; use PATCH to update it."
            if await self.tools.get(name) is not None:
                raise SearchToolAlreadyStoredError(conflict)
            pinned = default_to_pin(self.config, name, stored_names=stored_tool_names()) if kind == "search" else None
            if pinned is not None:
                # The default's row can collide too, with another create pinning at the same time.
                conflict = (
                    f"'{name}' was not stored: a stored tool of that name exists, or another create changed the "
                    "search instances at the same time. Reload and retry."
                )
                # Staged before the row, so the row's flush writes both and a
                # collision on either is this create's one conflict.
                await self.stage_setting(WEB_SEARCH_DEFAULT_TOOL, pinned.name)
            encrypted_api_key, last4 = _encrypted_key(request.api_key)
            try:
                row = await self.tools.insert(
                    name=name,
                    kind=kind,
                    provider=request.provider,
                    fetch_tool=entry["fetch_tool"],
                    api_base=request.api_base,
                    timeout_seconds=request.timeout,
                    options=restore_redacted_values(request.options, None) or {},
                    encrypted_api_key=encrypted_api_key,
                    last4=last4,
                )
            except SearchToolConflict:
                raise SearchToolAlreadyStoredError(conflict) from None
        if pinned is not None:
            apply_override(self.config, WEB_SEARCH_DEFAULT_TOOL, pinned.name)
            logger.info("Set web_search_default_tool to '%s' before adding a second search instance", pinned.name)
        shadows_config = name in config_file_tools(self.config, kind)
        if shadows_config:
            logger.warning(
                "Stored %s tool '%s' shadows the config.yml entry of the same name; the stored entry now wins.",
                kind,
                name,
            )
        await self._apply_write(f"writing '{name}'")
        return CreatedSearchToolSchema(
            **row.to_public_dict(),
            shadows_config=shadows_config,
            pinned_web_search_default_tool=pinned.name if pinned is not None else None,
            notice=pinned.notice if pinned is not None else None,
        )

    async def update(self, name: str, request: UpdateSearchToolRequest) -> StoredSearchToolSchema:
        """Update a stored instance under a row lock: an omitted field is kept, an explicit ``null`` clears it.

        Raises:
            StoredSearchToolNotFoundError: no stored instance has the name.
            SearchToolChangedError: ``expected_updated_at`` is not the stored version.
            SearchToolRefusedError: the instance rules refuse the result, or a change of kind.
            SearchToolKeyUnstorableError: a key came without a usable ``OTARI_SECRET_KEY``.
        """
        async with self.uow:
            existing = await self.tools.get_for_update(name)
            if existing is None:
                raise StoredSearchToolNotFoundError(f"No stored search tool '{name}'.")
            kind: ToolKind = "fetch" if existing.kind == "fetch" else "search"
            check_kind_unchanged(name, kind, request.kind)
            if request.expected_updated_at is not None:
                current = existing.updated_at.isoformat() if existing.updated_at else None
                if current != request.expected_updated_at:
                    raise SearchToolChangedError()

            # Distinguish "field omitted" (keep) from "field set to null" (clear), then
            # validate the instance as it will be after the patch.
            sent = request.model_fields_set
            fetch_tool = request.fetch_tool.strip() if request.fetch_tool is not None else None
            # An explicit null provider is meaningless (the column is non-null), so it
            # keeps the stored provider.
            sets_provider = "provider" in sent and bool(request.provider)
            merged: dict[str, Any] = {
                "provider": request.provider if sets_provider else existing.provider,
                "fetch_tool": fetch_tool if "fetch_tool" in sent else existing.fetch_tool,
                "api_base": request.api_base if "api_base" in sent else existing.api_base,
                "timeout": request.timeout if "timeout" in sent else existing.timeout_seconds,
                "options": request.options if "options" in sent else existing.options,
                # Only presence matters to the validator, and the stored key is never
                # decrypted here just to re-validate it.
                "api_key": request.api_key if "api_key" in sent else existing.encrypted_api_key,
            }
            check_tool_entry(name, merged, kind=kind, inherited_api_base=self.config.web_search_url)
            sets_fetch_tool = "fetch_tool" in sent
            if sets_fetch_tool and merged["fetch_tool"] is not None:
                # The fetch instance it names may have been stored through another replica.
                apply_stored_rows(self.config, await self.tools.list_committed())
            check_tool_write(
                self.config,
                name,
                merged,
                kind=kind,
                sets_name=False,
                sets_options="options" in sent or merged["provider"] != existing.provider,
                sets_fetch_tool=sets_fetch_tool,
            )

            changes: dict[str, Any] = {}
            if sets_provider:
                changes["provider"] = request.provider
            if sets_fetch_tool:
                changes["fetch_tool"] = merged["fetch_tool"]
            if "api_base" in sent:
                changes["api_base"] = request.api_base
            if "timeout" in sent:
                changes["timeout_seconds"] = request.timeout
            if "options" in sent:
                # A value echoed back as the mask keeps the stored one.
                changes["options"] = restore_redacted_values(request.options, existing.options) or {}
            if "api_key" in sent:
                changes["encrypted_api_key"], changes["last4"] = _encrypted_key(request.api_key)
            row = await self.tools.update(existing, changes)
        await self._apply_write(f"writing '{name}'")
        return StoredSearchToolSchema.from_model(row, shadows_config=name in config_file_tools(self.config, kind))

    async def delete(self, name: str) -> None:
        """Delete a stored instance. A config-file one, or ``builtin_fetch``, cannot be deleted here.

        Raises:
            StoredSearchToolNotFoundError: no stored instance has the name.
        """
        async with self.uow:
            row = await self.tools.get(name)
            if row is None:
                raise StoredSearchToolNotFoundError(self._not_stored(name))
            await self.tools.delete(row)
        await self._apply_write(f"writing '{name}'")

    async def reencrypt(self) -> ReencryptSearchToolsResponse:
        """Re-encrypt stored keys with the current primary OTARI_SECRET_KEY.

        A key the configured key set cannot decrypt is left untouched and counted
        as unreadable. A row an edit changed mid-rotation is counted as skipped:
        whoever wrote it already encrypted it with the primary key.

        Raises:
            SearchToolKeyUnstorableError: ``OTARI_SECRET_KEY`` is missing or invalid.
        """
        reencrypted = 0
        unreadable = 0
        skipped = 0
        async with self.uow:
            try:
                for row in await self.tools.list_encrypted():
                    original = row.encrypted_api_key
                    if original is None:
                        continue
                    try:
                        plaintext = decrypt_secret(original)
                    except SecretDecryptionError:
                        unreadable += 1
                        continue
                    replacement = encrypt_secret(plaintext)
                    if await self.tools.swap_ciphertext(row.name, expected=original, replacement=replacement):
                        reencrypted += 1
                    else:
                        skipped += 1
            except SecretBoxUnavailableError as exc:
                raise SearchToolKeyUnstorableError(str(exc)) from None
        await self._apply_write("re-encrypting keys")
        return ReencryptSearchToolsResponse(reencrypted=reencrypted, unreadable=unreadable, skipped=skipped)

    async def refresh_rows(self) -> set[str]:
        """Reload the stored rows, apply them, and return shadowed names.

        The rows only. A write to the tool settings reloads them to check a default
        against, and must not re-apply the stored settings it is about to replace.
        """
        async with self.uow:
            rows = await self.tools.list_committed()
        return apply_stored_rows(self.config, rows)

    async def _reload_stored_tools(self) -> set[str]:
        """Read the stored rows and the runtime tool settings, apply both, and return shadowed names.

        Runs in the caller's block, so both are read in one transaction, the rows
        first: a create that adds a second search instance stores the default
        naming the first in the same commit, so a read between the two statements
        gets at worst that default with the rows before the create, never the
        second row without the default.
        """
        rows = await self.tools.list_committed()
        return apply_stored_tools(self.config, rows, await self.read_settings())

    async def _apply_write(self, what: str) -> None:
        """Make a committed change take effect on this worker.

        The write is already committed, so a refresh failure is logged and the
        call succeeds; other workers converge within the refresh interval.
        """
        try:
            await self.refresh_rows()
        except DATABASE_ERRORS:
            logger.warning("Search tool overlay refresh failed after %s; converges within TTL", what)

    def _not_stored(self, name: str) -> str:
        """Why no stored instance called ``name`` can be deleted, naming the one the API cannot change."""
        if name in config_file_search_tools(self.config):
            return f"Search tool '{name}' is defined in the config file and cannot be deleted through the API."
        if name in config_file_tools(self.config, "fetch"):
            return f"Fetch tool '{name}' is defined in the config file and cannot be deleted through the API."
        if name == BUILTIN_FETCH:
            return f"{BUILTIN_FETCH}, the built-in fetcher, always exists and cannot be deleted."
        return f"No stored search tool '{name}'."
