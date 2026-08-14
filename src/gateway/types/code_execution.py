"""The code-execution wire contract, as the gateway consumes it.

Otari does not run code itself: it drives a code-execution backend over the
contract specified in ``docs/code-execution-protocol.md``. These are the
response shapes the gateway reads, so ``SandboxBackend`` parses a validated
object instead of walking untyped dicts.

Only the fields the gateway actually reads are modelled. The contract's
extension policy says a backend may return additional fields and a consumer
must ignore the ones it does not recognize, so unknown keys are dropped rather
than rejected, and the spec doc (not this module) is the full contract.

The same policy is why these models are liberal about the values they accept.
A field the gateway only renders absorbs a null or an unusable shape and keeps
going, because a payload that carries usable output must not be discarded over
a stream the backend chose to serialize as `null`. Validation is reserved for
the parts the gateway cannot proceed without: it fails on a missing session id
or a missing result payload, not on a cosmetic field.

The types live here, not next to ``services/sandbox_backend``, because they are
leaf types with no service dependencies, and ``check_architecture.py`` keeps
that direction one-way.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

__all__ = [
    "CodeExecutionFileRef",
    "CodeExecutionResult",
    "ExecResponse",
    "ResultBlock",
    "SessionHandle",
]


def _rendered_text(value: Any) -> Any:
    """Coerce whatever a backend put in a render-only field into text.

    ``null`` becomes empty: a backend that models an absent stream as an
    optional field serializes it that way rather than omitting it, and a plain
    ``str`` would reject it.

    Anything else non-string is stringified rather than dropped. Dropping it
    would be the more dangerous choice: a backend that puts a structure in
    ``stderr`` is still reporting a failure, and discarding it renders the run
    as clean, unmarked output that then bills as a success. Stringifying keeps
    the signal, and matches what the pre-typing renderer did by interpolating
    the value straight into the rendered section.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _file_refs_only(value: Any) -> Any:
    """Keep the entries that can describe a file, discard the rest.

    The nested ``content`` is the file-reference list, but for a text-editor
    view the same field carries the file's text instead, and it may be ``null``
    when nothing was produced. Neither names a file, so neither contributes a
    filename, and neither is worth failing an otherwise successful call over.
    """
    if not isinstance(value, list):
        return []
    return [entry for entry in value if isinstance(entry, dict)]


_RenderedStr = Annotated[str, BeforeValidator(_rendered_text)]
_FileRefs = Annotated[list["CodeExecutionFileRef"], BeforeValidator(_file_refs_only)]


class _ContractModel(BaseModel):
    """Base for every contract shape: tolerant of fields we do not know.

    ``extra="ignore"`` is Pydantic's default; it is spelled out because it is
    the extension policy rather than an incidental default, and a future
    ``model_config`` edit should not silently drop it.
    """

    model_config = ConfigDict(extra="ignore")


class SessionHandle(_ContractModel):
    """A leased session. The gateway needs only the id to address it."""

    session_id: str


class CodeExecutionFileRef(_ContractModel):
    """A file the execution produced (a chart, a generated CSV)."""

    filename: _RenderedStr = ""


class CodeExecutionResult(_ContractModel):
    """The ``content`` payload of a result block.

    Note this is a single object, not a list of mixed content blocks: the
    contract mirrors Anthropic's ``code_execution_result``, whose nested
    ``content`` is the file-reference list.
    """

    stdout: _RenderedStr = ""
    stderr: _RenderedStr = ""
    # Optional as well as defaulted: a backend that sends an explicit `null`
    # means "no exit status", which the renderer treats the same as 0. Making
    # this a bare `int` would reject that payload outright.
    return_code: int | None = 0
    content: _FileRefs = Field(default_factory=list)


class ResultBlock(_ContractModel):
    """One tool call's outcome.

    ``type`` is a plain string, deliberately not a closed union of the three
    tool kinds. The contract may add a fourth kind additively, and an older
    gateway must keep rendering its result rather than failing to parse it.
    """

    type: str = ""
    content: CodeExecutionResult


class ExecResponse(_ContractModel):
    """The response to one execute call."""

    result_block: ResultBlock
