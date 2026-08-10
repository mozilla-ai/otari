"""The code-execution wire contract, as the gateway consumes it.

Otari does not run code itself: it drives a code-execution backend over the
contract specified in ``docs/code-execution-protocol.md``. These are the
response shapes the gateway reads, so ``SandboxBackend`` parses a validated
object instead of walking untyped dicts.

Only the fields the gateway actually reads are modelled. The contract's
extension policy says a backend may return additional fields and a consumer
must ignore the ones it does not recognise, so unknown keys are dropped rather
than rejected, and the spec doc (not this module) is the full contract.

The types live here, not next to ``services/sandbox_backend``, because they are
leaf types with no service dependencies, and ``check_architecture.py`` keeps
that direction one-way.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "CodeExecutionFileRef",
    "CodeExecutionResult",
    "ExecResponse",
    "ResultBlock",
    "SessionHandle",
]


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

    filename: str = ""


class CodeExecutionResult(_ContractModel):
    """The ``content`` payload of a result block.

    Note this is a single object, not a list of mixed content blocks: the
    contract mirrors Anthropic's ``code_execution_result``, whose nested
    ``content`` is the file-reference list.
    """

    stdout: str = ""
    stderr: str = ""
    # Optional as well as defaulted: a backend that sends an explicit `null`
    # means "no exit status", which the renderer treats the same as 0. Making
    # this a bare `int` would reject that payload outright.
    return_code: int | None = 0
    content: list[CodeExecutionFileRef] = Field(default_factory=list)


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
