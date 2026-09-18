"""Bounded, envelope-specific inspection of structured file references."""

from typing import Any

from gateway.services.provider_files.contracts import FilesError

_MAX_NODES = 20000
_MAX_DEPTH = 32
_MAX_REFERENCES = 100


def collect_anthropic_file_references(value: Any) -> list[str]:
    """Collect references throughout message history without interpreting ordinary text."""
    found: dict[str, None] = {}
    pending = [(value, 0)]
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > _MAX_NODES or depth > _MAX_DEPTH:
            raise FilesError(400, "File reference structure exceeds configured limits")
        if isinstance(item, list):
            if len(item) + len(pending) > _MAX_NODES:
                raise FilesError(400, "File reference structure exceeds configured limits")
            pending.extend((child, depth + 1) for child in reversed(item))
        elif isinstance(item, dict):
            if len(item) + len(pending) > _MAX_NODES:
                raise FilesError(400, "File reference structure exceeds configured limits")
            nested_file = item.get("file")
            if isinstance(nested_file, dict) and "file_id" in nested_file:
                raise FilesError(400, "Use Messages for provider-native file references")
            file_id = item.get("file_id")
            if file_id is not None:
                kind = item.get("type")
                if kind not in {"file", "container_upload", "code_execution_output", "bash_code_execution_output"}:
                    raise FilesError(400, "Unsupported structured file reference")
                if not isinstance(file_id, str) or not file_id or len(file_id) > 255:
                    raise FilesError(400, "Invalid file reference")
                found[file_id] = None
                if len(found) > _MAX_REFERENCES:
                    raise FilesError(400, "Too many file references")
            # Text, schemas, and arbitrary tool inputs are not provider file-reference positions.
            pending.extend(
                (item[key], depth + 1) for key in ("messages", "content", "source", "output", "results") if key in item
            )
    return list(found)


def reject_openai_file_state(payload: dict[str, Any]) -> None:
    """Reject account-scoped OpenAI state until its inference ownership protocol exists."""
    if payload.get("previous_response_id") or payload.get("conversation"):
        raise FilesError(400, "Provider conversation reuse is not supported in hybrid mode")
    pending: list[tuple[Any, int]] = [(payload, 0)]
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > _MAX_NODES or depth > _MAX_DEPTH:
            raise FilesError(400, "File reference structure exceeds configured limits")
        if isinstance(item, list):
            if len(item) + len(pending) > _MAX_NODES:
                raise FilesError(400, "File reference structure exceeds configured limits")
            pending.extend((child, depth + 1) for child in item)
        elif isinstance(item, dict):
            kind = item.get("type")
            if (
                any(item.get(key) for key in ("file_id", "file_ids", "vector_store_ids", "container_id"))
                or (
                    isinstance(kind, str)
                    and kind
                    in {
                        "code_interpreter",
                        "file_search",
                        "shell",
                        "item_reference",
                        "container_reference",
                        "compaction",
                    }
                )
                or isinstance(item.get("container"), str)
            ):
                raise FilesError(400, "OpenAI provider file state is not supported in hybrid inference")
            variables = item.get("variables")
            if isinstance(variables, dict):
                # Prompt variable names are arbitrary; their values are typed input blocks or text.
                pending.append((list(variables.values()), depth + 1))
            pending.extend(
                (item[key], depth + 1)
                for key in (
                    "prompt",
                    "input",
                    "messages",
                    "content",
                    "file",
                    "tools",
                    "container",
                    "attachments",
                    "environment",
                    "input_image_mask",
                    "output",
                    "results",
                    "annotations",
                )
                if key in item
            )
