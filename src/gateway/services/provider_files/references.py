"""Bounded inspection of structured Anthropic file references."""

from typing import Any

from gateway.services.provider_files.contracts import FilesError

_MAX_NODES = 20000
_MAX_DEPTH = 32
_MAX_REFERENCES = 100


def collect_file_references(value: Any) -> list[str]:
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
                if kind not in {"file", "container_upload", "code_execution_output"}:
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
