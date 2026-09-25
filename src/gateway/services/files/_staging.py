"""An uploaded file handed to a code-execution sandbox, and the name it takes inside one."""

from collections.abc import Collection
from dataclasses import dataclass
from pathlib import PurePosixPath

# The purpose stamped on a file the code-execution sandbox produced, so a
# listing can tell a run's artifact from a user's upload.
CODE_EXECUTION_OUTPUT_PURPOSE = "code_execution_output"


@dataclass(frozen=True)
class StagedFile:
    """An uploaded file a request asked the code-execution sandbox to see.

    Carries the storage ref rather than the bytes so a large upload is read
    once, when the sandbox opens, and never held for longer than that.
    """

    file_id: str
    # The name the file has inside the session's working directory, which is
    # also what the model is told; see ``sandbox_path_for``.
    filename: str
    mime_type: str
    storage_ref: str


def sandbox_path_for(filename: str, taken: Collection[str]) -> str:
    """The name a staged upload gets inside the session's working directory.

    The upload's own name reduced to its last path segment, so a name carrying
    separators neither nests nor escapes, and suffixed ``-2``, ``-3``, ... when
    an earlier attachment already took it, so two uploads named alike are both
    there rather than one overwriting the other. A name with no usable segment
    becomes ``file``.
    """
    base = PurePosixPath(filename.replace("\\", "/")).name
    if base in ("", ".", ".."):
        base = "file"
    if base not in taken:
        return base
    stem, dot, ext = base.rpartition(".")
    if not dot or not stem:
        stem, ext = base, ""
    else:
        ext = f".{ext}"
    n = 2
    while f"{stem}-{n}{ext}" in taken:
        n += 1
    return f"{stem}-{n}{ext}"
