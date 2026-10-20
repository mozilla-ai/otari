"""What a start compares to decide whether a config.yml entry changed since it was last written."""

import hashlib
import json
from collections.abc import Mapping


def declaration_digest(entry: Mapping[str, object]) -> str:
    """A stable SHA-256 of a declared entry's fields, independent of key order.

    Stored beside the row the entry was written over, so the next start writes
    the entry back only when this changes.
    """
    canonical = json.dumps(entry, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()
