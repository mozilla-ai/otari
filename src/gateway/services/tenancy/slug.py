"""Organization slug generation. Pure functions, no database access."""

import re


def slugify(value: str) -> str:
    """Reduce a display name to a URL-safe slug, or ``organization`` if nothing survives."""
    lowered = value.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return slug or "organization"


__all__ = ["slugify"]
