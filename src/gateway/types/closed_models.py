"""Which models no open-weight plan covers.

Every model not ruled closed here counts as open-weight: the closed labs are few
and change slowly, where a list of open models would trail every release.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass

# A Bedrock-style id names its vendor, and sometimes a region, before the model:
# ``anthropic.claude-sonnet-4``, ``us.anthropic.claude-sonnet-4``. A dot followed
# by a digit is a version (``glm-4.6``, ``gpt-4.1``), not a vendor prefix.
_VENDOR_PREFIX = re.compile(r"^(?:[a-z]{2,4}\.)?[a-z][a-z0-9-]*\.(?=[a-z])")


def _model_name(model: str) -> str:
    """The model's own name: after any path (``openrouter``'s ``anthropic/``) and any vendor prefix."""
    name = model.rsplit("/", 1)[-1].lower()
    return _VENDOR_PREFIX.sub("", name)


def _is_family(name: str, family: str) -> bool:
    """Whether ``name`` belongs to ``family``: it starts with it, as a whole word unless the family ends in ``-``."""
    if not name.startswith(family):
        return False
    rest = name[len(family) :]
    return family.endswith("-") or not rest or not rest[0].isalnum()


@dataclass(frozen=True)
class ClosedModels:
    """Closed providers, closed model families, and the open families that are exceptions to both."""

    providers: frozenset[str]
    closed_families: tuple[str, ...]
    open_families: tuple[str, ...]

    @classmethod
    def of(
        cls, *, providers: Iterable[str], closed_families: Iterable[str], open_families: Iterable[str]
    ) -> "ClosedModels":
        """Build one from lists as a person writes them: case and surrounding blanks do not matter."""
        clean = [item.strip().lower() for item in providers if item.strip()]
        return cls(
            providers=frozenset(clean),
            closed_families=tuple(item.strip().lower() for item in closed_families if item.strip()),
            open_families=tuple(item.strip().lower() for item in open_families if item.strip()),
        )

    def is_closed(self, provider: str, model: str) -> bool:
        """Whether ``model`` on ``provider`` (a wire name) is closed.

        The model's own name decides first, whichever provider or aggregator
        serves it (``openrouter``'s ``anthropic/claude-sonnet-4``): an open family
        is open even on a closed provider, and a closed family closed anywhere.
        """
        name = _model_name(model)
        if any(_is_family(name, family) for family in self.open_families):
            return False
        return provider.lower() in self.providers or any(_is_family(name, family) for family in self.closed_families)
