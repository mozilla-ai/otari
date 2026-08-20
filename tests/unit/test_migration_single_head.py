"""The migration graph resolves to exactly one head.

``core.database`` and the ``otari migrate`` CLI both upgrade to ``"head"``,
singular, so a second head is not a style problem: Alembic refuses the argument
outright ("Multiple head revisions are present") and no deployment can migrate.

Nothing in per-PR CI catches it, which is why this test exists. Two revisions
authored against the same parent each pass on their own branch, and the graph
only forks once both have merged, so the first red run is on ``main`` after the
second merge. Asserting the invariant here moves the failure back onto the PR
that would create it.

The fix for a failure is a merge revision naming both heads
(``down_revision = ("<a>", "<b>")``), not renumbering either branch.
"""

from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_migration_graph_has_a_single_head() -> None:
    config = Config(str(_REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(_REPO_ROOT / "alembic"))

    heads = ScriptDirectory.from_config(config).get_heads()

    assert len(heads) == 1, (
        f"Alembic has {len(heads)} heads ({', '.join(sorted(heads))}), so 'alembic upgrade head' fails "
        "and no deployment can migrate. Add a merge revision whose down_revision names every head."
    )
