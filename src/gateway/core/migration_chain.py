"""Which tables otari's migration chain owns, read from the revisions without running them."""

import ast
from pathlib import Path

from alembic.script import Script, ScriptDirectory
from alembic.util import CommandError


def _table_name_argument(call: ast.Call, position: int, keyword: str) -> ast.expr | None:
    """Return the argument a table operation received for one parameter, by position or by keyword."""
    if len(call.args) > position:
        return call.args[position]
    return next((argument.value for argument in call.keywords if argument.arg == keyword), None)


def tables_created_by(script: Script) -> frozenset[str]:
    """Return the tables one revision creates, or renames a table to.

    A revision must name each such table with a string literal or a module-level constant.

    Raises:
        CommandError: The revision names a table any other way, so its ownership cannot be read.

    """
    tree = ast.parse(Path(script.path).read_text(encoding="utf-8"), filename=script.path)
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr == "create_table":
            argument = _table_name_argument(node, 0, "table_name")
        elif node.func.attr == "rename_table":
            argument = _table_name_argument(node, 1, "new_table_name")
        else:
            continue
        if isinstance(argument, ast.Constant):
            name = argument.value
        elif isinstance(argument, ast.Name):
            name = getattr(script.module, argument.id, None)
        else:
            name = None
        if not isinstance(name, str):
            written = ast.unparse(argument) if argument is not None else "no name"
            msg = (
                f"Revision {script.revision} names a table with {written}, which is neither a string nor a "
                "module-level constant, so autogenerate cannot tell whether the table belongs to otari's chain."
            )
            raise CommandError(msg)
        names.add(name)
    return frozenset(names)


def tables_the_chain_creates(scripts: ScriptDirectory) -> frozenset[str]:
    """Return every table a revision in the chain creates, or renames a table to."""
    return frozenset(name for script in scripts.walk_revisions() for name in tables_created_by(script))
