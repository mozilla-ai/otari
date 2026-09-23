"""The `otari` console script: the light commands, plus the gateway's when it is installed."""

from __future__ import annotations

import importlib.util
from typing import Any

import click

from otari_agent.hook import gates, hook
from otari_agent.usage_import import import_group


class OtariGroup(click.Group):
    """A group that attaches the gateway's server commands only when something asks for them.

    `otari hook` runs on every tool call of a coding agent, so it must not pay
    for importing gateway.cli (any-llm, SQLModel, uvicorn) in a checkout where
    the gateway is installed too. The attach happens on a listing (--help) or
    on a subcommand this group does not know. `find_spec`, not try/except
    ImportError: a broken gateway install should fail loudly, not vanish from
    the help.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._gateway_attached = False

    def _attach_gateway_commands(self) -> None:
        if self._gateway_attached:
            return
        self._gateway_attached = True
        if importlib.util.find_spec("gateway") is None:
            return
        from gateway.cli import register

        register(self)

    def list_commands(self, ctx: click.Context) -> list[str]:
        self._attach_gateway_commands()
        return super().list_commands(ctx)

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        command = super().get_command(ctx, cmd_name)
        if command is None:
            self._attach_gateway_commands()
            command = super().get_command(ctx, cmd_name)
        return command


@click.group(cls=OtariGroup)
def cli() -> None:
    """Otari CLI."""


cli.add_command(hook)
cli.add_command(gates)
cli.add_command(import_group)


def main() -> None:
    """Entry point for the `otari` console script."""
    cli()


if __name__ == "__main__":
    main()
