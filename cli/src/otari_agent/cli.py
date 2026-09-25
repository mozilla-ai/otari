"""The `otari` console script: the light commands, plus the gateway's when it is installed."""

from __future__ import annotations

import importlib.util
import os
from typing import Any

import click

from otari_agent import __version__
from otari_agent.hook import guardrails, hook
from otari_agent.usage_import import import_group

# What gateway.cli.register attaches, named here so a light install can say why
# the command is missing rather than that it does not exist.
SERVER_COMMANDS = frozenset({"serve", "init-db", "migrate", "gen-secret-key", "routing"})


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
        if command is None and cmd_name in SERVER_COMMANDS:
            ctx.fail(
                f"'{cmd_name}' is a server command, and this install does not include the otari gateway. "
                "Run it from the gateway's Docker image or a source checkout."
            )
        return command


# The Docker image installs from the tree, where __version__ is the unstamped
# 0.0.0, and names its version in OTARI_VERSION instead, the variable
# src/gateway/version.py reads. A stamped build (Homebrew) sets no such variable.
_REPORTED_VERSION = os.environ.get("OTARI_VERSION") or __version__


@click.group(cls=OtariGroup)
@click.version_option(_REPORTED_VERSION, "--version", prog_name="otari")
def cli() -> None:
    """Otari CLI."""


cli.add_command(hook)
cli.add_command(guardrails)
cli.add_command(import_group)


def main() -> None:
    """Entry point for the `otari` console script."""
    cli()


if __name__ == "__main__":
    main()
