"""otari, the laptop-side CLI: `otari hook`, `otari hook setup`, `otari import claude-code`.

Installed on its own (Homebrew) this is the whole program. Installed beside the
`gateway` distribution (Docker, a development checkout) it also carries the
server commands, which `gateway.cli.register` attaches; see `otari_agent.cli`.
"""

# Written at build time by setuptools-scm from the repository's Git tag (see
# cli/pyproject.toml). Absent only in a source tree nothing has built yet,
# where there is no version to report.
try:
    from otari_agent._version import __version__
except ImportError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
