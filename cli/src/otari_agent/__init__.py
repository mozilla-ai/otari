"""otari, the laptop-side CLI: `otari hook`, `otari hook setup`, `otari import claude-code`.

Installed on its own (Homebrew) this is the whole program. Installed beside the
`gateway` distribution (Docker, a development checkout) it also carries the
server commands, which `gateway.cli.register` attaches; see `otari_agent.cli`.
"""

# Stamped by the release workflow (a sed on this line) right before
# `uv build --package otari-agent`. A literal, not importlib.metadata: it costs
# nothing at import and is correct from a source tree.
__version__ = "0.0.0"

__all__ = ["__version__"]
