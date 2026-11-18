"""Where Claude Code keeps the files on this machine that Otari reads.

Claude Code puts its user settings, session history and plugins in one
directory, `~/.claude` by default. `CLAUDE_CONFIG_DIR` moves all of it, so a
path built from the home directory finds nothing on a machine that sets the
variable.

NOTE: The variable is read on each call, not at import, because a caller may
set it after this module is imported and a Click option default is built while
the module loads.
"""

import os
from pathlib import Path

CLAUDE_CONFIG_DIR_ENV = "CLAUDE_CONFIG_DIR"


def claude_config_dir() -> Path:
    """Claude Code's own configuration directory on this machine."""
    configured = os.environ.get(CLAUDE_CONFIG_DIR_ENV)
    return Path(configured) if configured else Path.home() / ".claude"


def claude_user_settings_path() -> Path:
    """The user-level `settings.json`, which applies in every repository on this machine."""
    return claude_config_dir() / "settings.json"


def claude_transcripts_dir() -> Path:
    """Where Claude Code writes one JSONL transcript per session."""
    return claude_config_dir() / "projects"
