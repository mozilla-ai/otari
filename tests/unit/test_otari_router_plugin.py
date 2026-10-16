"""The otari-router Claude Code plugin and the marketplace file that lists it.

`claude plugin validate` and `claude plugin test` cover the plugin itself and
run where Claude Code is installed. This pins what has no gate otherwise: that
the repository's marketplace file, the plugin's manifest and its hooks file
agree with each other, and that the manifest asks for what the module needs.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MARKETPLACE = REPO_ROOT / ".claude-plugin" / "marketplace.json"
PLUGIN = REPO_ROOT / "plugins" / "otari-router"


def _json(path: Path) -> dict[str, object]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict), path
    return parsed


def test_the_marketplace_lists_the_plugin_where_it_lives() -> None:
    marketplace = _json(MARKETPLACE)
    manifest = _json(PLUGIN / ".claude-plugin" / "plugin.json")

    assert marketplace["name"] == "otari"
    plugins = marketplace["plugins"]
    assert isinstance(plugins, list)
    (entry,) = plugins
    assert entry["name"] == manifest["name"] == "otari-router"
    assert (REPO_ROOT / entry["source"]).resolve() == PLUGIN
    assert (PLUGIN / "README.md").is_file()


def test_every_hooks_module_the_plugin_names_exists() -> None:
    hooks = _json(PLUGIN / "hooks" / "hooks.json")

    modules = hooks["modules"]
    assert isinstance(modules, list) and modules
    for module in modules:
        assert isinstance(module, str) and module.startswith("./"), module
        assert (PLUGIN / "hooks" / module[2:]).is_file(), module


def test_the_plugin_requires_the_gateway_and_the_key() -> None:
    manifest = _json(PLUGIN / ".claude-plugin" / "plugin.json")

    options = manifest["userConfig"]
    assert isinstance(options, dict)
    assert set(options) == {"otari_url", "otari_api_key"}
    for option in options.values():
        assert option["type"] == "string"
        assert option["required"] is True, "the module reads both without a fallback"
    assert options["otari_api_key"]["sensitive"] is True
    assert "sensitive" not in options["otari_url"]
