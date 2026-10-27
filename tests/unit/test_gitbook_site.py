"""The GitBook site build, run against the real docs so a page missing from the menu fails a PR."""

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "prepare_gitbook_site.py"
_DOCS_DIR = _REPO_ROOT / "docs"
_URL = "https://github.com/mozilla-ai/otari"
_SPEC = {
    "info": {"title": "otari", "version": "0.0.0-dev"},
    "paths": {
        "/api/v1/keys": {"get": {"tags": ["keys"]}},
        "/api/v1/chat/completions": {"post": {"tags": ["chat"]}, "get": {"tags": ["chat"]}},
    },
}


_SITE_CONFIG = """site:
  title: Otari
  structure:
    - type: section
      children:
        - type: space
          content:
            directory: ./home
    - type: section
      children:
        - type: space
          content:
            directory: ./guides
"""


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("prepare_gitbook_site", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gitbook = _load()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    (docs / "providers").mkdir(parents=True)
    (docs / "public").mkdir()
    (tmp_path / "deploy").mkdir()
    pages = ("index.md", "guide.md", "internal.md", "api-reference.md", "providers/openai.md")
    for path in (*(f"docs/{page}" for page in pages), "ARCHITECTURE.md"):
        (tmp_path / path).write_text("# Page\n", encoding="utf-8")
    (docs / "public" / "openapi.json").write_text(json.dumps(_SPEC), encoding="utf-8")
    (docs / "SUMMARY.md").write_text(
        "# Home\n\n* [Otari](index.md)\n\n# Guides\n\n* [Guide](guide.md)\n\n## Setup\n\n* [API](api-reference.md)\n"
        "* [OpenAI](providers/openai.md)\n",
        encoding="utf-8",
    )
    (docs / "gitbook-docs.yaml").write_text(_SITE_CONFIG, encoding="utf-8")
    (tmp_path / ".gitbook-branch-readme.md").write_text("# Branch\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def site(repo: Path) -> Any:
    return gitbook.Site.from_summary(repo / "docs", "v1.2.3")


def test_summary_pages_reads_entries_in_menu_order() -> None:
    summary = "# Summary\n\n* [Otari](index.md)\n\n## Group\n\n* [B](b.md)\n- [A](sub/a.md)\n"
    assert gitbook.summary_pages(summary) == ["index.md", "b.md", "sub/a.md"]


def test_summary_spaces_splits_the_menu_at_each_top_heading() -> None:
    summary = "# Home\n\n* [Otari](index.md)\n\n# Self-Hosting\n\n## Run it\n\n* [Deploy](deploy.md)\n"
    spaces = gitbook.summary_spaces(summary)
    assert [(space.title, space.directory, space.pages) for space in spaces] == [
        ("Home", "home", ["index.md"]),
        ("Self-Hosting", "self-hosting", ["deploy.md"]),
    ]
    assert spaces[1].menu == "## Run it\n\n* [Deploy](deploy.md)\n"


@pytest.mark.parametrize(
    "summary", ["* [Otari](index.md)\n# Home\n\n* [B](b.md)\n", "# Home\n\n* [B](b.md)\n\n# Empty\n\n## Group\n"]
)
def test_summary_spaces_refuses_a_page_outside_a_space_or_an_empty_space(summary: str) -> None:
    with pytest.raises(ValueError):
        gitbook.summary_spaces(summary)


def test_build_copies_assets_into_every_space_and_keeps_links_to_them(repo: Path) -> None:
    (repo / "docs" / "assets").mkdir()
    (repo / "docs" / "assets" / "flow.webp").write_bytes(b"RIFF")
    site = gitbook.Site.from_summary(repo / "docs", "main")
    assert site.rewrite_links("![Flow](../assets/flow.webp)", "providers/openai.md") == "![Flow](../assets/flow.webp)"
    site.build(repo / "site")
    assert (repo / "site" / "home" / "assets" / "flow.webp").is_file()
    assert (repo / "site" / "guides" / "assets" / "flow.webp").is_file()


def test_link_within_a_space_is_kept(site: Any) -> None:
    text = "See [API](../api-reference.md#auth)."
    assert site.rewrite_links(text, "providers/openai.md") == text


def test_link_to_another_space_gets_the_page_url_on_the_site(site: Any) -> None:
    assert site.rewrite_links("[Home](../index.md#top)", "providers/openai.md") == "[Home](https://docs.otari.ai/#top)"
    assert site.rewrite_links("[Guide](guide.md)", "index.md") == "[Guide](https://docs.otari.ai/guides)"
    assert site.rewrite_links("[OpenAI](providers/openai.md)", "index.md") == (
        "[OpenAI](https://docs.otari.ai/guides/setup/openai)"
    )


def test_page_paths_follow_the_group_the_parent_and_the_file_name() -> None:
    menu = "* [Start](start.md)\n\n## Run It Well\n\n* [API](api-reference.md)\n  * [Chat](api-endpoints/chat.md)\n"
    space = gitbook.Space("Reference", "reference", menu)
    assert space.page_paths() == {
        "start.md": "",
        "api-reference.md": "run-it-well/api-reference",
        "api-endpoints/chat.md": "run-it-well/api-reference/chat",
    }


def test_site_url_option_moves_links_between_spaces(repo: Path) -> None:
    site = gitbook.Site.from_summary(repo / "docs", "main", "https://preview.example/token/")
    assert site.rewrite_links("[API](api-reference.md)", "index.md") == (
        "[API](https://preview.example/token/guides/setup/api-reference)"
    )


def test_link_to_an_unpublished_page_points_at_github(site: Any) -> None:
    assert site.rewrite_links("[Internal](internal.md#rules)", "index.md") == (
        f"[Internal]({_URL}/blob/v1.2.3/docs/internal.md#rules)"
    )


def test_link_outside_docs_points_at_github(site: Any) -> None:
    text = '[Arch](../ARCHITECTURE.md "Architecture") and [Deploy](../deploy/)'
    assert site.rewrite_links(text, "index.md") == (
        f'[Arch]({_URL}/blob/v1.2.3/ARCHITECTURE.md "Architecture") and [Deploy]({_URL}/tree/v1.2.3/deploy)'
    )


def test_link_to_a_non_markdown_file_points_at_github(site: Any) -> None:
    assert site.rewrite_links("[Spec](public/openapi.json)", "index.md") == (
        f"[Spec]({_URL}/blob/v1.2.3/docs/public/openapi.json)"
    )


def test_external_and_anchor_links_are_kept(site: Any) -> None:
    text = "[Site](https://otari.ai) [Mail](mailto:a@b.c) [Here](#here)"
    assert site.rewrite_links(text, "index.md") == text


def test_links_in_code_are_kept(site: Any) -> None:
    text = "```markdown\n[Internal](internal.md)\n```\nUse `[x](internal.md)` here.\n~~~~\n```\n[A](internal.md)\n~~~~"
    assert site.rewrite_links(text, "index.md") == text


def test_link_with_a_code_label_is_rewritten(site: Any) -> None:
    assert site.rewrite_links("Start with [`ARCHITECTURE.md`](../ARCHITECTURE.md).", "index.md") == (
        f"Start with [`ARCHITECTURE.md`]({_URL}/blob/v1.2.3/ARCHITECTURE.md)."
    )


def test_link_after_a_code_block_is_rewritten(site: Any) -> None:
    text = "````text\n```\n````\n[Internal](internal.md)"
    assert site.rewrite_links(text, "index.md").endswith(f"[Internal]({_URL}/blob/v1.2.3/docs/internal.md)")


@pytest.mark.parametrize("target", ["missing.md", "../../outside.md"])
def test_link_to_no_file_in_the_repository_fails(site: Any, target: str) -> None:
    with pytest.raises(ValueError, match=re.escape(target)):
        site.rewrite_links(f"[Broken]({target})", "index.md")


def test_build_writes_only_the_published_pages(repo: Path, site: Any) -> None:
    out = repo / "site"
    (out / "stale.md").parent.mkdir()
    (out / "stale.md").write_text("old", encoding="utf-8")

    site.build(out)

    written = sorted(path.relative_to(out).as_posix() for path in out.rglob("*") if path.is_file())
    assert written == [
        "README.md",
        "api/openapi.json",
        "gitbook-docs.yaml",
        "guides/.gitbook.yaml",
        "guides/SUMMARY.md",
        "guides/api-endpoints/chat.md",
        "guides/api-endpoints/keys.md",
        "guides/api-reference.md",
        "guides/guide.md",
        "guides/providers/openai.md",
        "home/.gitbook.yaml",
        "home/SUMMARY.md",
        "home/index.md",
    ]


def test_build_gives_each_space_its_first_page_as_its_landing_page(repo: Path, site: Any) -> None:
    site.build(repo / "site")

    assert (repo / "site" / "guides" / ".gitbook.yaml").read_text(encoding="utf-8") == (
        "root: ./\n\nstructure:\n  readme: guide.md\n  summary: SUMMARY.md\n"
    )


def test_build_refuses_a_heading_the_site_config_does_not_map(repo: Path, site: Any) -> None:
    (repo / "docs" / "gitbook-docs.yaml").write_text(_SITE_CONFIG.replace("./guides", "./elsewhere"), encoding="utf-8")

    with pytest.raises(ValueError, match="elsewhere"):
        site.build(repo / "site")


def test_build_nests_a_page_per_tag_under_the_api_reference(repo: Path, site: Any) -> None:
    site.build(repo / "site")

    assert (repo / "site" / "guides" / "SUMMARY.md").read_text(encoding="utf-8") == (
        "# Summary\n\n* [Guide](guide.md)\n\n## Setup\n\n* [API](api-reference.md)\n"
        "  * [Chat](api-endpoints/chat.md)\n  * [Keys](api-endpoints/keys.md)\n* [OpenAI](providers/openai.md)\n"
    )


def test_endpoint_page_names_each_operation_by_path_then_method(repo: Path, site: Any) -> None:
    site.build(repo / "site")

    spec_url = "https://raw.githubusercontent.com/mozilla-ai/otari/gitbook-docs/api/openapi.json"
    block = (
        '{{% openapi-operation spec="otari-openapi-spec" path="/api/v1/chat/completions" method="{method}" %}}\n'
        f"[OpenAPI otari-openapi-spec]({spec_url})\n"
        "{{% endopenapi-operation %}}\n"
    )
    assert (repo / "site" / "guides" / "api-endpoints" / "chat.md").read_text(encoding="utf-8") == (
        f"# Chat\n\n{block.format(method='get')}\n{block.format(method='post')}"
    )


@pytest.mark.parametrize(("ref", "version"), [("v1.2.3", "1.2.3"), ("main", "0.0.0-dev")])
def test_build_stamps_a_release_version_on_the_spec(repo: Path, ref: str, version: str) -> None:
    gitbook.Site.from_summary(repo / "docs", ref).build(repo / "site")

    published = json.loads((repo / "site" / "api" / "openapi.json").read_text(encoding="utf-8"))
    assert published["info"]["version"] == version
    assert published["paths"] == _SPEC["paths"]


@pytest.mark.parametrize(
    ("tags", "page"), [([], "other"), (["chat", "keys"], "chat"), (["Chat Completions"], "chat-completions")]
)
def test_operation_is_paged_under_its_first_tag(tags: list[str], page: str) -> None:
    spec = {"paths": {"/api/v1/odd": {"get": {"tags": tags}}}}
    assert gitbook.operations_by_page(spec) == {page: [("/api/v1/odd", "get")]}


@pytest.mark.parametrize(
    ("page", "title"),
    [
        ("chat", "Chat"),
        ("organization-budgets", "Organization budgets"),
        ("mcp-servers", "MCP servers"),
        ("otel", "OTel"),
    ],
)
def test_page_title_is_the_page_name_in_sentence_case(page: str, title: str) -> None:
    assert gitbook.page_title(page) == title


def test_operations_on_a_page_list_reads_before_writes() -> None:
    methods = ("delete", "patch", "post", "put", "get")
    spec = {"paths": {"/api/v1/keys": {method: {"tags": ["keys"]} for method in methods}}}
    assert [method for _, method in gitbook.operations_by_page(spec)["keys"]] == [
        "get",
        "post",
        "put",
        "patch",
        "delete",
    ]


def test_endpoint_pages_follow_the_api_reference_children() -> None:
    summary = "* [API](api-reference.md)\n  * [Errors](errors.md)\n    * [Codes](codes.md)\n* [Guide](guide.md)"
    assert gitbook.nest_endpoint_pages(summary, ["chat"]) == (
        "* [API](api-reference.md)\n  * [Errors](errors.md)\n    * [Codes](codes.md)\n"
        "  * [Chat](api-endpoints/chat.md)\n* [Guide](guide.md)"
    )


def test_build_fails_when_a_docs_page_is_in_the_endpoint_pages_directory(repo: Path) -> None:
    (repo / "docs" / "api-endpoints").mkdir()
    (repo / "docs" / "api-endpoints" / "chat.md").write_text("# Chat\n", encoding="utf-8")
    summary = repo / "docs" / "SUMMARY.md"
    summary.write_text(summary.read_text(encoding="utf-8") + "* [Chat](api-endpoints/chat.md)\n", encoding="utf-8")
    site = gitbook.Site.from_summary(repo / "docs", "main")
    with pytest.raises(ValueError, match="api-endpoints"):
        site.build(repo / "site")


def _one_space(repo: Path, page: str) -> None:
    (repo / "docs" / "SUMMARY.md").write_text(f"# Home\n\n* [Page]({page})\n", encoding="utf-8")
    (repo / "docs" / "gitbook-docs.yaml").write_text("        directory: ./home\n", encoding="utf-8")


def test_build_fails_when_the_menu_has_no_api_reference(repo: Path) -> None:
    _one_space(repo, "guide.md")
    site = gitbook.Site.from_summary(repo / "docs", "main")
    with pytest.raises(ValueError, match="api-reference.md"):
        site.build(repo / "site")


def test_build_fails_when_the_menu_names_a_missing_page(repo: Path) -> None:
    _one_space(repo, "gone.md")
    site = gitbook.Site.from_summary(repo / "docs", "main")
    with pytest.raises(ValueError, match="gone.md"):
        site.build(repo / "site")


def test_pages_off_the_menu_lists_pages_the_menu_does_not_name(repo: Path, site: Any) -> None:
    assert gitbook.pages_off_the_menu(repo / "docs", site.published) == ["internal.md"]


def test_every_docs_page_is_on_the_menu() -> None:
    site = gitbook.Site.from_summary(_DOCS_DIR, "main")
    assert gitbook.pages_off_the_menu(_DOCS_DIR, site.published) == [], (
        "List each new page in docs/SUMMARY.md, or move a page for contributors out of docs/"
    )


def test_real_docs_build(tmp_path: Path) -> None:
    site = gitbook.Site.from_summary(_DOCS_DIR, "v0.0.0")
    site.build(tmp_path / "site")
    assert (tmp_path / "site" / "home" / "index.md").is_file()
