"""The published API lives under one prefix, and the security stamps say so.

Three things ``make openapi-check`` cannot catch. First, that every path in the
generated document sits under ``API_ROOT`` or ``OTLP_ROOT``. Second, that the
allowlists in ``gateway.main`` name routes that exist: the generator reads
those same lists, so the committed document and the generator agree even when
both are stale. The routing table is the independent source of truth here.
Third, that the operation ids hold in every mode, not only in the standalone
mode the committed document is generated from. And a fourth that no document
shows at all: that the roots are spelled nowhere but in their owner, since a
path built in a helper or a message from the literal is invisible above.

The stamp checks are weaker on purpose. ``custom_openapi`` reads the same
lists, so a mounted path wrongly added to ``_UNAUTHENTICATED_PATHS`` stamps
itself consistent and passes; only the gateway-token check asserts an exact
value. Whether a listed path really answers without a credential is a runtime
question, and not this test's.

On the worker's PostgreSQL like the rest of ``tests/integration``, because
``create_app`` is the honest way to get a routing table.
"""

import re
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI

import gateway
from gateway.api.routes import (
    audio,
    batches,
    chat,
    embeddings,
    hosted_mode,
    hybrid_mode,
    images,
    mcp,
    messages,
    moderations,
    rerank,
    responses,
    search,
    web_search_backend,
)
from gateway.core.config import API_ROOT, OTLP_ROOT, PLATFORM_TOKEN_ENV_VAR, GatewayConfig
from gateway.main import (
    _COOKIE_AUTH_PREFIXES,
    _GATEWAY_TOKEN_PATHS,
    _PUBLIC_PREFIXES,
    _UNAUTHENTICATED_PATHS,
    create_app,
)
from gateway.main import (
    _under as _main_under,
)

# App-level paths that deliberately stay at the origin root. Each has a reason
# outside this repository's control; see the spec's D8.
FROZEN_ROOT_PATHS = frozenset({"/auth/{provider}/callback", "/metrics"})
# The SPA shell and its assets. Not API, never versioned.
SHELL_PATHS = frozenset({"/", "/welcome", "/favicon.svg", "/dashboard-build.json"})
SHELL_MOUNTS = ("/assets", "/pwa", "/fonts")


def _config(postgres_url: str, mode: str | None, **overrides: Any) -> GatewayConfig:
    return GatewayConfig(
        mode=mode,
        database_url=postgres_url,
        master_key="test-master-key",
        auto_migrate=False,
        require_pricing=False,
        model_discovery=False,
        bootstrap_api_key=False,
        **overrides,
    )


def _mounted(app: FastAPI) -> set[str]:
    paths = {route.path for route in app.routes if hasattr(route, "path")}
    # A newer FastAPI may stop flattening nested routers into app.routes; an
    # empty set here would make every assertion below pass vacuously.
    assert paths, "app.routes carried no paths"
    return paths


@pytest.fixture(scope="module")
def standalone(postgres_url: str) -> FastAPI:
    # Web search configured, so web_search_backend.router mounts and the
    # gateway-token path exists to be checked.
    return create_app(
        _config(
            postgres_url,
            "standalone",
            enable_metrics=True,
            web_search_provider="tavily",
            web_search_provider_api_key="test-search-key",
            web_search_backend_token="test-gateway-token",
        )
    )


@pytest.fixture(scope="module")
def hosted(postgres_url: str) -> FastAPI:
    return create_app(_config(postgres_url, "hosted"))


@pytest.fixture(scope="module")
def hybrid(postgres_url: str) -> FastAPI:
    # The token is read once, while the config loads, so it need only be in
    # the environment for the construction. Left set, it would make the next
    # standalone app built in this process refuse to start.
    with pytest.MonkeyPatch.context() as env:
        env.setenv(PLATFORM_TOKEN_ENV_VAR, "test-platform-token")
        return create_app(_config(postgres_url, "hybrid", platform={"base_url": "http://localhost:8100/api/v1"}))


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(f"{root}/")


def test_every_mounted_route_is_under_a_root_or_frozen(standalone: FastAPI, hosted: FastAPI) -> None:
    for app in (standalone, hosted):
        stray = {
            p
            for p in _mounted(app)
            if not (
                _under(p, API_ROOT)
                or _under(p, OTLP_ROOT)
                or p in FROZEN_ROOT_PATHS
                or p in SHELL_PATHS
                or p.startswith(SHELL_MOUNTS)
            )
        }
        assert not stray, f"routes outside {API_ROOT} and {OTLP_ROOT}: {sorted(stray)}"


def test_every_published_path_is_under_a_root(standalone: FastAPI) -> None:
    paths = standalone.openapi()["paths"]
    assert paths, "the generated document has no paths"
    stray = {p for p in paths if not (_under(p, API_ROOT) or _under(p, OTLP_ROOT))}
    assert not stray, f"published outside {API_ROOT} and {OTLP_ROOT}: {sorted(stray)}"


def test_frozen_root_paths_still_exist(standalone: FastAPI) -> None:
    mounted = _mounted(standalone)
    missing = FROZEN_ROOT_PATHS - mounted
    assert not missing, f"frozen root paths were moved or dropped: {sorted(missing)}"


def test_exact_match_allowlists_name_mounted_routes(standalone: FastAPI) -> None:
    mounted = _mounted(standalone)
    for name, paths in (
        ("_UNAUTHENTICATED_PATHS", _UNAUTHENTICATED_PATHS),
        ("_GATEWAY_TOKEN_PATHS", _GATEWAY_TOKEN_PATHS),
    ):
        assert paths, f"{name} is empty"
        missing = set(paths) - mounted
        assert not missing, f"{name} names routes that are not mounted: {sorted(missing)}"


def test_prefix_allowlists_cover_at_least_one_route(standalone: FastAPI) -> None:
    mounted = _mounted(standalone)
    for name, prefixes in (
        ("_PUBLIC_PREFIXES", _PUBLIC_PREFIXES),
        ("_COOKIE_AUTH_PREFIXES", _COOKIE_AUTH_PREFIXES),
    ):
        assert prefixes, f"{name} is empty"
        for prefix in prefixes:
            assert any(_under(p, prefix) for p in mounted), f"{name}: nothing mounted under {prefix}"


def test_generated_document_stamps_security_as_the_allowlists_say(standalone: FastAPI) -> None:
    paths = standalone.openapi()["paths"]
    for path in _UNAUTHENTICATED_PATHS:
        for method, operation in paths[path].items():
            if isinstance(operation, dict):
                assert "security" not in operation, f"{method.upper()} {path} is stamped but listed as unauthenticated"
    for path in _GATEWAY_TOKEN_PATHS:
        for method, operation in paths[path].items():
            if isinstance(operation, dict):
                assert operation.get("security") == [{"GatewayTokenAuth": []}], (
                    f"{method.upper()} {path} must be stamped GatewayTokenAuth only"
                )


def _operation_ids(app: FastAPI) -> Counter[str]:
    ids = Counter(
        operation["operationId"]
        for path_item in app.openapi()["paths"].values()
        for operation in path_item.values()
        if isinstance(operation, dict)
    )
    assert ids, "the generated document has no operations"
    return ids


def test_no_operation_id_names_a_mount_root(standalone: FastAPI, hosted: FastAPI, hybrid: FastAPI) -> None:
    """The id is the method name a generated SDK exposes, so it must outlive a path move.

    FastAPI's default folds the whole path into the id, which is how every
    operation was renamed when the root moved. The roots are looked for in the
    form that default writes them, with every non-word character an underscore.
    """
    roots = tuple(re.sub(r"\W", "_", root) for root in (API_ROOT, OTLP_ROOT))
    for mode, app in (("standalone", standalone), ("hosted", hosted), ("hybrid", hybrid)):
        stray = sorted(op for op in _operation_ids(app) if any(root in op for root in roots))
        assert not stray, f"{mode}: operation ids carry a mount root: {stray}"


def test_operation_ids_are_tag_and_handler(standalone: FastAPI) -> None:
    """One anchor, so a scheme change cannot pass as long as it avoids the root."""
    assert "keys-create_key" in _operation_ids(standalone)


def test_the_mode_stubs_are_not_published(hosted: FastAPI, hybrid: FastAPI) -> None:
    """A refusal is a deployment posture, not an operation a client can call.

    Pinned by name rather than left to the duplicate check, which the stubs
    would also pass if they were split into one route per method.
    """
    hosted_prefixes = [f"{API_ROOT}{prefix}" for prefix, _ in hosted_mode.DATA_PLANE_PREFIXES]
    hybrid_prefixes = [f"{API_ROOT}{route.path}" for route in hybrid_mode.router.routes if hasattr(route, "path")]
    for mode, app, prefixes in (("hosted", hosted, hosted_prefixes), ("hybrid", hybrid, hybrid_prefixes)):
        assert prefixes, f"{mode}: no stub prefixes to check"
        published = {p for p in app.openapi()["paths"] if any(_under(p, prefix) for prefix in prefixes)}
        assert not published, f"{mode}: refused paths reached the document: {sorted(published)}"


def test_operation_ids_are_unique_in_every_mode(standalone: FastAPI, hosted: FastAPI, hybrid: FastAPI) -> None:
    """A duplicate id makes the document invalid, and only one mode's document is committed.

    The mode stubs answer seven methods at two paths from one handler, and an
    id is derived once per route, so any stub that reaches the document
    duplicates itself. Standalone mounts no stub, which is why the committed
    document never showed it.
    """
    for mode, app in (("standalone", standalone), ("hosted", hosted), ("hybrid", hybrid)):
        duplicated = sorted(op for op, count in _operation_ids(app).items() if count > 1)
        assert not duplicated, f"{mode}: duplicate operation ids: {duplicated}"


# Paths that may appear in published prose without being ours to move. The usage
# filters quote the frozen label a row carries, not a route; the rest belong to
# somebody else's contract.
PROSE_EXEMPT_PATHS = frozenset(
    {
        "/v1/chat/completions",
        "/v1/messages/count_tokens",
        "/v1/search",
    }
)


def _described_paths(document: dict[str, Any]) -> set[str]:
    """Every ``/v1`` path named in a description anywhere in the document."""
    found: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "description" and isinstance(value, str):
                    found.update(re.findall(r"(?<!/api)(?<!/otlp)/v1/[\w/{}.-]*", value))
                else:
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(document)
    return {path.rstrip(".,)`") for path in found}


def test_no_published_description_names_a_moved_path(standalone: FastAPI) -> None:
    """Published prose is API documentation, so a stale path in it misleads a caller.

    A docstring cannot interpolate, so these paths are written out by hand and
    nothing but this stops them drifting when the routes move. The exemptions
    are paths that did not move: a frozen usage label a caller filters on, and
    two foreign contracts Otari mirrors rather than owns.
    """
    stray = _described_paths(standalone.openapi()) - PROSE_EXEMPT_PATHS
    assert not stray, f"published prose still names paths that moved: {sorted(stray)}"


def test_a_near_miss_of_a_public_prefix_is_not_treated_as_public() -> None:
    """A sibling that merely starts the same way must not inherit the exemption.

    The exemption skips ``Cache-Control: private, no-store`` and ``Vary:
    Authorization``, so a byte-prefix comparison here fails open: a future
    ``/api/v1/health-internal`` would be served to a shared cache without
    either. The same comparison decides which operations the published document
    leaves unstamped.
    """
    for prefix in _PUBLIC_PREFIXES:
        assert _main_under(prefix, _PUBLIC_PREFIXES)
        assert _main_under(f"{prefix}/readiness", _PUBLIC_PREFIXES)
        assert not _main_under(f"{prefix}-internal", _PUBLIC_PREFIXES)
        assert not _main_under(f"{prefix}z", _PUBLIC_PREFIXES)
        assert not _main_under(f"{prefix}x/readiness", _PUBLIC_PREFIXES)


GATEWAY_SRC = Path(gateway.__file__).resolve().parent
WEB_SRC = GATEWAY_SRC.parents[1] / "web" / "src"

# The identifiers that look like routes and are not. Each is written to a
# usage-log or in-flight row and keeps the value it had when routes lived at
# /v1, so a source line may name one anywhere except where a route is
# declared. A new name belongs here only after reading what it feeds.
FROZEN_LABELS = frozenset(
    {
        audio.USAGE_ENDPOINT_SPEECH,
        audio.USAGE_ENDPOINT_TRANSCRIPTIONS,
        batches.USAGE_ENDPOINT,
        batches.USAGE_ENDPOINT_RESULTS,
        chat.USAGE_ENDPOINT,
        embeddings.USAGE_ENDPOINT,
        images.USAGE_ENDPOINT,
        mcp.EXECUTE_ENDPOINT,
        mcp.TOOLS_ENDPOINT,
        messages.USAGE_ENDPOINT,
        moderations.USAGE_ENDPOINT,
        rerank.USAGE_ENDPOINT,
        responses.USAGE_ENDPOINT,
        search.SEARCH_ENDPOINT,
        web_search_backend.SEARCH_ENDPOINT,
    }
)
# OpenAI's Batch API names the operation each line runs by OpenAI's own path,
# so batches.py sends this to the provider. Provider contract, not ours.
BATCH_WIRE_VALUES = frozenset({"/v1/chat/completions"})

_SPELLED_API_ROOT = re.compile(r"""["']/api/v1""")
# The quoted value, up to the closing quote or the end of the line, so a
# literal that runs on to the next line is judged on what is visible and fails.
_QUOTED_OLD_ROOT = re.compile(r"""["'](/v1/[^"']*)""")
_DECLARES_A_ROUTE = re.compile(r"\bprefix=|@\w+\.(?:get|post|put|patch|delete|head|options|api_route|websocket)\(")


def _source_lines(root: Path, suffixes: tuple[str, ...]) -> list[tuple[Path, int, str]]:
    assert root.is_dir(), f"{root} is not a directory; the guard has nothing to read"
    lines = [
        (path.relative_to(root), number, line)
        for path in sorted(root.rglob("*"))
        if path.suffix in suffixes and path.is_file()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
    ]
    assert lines, f"no {suffixes} source under {root}"
    return lines


def _cite(path: Path, number: int, line: str) -> str:
    return f"{path}:{number}: {line.strip()}"


def test_no_gateway_source_spells_the_api_root() -> None:
    """The root has one owner, so a path built anywhere else takes the constant.

    The check is on the literal text. A path built as ``f"{API_ROOT}/keys"``
    never contains it, which is what leaves the constant as the only way
    through.
    """
    stray = [
        _cite(path, number, line)
        for path, number, line in _source_lines(GATEWAY_SRC, (".py",))
        if path != Path("core/config.py") and _SPELLED_API_ROOT.search(line)
    ]
    assert not stray, "the API root is spelled outside core/config.py:\n" + "\n".join(stray)


def test_no_gateway_source_spells_the_old_root() -> None:
    """A quoted ``/v1/`` is a frozen label, a wire value, or OTel's tail; anything else is a stale route.

    A string equal to a frozen label may be named anywhere, in its definition,
    a docstring or a filter description, except on a line that declares a
    route: there it is a version in a prefix or a decorator, whatever its
    value. The OTLP module is read past whole, because OTel owns the
    ``/v1/{traces,logs,metrics}`` tail it declares.
    """
    allowed = FROZEN_LABELS | BATCH_WIRE_VALUES
    stray: list[str] = []
    for path, number, line in _source_lines(GATEWAY_SRC, (".py",)):
        if path == Path("api/routes/otlp.py"):
            continue
        declares_a_route = _DECLARES_A_ROUTE.search(line) is not None
        stray.extend(
            _cite(path, number, line)
            for match in _QUOTED_OLD_ROOT.finditer(line)
            if match.group(1) not in allowed or declares_a_route
        )
    assert not stray, "the old root is spelled where a route or a built path should be:\n" + "\n".join(stray)


def test_no_dashboard_source_doubles_the_root() -> None:
    """``apiFetch`` prepends the root, so a call site or a mock that spells it too says ``/api/api/v1``.

    Only the doubled form is checked. The dashboard still spells the root on
    its own in copy and in tests that stub ``fetch``, which see the whole URL.
    """
    stray = [
        _cite(path, number, line)
        for path, number, line in _source_lines(WEB_SRC, (".ts", ".tsx", ".md", ".css"))
        if "/api/api/v1" in line
    ]
    assert not stray, "the dashboard doubles the API root:\n" + "\n".join(stray)
