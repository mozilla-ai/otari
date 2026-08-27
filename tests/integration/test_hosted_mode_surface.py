"""A hosted control plane serves the management plane and refuses inference.

The mirror of ``test_hybrid_mode_surface``, which asserts the opposite posture
for the opposite deployment. Regression cover for otari#822: hosted mode used to
mount every data-plane router alongside the management API, so a workspace key
pointed at the control plane host ran real inference that no wallet was ever
debited for.

On the worker's PostgreSQL like the rest of ``tests/integration``, not on a
SQLite file of its own: a hosted app boots against a real control plane's
database, and asserting a deployment posture on a backend the deployment never
runs on is the kind of shortcut that makes a green test mean less than it looks.
"""

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from gateway.api.routes.hosted_mode import DATA_PLANE_PREFIXES
from gateway.core.config import GatewayConfig
from gateway.main import create_app

from .conftest import build_test_client

# Every data-plane path otari#822 found served on the control plane, plus the
# sub-paths that ride the same catch-all.
INFERENCE_PATHS = (
    "/v1/chat/completions",
    "/v1/messages",
    "/v1/messages/count_tokens",
    "/v1/responses",
    "/v1/embeddings",
    "/v1/images/generations",
    "/v1/audio/transcriptions",
    "/v1/audio/speech",
    "/v1/rerank",
    "/v1/batches",
    "/v1/moderations",
    "/v1/search",
    "/v1/files",
)

EXPECTED_DETAIL = (
    "This deployment is a control plane and does not serve inference. "
    "Send inference requests to your Otari gateway instead."
)


def _config(postgres_url: str, mode: str | None, data_plane_url: str | None = None) -> GatewayConfig:
    return GatewayConfig(
        mode=mode,
        database_url=postgres_url,
        data_plane_url=data_plane_url,
        master_key="test-master-key",
        auto_migrate=False,
        require_pricing=False,
        model_discovery=False,
        # Nothing here authenticates, so there is no key to mint and no reason
        # to print one into the suite's output.
        bootstrap_api_key=False,
    )


@pytest.fixture
def hosted_client(postgres_url: str) -> Generator[TestClient]:
    yield from build_test_client(_config(postgres_url, "hosted"))


@pytest.fixture
def hosted_client_knowing_its_data_plane(postgres_url: str) -> Generator[TestClient]:
    """The same control plane, configured with the address of its data plane."""
    yield from build_test_client(_config(postgres_url, "hosted", "https://gateway.example.com"))


def test_hosted_mode_refuses_every_inference_endpoint(hosted_client: TestClient) -> None:
    """The leak itself: no data-plane path may be served by a control plane.

    Asserted unauthenticated on purpose. A refusal that only a valid key reaches
    would still leave the endpoint mounted, and the bug was that it ran the
    request at all, so the refusal has to come before auth rather than after it.
    """
    for path in INFERENCE_PATHS:
        response = hosted_client.post(path, json={})

        assert response.status_code == 404, f"{path} was served: {response.status_code}"
        assert response.json() == {"detail": EXPECTED_DETAIL}, path


def test_hosted_mode_refuses_inference_on_every_verb(hosted_client: TestClient) -> None:
    """A GET must not slip past a stub registered only for POST.

    The reads are refused with the writes, deliberately, so a tenant holding a
    file or a batch created before the gate went in loses the API to it rather
    than keeping a read-only window onto a plane this deployment no longer
    serves. See ``hosted_mode``'s module docstring.
    """
    for method in ("get", "put", "patch", "delete"):
        response = getattr(hosted_client, method)("/v1/chat/completions")

        assert response.status_code == 404, method
        assert response.json() == {"detail": EXPECTED_DETAIL}, method

    for method, path in (
        ("get", "/v1/files"),
        ("get", "/v1/files/file-abc"),
        ("get", "/v1/files/file-abc/content"),
        ("delete", "/v1/files/file-abc"),
        ("get", "/v1/batches"),
        ("get", "/v1/batches/batch-abc"),
    ):
        response = getattr(hosted_client, method)(path)

        assert response.status_code == 404, path
        assert response.json() == {"detail": EXPECTED_DETAIL}, path


def test_hosted_mode_still_serves_the_management_plane(hosted_client: TestClient) -> None:
    """The refusal is scoped to the data plane and takes no management route with it.

    A 404 here would mean the stub's catch-all swallowed a management prefix.
    Anything else, including the 401 these return unauthenticated, means the
    real router answered.
    """
    for path in (
        "/v1/keys",
        "/v1/usage",
        "/v1/organizations/me/provider-keys",
        # Discovery, not dispatch, and a surface bootstrap publishes for a
        # hosted deployment, so it stays mounted.
        "/v1/models",
        # The catalog POST /v1/search dispatches against. Management, and the
        # one prefix a careless /v1/search stub could shadow.
        "/v1/search-tools",
    ):
        response = hosted_client.get(path)

        assert response.status_code != 404, f"{path} was swallowed by the stubs"

    # Unauthenticated and mounted in every mode: it is how a browser learns
    # which deployment it reached.
    bootstrap = hosted_client.get("/v1/bootstrap")
    assert bootstrap.status_code == 200
    assert bootstrap.json()["deployment_type"] == "hosted"


@pytest.mark.parametrize("mode", ["standalone", None])
def test_non_hosted_modes_keep_serving_inference(mode: str | None, postgres_url: str) -> None:
    """Standalone legitimately serves both planes from one process, and still does.

    ``None`` covers the derived case: mode unset and no platform token also
    resolves to standalone, and that is what an ordinary self-hosted deployment
    runs, so it must not pick up the control plane's posture.

    Which routers mount is settled when the app is constructed, so this reads
    the mounted set rather than booting a client to ask it.
    """
    config = _config(postgres_url, mode)
    assert not config.is_hosted_mode
    served = {route.path for route in create_app(config).routes if hasattr(route, "path")}

    # Each path as the data plane spells it, with no prefix fallback: a looser
    # match would pass on a route that had moved out from under the caller.
    for path in INFERENCE_PATHS:
        assert path in served, path
    # And the refusal stubs are absent entirely, rather than mounted behind the
    # real routes where a path change could one day expose them.
    assert "/v1/chat/{path:path}" not in served


def test_every_gated_router_has_a_stub_standing_in_for_it(postgres_url: str) -> None:
    """Nothing may be gated off hosted mode without a stub to answer for it.

    The gated-router list lives in ``main._register_core_routers`` and the
    prefix list in ``hosted_mode``, with nothing tying the two together, so a
    router dropped from the first without a prefix added to the second would
    ship the bare 404 this whole module exists to avoid, and would do it
    silently. Derive the tie instead of restating either list: whatever
    standalone serves and hosted does not must fall under a prefix the stubs
    claim.

    It cannot catch a *new* data-plane router added outside the guard, since
    that one is mounted in both modes and so never shows up in the difference.
    The half it does catch is the half that fails quietly.
    """
    def mounted(mode: str) -> set[str]:
        app = create_app(_config(postgres_url, mode))
        return {route.path for route in app.routes if hasattr(route, "path")}

    dropped = mounted("standalone") - mounted("hosted")
    assert dropped, "hosted mode dropped no routes at all, so the gate is not doing anything"

    prefixes = [prefix for prefix, _why in DATA_PLANE_PREFIXES]
    uncovered = {
        path for path in dropped if not any(path == prefix or path.startswith(f"{prefix}/") for prefix in prefixes)
    }
    assert not uncovered, f"gated off hosted mode with no stub to answer for them: {sorted(uncovered)}"


def test_the_refusal_names_the_data_plane_when_the_deployment_knows_it(
    hosted_client_knowing_its_data_plane: TestClient,
) -> None:
    """A caller who mis-pointed an SDK needs the address, not the category.

    "Send it to your Otari gateway" is what they already believed they were
    doing. Where ``data_plane_url`` is set, which is the same value
    ``GET /v1/bootstrap`` hands the dashboard, the refusal names the host.
    """
    response = hosted_client_knowing_its_data_plane.post("/v1/chat/completions", json={})

    assert response.status_code == 404
    assert response.json() == {
        "detail": (
            "This deployment is a control plane and does not serve inference. "
            "Send inference requests to https://gateway.example.com instead."
        )
    }
