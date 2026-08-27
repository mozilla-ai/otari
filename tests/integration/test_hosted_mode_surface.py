"""A hosted control plane serves the management plane and refuses inference.

The mirror of ``test_hybrid_mode_surface``, which asserts the opposite posture
for the opposite deployment. Regression cover for otari#822: hosted mode used to
mount every data-plane router alongside the management API, so a workspace key
pointed at the control plane host ran real inference that no wallet was ever
debited for.
"""

import tempfile
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gateway.api.deps import reset_config
from gateway.core.config import GatewayConfig
from gateway.core.database import reset_db
from gateway.main import create_app

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


def _hosted_app(tmpdir: str) -> FastAPI:
    config = GatewayConfig(
        mode="hosted",
        database_url=f"sqlite:///{Path(tmpdir) / 'hosted.db'}",
        bootstrap_api_key=False,
    )
    return create_app(config)


def test_hosted_mode_refuses_every_inference_endpoint() -> None:
    """The leak itself: no data-plane path may be served by a control plane.

    Asserted unauthenticated on purpose. A refusal that only a valid key reaches
    would still leave the endpoint mounted, and the bug was that it ran the
    request at all, so the refusal has to come before auth rather than after it.
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            app = _hosted_app(tmpdir)
            with TestClient(app) as client:
                for path in INFERENCE_PATHS:
                    response = client.post(path, json={})

                    assert response.status_code == 404, f"{path} was served: {response.status_code}"
                    assert response.json() == {"detail": EXPECTED_DETAIL}, path
    finally:
        reset_config()
        reset_db()


def test_hosted_mode_refuses_inference_on_every_verb() -> None:
    """A GET must not slip past a stub registered only for POST."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            app = _hosted_app(tmpdir)
            with TestClient(app) as client:
                for method in ("get", "put", "patch", "delete"):
                    response = getattr(client, method)("/v1/chat/completions")

                    assert response.status_code == 404, method
                    assert response.json() == {"detail": EXPECTED_DETAIL}, method
    finally:
        reset_config()
        reset_db()


def test_hosted_mode_still_serves_the_management_plane() -> None:
    """The refusal is scoped to the data plane and takes no management route with it.

    A 404 here would mean the stub's catch-all swallowed a management prefix.
    Anything else, including the 401 these return unauthenticated, means the
    real router answered.
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            app = _hosted_app(tmpdir)
            with TestClient(app) as client:
                for path in (
                    "/v1/keys",
                    "/v1/usage",
                    "/v1/organizations/me/provider-keys",
                    # Discovery, not dispatch, and a surface bootstrap publishes
                    # for a hosted deployment, so it stays mounted.
                    "/v1/models",
                    # The catalog POST /v1/search dispatches against. Management,
                    # and the one prefix a careless /v1/search stub could shadow.
                    "/v1/search-tools",
                ):
                    response = client.get(path)

                    assert response.status_code != 404, f"{path} was swallowed by the stubs"

                # Unauthenticated and mounted in every mode: it is how a browser
                # learns which deployment it reached.
                bootstrap = client.get("/v1/bootstrap")
                assert bootstrap.status_code == 200
                assert bootstrap.json()["deployment_type"] == "hosted"
    finally:
        reset_config()
        reset_db()


@pytest.mark.parametrize("mode", ["standalone", None])
def test_non_hosted_modes_keep_serving_inference(mode: str | None) -> None:
    """Standalone legitimately serves both planes from one process, and still does.

    ``None`` covers the derived case: mode unset and no platform token also
    resolves to standalone, and that is what an ordinary self-hosted deployment
    runs, so it must not pick up the control plane's posture.
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = GatewayConfig(
                mode=mode,
                database_url=f"sqlite:///{Path(tmpdir) / 'standalone.db'}",
                bootstrap_api_key=False,
            )
            assert not config.is_hosted_mode
            app = create_app(config)
            served = {route.path for route in app.routes if hasattr(route, "path")}

            # Each path as the data plane spells it, with no prefix fallback: a
            # looser match would pass on a route that had moved out from under
            # the caller.
            for path in INFERENCE_PATHS:
                assert path in served, path
            # And the refusal stubs are absent entirely, rather than mounted
            # behind the real routes where a path change could one day expose
            # them.
            assert "/v1/chat/{path:path}" not in served
    finally:
        reset_config()
        reset_db()
