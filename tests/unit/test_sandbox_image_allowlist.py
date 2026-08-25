"""Which sandbox images a workspace's code-execution policy may pin (#740).

The operator's answer, derived from the deployment config alone. A workspace
image is a supply-chain surface rather than a string, so the default is an empty
set: a deployment that curated nothing has vetted nothing.
"""

from gateway.core.config import GatewayConfig


def test_no_images_configured_means_a_workspace_may_pin_nothing() -> None:
    assert GatewayConfig().pinnable_sandbox_images() == ()


def test_the_deployment_image_is_pinnable_without_being_listed() -> None:
    """Naming the image every request already gets asks for nothing new."""
    config = GatewayConfig(sandbox_session_image="mzdotai/otari-sandbox-container:latest")
    assert config.pinnable_sandbox_images() == ("mzdotai/otari-sandbox-container:latest",)


def test_the_curated_list_is_split_trimmed_and_deduped() -> None:
    config = GatewayConfig(
        sandbox_session_image="mzdotai/otari-sandbox-container:latest",
        sandbox_allowed_session_images=(
            " ghcr.io/acme/sandbox:2 ,mzdotai/otari-sandbox-container:latest,, ghcr.io/acme/gpu:1"
        ),
    )
    assert config.pinnable_sandbox_images() == (
        "mzdotai/otari-sandbox-container:latest",
        "ghcr.io/acme/sandbox:2",
        "ghcr.io/acme/gpu:1",
    )


def test_a_curated_list_without_a_deployment_image_stands_on_its_own() -> None:
    config = GatewayConfig(sandbox_allowed_session_images="ghcr.io/acme/sandbox:2")
    assert config.pinnable_sandbox_images() == ("ghcr.io/acme/sandbox:2",)


def test_a_blank_list_is_the_same_as_none() -> None:
    assert GatewayConfig(sandbox_allowed_session_images="  ,  ").pinnable_sandbox_images() == ()
