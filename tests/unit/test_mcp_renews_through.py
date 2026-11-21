"""Whether an organization's server can renew a copy's sign-in it would carry, without a database."""

import uuid

import pytest

from gateway.models.tools import McpUserConnection, WorkspaceMcpServer
from gateway.services.tools._mcp_connections import StoredMcpOAuthClient, renews_through

TOKEN_ENDPOINT = "https://auth.example.com/token"


def _server(*, secret: str | None) -> WorkspaceMcpServer:
    return WorkspaceMcpServer(
        organization_id=uuid.uuid4(),
        workspace_id=None,
        name="drive",
        url="https://mcp.example.com/mcp",
        oauth_client=StoredMcpOAuthClient(client_id="client", encrypted_secret=secret).model_dump(),
        oauth_client_token_endpoint=TOKEN_ENDPOINT,
    )


def _sign_in(auth_method: str) -> McpUserConnection:
    return McpUserConnection(
        server_id=uuid.uuid4(),
        user_id="ada",
        encrypted_access_token="access",
        encrypted_refresh_token="refresh",
        client_id="client",
        client_auth_method=auth_method,
        token_endpoint=TOKEN_ENDPOINT,
    )


@pytest.mark.parametrize(
    ("auth_method", "secret", "renews"),
    [
        # A public client sends no secret, so the new server needs none.
        ("none", None, True),
        ("none", "encrypted-secret", True),
        # A confidential client's renewal needs the secret the sign-in was made with.
        ("client_secret_basic", None, False),
        ("client_secret_post", None, False),
        ("client_secret_basic", "encrypted-secret", True),
    ],
)
def test_a_sign_in_is_renewed_only_where_the_server_can_authenticate_it_as_it_was(
    auth_method: str, secret: str | None, renews: bool
) -> None:
    assert renews_through(_server(secret=secret), _sign_in(auth_method)) is renews
