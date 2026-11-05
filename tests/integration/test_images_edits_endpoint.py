"""Tests for the POST /api/v1/images/edits endpoint."""

from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from any_llm.exceptions import UnsupportedParameterError
from any_llm.types.image import Image, ImagesResponse
from fastapi import Request
from fastapi.testclient import TestClient
from openai.types.images_response import Usage, UsageInputTokensDetails
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.requests import ClientDisconnect

from gateway.core.config import API_ROOT
from gateway.core.database import release_session

from .conftest import held_reservations

_PATCH_TARGET = "gateway.api.routes.images.aimage_edit"
_EDITS_URL = f"{API_ROOT}/images/edits"
_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16


def _edit_response(n: int = 1, usage: Usage | None = None) -> ImagesResponse:
    return ImagesResponse(
        created=1700000000,
        data=[Image(b64_json=f"aW1hZ2Ux{i}") for i in range(n)],
        output_format="jpeg",
        usage=usage,
    )


def _form(**fields: Any) -> dict[str, Any]:
    return {"model": "openai:gpt-image-1", "prompt": "make it green", **fields}


def _price_gpt_image(client: TestClient, master_key_header: dict[str, str]) -> None:
    resp = client.post(
        f"{API_ROOT}/pricing",
        json={
            "model_key": "openai:gpt-image-1",
            "input_price_per_million": 40000,
            "output_price_per_million": 0.0,
            "unit": "image",
        },
        headers=master_key_header,
    )
    assert resp.status_code in (200, 201), resp.text


def test_edits_requires_auth(client: TestClient) -> None:
    resp = client.post(_EDITS_URL, data=_form(), files={"image": ("cat.png", _PNG, "image/png")})
    assert resp.status_code == 401


def test_edits_single_image(client: TestClient, api_key_header: dict[str, str]) -> None:
    with patch(_PATCH_TARGET, new_callable=AsyncMock, return_value=_edit_response()) as mock:
        resp = client.post(
            _EDITS_URL, data=_form(), files={"image": ("cat.png", _PNG, "image/png")}, headers=api_key_header
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["data"][0]["b64_json"] == "aW1hZ2Ux0"
    assert body["output_format"] == "jpeg"

    kwargs = mock.call_args.kwargs
    assert kwargs["model"] == "gpt-image-1"
    assert kwargs["provider"] == "openai"
    assert kwargs["prompt"] == "make it green"
    assert kwargs["image"] == ("cat.png", _PNG, "image/png")


def test_edits_frees_the_auth_connection_before_reading_the_body(
    client: TestClient, api_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slow client must not pin the connection authentication checked out."""
    events: list[str] = []
    form = Request.form

    async def recording_release(session: AsyncSession) -> bool:
        events.append("release")
        return await release_session(session)

    def recording_form(self: Request, **kwargs: Any) -> Any:
        events.append("form")
        return form(self, **kwargs)

    monkeypatch.setattr("gateway.api.deps.release_session", recording_release)
    monkeypatch.setattr(Request, "form", recording_form)
    with patch(_PATCH_TARGET, new_callable=AsyncMock, return_value=_edit_response()):
        resp = client.post(
            _EDITS_URL, data=_form(), files={"image": ("cat.png", _PNG, "image/png")}, headers=api_key_header
        )

    assert resp.status_code == 200, resp.text
    assert events == ["release", "form"]


@pytest.mark.parametrize("field", ["image", "image[]"])
def test_edits_several_images(client: TestClient, api_key_header: dict[str, str], field: str) -> None:
    files = [(field, ("a.png", _PNG, "image/png")), (field, ("b.jpg", _JPEG, "image/jpeg"))]
    with patch(_PATCH_TARGET, new_callable=AsyncMock, return_value=_edit_response()) as mock:
        resp = client.post(_EDITS_URL, data=_form(), files=files, headers=api_key_header)
    assert resp.status_code == 200, resp.text
    assert mock.call_args.kwargs["image"] == [("a.png", _PNG, "image/png"), ("b.jpg", _JPEG, "image/jpeg")]


def test_edits_names_a_generic_content_type_from_the_bytes(client: TestClient, api_key_header: dict[str, str]) -> None:
    """OpenAI refuses an application/octet-stream image part, and Gemini needs the type."""
    files = [("image", ("photo", _JPEG, "application/octet-stream")), ("image", ("blob", b"not-an-image", ""))]
    with patch(_PATCH_TARGET, new_callable=AsyncMock, return_value=_edit_response()) as mock:
        resp = client.post(_EDITS_URL, data=_form(), files=files, headers=api_key_header)
    assert resp.status_code == 200, resp.text
    first, second = mock.call_args.kwargs["image"]
    assert first == ("photo", _JPEG, "image/jpeg")
    assert second[2] is None


def test_edits_forwards_the_mask(client: TestClient, api_key_header: dict[str, str]) -> None:
    files = {"image": ("cat.png", _PNG, "image/png"), "mask": ("mask.png", _PNG, "image/png")}
    with patch(_PATCH_TARGET, new_callable=AsyncMock, return_value=_edit_response()) as mock:
        resp = client.post(_EDITS_URL, data=_form(), files=files, headers=api_key_header)
    assert resp.status_code == 200, resp.text
    assert mock.call_args.kwargs["mask"] == ("mask.png", _PNG, "image/png")


@pytest.mark.parametrize(
    ("field", "sent", "forwarded"),
    [
        ("n", "2", 2),
        ("size", "1536x1024", "1536x1024"),
        ("quality", "high", "high"),
        ("response_format", "b64_json", "b64_json"),
    ],
)
def test_edits_forwards_optional_fields(
    client: TestClient, api_key_header: dict[str, str], field: str, sent: str, forwarded: Any
) -> None:
    with patch(_PATCH_TARGET, new_callable=AsyncMock, return_value=_edit_response()) as mock:
        resp = client.post(
            _EDITS_URL,
            data=_form(**{field: sent}),
            files={"image": ("cat.png", _PNG, "image/png")},
            headers=api_key_header,
        )
    assert resp.status_code == 200, resp.text
    assert mock.call_args.kwargs[field] == forwarded


def test_edits_leaves_unset_fields_out(client: TestClient, api_key_header: dict[str, str]) -> None:
    with patch(_PATCH_TARGET, new_callable=AsyncMock, return_value=_edit_response()) as mock:
        resp = client.post(
            _EDITS_URL, data=_form(), files={"image": ("cat.png", _PNG, "image/png")}, headers=api_key_header
        )
    assert resp.status_code == 200, resp.text
    assert not {"n", "size", "quality", "response_format", "mask", "user"} & mock.call_args.kwargs.keys()


def test_edits_strips_user(client: TestClient, master_key_header: dict[str, str], test_user: dict[str, Any]) -> None:
    """``user`` picks who is billed; it is not sent to the provider."""
    with patch(_PATCH_TARGET, new_callable=AsyncMock, return_value=_edit_response()) as mock:
        resp = client.post(
            _EDITS_URL,
            data=_form(user=test_user["user_id"]),
            files={"image": ("cat.png", _PNG, "image/png")},
            headers=master_key_header,
        )
    assert resp.status_code == 200, resp.text
    assert "user" not in mock.call_args.kwargs

    rows = client.get(f"{API_ROOT}/usage", params={"endpoint": "/v1/images/edits"}, headers=master_key_header).json()
    assert rows[0]["user_id"] == test_user["user_id"]


def test_edits_master_key_requires_user(client: TestClient, master_key_header: dict[str, str]) -> None:
    with patch(_PATCH_TARGET, new_callable=AsyncMock, return_value=_edit_response()) as mock:
        resp = client.post(
            _EDITS_URL, data=_form(), files={"image": ("cat.png", _PNG, "image/png")}, headers=master_key_header
        )
    assert resp.status_code == 400
    assert "user" in resp.json()["detail"].lower()
    mock.assert_not_called()


def test_edits_missing_image(client: TestClient, api_key_header: dict[str, str]) -> None:
    with patch(_PATCH_TARGET, new_callable=AsyncMock) as mock:
        resp = client.post(
            _EDITS_URL, data=_form(), files={"other": ("x.png", _PNG, "image/png")}, headers=api_key_header
        )
    assert resp.status_code == 422
    assert resp.json()["code"] == "invalid_request"
    assert resp.json()["detail"][0]["loc"] == ["body", "image"]
    mock.assert_not_called()


def test_edits_missing_prompt(client: TestClient, api_key_header: dict[str, str]) -> None:
    with patch(_PATCH_TARGET, new_callable=AsyncMock) as mock:
        resp = client.post(
            _EDITS_URL,
            data={"model": "openai:gpt-image-1"},
            files={"image": ("cat.png", _PNG, "image/png")},
            headers=api_key_header,
        )
    assert resp.status_code == 422
    assert ["body", "prompt"] in [error["loc"] for error in resp.json()["detail"]]
    mock.assert_not_called()


def test_edits_refuses_an_image_sent_as_text(client: TestClient, api_key_header: dict[str, str]) -> None:
    """A text ``image`` must never reach any-llm, which reads a string image as a path on this server."""
    with patch(_PATCH_TARGET, new_callable=AsyncMock) as mock:
        resp = client.post(_EDITS_URL, data=_form(image="/etc/passwd"), headers=api_key_header)
    assert resp.status_code == 422
    mock.assert_not_called()


def test_edits_refuses_too_many_images(client: TestClient, api_key_header: dict[str, str]) -> None:
    files = [("image", (f"{i}.png", _PNG, "image/png")) for i in range(17)]
    with patch(_PATCH_TARGET, new_callable=AsyncMock) as mock:
        resp = client.post(_EDITS_URL, data=_form(), files=files, headers=api_key_header)
    assert resp.status_code == 400
    assert "16" in resp.json()["detail"]
    mock.assert_not_called()


@pytest.mark.parametrize("field", ["image", "mask"])
def test_edits_refuses_an_upload_over_the_cap(client: TestClient, api_key_header: dict[str, str], field: str) -> None:
    """The cap is on the whole body, so a large mask counts as much as a large image."""
    files = {"image": ("cat.png", _PNG, "image/png"), field: (f"{field}.png", b"\x00" * 4096, "image/png")}
    with (
        patch("gateway.api.routes.images._MAX_EDIT_BODY_BYTES", 2048),
        patch(_PATCH_TARGET, new_callable=AsyncMock) as mock,
    ):
        resp = client.post(_EDITS_URL, data=_form(), files=files, headers=api_key_header)
    assert resp.status_code == 413
    assert "larger than" in resp.json()["detail"]
    mock.assert_not_called()


def test_edits_refuses_files_over_the_cap_inside_the_body_cap(
    client: TestClient, api_key_header: dict[str, str]
) -> None:
    """The body cap leaves room for the form framing, so files just over their own cap are still refused."""
    files = [("image", ("a.png", b"\x00" * 600, "image/png")), ("mask", ("m.png", b"\x00" * 600, "image/png"))]
    with (
        patch("gateway.api.routes.images._MAX_EDIT_UPLOAD_BYTES", 1000),
        patch(_PATCH_TARGET, new_callable=AsyncMock) as mock,
    ):
        resp = client.post(_EDITS_URL, data=_form(), files=files, headers=api_key_header)
    assert resp.status_code == 413
    mock.assert_not_called()


def test_edits_parses_no_upload_without_credentials(client: TestClient) -> None:
    """The auth check answers before the body is read, so an anonymous upload is never stored."""
    with patch("gateway.api.routes.images.bounded_body") as bounded:
        resp = client.post(_EDITS_URL, data=_form(), files={"image": ("cat.png", b"\x00" * 4096, "image/png")})
    assert resp.status_code == 401
    bounded.assert_not_called()


def test_edits_interrupted_upload(client: TestClient, api_key_header: dict[str, str]) -> None:
    body = MagicMock(form=MagicMock(side_effect=ClientDisconnect()))
    with patch("gateway.api.routes.images.bounded_body", return_value=body):
        resp = client.post(
            _EDITS_URL, data=_form(), files={"image": ("cat.png", _PNG, "image/png")}, headers=api_key_header
        )
    assert resp.status_code == 400
    assert "interrupted" in resp.json()["detail"]


@pytest.mark.parametrize("field", ["prompt", "model"])
def test_edits_refuses_a_text_field_sent_as_a_file(
    client: TestClient, api_key_header: dict[str, str], field: str
) -> None:
    form = _form()
    del form[field]
    files = [("image", ("cat.png", _PNG, "image/png")), (field, ("x.txt", b"hi", "text/plain"))]
    with patch(_PATCH_TARGET, new_callable=AsyncMock) as mock:
        resp = client.post(_EDITS_URL, data=form, files=files, headers=api_key_header)
    assert resp.status_code == 422
    assert resp.json()["detail"][0]["loc"] == ["body", field]
    mock.assert_not_called()


def test_edits_refuses_a_mask_sent_as_text(client: TestClient, api_key_header: dict[str, str]) -> None:
    with patch(_PATCH_TARGET, new_callable=AsyncMock) as mock:
        resp = client.post(
            _EDITS_URL,
            data=_form(mask="/etc/passwd"),
            files={"image": ("cat.png", _PNG, "image/png")},
            headers=api_key_header,
        )
    assert resp.status_code == 422
    assert resp.json()["detail"][0]["loc"] == ["body", "mask"]
    mock.assert_not_called()


def test_edits_provider_error(
    client: TestClient, master_key_header: dict[str, str], api_key_header: dict[str, str]
) -> None:
    with patch(_PATCH_TARGET, new_callable=AsyncMock, side_effect=RuntimeError("provider down")):
        resp = client.post(
            _EDITS_URL, data=_form(), files={"image": ("cat.png", _PNG, "image/png")}, headers=api_key_header
        )
    assert resp.status_code == 502
    assert "provider" in resp.json()["detail"].lower()

    rows = client.get(
        f"{API_ROOT}/usage", params={"endpoint": "/v1/images/edits", "status": "error"}, headers=master_key_header
    ).json()
    assert "provider down" in rows[0]["error_message"]


def test_edits_unsupported_parameter_is_the_callers_400(client: TestClient, api_key_header: dict[str, str]) -> None:
    """Gemini refuses response_format=url and a mask this way."""
    with patch(
        _PATCH_TARGET, new_callable=AsyncMock, side_effect=UnsupportedParameterError("response_format", "gemini")
    ):
        resp = client.post(
            _EDITS_URL,
            data=_form(response_format="url"),
            files={"image": ("cat.png", _PNG, "image/png")},
            headers=api_key_header,
        )
    assert resp.status_code == 400
    assert "response_format" in resp.json()["detail"]


def test_edits_refuses_zero_images(client: TestClient, api_key_header: dict[str, str]) -> None:
    with patch(_PATCH_TARGET, new_callable=AsyncMock) as mock:
        resp = client.post(
            _EDITS_URL, data=_form(n="0"), files={"image": ("cat.png", _PNG, "image/png")}, headers=api_key_header
        )
    assert resp.status_code == 422
    assert ["body", "n"] in [error["loc"] for error in resp.json()["detail"]]
    mock.assert_not_called()


def _blocked_user(client: TestClient, headers: dict[str, str]) -> str:
    assert client.post(f"{API_ROOT}/users", json={"user_id": "edit-user", "blocked": True}, headers=headers).is_success
    return "edit-user"


def _over_budget_user(client: TestClient, headers: dict[str, str]) -> str:
    budget = client.post(f"{API_ROOT}/budgets", json={"max_budget": 0.0}, headers=headers).json()
    created = client.post(
        f"{API_ROOT}/users", json={"user_id": "edit-user", "budget_id": budget["budget_id"]}, headers=headers
    )
    assert created.is_success
    return "edit-user"


@pytest.mark.parametrize(
    ("make_user", "status_code"),
    [
        pytest.param(lambda _client, _headers: None, 400, id="master-key-without-user"),
        pytest.param(_blocked_user, 403, id="blocked-user"),
        pytest.param(_over_budget_user, 403, id="over-budget-user"),
    ],
)
def test_edits_reads_no_file_for_a_refused_request(
    client: TestClient, master_key_header: dict[str, str], make_user: Any, status_code: int
) -> None:
    """A request the gateway refuses before the provider call never loads its files."""
    user = make_user(client, master_key_header)
    form = _form() if user is None else _form(user=user)
    with (
        patch("gateway.api.routes.images._upload_part", new_callable=AsyncMock) as read,
        patch(_PATCH_TARGET, new_callable=AsyncMock) as mock,
    ):
        resp = client.post(
            _EDITS_URL, data=form, files={"image": ("cat.png", _PNG, "image/png")}, headers=master_key_header
        )
    assert resp.status_code == status_code, resp.text
    read.assert_not_called()
    mock.assert_not_called()


def test_edits_bills_per_image_and_logs_usage(
    client: TestClient, master_key_header: dict[str, str], api_key_header: dict[str, str]
) -> None:
    _price_gpt_image(client, master_key_header)
    usage = Usage(
        input_tokens=1132,
        input_tokens_details=UsageInputTokensDetails(image_tokens=1120, text_tokens=12),
        output_tokens=1751,
        total_tokens=2883,
    )
    with (
        held_reservations() as held,
        patch(_PATCH_TARGET, new_callable=AsyncMock, return_value=_edit_response(n=2, usage=usage)),
    ):
        resp = client.post(
            _EDITS_URL,
            data=_form(n="3"),
            files={"image": ("cat.png", _PNG, "image/png")},
            headers=api_key_header,
        )
    assert resp.status_code == 200, resp.text
    # Three images held before the call, two billed after it.
    assert held == [Decimal("0.12")]

    [row] = client.get(f"{API_ROOT}/usage", params={"endpoint": "/v1/images/edits"}, headers=master_key_header).json()
    assert row["status"] == "success"
    assert row["model"] == "gpt-image-1"
    assert row["prompt_tokens"] == 1132
    assert row["completion_tokens"] == 1751
    assert row["billing_meters"] == {"images": 2}
    assert row["pricing_breakdown"] == [
        {"meter": "images", "units": 2, "unit": "image", "unit_rate": 0.04, "cost": pytest.approx(0.08)}
    ]
    assert row["cost"] == pytest.approx(0.08)
    # The settled cost rides the provider's usage, beside its token counts.
    assert resp.json()["usage"] == {**usage.model_dump(), "cost_usd": "0.080000", "pricing_source": "deployment"}


def test_edits_without_pricing_carry_no_inline_cost(client: TestClient, api_key_header: dict[str, str]) -> None:
    usage = Usage(
        input_tokens=1132,
        input_tokens_details=UsageInputTokensDetails(image_tokens=1120, text_tokens=12),
        output_tokens=1751,
        total_tokens=2883,
    )
    with patch(_PATCH_TARGET, new_callable=AsyncMock, return_value=_edit_response(usage=usage)):
        resp = client.post(
            _EDITS_URL, data=_form(), files={"image": ("cat.png", _PNG, "image/png")}, headers=api_key_header
        )
    assert resp.status_code == 200, resp.text
    assert resp.json()["usage"] == usage.model_dump()


def test_edits_drop_a_providers_own_cost_fields(client: TestClient, api_key_header: dict[str, str]) -> None:
    """An upstream that reports its own cost is not passed off as the gateway's settled cost."""
    upstream = Usage.model_validate(
        {
            "input_tokens": 12,
            "input_tokens_details": {"image_tokens": 0, "text_tokens": 12},
            "output_tokens": 1120,
            "total_tokens": 1132,
            "cost_usd": "9.000000",
            "pricing_source": "upstream",
        }
    )
    with patch(_PATCH_TARGET, new_callable=AsyncMock, return_value=_edit_response(usage=upstream)):
        resp = client.post(
            _EDITS_URL, data=_form(), files={"image": ("cat.png", _PNG, "image/png")}, headers=api_key_header
        )
    assert resp.status_code == 200, resp.text
    usage = resp.json()["usage"]
    assert "cost_usd" not in usage
    assert "pricing_source" not in usage
    assert usage["total_tokens"] == 1132
