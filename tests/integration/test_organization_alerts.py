"""Organization alert rules: CRUD, authorization, tenant isolation, and the SSRF gate.

Exercised at the service layer, matching `test_organization_guardrails.py`: the
API can only ever act as the one bootstrap operator identity a standalone
deployment has, who is always an owner, so the rules that matter most (a plain
member refused, another organization's rule invisible) are only reachable by
calling the service with identities built at whatever role a case needs.

Webhook destinations here use IP literals in public ranges, or are rejected
before any lookup happens. ``validate_alert_destination_url`` resolves a
hostname through DNS, so a test naming one would pass or fail on whether the
runner has egress.

The pure request-body validation cases live in
``tests/unit/test_alert_rule_schemas.py`` instead: they need no database, and
keeping them here made them unrunnable on a machine without Docker.
"""

from collections.abc import Iterator

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.entities import AlertDelivery, AlertRule
from gateway.models.tenancy import Organization, User
from gateway.repositories.tenancy import OrganizationMemberRepository, OrganizationRepository, UserRepository
from gateway.services.secret_box import decrypt_secret, generate_secret_key
from gateway.services.tenancy.errors import (
    AlertRuleAlreadyExistsError,
    AlertRuleInertError,
    AlertRuleLimitReachedError,
    AlertRuleNotFoundError,
    AlertRuleUnsafeDestinationError,
    AlertRuleUnsupportedDestinationError,
    NotAuthorizedError,
)
from gateway.services.tenancy.organization_alert_service import (
    MAX_ALERT_RULES_PER_ORGANIZATION,
    AlertRuleCreate,
    AlertRuleUpdate,
    OrganizationAlertService,
)

pytestmark = pytest.mark.asyncio

# A vendor schema: its endpoint is compiled into the Apprise plugin, so storing
# one never reaches the address check and never touches DNS.
SLACK_DESTINATION = "slack://xoxb-AAA/xoxb-BBB/xoxb-CCC/#alerts"
# A public IP literal, so the address check resolves nothing.
PUBLIC_WEBHOOK = "jsons://93.184.216.34/incoming/hook"


async def _organization(db: AsyncSession, *, slug: str = "acme") -> Organization:
    return await OrganizationRepository(db).create_organization(name=slug.title(), slug=slug, created_by_user_id=None)


async def _member(db: AsyncSession, organization: Organization, *, role: str, full_name: str) -> User:
    user = await UserRepository(db).create_local_identity(
        full_name=full_name,
        active_organization_id=organization.id,
    )
    await OrganizationMemberRepository(db).create_membership(
        organization_id=organization.id, user_id=user.id, role=role
    )
    return user


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    yield


def _create(**overrides: object) -> AlertRuleCreate:
    fields: dict[str, object] = {"name": "Platform Slack", "destination": SLACK_DESTINATION}
    fields.update(overrides)
    return AlertRuleCreate(**fields)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# CRUD
# --------------------------------------------------------------------------


async def test_crud_round_trip(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = OrganizationAlertService(async_db)

    created = await service.create_rule(user=owner, request=_create())
    assert created.name == "Platform Slack"
    assert created.warn_at_percent == 80
    assert created.notify_on_exceeded is True
    assert created.enabled is True

    listed = await service.list_rules(user=owner)
    assert listed.count == 1
    assert [rule.id for rule in listed.data] == [created.id]

    updated = await service.update_rule(
        user=owner, rule_id=created.id, request=AlertRuleUpdate(warn_at_percent=95, enabled=False)
    )
    assert updated.warn_at_percent == 95
    assert updated.enabled is False

    await service.delete_rule(user=owner, rule_id=created.id)
    assert (await service.list_rules(user=owner)).count == 0


async def test_the_destination_is_encrypted_and_never_returned(async_db: AsyncSession) -> None:
    """The stored row holds ciphertext; the API shape holds only the redaction.

    The single most important assertion in this file: an Apprise URL is a live
    bot token, and a rule list is readable by every owner and admin of the
    organization.
    """
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = OrganizationAlertService(async_db)

    created = await service.create_rule(user=owner, request=_create())

    row = (await async_db.execute(select(AlertRule).where(AlertRule.id == created.id))).scalar_one()
    assert SLACK_DESTINATION not in row.encrypted_destination
    assert decrypt_secret(row.encrypted_destination) == SLACK_DESTINATION

    # Neither the returned shape nor the stored redaction carries any token.
    for value in (created.destination, row.redacted_destination):
        assert "xoxb-AAA" not in value
        assert "xoxb-BBB" not in value
        assert "xoxb-CCC" not in value
        assert value.startswith("slack://")


async def test_one_rule_name_per_organization(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = OrganizationAlertService(async_db)

    await service.create_rule(user=owner, request=_create(name="Duplicate"))
    with pytest.raises(AlertRuleAlreadyExistsError):
        await service.create_rule(user=owner, request=_create(name="Duplicate"))


async def test_the_same_name_is_free_in_another_organization(async_db: AsyncSession) -> None:
    """The uniqueness is per organization, not deployment-wide."""
    first = await _organization(async_db, slug="acme")
    second = await _organization(async_db, slug="globex")
    owner_a = await _member(async_db, first, role="owner", full_name="A")
    owner_b = await _member(async_db, second, role="owner", full_name="B")
    service = OrganizationAlertService(async_db)

    await service.create_rule(user=owner_a, request=_create(name="Shared"))
    assert await service.create_rule(user=owner_b, request=_create(name="Shared"))


async def test_the_rule_limit_is_enforced(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = OrganizationAlertService(async_db)

    for index in range(MAX_ALERT_RULES_PER_ORGANIZATION):
        await service.create_rule(user=owner, request=_create(name=f"Rule {index}"))
    with pytest.raises(AlertRuleLimitReachedError):
        await service.create_rule(user=owner, request=_create(name="One too many"))


async def test_deleting_a_rule_discards_its_deliveries(async_db: AsyncSession) -> None:
    """The cascade is what re-arms an alert when a rule is recreated."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = OrganizationAlertService(async_db)

    created = await service.create_rule(user=owner, request=_create())
    async_db.add(AlertDelivery(alert_rule_id=created.id, scoped_budget_id="ceiling-1", kind="warning"))
    await async_db.commit()

    await service.delete_rule(user=owner, rule_id=created.id)
    remaining = (
        await async_db.execute(select(AlertDelivery).where(AlertDelivery.alert_rule_id == created.id))
    ).scalars().all()
    assert list(remaining) == []


# --------------------------------------------------------------------------
# Authorization and tenant isolation
# --------------------------------------------------------------------------


async def test_a_plain_member_may_not_read_or_write(async_db: AsyncSession) -> None:
    """One gate for reads and writes: a rule names an endpoint this gateway posts to."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    member = await _member(async_db, organization, role="member", full_name="Member")
    service = OrganizationAlertService(async_db)

    created = await service.create_rule(user=owner, request=_create())

    with pytest.raises(NotAuthorizedError):
        await service.list_rules(user=member)
    with pytest.raises(NotAuthorizedError):
        await service.create_rule(user=member, request=_create(name="Sneaky"))
    with pytest.raises(NotAuthorizedError):
        await service.update_rule(user=member, rule_id=created.id, request=AlertRuleUpdate(enabled=False))
    with pytest.raises(NotAuthorizedError):
        await service.delete_rule(user=member, rule_id=created.id)
    with pytest.raises(NotAuthorizedError):
        await service.test_rule(user=member, rule_id=created.id)


async def test_an_admin_may_manage_rules(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    admin = await _member(async_db, organization, role="admin", full_name="Admin")
    service = OrganizationAlertService(async_db)
    assert await service.create_rule(user=admin, request=_create())


async def test_another_organizations_rule_is_not_found(async_db: AsyncSession) -> None:
    """404 rather than 403, so the id is not an existence oracle across tenants."""
    first = await _organization(async_db, slug="acme")
    second = await _organization(async_db, slug="globex")
    owner_a = await _member(async_db, first, role="owner", full_name="A")
    owner_b = await _member(async_db, second, role="owner", full_name="B")
    service = OrganizationAlertService(async_db)

    theirs = await service.create_rule(user=owner_a, request=_create())

    with pytest.raises(AlertRuleNotFoundError):
        await service.update_rule(user=owner_b, rule_id=theirs.id, request=AlertRuleUpdate(enabled=False))
    with pytest.raises(AlertRuleNotFoundError):
        await service.delete_rule(user=owner_b, rule_id=theirs.id)
    with pytest.raises(AlertRuleNotFoundError):
        await service.test_rule(user=owner_b, rule_id=theirs.id)
    # And it does not appear in their list at all.
    assert (await service.list_rules(user=owner_b)).count == 0


# --------------------------------------------------------------------------
# Destination validation and the SSRF gate
# --------------------------------------------------------------------------


async def test_an_unparseable_destination_is_refused(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = OrganizationAlertService(async_db)

    with pytest.raises(AlertRuleUnsupportedDestinationError):
        await service.create_rule(user=owner, request=_create(destination="definitelynotascheme://host"))


@pytest.mark.parametrize(
    "destination",
    [
        "json://127.0.0.1/hook",
        "json://10.0.0.5/hook",
        "jsons://192.168.1.10/hook",
        "json://169.254.169.254/latest/meta-data",
    ],
)
async def test_a_webhook_pointed_inside_the_deployment_is_refused(
    async_db: AsyncSession, destination: str
) -> None:
    """Fail-closed by default. The cloud metadata endpoint is the case that matters."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = OrganizationAlertService(async_db)

    with pytest.raises(AlertRuleUnsafeDestinationError):
        await service.create_rule(user=owner, request=_create(destination=destination))


async def test_a_public_webhook_is_accepted(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = OrganizationAlertService(async_db)
    assert await service.create_rule(user=owner, request=_create(destination=PUBLIC_WEBHOOK))


async def test_a_vendor_schema_skips_the_address_check(async_db: AsyncSession) -> None:
    """A slack:// URL's first token parses into the netloc and is not a host.

    Address-checking it would reject a perfectly good rule (and, worse, could
    resolve a token as a hostname), which is why only the webhook-shaped
    schemas are checked.
    """
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = OrganizationAlertService(async_db)
    assert await service.create_rule(user=owner, request=_create(destination="slack://10/0/0/1"))


async def test_the_private_host_override_opens_the_gate(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The self-hosted case: an internal chat server on the deployment's network."""
    monkeypatch.setenv("OTARI_ALERT_ALLOW_PRIVATE_HOSTS", "true")
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = OrganizationAlertService(async_db)
    assert await service.create_rule(user=owner, request=_create(destination="json://10.0.0.5/hook"))


async def test_the_web_search_override_does_not_open_the_alert_gate(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each SSRF gate has its own override; turning one on must not widen another."""
    monkeypatch.setenv("OTARI_WEB_SEARCH_ALLOW_PRIVATE_HOSTS", "true")
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = OrganizationAlertService(async_db)

    with pytest.raises(AlertRuleUnsafeDestinationError):
        await service.create_rule(user=owner, request=_create(destination="json://10.0.0.5/hook"))


# --------------------------------------------------------------------------
# Update semantics
# --------------------------------------------------------------------------


async def test_an_omitted_destination_keeps_the_stored_one(async_db: AsyncSession) -> None:
    """The column is NOT NULL, so omission is the only "leave it alone" state."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = OrganizationAlertService(async_db)

    created = await service.create_rule(user=owner, request=_create())
    await service.update_rule(user=owner, rule_id=created.id, request=AlertRuleUpdate(name="Renamed"))

    row = (await async_db.execute(select(AlertRule).where(AlertRule.id == created.id))).scalar_one()
    assert decrypt_secret(row.encrypted_destination) == SLACK_DESTINATION
    assert row.name == "Renamed"


async def test_a_null_warn_threshold_clears_the_warning(async_db: AsyncSession) -> None:
    """The one field where an explicit null is a value rather than an omission."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = OrganizationAlertService(async_db)

    created = await service.create_rule(user=owner, request=_create())
    updated = await service.update_rule(
        user=owner,
        rule_id=created.id,
        request=AlertRuleUpdate.model_validate({"warn_at_percent": None}),
    )
    assert updated.warn_at_percent is None
    assert updated.notify_on_exceeded is True


async def test_a_rule_created_with_no_warning_keeps_it_null(async_db: AsyncSession) -> None:
    """The service must store an explicit null rather than the schema's 80.

    The end-to-end form of the bug ``warn_at_percent``'s model comment
    describes: a column default fired on the explicit None and handed the
    operator the early warnings they had just declined.
    ``tests/unit/test_alert_rule_schemas.py`` guards the column itself.
    """
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = OrganizationAlertService(async_db)

    created = await service.create_rule(user=owner, request=_create(warn_at_percent=None))
    assert created.warn_at_percent is None

    row = (await async_db.execute(select(AlertRule).where(AlertRule.id == created.id))).scalar_one()
    assert row.warn_at_percent is None


async def test_a_patch_cannot_reach_the_inert_state_either(async_db: AsyncSession) -> None:
    """The create body's rule, re-checked on the merged row.

    A PATCH can reach the dead state by sending only one of the two halves,
    which is why the check lives in the service as well as in the schema.
    """
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = OrganizationAlertService(async_db)

    created = await service.create_rule(user=owner, request=_create(warn_at_percent=None))
    with pytest.raises(AlertRuleInertError):
        await service.update_rule(
            user=owner, rule_id=created.id, request=AlertRuleUpdate(notify_on_exceeded=False)
        )
