"""Give an organization key with a custom endpoint a selector name of its own.

A key with an ``api_base`` (an OpenAI-compatible host such as Synthetic or a
vLLM server) is reached as ``<slug>:<model>`` rather than ``<provider>:<model>``,
so it no longer shares a name with the provider it speaks the protocol of.

Each such key gets a slug made from its name, stepping around a provider's name,
an instance name the deployment has stored and a slug already taken. The
organization's rates for the models it offers, under either spelling, are
copied to every such key offering the model, and removed from the provider
unless a key of the bare provider still serves the model (one offering nothing
serves every model). A keyed key is no provider's default, so the flag is
cleared on it and the default index leaves it out.

A spending ceiling narrowed to the provider does not follow: it is a limit on
the provider's traffic, and a copy would start a second count.

Revision ID: b3e7d1a9c5f2
Revises: 9e4b7a2c6d1f
Create Date: 2026-11-05
"""

import re
import uuid
from collections import defaultdict
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from any_llm import LLMProvider

revision: str = "b3e7d1a9c5f2"
down_revision: str | Sequence[str] | None = "9e4b7a2c6d1f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Frozen here: a slug a later release would refuse is still one this one handed out.
_TAKEN = frozenset(
    {provider.value for provider in LLMProvider}
    | {"openai-compatible", "openai_compatible", "anthropic-compatible", "anthropic_compatible", "otari", "hosted"}
)


def _slug(name: str, provider: str, taken: frozenset[str], used: set[str]) -> str:
    base = re.sub(r"[^a-z0-9._-]+", "-", name.strip().lower()).strip("-._")[:56] or provider
    if base in taken:
        base = f"{base}-custom"
    slug, n = base, 2
    while slug in used:
        slug, n = f"{base}-{n}", n + 1
    used.add(slug)
    return slug


def _stored_instance_names(bind: sa.Connection) -> set[str]:
    """Instance names the deployment holds in its own tables, which a selector resolves before a key."""
    inspector = sa.inspect(bind)
    names: set[str] = set()
    for table, column in (
        ("provider_credentials", "instance"),
        ("hosted_providers", "name"),
        ("configured_provider_imports", "name"),
    ):
        if inspector.has_table(table):
            names.update(bind.execute(sa.text(f"SELECT {column} FROM {table}")).scalars())  # noqa: S608
    return names


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("org_provider_keys", sa.Column("slug", sa.String(length=64), nullable=True))

    bind = op.get_bind()
    metadata = sa.MetaData()
    keys = sa.Table("org_provider_keys", metadata, autoload_with=bind)
    offered = sa.Table("org_provider_key_models", metadata, autoload_with=bind)
    rates = sa.Table("organization_model_pricing", metadata, autoload_with=bind)
    taken = _TAKEN | _stored_instance_names(bind)

    rows = bind.execute(
        sa.select(keys.c.id, keys.c.organization_id, keys.c.provider, keys.c.name, keys.c.api_base).order_by(
            keys.c.created_at, keys.c.id
        )
    ).all()
    used: dict[uuid.UUID, set[str]] = {}
    keyed: list[tuple[uuid.UUID, uuid.UUID, str, str]] = []
    for key_id, organization_id, provider, name, api_base in rows:
        if not api_base:
            continue
        slug = _slug(name, provider, taken, used.setdefault(organization_id, set()))
        bind.execute(keys.update().where(keys.c.id == key_id).values(slug=slug, is_org_default=False))
        keyed.append((key_id, organization_id, provider, slug))

    # Which slugs need each (organization, provider, model) rate.
    targets: dict[tuple[uuid.UUID, str, str], list[str]] = defaultdict(list)
    for key_id, organization_id, provider, slug in keyed:
        for model in bind.execute(sa.select(offered.c.model).where(offered.c.org_provider_key_id == key_id)).scalars():
            targets[(organization_id, provider, model)].append(slug)

    # A live bare key still serves a model it offers, and every model when it offers none.
    bare = sa.select(keys.c.id, keys.c.organization_id, keys.c.provider).where(
        keys.c.slug.is_(None), keys.c.archived_at.is_(None)
    )
    bare_offers: set[tuple[uuid.UUID, str, str]] = set()
    bare_unnarrowed: set[tuple[uuid.UUID, str]] = set()
    for key_id, organization_id, provider in bind.execute(bare).all():
        models = list(bind.execute(sa.select(offered.c.model).where(offered.c.org_provider_key_id == key_id)).scalars())
        if not models:
            bare_unnarrowed.add((organization_id, provider))
        bare_offers.update((organization_id, provider, model) for model in models)

    for (organization_id, provider, model), slugs in targets.items():
        matching = (rates.c.organization_id == organization_id) & rates.c.model_key.in_(
            [f"{provider}:{model}", f"{provider}/{model}"]
        )
        for row in bind.execute(sa.select(rates).where(matching)).mappings().all():
            for slug in slugs:
                # SQLite reflects the id as its stored hex, PostgreSQL as a UUID.
                new_id = uuid.uuid4() if isinstance(row["id"], uuid.UUID) else uuid.uuid4().hex
                bind.execute(rates.insert().values({**row, "id": new_id, "model_key": f"{slug}:{model}"}))
        if (organization_id, provider, model) not in bare_offers and (organization_id, provider) not in bare_unnarrowed:
            bind.execute(rates.delete().where(matching))

    op.drop_index("uq_org_provider_keys_org_default", table_name="org_provider_keys")
    op.create_index(
        "uq_org_provider_keys_org_default",
        "org_provider_keys",
        ["organization_id", "provider"],
        unique=True,
        postgresql_where=sa.text("is_org_default AND archived_at IS NULL AND slug IS NULL"),
        sqlite_where=sa.text("is_org_default AND archived_at IS NULL AND slug IS NULL"),
    )
    op.create_index(
        "uq_org_provider_keys_org_slug",
        "org_provider_keys",
        ["organization_id", "slug"],
        unique=True,
        postgresql_where=sa.text("slug IS NOT NULL"),
        sqlite_where=sa.text("slug IS NOT NULL"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    bind = op.get_bind()
    metadata = sa.MetaData()
    keys = sa.Table("org_provider_keys", metadata, autoload_with=bind)
    rates = sa.Table("organization_model_pricing", metadata, autoload_with=bind)

    for organization_id, provider, slug in bind.execute(
        sa.select(keys.c.organization_id, keys.c.provider, keys.c.slug).where(keys.c.slug.is_not(None))
    ).all():
        prefix = f"{slug}:"
        for row_id, model_key in bind.execute(
            sa.select(rates.c.id, rates.c.model_key).where(
                rates.c.organization_id == organization_id, rates.c.model_key.startswith(prefix, autoescape=True)
            )
        ).all():
            old_key = f"{provider}:{model_key[len(prefix) :]}"
            twin = bind.execute(
                sa.select(rates.c.id).where(rates.c.organization_id == organization_id, rates.c.model_key == old_key)
            ).first()
            if twin is not None:
                bind.execute(rates.delete().where(rates.c.id == row_id))
            else:
                bind.execute(rates.update().where(rates.c.id == row_id).values(model_key=old_key))

    op.drop_index("uq_org_provider_keys_org_slug", table_name="org_provider_keys")
    op.drop_index("uq_org_provider_keys_org_default", table_name="org_provider_keys")
    op.create_index(
        "uq_org_provider_keys_org_default",
        "org_provider_keys",
        ["organization_id", "provider"],
        unique=True,
        postgresql_where=sa.text("is_org_default AND archived_at IS NULL"),
        sqlite_where=sa.text("is_org_default AND archived_at IS NULL"),
    )
    op.drop_column("org_provider_keys", "slug")
