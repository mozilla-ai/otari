"""Give an organization key with a custom endpoint a selector name of its own.

A key with an ``api_base`` (an OpenAI-compatible host such as Synthetic or a
vLLM server) is reached as ``<slug>:<model>`` rather than ``<provider>:<model>``,
so it no longer shares a name with the provider it speaks the protocol of.

Each such key gets a slug made from its name. The organization's rates for the
models it offers move to the new key; a rate a key of the bare provider also
needs is copied instead. A keyed key is no provider's default, so the flag is
cleared on it and the default index leaves it out.

Revision ID: b3e7d1a9c5f2
Revises: 9e4b7a2c6d1f
Create Date: 2026-11-05
"""

import re
import uuid
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


def _slug(name: str, provider: str, used: set[str]) -> str:
    base = re.sub(r"[^a-z0-9._-]+", "-", name.strip().lower()).strip("-._")[:56] or provider
    if base in _TAKEN:
        base = f"{base}-custom"
    slug, n = base, 2
    while slug in used:
        slug, n = f"{base}-{n}", n + 1
    used.add(slug)
    return slug


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("org_provider_keys", sa.Column("slug", sa.String(length=64), nullable=True))

    bind = op.get_bind()
    metadata = sa.MetaData()
    keys = sa.Table("org_provider_keys", metadata, autoload_with=bind)
    offered = sa.Table("org_provider_key_models", metadata, autoload_with=bind)
    rates = sa.Table("organization_model_pricing", metadata, autoload_with=bind)

    rows = bind.execute(
        sa.select(
            keys.c.id, keys.c.organization_id, keys.c.provider, keys.c.name, keys.c.api_base, keys.c.archived_at
        ).order_by(keys.c.created_at, keys.c.id)
    ).all()
    used: dict[uuid.UUID, set[str]] = {}
    keyed: list[tuple[uuid.UUID, uuid.UUID, str, str]] = []
    for key_id, organization_id, provider, name, api_base, _archived_at in rows:
        if not api_base:
            continue
        slug = _slug(name, provider, used.setdefault(organization_id, set()))
        bind.execute(keys.update().where(keys.c.id == key_id).values(slug=slug, is_org_default=False))
        keyed.append((key_id, organization_id, provider, slug))

    # A bare key of the same provider still offering a model keeps its rate.
    bare_offers: set[tuple[uuid.UUID, str, str]] = {
        (organization_id, provider, model)
        for organization_id, provider, model in bind.execute(
            sa.select(keys.c.organization_id, keys.c.provider, offered.c.model)
            .join(offered, offered.c.org_provider_key_id == keys.c.id)
            .where(keys.c.slug.is_(None), keys.c.archived_at.is_(None))
        ).all()
    }
    for key_id, organization_id, provider, slug in keyed:
        models = bind.execute(sa.select(offered.c.model).where(offered.c.org_provider_key_id == key_id)).scalars()
        for model in models:
            old_key, new_key = f"{provider}:{model}", f"{slug}:{model}"
            matching = (rates.c.organization_id == organization_id) & (rates.c.model_key == old_key)
            if (organization_id, provider, model) not in bare_offers:
                bind.execute(rates.update().where(matching).values(model_key=new_key))
                continue
            for row in bind.execute(sa.select(rates).where(matching)).mappings().all():
                # SQLite reflects the id as its stored hex, PostgreSQL as a UUID.
                new_id = uuid.uuid4() if isinstance(row["id"], uuid.UUID) else uuid.uuid4().hex
                bind.execute(rates.insert().values({**row, "id": new_id, "model_key": new_key}))

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
