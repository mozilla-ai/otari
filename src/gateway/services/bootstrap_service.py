import uuid

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.auth import generate_api_key, hash_key, key_prefix
from gateway.core.config import GatewayConfig
from gateway.log_config import log_secret
from gateway.models.entities import APIKey
from gateway.repositories.users_repository import get_or_create_default_user


async def bootstrap_first_api_key(config: GatewayConfig, db: AsyncSession) -> None:
    """Create a first API key for new installations."""

    if not config.bootstrap_api_key:
        return

    existing_key = (await db.execute(select(APIKey.id).limit(1))).scalar_one_or_none()
    if existing_key:
        return

    api_key = generate_api_key()
    key_id = str(uuid.uuid4())

    # The bootstrap key has no explicit owner, so it lands on the shared "default"
    # user like any other no-user key, rather than a per-key virtual user.
    user = await get_or_create_default_user(db)

    db_key = APIKey(
        id=key_id,
        key_hash=hash_key(api_key),
        key_prefix=key_prefix(api_key),
        key_name="bootstrap",
        user_id=user.user_id,
        metadata_={"bootstrap": True},
    )
    db.add(db_key)

    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise

    log_secret(
        "No API keys found. Created bootstrap key for first run. Save this key now:",
        api_key,
    )
