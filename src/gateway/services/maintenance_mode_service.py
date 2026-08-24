"""The login freeze: whether this deployment is refusing new dashboard sign-ins.

An operator turns this on to redeploy without anyone signing in mid-migration,
and off again when the deployment is back. It refuses *new* sessions only:
a session already minted keeps working, so the operator running the redeploy is
never locked out of the dashboard by the switch they just flipped.

**Why this is not a key in ``runtime_settings_service``**, whose table it does
share. That service exists to override a ``GatewayConfig`` field: every key in
its ``_SPECS`` names one, an override is applied by mutating this worker's
in-memory config, and its own docstring records that other workers pick a change
up "on their next startup, not live". Both halves are wrong for a freeze. There
is no config field to mutate, and a freeze that holds on the replica that served
the toggle while the other replicas keep signing people in is not a freeze. So
the flag is read from the stored row on every sign-in attempt instead. That read
is a single-row primary-key lookup against a request that already pays for a
bcrypt verification, which is several orders of magnitude more expensive.

``load_overrides`` skips any row whose key is not in ``_SPECS``, so this row is
invisible to that service and cannot be applied to the config by accident. The
value encoding ("true"/"false") is deliberately the same one it uses for a bool,
so the table holds one vocabulary rather than two.

Nothing here touches the data plane. A frozen deployment still serves
``/v1/chat/completions`` and the rest of the management API to a caller
presenting the master key or an API key through the header; what stops is the
dashboard sign-in that mints a cookie. That boundary is what lets an operator
turn the freeze back off without signing in: ``PATCH /v1/settings/maintenance-mode``
is master-key gated and reachable through the header, which never passes through
the door the freeze closes. It does presuppose the operator still holds that key;
one who does not recovers by setting ``OTARI_MASTER_KEY`` and restarting.
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.entities import RuntimeSetting

# The ``runtime_settings`` row this flag lives in. Deliberately absent from
# ``runtime_settings_service._SPECS``: see the module docstring.
MAINTENANCE_MODE_KEY = "maintenance_mode"

_TRUE = "true"
_FALSE = "false"


async def is_maintenance_mode(session: AsyncSession) -> bool:
    """Return whether new dashboard sign-ins are frozen right now.

    Reads the stored row rather than any cached value, because the freeze has to
    hold on every replica the moment it is set, not from each replica's next
    startup. Absent row means not frozen, which is the fail-open default a fresh
    deployment starts from: an operator who has never touched this must not have
    to turn something off before anyone can sign in.
    """
    value = (
        await session.execute(select(RuntimeSetting.value).where(RuntimeSetting.key == MAINTENANCE_MODE_KEY))
    ).scalar_one_or_none()
    return value == _TRUE


async def stage_maintenance_mode(session: AsyncSession, *, enabled: bool) -> None:
    """Stage the freeze flag for persistence; the caller commits.

    Persistence is the whole of the change: unlike a ``runtime_settings_service``
    override there is nothing to apply to the running worker afterwards, because
    every reader goes back to the row.
    """
    row = await session.get(RuntimeSetting, MAINTENANCE_MODE_KEY)
    serialized = _TRUE if enabled else _FALSE
    if row is None:
        session.add(RuntimeSetting(key=MAINTENANCE_MODE_KEY, value=serialized))
    else:
        row.value = serialized
