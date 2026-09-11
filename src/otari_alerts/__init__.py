"""Budget alerts for Otari, as a bootstrap plugin.

Tell an organization that one of its budget ceilings is running out, before it
starts refusing requests. A rule is a destination (an Apprise URL, so Slack,
Discord, PagerDuty, mail and a plain webhook are one column) plus the
thresholds that reach it; a background worker reads the ceilings on a timer and
sends each crossing once.

Install it and point Otari at it::

    pip install 'gateway[alerts]'
    export OTARI_BOOTSTRAP=otari_alerts:register

That is the whole of the wiring. :func:`register` contributes a router, a
background task and a migration chain to the composition container, and edits
no Otari source file, which is what ``ARCHITECTURE.md`` rule 6 asks of anything
layered on top.

**Why ``otari_alerts`` and not ``gateway_alerts``.** ``AGENTS.md`` reserves the
exact name ``otari`` on PyPI for the Otari client SDK, which is why the gateway
itself imports as ``gateway``; ``otari_alerts`` collides with nothing, so the
rule's other half applies, that anything user-facing is named ``otari``. The
selector a deployment writes in its environment,
``OTARI_BOOTSTRAP=otari_alerts:register``, is about as user-facing as a name
gets.

The modules, split by what each knows:

- :mod:`otari_alerts.dispatcher` delivers a message to an Apprise URL and knows
  nothing about budgets.
- :mod:`otari_alerts.destinations` decides which destinations are acceptable
  and how one is masked for display.
- :mod:`otari_alerts.evaluator` knows which ceilings have crossed a threshold,
  claims the right to alert on each, and asks the dispatcher to send. It is the
  contributed background task.
- :mod:`otari_alerts.service` is the CRUD surface a tenant configures rules
  through, and :mod:`otari_alerts.routes` the router over it.
"""

from pathlib import Path
from typing import Final

from gateway.container import (
    BackgroundTaskContribution,
    Container,
    MigrationContribution,
    RouterContribution,
)
from otari_alerts.evaluator import run_budget_alert_evaluator
from otari_alerts.routes import router

# Resolved from this file rather than from the repository root, so it is
# correct in an installed wheel as well as in a source checkout.
ALEMBIC_DIR: Final = Path(__file__).parent / "alembic"

# The table this plugin's chain stamps. Declared to Otari so it can refuse a
# collision, and read back by the chain's own ``env.py``, so the declared name
# and the stamped one cannot drift.
VERSION_TABLE: Final = "alerts_alembic_version"

__all__ = ["ALEMBIC_DIR", "VERSION_TABLE", "register"]


def register(container: Container) -> None:
    """Wire budget alerts into Otari's composition container.

    Called once at startup by ``build_container`` when ``OTARI_BOOTSTRAP``
    names it. Binds no port: alerting has one implementation and, per
    ``ARCHITECTURE.md`` rule 7, a port that will only ever have one is
    ceremony. All three contributions are additive.
    """
    # Imported for its side effect: reading the settings here turns a bad
    # OTARI_ALERT_* value into a failed boot rather than a worker that dies on
    # its first tick.
    import otari_alerts.config  # noqa: F401

    # ``capability=None``: the surface is present exactly when this module is
    # installed. A capability names a licensing axis, and inventing one here
    # would add a decision nobody makes and a second place the real answer
    # could be read from. The router declares its own credential, since the
    # mount point adds none.
    container.contribute_router(RouterContribution(capability=None, router=router))
    container.contribute_background_task(
        BackgroundTaskContribution(name="budget-alerts", start=run_budget_alert_evaluator)
    )
    container.contribute_migrations(
        MigrationContribution(
            name="alerts",
            script_location=str(ALEMBIC_DIR),
            version_table=VERSION_TABLE,
        )
    )
