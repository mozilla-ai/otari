"""The dashboard forgets a kept answer once a dashboard session would have expired.

`web/src/shared/api/keptAnswers.ts` cannot read the configured session lifetime
before it has a session, so it takes the default. The two numbers are written in
two languages, which no type checker spans, so this test is what keeps them equal.
"""

import re
from pathlib import Path

from gateway.core.config import GatewayConfig

KEPT_ANSWERS = Path(__file__).resolve().parents[2] / "web" / "src" / "shared" / "api" / "keptAnswers.ts"


def test_a_kept_answer_lives_as_long_as_a_default_dashboard_session() -> None:
    match = re.search(r"export const KEPT_ANSWER_HOURS = (\d+)\n", KEPT_ANSWERS.read_text())
    assert match is not None, "KEPT_ANSWER_HOURS is no longer a literal in keptAnswers.ts"
    assert int(match.group(1)) == GatewayConfig.model_fields["dashboard_session_ttl_hours"].default
