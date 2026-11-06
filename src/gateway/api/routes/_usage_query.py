"""Query parameters the usage reads share, so every usage endpoint describes a filter alike.

The deployment's, an organization's and a calling key's usage reads take the
same filters over the same rows, and a filter documented one way on one of them
and another way on the next would read as two different filters.
"""

from typing import Annotated

from fastapi import Query

from gateway.api.routes._request_tags import MAX_TAG_KEY_LENGTH
from gateway.core.sql import MAX_FILTER_VALUES
from gateway.schemas.usage_and_telemetry import GATEWAY_TOOL_NAMES

# How many request groups one call may ask for. The dashboard batches the groups
# visible on a page of the activity log into a single lookup, so the bound tracks
# the largest page size (1000) rather than a plan's candidate count; it exists to
# keep a caller from posting an unbounded IN list.
MAX_REQUEST_GROUPS = 1000

START_DESC = "Return logs with timestamp >= start_date (ISO 8601 or Unix epoch seconds)"
END_DESC = "Return logs with timestamp < end_date (ISO 8601 or Unix epoch seconds)"
STATUS_DESC = (
    "Filter to a single status: 'success', 'error', or 'absorbed' (an attempt a routing policy "
    "recovered from, excluded from error_count and request_count)"
)
STATUS_CODE_DESC = (
    "Filter to a single failure status code (e.g. 429 for provider rate limits, "
    "402 for missing-pricing rejections). Only error rows carry one, so this "
    "filter also restricts to status='error' unless 'status' is given explicitly"
)
ENDPOINT_DESC = "Filter to a single endpoint (e.g. '/v1/chat/completions')"
PROVIDER_DESC = "Filter to a single provider (e.g. 'openai')"
SOURCE_DESC = "Filter to a single provenance source (e.g. 'gateway' or 'claude_code')"
SOURCE_LABEL_DESC = (
    "Filter to a single session (source_label): the session's trace id on a gateway request that named one, "
    "or the session/project label imported usage carried"
)
TAG_DESC = (
    "Filter by a request tag (what a request sent in its `metadata`), as `key:value`. Repeatable: "
    "values for the same key match any of them, and different keys must all match. "
    f"At most {MAX_FILTER_VALUES} per call."
)
TagFilter = Annotated[list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=TAG_DESC)]
GROUP_BY_TAG_DESC = (
    "A tag key to break spend down by, returned as `by_tag`. Rows that do not carry the tag group under a null key."
)
GroupByTag = Annotated[str | None, Query(min_length=1, max_length=MAX_TAG_KEY_LENGTH, description=GROUP_BY_TAG_DESC)]
REQUEST_GROUP_DESC = (
    "Filter to the rows of one or more request groups; repeatable "
    "(request_group_id=a&request_group_id=b). A routed request writes one row per "
    "attempt, all sharing a request_group_id, so this returns a request's whole plan: "
    "its absorbed attempts and the attempt that served it. Ignore ordering by "
    "timestamp and read attempt_position to reconstruct the plan. At most "
    f"{MAX_REQUEST_GROUPS} ids per call."
)
PRICED_DESC = (
    "Filter by token-pricing state: true = only rows whose model tokens were priced, "
    "false = only rows that still need pricing (no cost at all, or tokens that were "
    "never metered because the model had no rate). A row charged only for gateway-run "
    "tool calls still counts as needing pricing."
)
TOOL_DESC = (
    "Filter to requests that ran a gateway-run tool. 'any' matches any tool; a tool "
    f"name ({', '.join(GATEWAY_TOOL_NAMES)}) matches that tool specifically."
)
COUNTS_DESC = (
    "Filter by budget participation, which is not the same question as provenance: "
    "true = only enforced gateway rows, false = every row that never touches a budget, "
    "meaning imported usage and also gateway traffic on a budget-exempt key"
)
WORKSPACE_DESC = "Only usage recorded in this workspace."
# The three entity filters are repeatable on every usage endpoint, so a chart or a
# log view can compare a handful of models / users / keys instead of one at a time.
# The bulk delete / set-price selection body takes the same form (see
# UsageSelection): "all N matching" is counted over these filters and re-derived
# from that body, so a filter one side could not express would target a different
# set of rows than the operator was shown.
USER_MULTI_DESC = (
    "Filter to one or more users; repeatable (user_id=a&user_id=b). Several values match any of "
    f"them. At most {MAX_FILTER_VALUES} per call."
)
MODEL_MULTI_DESC = (
    "Filter to one or more models; repeatable (model=a&model=b). Several values match any of them. "
    f"At most {MAX_FILTER_VALUES} per call."
)
API_KEY_MULTI_DESC = (
    "Filter to one or more API key ids; repeatable (api_key_id=a&api_key_id=b). Several values "
    f"match any of them. At most {MAX_FILTER_VALUES} per call."
)
DIMENSIONS_DESC = (
    "Which breakdowns to compute; repeatable (dimensions=model&dimensions=user). Each value names the "
    "'by_<value>' response field it fills, except 'status_code', which fills the failure taxonomy in "
    "'errors_by_status_code'. Omit for every breakdown (the default); pass 'none' for a "
    "totals-and-series-only response. Each dimension left out skips one GROUP BY scan, so a caller that "
    "reads only the tiles or the time series should say so. Fields that were not requested come back empty."
)
