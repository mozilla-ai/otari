// The shapes the OpenAPI spec does not describe, and why each one is here.
//
// Everything else the dashboard sends or receives comes from `./schema.ts`,
// generated from the gateway's spec. This file is the exception list, kept short
// and justified on purpose: a type that could come from the spec and does not is
// a type that will drift from the API without anyone noticing.

// ---------------------------------------------------------------------------
// Routing policy bodies.
//
// `PolicyRequest.spec` and `PolicyResponse.spec` are declared as free-form
// objects on the wire, so the generator can only emit `Record<string, unknown>`
// for them, and the Routing page reads and writes their internals.
//
// The gateway does have a real `PolicySpec` model and the route already
// validates every write into it, so the wire contract is narrower than it says.
// Two things stop this PR from just publishing that model:
//
//   * On the request, the route hand-rolls a 400 that names the offending field
//     so the policy form can bind each error to its input. Annotating the body
//     as `PolicySpec` would hand that to FastAPI's generic 422 and lose it.
//   * On the response, `PolicySpec` carries defaults, so serializing through it
//     would add `spec_version`, `guardrails: []` and six null keys per select
//     entry to every policy. Suppressing that means a serializer on `PolicySpec`
//     itself, which is also what the two write paths dump before storing, so it
//     would change what is written to the database, not just what is read back.
//
// Both are fixable and worth fixing; neither belongs in the same change as the
// client generation. Until then these stay hand-written and can drift, which is
// the cost being accepted here.
// ---------------------------------------------------------------------------

export interface PolicyThreshold {
  gte?: number
  gt?: number
  lte?: number
  lt?: number
}

export interface PolicyWhen {
  budget_used_pct?: PolicyThreshold
  budget_remaining_usd?: PolicyThreshold
  user_id?: string | string[]
  key_id?: string | string[]
}

export interface PolicySelectEntry {
  when?: PolicyWhen
  target?: string
  /** The fallthrough. Exactly one entry carries it, and it must come last. */
  default?: string
  /** A router backend that orders `candidates` per request: "weighted" to split
   *  traffic by share, "knn" to learn which prompts a cheaper model handles. */
  router?: string
  /** The pool a `router` entry orders. Required there, meaningless elsewhere. */
  candidates?: string[]
  /** Share of traffic per candidate, for a `router: "weighted"` entry only.
   *  Relative, not percentages: {a: 70, b: 30} and {a: 7, b: 3} are one split. A
   *  candidate left out takes no traffic and stays in the plan as a failover. */
  weights?: Record<string, number>
}

export interface PolicyGuardrail {
  profile: string
  /** Required: the per-request field defaults to "monitor", so an omitted mode
   *  here would look like a mandate and behave as shadow mode. */
  mode: "block" | "monitor"
  on_unavailable?: "block" | "monitor"
  url?: string | null
}

export interface PolicySpec {
  spec_version?: number
  select: PolicySelectEntry[]
  on_failure?: string[]
  guardrails?: PolicyGuardrail[]
}

// ---------------------------------------------------------------------------
// Not served by the gateway's API at all.
// ---------------------------------------------------------------------------

/**
 * `/dashboard-build.json`, served by the gateway outside the OpenAPI surface
 * (`include_in_schema=False`), so this declaration is its only client contract.
 * Both fields are real: see the route in `src/gateway/main.py`.
 */
export interface DashboardBuild {
  build: string
  version: string
}

// ---------------------------------------------------------------------------
// Client-side only: never sent or received as a body.
// ---------------------------------------------------------------------------

/**
 * The filter set a usage view holds, which the hooks turn into a query string.
 *
 * Deliberately not a wire shape: the endpoints take these as individual query
 * parameters, several of them repeatable, so there is no request body in the
 * spec for this to be generated from.
 */
export interface UsageFilters {
  // The workspace a row was recorded in. Set from the sidebar's switcher rather
  // than by the operator, so the request log and its counts show the workspace
  // the shell is looking at. Not on `/v1/usage/summary`, which the gateway does
  // not scope yet, so the charts above the log stay deployment-wide.
  workspace_id?: string
  start_date?: string
  // Upper bound (exclusive). Omitted for a live "up to now" window; set by the
  // analytics previous-period query so its window does not overlap the current one.
  end_date?: string
  status?: string
  // The three entity filters accept several values on every usage endpoint: they go
  // on the wire as repeated query params and match any of them, so one chart can
  // compare a handful of models / users / keys and the request log can be scoped to
  // the same set a drill-down arrived with.
  model?: string | string[]
  endpoint?: string
  provider?: string
  user_id?: string | string[]
  api_key_id?: string | string[]
  source?: string
  // Session/project attribution (a row's `source_label`), so the log can be
  // scoped to the one agent session a breakdown row points at.
  source_label?: string
  // Pricing state: true = only rows whose model tokens were priced, false = only
  // rows that still need pricing. A row charged only for gateway-run tool calls
  // counts as needing pricing, so a tool charge cannot hide it from that view.
  priced?: boolean
  // Gateway-run tool usage. "any" matches any tool (including MCP tools, whose
  // names come from the caller's server); a name matches that tool specifically.
  // "any" matches any tool; anything else is a specific tool name. Not a literal
  // union: MCP tool names come from the caller's own server, so the set is open
  // and pinning it to the two built-ins forced a cast at the one call site that
  // drills into a named tool.
  tool?: string
  // Budget participation: false scopes to imported rows (the bulk-op target set).
  counts_toward_budget?: boolean
}

/** Which tool/guardrail service a settings test targets. A path parameter in the
 *  spec rather than a named schema, so it is restated here. */
export type ToolServiceName = "web_search" | "sandbox" | "guardrails"
