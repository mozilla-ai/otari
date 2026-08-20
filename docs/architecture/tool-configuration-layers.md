# Per-workspace tool configuration

Status: a design decision for work that is not built yet. It records the answer to
[mozilla-ai/otari#655](https://github.com/mozilla-ai/otari/issues/655) so that the four
implementation issues it gates ([#654](https://github.com/mozilla-ai/otari/issues/654) guardrails,
[#656](https://github.com/mozilla-ai/otari/issues/656) web search,
[#657](https://github.com/mozilla-ai/otari/issues/657) code execution,
[#658](https://github.com/mozilla-ai/otari/issues/658) MCP servers) build one mechanism instead of
four. The user-facing pages ([Built-in tools](../tools.md), [Guardrails](../guardrails.md),
[MCP](../mcp.md), [Access control](../access-control.md)) describe what ships; each of those issues
updates its own page when it lands.

## The problem

Every tool surface here is configured deployment-wide today: gated on the master key, stored in
`runtime_settings` or a credential table, and merged over `config.yml` into a process-wide in-memory
overlay so that a read site with no database session can still answer
(`services/search_tool_store_service.py`, `services/provider_store_service.py`,
`services/tool_settings_service.py`). A process-wide overlay holds exactly one answer, and
per-workspace configuration needs the answer for *this* request's workspace.

Two questions had to be settled before any of the four could be built: how the per-workspace answer
reaches the request path, and what it means when it disagrees with the deployment default.

## Answer 1, resolution: the workspace travels as an identity, the policy is resolved at admission

**The overlay does not become workspace-keyed, and the dispatch path does not gain a session.** The
workspace reaches the request path as an *identity* resolved once at auth time, and the tool policy
for that workspace is read from the database at admission, in the same async preamble that already
holds a session, and materialized onto the request-scoped object the dispatch path already reads.

Concretely, for all four surfaces:

1. **Carrier.** `RequestContext.workspace_id`, resolved once in the preamble by
   `services/workspace_scope.resolve_workspace_id` off the authenticating API key, never off a
   header. That field is introduced by [#670](https://github.com/mozilla-ai/otari/pull/670) and is
   reused here rather than resolved a second time; until it lands, it is the same one-line
   resolution in the same place. A master-key request lands in the deployment's default workspace,
   which is what the master key already means everywhere else.
   `RequestContext.organization_id` is already resolved in that spot, for the same reason, and is
   what per-organization pricing reads.
2. **Read site.** `prepare_gateway_tools` in `api/routes/_pipeline.py`, where `ctx.db` is a live
   session in standalone mode. One read per request per surface the request actually uses, next to
   the pricing lookup and the budget reservation that already happen there.
3. **Handoff.** The resolved values land on `ToolContext`, which is built at admission and is
   already what the tool loop and the streaming path read. No new plumbing: the fields for a
   sandbox purpose hint, an iteration ceiling, a web-search entry, and a list of MCP servers are all
   there today.

### Why not workspace-key the overlay

The overlay exists for one reason: a consumer that cannot await. `resolve_search_tool(config, name)`
is synchronous, and so is the provider-kwargs builder. Nothing about tool *policy* has that
constraint, because policy is consumed at admission. Keying the overlay by workspace would buy
nothing and cost the two things a process-wide overlay always costs: a staleness window on a
security-relevant toggle (a workspace disabling a tool would keep working for up to the refresh TTL,
on every replica), and memory proportional to workspaces times tools on every worker.

### Why not give the dispatch path a session

Because the seam that needs the answer is not the dispatch path. It is admission, which already has
one. Pushing the read later is also structurally worse on the streaming path: the tool loop advances
while the response body is produced, past the point where the handler's session dependency has
exited, which is why the usage log writer opens its own session rather than borrowing the request's.
A mid-loop policy read would mean a new session per tool call on the hot path, to answer a question
that was already answerable before the provider was called.

### Credentials stay where they are

The split that makes this work: **the deployment layer owns backends, endpoints, and secrets; the
workspace layer owns policy and holds no secret.** `search_tool_credentials`, `provider_credentials`,
and the `*_url` settings keep their process-wide overlays, unchanged and not workspace-keyed. This is
already the shape #656 assumes ("It stores no secret, because the credential is held by the
gateway-side adapter"), and it is what keeps a workspace admin from acquiring egress or a credential
that the operator did not configure.

### Hybrid mode is unaffected

Per-workspace tool configuration is standalone-only, exactly like the credential stores. In hybrid
mode the platform owns the per-workspace policy and the gateway already resolves it over HTTP at
this same seam (`_resolve_platform_code_execution`, `_resolve_platform_web_search`,
`_resolve_platform_mcp_servers` in `prepare_gateway_tools`). `ctx.db` is `None` there and no local
row is consulted. The local implementation mirrors the hybrid shape rather than inventing one, which
is the strongest argument for this answer: the hybrid path has resolved per-workspace tool policy at
admission, from the request's identity, into `ToolContext`, since before the question was asked.

## Answer 2, composition: a lower layer may narrow, never widen

**The invariant: no layer may widen the set of requests that succeed beyond what the layer above it
explicitly permits.** The deployment offer is a ceiling, not a default with exceptions. A workspace
row is therefore a veto and a refinement, and never a grant.

That single rule decides the three-way question ("grant, veto, or refinement?") for every surface at
once, and it is what makes a whole class of bug impossible by construction: a workspace admin cannot
reach a backend, a URL, or a credential the operator did not configure, whatever the row says.

Three corollaries do the rest of the work.

**Absence means "no narrowing," not "denied."** No row at a layer means that layer imposes no
constraint. This is how per-key model allow-lists already read (`None` is unrestricted, `[]` is deny
all, and callers must branch on `is None`), and how per-organization pricing already reads (no
`organization_id` resolves exactly as it did before the override table existed). Absence must be
represented as the absence of a row, never as a seeded row carrying default values: a seeded row is
a value someone can change and then cannot get back to "today."

This is also the answer to what a newly created workspace inherits, which #654 asks for by name: it
is created with no rows of its own and therefore inherits the layer above by construction.
Inheritance is never materialized by copying the parent's rows at creation time. A copy freezes the
parent's value at the moment of creation and then diverges silently, so a later change to the
organization's configuration would reach the workspaces created after it and not the ones created
before.

**Every field is either a default or a bound, and the implementation must say which.** A *default*
supplies a value when the request said nothing meaningful, and the request wins when it did. A
*bound* is enforced regardless of what the request asked, and the strictest layer wins: `min()` for a
numeric ceiling, intersection for an allow-list, union for a deny-list. Getting this backwards is
the quiet failure mode, because a cost or safety control expressed as a default is not a control at
all: any caller can ask past it. An empty or absent value from a layer is "no preference" and does
not participate in the intersection, so a request that supplies an empty allow-list falls back to
the workspace's rather than clearing it.

**Delegation is explicit.** A layer may hand a toggle down ("this guardrail is optional", "this
workspace may enable search for itself"). What it cannot do is let a lower layer take something it
was not given. "Explicitly permits" in the invariant is doing that work.

### Per surface

Where the four differ, they differ deliberately. The differences are all one thing: a guardrail is a
*restriction*, and the other three are *capabilities*, so "narrow" points the other way for
guardrails.

**Guardrails ([#654](https://github.com/mozilla-ai/otari/issues/654)).** The deployment owns
`guardrails_url` and the SSRF gate, unchanged. The organization owns the guardrail entries: profile,
credential, mode, whether the entry is `required` or `optional`, and which of its workspaces the
entry applies to. A workspace row may raise strictness on an entry scoped to it (`monitor` to
`block`, a tighter threshold), and may turn off an entry the organization marked `optional`. It may
not create an entry, may not reach an entry the organization did not scope to it, and may not turn
off or loosen a `required` one. So, for the three questions #654 asked in its own words: a workspace
cannot enable a guardrail its organization has not configured, cannot disable one the organization
requires, and can narrow the parameters of one it has been scoped into. `optional` exists so an
organization can delegate a toggle on purpose; an organization that wants no delegation marks the
entry `required`.

**Web search ([#656](https://github.com/mozilla-ai/otari/issues/656)).** Deployment ceiling: with no
`web_search_url` configured the request is refused as it is today, whatever any row says. The
enabled toggle is delegated to the workspace, which is the self-serve behavior #656 asks for; off
means the request is refused rather than silently served. Bounds: `max_results`, `allowed_domains`
(intersection), `blocked_domains` (union). Defaults: `purpose_hint`, `engines`, `provider_options`
(shallow-merged, request keys win). No credential in the workspace layer. Note that the hybrid path
today applies `max_results` and the domain lists as defaults; classifying them as bounds is a
deliberate change, and aligning the two is #656's work rather than a divergence to leave standing.

**Code execution ([#657](https://github.com/mozilla-ai/otari/issues/657)).** Deployment ceiling: no
`sandbox_url`, no code execution. Bounds: the permitted tool set (intersection), the execution
timeout and the loop-iteration ceiling (`min()`, which is how `ToolContext.max_tool_iterations`
already folds the workspace value in). Defaults: `purpose_hint`. Disabled means 403 at admission,
before the provider is called, which is what the hybrid path already does. The sandbox wire contract
and its conformance check are untouched: this is about who may ask for what.

**MCP servers ([#658](https://github.com/mozilla-ai/otari/issues/658)).** Composition does not
apply. There is no deployment-level server list to compose over, so a workspace row is the only
source and creating one is not a widening of anything. Three rules still bind. The deployment's SSRF
gate (`mcp_allow_loopback`, `mcp_allow_private_hosts`, `services/url_safety.validate_mcp_url`) is a
ceiling that applies to a stored workspace server exactly as it applies to a request-supplied one,
and no row can lift it; this is the one place a workspace row looks like a grant, and the reason it
is not is that the egress it implies is still bounded by the deployment. A workspace's stored servers
are additive to what a request names, merged into `ToolContext.mcp_server_configs` the way the
hybrid `mcp_server_ids` path already merges platform-resolved servers. And tokens are encrypted at
rest and absent from responses, matching every other stored secret here.

### Who may write

Follows from the above rather than being a fourth decision, but worth stating so it is not invented
four times. The deployment layer keeps its master-key gate. The organization and workspace layers go
through the existing tenancy authorization (`services/tenancy/organization_service.py`,
`services/tenancy/workspace_service.py`), not the master key: organization owner or admin for an
organization-scoped write, and organization owner or admin *or* that workspace's own owner or admin
for a workspace-scoped one. That is the gate
[#670](https://github.com/mozilla-ai/otari/pull/670) uses for the same shape of row.

## A deployment that configures nothing per-workspace behaves as it does today

This is a **requirement on each of the four implementations**, not a property to be assumed from the
composition rule. What it means concretely:

- With no rows in any new per-workspace table, every surface resolves to the value it resolves to
  today. Each implementation carries a test that asserts this with zero rows present, rather than
  only testing the configured paths.
- The single-workspace deployment that every migration seeds is the degenerate case of the general
  rule, not a special-cased branch.
- The admission read must not be able to fail a request that would otherwise be served. A
  *capability* whose policy cannot be resolved (the workspace cannot be determined, the query errors)
  falls back to the deployment configuration and logs, the way
  `workspace_scope.organization_for_workspace_id` returns `None` rather than raising. A *restriction*
  fails closed, because that is already the posture for a `block`-mode guardrail that cannot be
  evaluated.
- No new query on a request that uses none of these tools. The read happens on the branch that
  handles the tool, not in the preamble unconditionally.

## How this reads against the M4 decision

`otari-ai`'s [`docs/architecture/m4-reconciliation.md`](https://github.com/mozilla-ai/otari-ai/blob/main/docs/architecture/m4-reconciliation.md)
settles that guardrails, web search, MCP and code execution converge on **one workspace
configuration plane** ([otari-ai#1597](https://github.com/mozilla-ai/otari-ai/issues/1597)), that
the gateway keeps execution while the control plane keeps configuration, and that the
`runtime_settings` tool keys migrate into that plane. This page is the request-path half of that
row: the plane is where configuration lives, and what follows is how a request reads it and what it
means when two tiers of it disagree.

Two things that row does not settle, decided here.

**The plane has two tiers, and the `runtime_settings` tool keys split across them.** "Tool keys
migrate in" means they stop living in a generic deployment key/value table, not that a workspace may
set any of them. `guardrails_url`, `web_search_url` and `sandbox_url` are endpoints, and the operator
owns egress: `services/tool_settings_service.py` validates them structurally rather than against an
SSRF deny-list precisely because the operator is already fully trusted, and the SSRF gates themselves
(`mcp_allow_*`, `web_search_allow_private_hosts`) stay display-only and are not widened from the
dashboard. A workspace admin does not inherit that trust, so those three migrate in as the plane's
deployment tier. The web-search knobs (`web_search_engines`, `web_search_max_results`,
`web_search_extract`, `web_search_purpose_hint`) are policy, and they do become per-workspace under
the default-or-bound classification above.

**A capability the deployment cannot execute is not offered per workspace.** otari-ai#1597's
contract, that a capability shown as available must be resolvable and executable by the current
deployment, is this page's ceiling expressed on the dashboard instead of the request path. A
workspace toggle for a tool with no configured backend reads as unavailable with a reason, not as an
enabled switch that produces a runtime failure. The request-path refusal is unchanged (the existing
400 for an unconfigured backend), and the two surfaces must agree, which is the whole point of
#1597.

## Relationship to the provider-key decision

[mozilla-ai/otari-ai#1748](https://github.com/mozilla-ai/otari-ai/issues/1748), implemented in
[#670](https://github.com/mozilla-ai/otari/pull/670), settled the same in-memory-overlay problem for
provider credentials and reached a `(workspace_id, provider)`-keyed in-memory overlay. The two
answers are compatible, and deliberately share the carrier: #670 adds
`RequestContext.workspace_id`, resolved once in the preamble off the authenticating key and `None`
in hybrid mode, and this decision reads that same field rather than resolving the workspace a second
time.

They differ on the read mechanism because they have different consumers, and one rule covers both:
**resolve tenant-scoped configuration from the database at the latest point that can await, and
cache it in memory only for a consumer that cannot.** A provider credential is consumed by the
synchronous kwargs builder inside dispatch, which cannot await, so it has to be in memory before the
request arrives. Tool policy is consumed at admission, which can. So this is one mechanism applied
to two consumers, not two mechanisms for one constraint.

The related note in #1748 that `search_tool_credentials` was explicitly deferred to a separate
decision still holds and is not reopened here: the *credential* stays deployment-scoped and
instance-addressed. What #656 adds above it is policy, which holds no secret.
