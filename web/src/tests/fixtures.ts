// Builders for the API shapes a test needs whole.
//
// These live here because the shapes now come from the generated client, so a
// field the gateway adds arrives in every test at once. Written out inline, that
// meant six files failing to compile over a field none of them cared about. A
// builder fills the whole shape with neutral values and takes an override for
// the part under test, so a test states what it is about and nothing else.

import type {
  ActivationAttempt,
  Budget,
  DeploymentBootstrap,
  Organization,
  OrganizationContext,
  OrganizationGuardrail,
  OrganizationMember,
  PricingResponse,
  ScopedBudget,
  UsageSeriesPoint,
  UsageTotals,
  User,
  Workspace,
  WorkspaceActivation,
  WorkspaceBudgetDefault,
  WorkspaceCodeExecutionPolicy,
  WorkspaceMember,
  WorkspaceWebSearchConfig,
} from "@/client"

export function usageTotals(overrides: Partial<UsageTotals> = {}): UsageTotals {
  return {
    cost: 0,
    prompt_tokens: 0,
    completion_tokens: 0,
    total_tokens: 0,
    cache_read_tokens: 0,
    cache_write_tokens: 0,
    cache_write_1h_tokens: 0,
    billed_input_tokens: 0,
    billed_output_tokens: 0,
    request_count: 0,
    error_count: 0,
    unpriced_requests: 0,
    avg_latency_ms: null,
    ...overrides,
  }
}

export function seriesPoint(
  overrides: Partial<UsageSeriesPoint> & Pick<UsageSeriesPoint, "bucket_start">,
): UsageSeriesPoint {
  return {
    cost: 0,
    tokens: 0,
    requests: 0,
    errors: 0,
    input_tokens: 0,
    output_tokens: 0,
    cache_read_tokens: 0,
    cache_write_tokens: 0,
    ...overrides,
  }
}

export function pricingResponse(
  overrides: Partial<PricingResponse> & Pick<PricingResponse, "model_key">,
): PricingResponse {
  return {
    effective_at: "2026-01-01T00:00:00Z",
    input_price_per_million: 0,
    output_price_per_million: 0,
    cache_read_price_per_million: null,
    cache_write_price_per_million: null,
    cache_write_1h_price_per_million: null,
    pricing_tiers: [],
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    ...overrides,
  }
}

// The surface names a standalone gateway hosts, kept in step with
// STANDALONE_SURFACES in src/gateway/api/routes/bootstrap.py. A test that wants
// a surface hidden overrides `surfaces` rather than editing this.
const STANDALONE_SURFACES = [
  "budgets",
  "keys",
  "models",
  "organizations",
  "pricing",
  "providers",
  "routing",
  "settings",
  "tools",
  "usage",
  "users",
  "workspaces",
]

/** The deployment bootstrap, standalone by default. See useDeployment. */
export function bootstrap(
  overrides: Partial<DeploymentBootstrap> = {},
): DeploymentBootstrap {
  return {
    deployment_type: "standalone",
    session_type: "local_operator",
    surfaces: [...STANDALONE_SURFACES],
    // The master key, because an unclaimed deployment is what a fixture
    // describes by default; a test about the password login overrides it.
    sign_in_methods: ["master_key"],
    management_url: null,
    // Null, because a fixture describes a deployment whose documentation is the
    // bundled guide; the tests about an operator-configured docs site set it.
    docs_url: null,
    // Not frozen, because a fixture describes a deployment somebody can sign
    // in to; the maintenance-mode tests override it.
    maintenance_mode: false,
    // Off by default, matching a deployment that has not set public_base_url:
    // the passkey tests turn it on rather than every other test turning it off.
    passkeys_ready: false,
    mail_ready: false,
    ...overrides,
  }
}

// ---------- tenancy ----------

const ORGANIZATION_ID = "11111111-1111-1111-1111-111111111111"

export function organization(
  overrides: Partial<Organization> = {},
): Organization {
  return {
    id: ORGANIZATION_ID,
    name: "Default Organization",
    slug: "default-organization",
    created_by_user_id: null,
    created_at: "2026-01-01T00:00:00+00:00",
    updated_at: null,
    ...overrides,
  }
}

/** The caller's organization plus their standing in it, as an owner by default. */
export function organizationContext(
  overrides: Partial<OrganizationContext> = {},
): OrganizationContext {
  return {
    organization_member_id: "22222222-2222-2222-2222-222222222222",
    role: "owner",
    status: "active",
    organization: organization(),
    // Empty by default: a context is about the organization, and a test that is
    // about the workspace switcher names the memberships it needs. The server
    // omits the field entirely for a caller in no workspace, which is why the
    // shell treats absent and empty the same way.
    workspace_memberships: [],
    byo_provider_keys_allowed: true,
    ...overrides,
  }
}

export function organizationMember(
  overrides: Partial<OrganizationMember> = {},
): OrganizationMember {
  return {
    organization_member_id: "22222222-2222-2222-2222-222222222222",
    user_id: "33333333-3333-3333-3333-333333333333",
    // The request-plane owner this member bills through, which the server mints
    // as the identity's UUID rendered as a string. Set by default because a
    // member who can hold a key is the ordinary case; a test that is about the
    // other one passes null.
    attribution_user_id: "33333333-3333-3333-3333-333333333333",
    invitation_id: null,
    // A standalone operator identity has no sign-in address; a ported platform
    // row does. Both shapes have to render, so the builder ships neither and a
    // test names the one it is about.
    email: null,
    full_name: "Operator",
    role: "owner",
    status: "active",
    created_at: "2026-01-01T00:00:00+00:00",
    updated_at: null,
    ...overrides,
  }
}

export function workspace(overrides: Partial<Workspace> = {}): Workspace {
  return {
    id: "44444444-4444-4444-4444-444444444444",
    name: "Default Workspace",
    description: null,
    organization_id: ORGANIZATION_ID,
    created_by_user_id: null,
    created_at: "2026-01-01T00:00:00+00:00",
    updated_at: null,
    ...overrides,
  }
}

export function workspaceMember(
  overrides: Partial<WorkspaceMember> = {},
): WorkspaceMember {
  return {
    id: "55555555-5555-5555-5555-555555555555",
    workspace_id: "44444444-4444-4444-4444-444444444444",
    user_id: "33333333-3333-3333-3333-333333333333",
    role: "owner",
    status: "active",
    created_at: "2026-01-01T00:00:00+00:00",
    updated_at: null,
    ...overrides,
  }
}

/**
 * A ceiling on one tenancy scope, with its own counters.
 *
 * Not a variant of `budget`: a budget is a limit template with no counters,
 * replicated per person, while this pools over whatever its scope names. The
 * workspace_member ones are what a workspace's default materializes.
 */
export function scopedBudget(
  overrides: Partial<ScopedBudget> = {},
): ScopedBudget {
  return {
    id: "88888888-8888-8888-8888-888888888888",
    scope_type: "workspace_member",
    scope_id: "99999999-9999-9999-9999-999999999999",
    provider_key_id: null,
    name: null,
    // The budget this ceiling enforces. `max_budget` and the cadence below are
    // read off it and travel on the wire; they are not stored on the ceiling.
    budget_id: "77777777-7777-7777-7777-777777777777",
    max_budget: 50,
    current_spend: 0,
    reserved_spend: 0,
    budget_duration_sec: null,
    reset_alignment: null,
    period_start: null,
    period_end: null,
    created_at: "2026-01-01T00:00:00+00:00",
    updated_at: "2026-01-01T00:00:00+00:00",
    ...overrides,
  }
}

/** A spending limit and its reset period, as the budgets list reports one. */
export function budget(overrides: Partial<Budget> = {}): Budget {
  return {
    budget_id: "77777777-7777-7777-7777-777777777777",
    name: "Team standard",
    max_budget: 100,
    budget_duration_sec: 2_592_000,
    // The calendar cadence, which lives here rather than on the rows enforcing
    // this budget: a limit and the period it is spent over are one decision.
    reset_alignment: null,
    user_count: 0,
    total_spend: 0,
    total_reserved: 0,
    created_at: "2026-01-01T00:00:00+00:00",
    updated_at: "2026-01-01T00:00:00+00:00",
    ...overrides,
  }
}

/**
 * A gateway spend row: what a key charges against.
 *
 * Shared rather than per-file because the organization roster now joins these
 * onto its members (`attribution_user_id`) to show model access and spend, so
 * more than one suite needs the shape.
 */
export function user(overrides: Partial<User> = {}): User {
  return {
    user_id: "33333333-3333-3333-3333-333333333333",
    alias: null,
    spend: 0,
    reserved: 0,
    budget_id: null,
    allowed_models: null,
    budget_started_at: null,
    next_budget_reset_at: null,
    blocked: false,
    created_at: "2026-01-01T00:00:00+00:00",
    updated_at: "2026-01-01T00:00:00+00:00",
    metadata: {},
    ...overrides,
  }
}

export function workspaceBudgetDefault(
  overrides: Partial<WorkspaceBudgetDefault> = {},
): WorkspaceBudgetDefault {
  return {
    id: "66666666-6666-6666-6666-666666666666",
    workspace_id: "44444444-4444-4444-4444-444444444444",
    // The budget the default hands out. Name, limit and period are read off it
    // rather than stored on the default, and travel on the wire so a caller can
    // render one without fetching every budget to resolve an id.
    budget_id: "77777777-7777-7777-7777-777777777777",
    provider_key_id: null,
    name: "Default member budget",
    max_budget: 50.0,
    budget_duration_sec: 2_592_000,
    // The other arm of the period pair, exclusive with the one above: a default
    // whose budget snaps to a calendar boundary carries this instead.
    reset_alignment: null,
    created_at: "2026-01-01T00:00:00+00:00",
    updated_at: "2026-01-01T00:00:00+00:00",
    ...overrides,
  }
}

/** A request the setup guide reports, successful by default. */
export function activationAttempt(
  overrides: Partial<ActivationAttempt> = {},
): ActivationAttempt {
  return {
    occurred_at: "2026-01-01T00:00:00+00:00",
    request_id: "77777777-7777-7777-7777-777777777777",
    status: "success",
    provider: "openai",
    model: "openai:gpt-4o-mini",
    error_category: null,
    cost_usd: 0.000123,
    latency_ms: 412,
    ...overrides,
  }
}

/**
 * Where a workspace stands on its first request: waiting and on offer, which is
 * the state the guide exists for. A test about the payoff or a failure overrides
 * `status` and the attempt beside it.
 */
export function workspaceActivation(
  overrides: Partial<WorkspaceActivation> = {},
): WorkspaceActivation {
  return {
    status: "waiting",
    activation_attempt: null,
    latest_attempt: null,
    experience_eligible: true,
    dismissed: false,
    ...overrides,
  }
}

export function organizationGuardrail(
  overrides: Partial<OrganizationGuardrail> = {},
): OrganizationGuardrail {
  return {
    id: "55555555-5555-5555-5555-555555555555",
    organization_id: "11111111-1111-1111-1111-111111111111",
    profile: "prompt-injection",
    // The ordinary entry: no endpoint of its own, so it is sent to the
    // deployment's guardrails URL, and no credential to authenticate with.
    url: null,
    has_credential: false,
    mode: "monitor",
    on_unavailable: "block",
    validate_kwargs: null,
    enabled: true,
    applies_to_all_workspaces: false,
    workspace_ids: [],
    created_at: "2026-08-24T00:00:00+00:00",
    updated_at: "2026-08-24T00:00:00+00:00",
    ...overrides,
  }
}

export function workspaceCodeExecutionPolicy(
  overrides: Partial<WorkspaceCodeExecutionPolicy> = {},
): WorkspaceCodeExecutionPolicy {
  return {
    workspace_id: "44444444-4444-4444-4444-444444444444",
    // The zero-rows state, which is what a workspace has until somebody sets a
    // policy: nothing configured, nothing narrowed.
    configured: false,
    sandbox_configured: true,
    enabled: true,
    default_purpose_hint: null,
    max_iterations: null,
    exec_timeout_s: null,
    created_at: null,
    updated_at: null,
    ...overrides,
  }
}

export function workspaceWebSearchConfig(
  overrides: Partial<WorkspaceWebSearchConfig> = {},
): WorkspaceWebSearchConfig {
  return {
    workspace_id: "44444444-4444-4444-4444-444444444444",
    // The zero-rows state, which is what a workspace has until somebody sets
    // one: nothing configured, nothing narrowed.
    configured: false,
    web_search_configured: true,
    enabled: true,
    max_results: null,
    purpose_hint: null,
    allowed_domains: null,
    blocked_domains: null,
    provider_options: null,
    created_at: null,
    updated_at: null,
    ...overrides,
  }
}
