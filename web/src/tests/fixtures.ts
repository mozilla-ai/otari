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
  CallerOrganizationMembership,
  DeploymentBootstrap,
  DeploymentUser,
  Organization,
  OrganizationContext,
  OrganizationGuardrail,
  OrganizationMember,
  OrgProviderKey,
  PricingResponse,
  ScopedBudget,
  UsageSeriesPoint,
  UsageTotals,
  User,
  Workspace,
  WorkspaceActivation,
  WorkspaceBudgetDefault,
  WorkspaceCodeExecutionPolicy,
  WorkspaceMcpServer,
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
  "admin",
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

// The same list for a hosted (multi-tenant) deployment, kept in step with
// HOSTED_SURFACES beside it: the process-global provider page drops and the
// organization-scoped one takes its place.
export const HOSTED_SURFACES = [
  ...STANDALONE_SURFACES.filter((surface) => surface !== "providers"),
  "organization_providers",
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
    // Null, because a standalone gateway is its own data plane: the address that
    // served this page is the address that serves the API. The hosted-mode
    // snippet tests set it.
    data_plane_url: null,
    // Null, because a fixture describes a deployment whose documentation is the
    // bundled guide; the tests about an operator-configured docs site set it.
    docs_url: null,
    // Null both, because a self-hosted gateway publishes no legal pages of its
    // own: Terms of service is then absent from the account menu and Data &
    // Privacy is the disabled row. The account-menu tests set them.
    terms_url: null,
    privacy_url: null,
    // Not frozen, because a fixture describes a deployment somebody can sign
    // in to; the maintenance-mode tests override it.
    maintenance_mode: false,
    // Off by default, matching a deployment that has not set public_base_url:
    // the passkey tests turn it on rather than every other test turning it off.
    passkeys_ready: false,
    // Empty by default, matching a deployment that registered no OAuth client:
    // the OAuth tests name the providers they need rather than every other test
    // clearing a list it does not care about.
    oauth_providers: [],
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
    // The identity a standalone first boot provisions: a name (OPERATOR_FULL_NAME
    // in provisioning_service.py) and no address, since the master key is what
    // signs it in. A test about a rostered member overrides it.
    caller: {
      user_id: "33333333-3333-3333-3333-333333333333",
      email: null,
      full_name: "Operator",
    },
    role: "owner",
    status: "active",
    organization: organization(),
    // Empty by default: a context is about the organization, and a test that is
    // about the workspace switcher names the memberships it needs. The server
    // omits the field entirely for a caller in no workspace, which is why the
    // shell treats absent and empty the same way.
    workspace_memberships: [],
    byo_provider_keys_allowed: true,
    // True by default because the fixture models the standalone deployment whose
    // one identity is both its operator and its organization's owner, which is
    // what most tests here are about. A test about a tenant who is *not* an
    // operator says so, and that is the case that reads the organization-scoped
    // usage routes rather than the deployment-wide ones.
    deployment_operator: true,
    provider_key_encryption_available: true,
    ...overrides,
  }
}

/**
 * One row of the caller's own organization memberships, as the switcher reads them.
 *
 * The active one by default, so a single-membership fixture describes the
 * ordinary deployment: one organization, provisioned at first boot, and the
 * caller in it.
 */
export function callerOrganizationMembership(
  overrides: Partial<CallerOrganizationMembership> = {},
): CallerOrganizationMembership {
  return {
    organization_member_id: "22222222-2222-2222-2222-222222222222",
    organization: organization(),
    role: "owner",
    status: "active",
    is_active_organization: true,
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

/**
 * One account as the deployment administration page lists it.
 *
 * An ordinary member by default: active, not an operator, one active
 * membership, and never signed in. Every case the page is really about (an
 * operator, the bootstrap row, the caller's own, a deactivated or a suspended
 * one) is a flag away, and stating them here would make the default row the
 * exception rather than the norm.
 */
export function deploymentUser(
  overrides: Partial<DeploymentUser> = {},
): DeploymentUser {
  return {
    id: "33333333-3333-3333-3333-333333333333",
    email: "ada@example.com",
    full_name: "Ada Lovelace",
    is_active: true,
    is_superuser: false,
    is_bootstrap_operator: false,
    is_self: false,
    last_sign_in_at: null,
    created_at: "2026-01-01T00:00:00+00:00",
    organizations: [
      {
        organization_id: ORGANIZATION_ID,
        name: "Default organization",
        slug: "default",
        role: "member",
        status: "active",
      },
    ],
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

export function orgProviderKey(
  overrides: Partial<OrgProviderKey> = {},
): OrgProviderKey {
  return {
    id: "66666666-6666-6666-6666-666666666666",
    organization_id: "11111111-1111-1111-1111-111111111111",
    provider: "openai",
    name: "Production",
    api_base: null,
    client_args: null,
    // What a stored credential looks like coming back: the ciphertext stays on
    // the server and only the tail of the key is ever published.
    last4: "abcd",
    is_org_default: false,
    archived_at: null,
    created_at: "2026-08-24T00:00:00+00:00",
    updated_at: null,
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
    // The default deployment has curated no images, so a workspace may pin
    // none. `available_tools` is what this deployment's sandbox serves, which
    // is one tool today, not the wider vocabulary a policy may be written in.
    allowed_images: [],
    available_tools: ["code_execution"],
    enabled: true,
    default_purpose_hint: null,
    max_iterations: null,
    exec_timeout_s: null,
    image: null,
    tools: null,
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

export function workspaceMcpServer(
  overrides: Partial<WorkspaceMcpServer> = {},
): WorkspaceMcpServer {
  return {
    id: "55555555-5555-5555-5555-555555555555",
    workspace_id: "44444444-4444-4444-4444-444444444444",
    name: "github",
    url: "https://mcp.example.com/github",
    purpose_hint: null,
    // Null is how "no allow-list" is stored, which is the neutral state a
    // builder wants. Not interchangeable with `[]` as a stored value, even
    // though `mcp_client` happens to read both as "expose everything".
    allowed_tools: null,
    enabled: true,
    // The token is write-only, so this is all a response ever says about it.
    has_token: false,
    created_at: "2026-08-01T00:00:00+00:00",
    updated_at: "2026-08-01T00:00:00+00:00",
    ...overrides,
  }
}
