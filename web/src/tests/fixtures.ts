// Builders for the API shapes a test needs whole.
//
// These live here because the shapes now come from the generated client, so a
// field the gateway adds arrives in every test at once. Written out inline, that
// meant six files failing to compile over a field none of them cared about. A
// builder fills the whole shape with neutral values and takes an override for
// the part under test, so a test states what it is about and nothing else.

import type {
  DeploymentBootstrap,
  Organization,
  OrganizationContext,
  OrganizationMember,
  PricingResponse,
  UsageSeriesPoint,
  UsageTotals,
  Workspace,
  WorkspaceBudgetDefault,
  WorkspaceMember,
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
    management_url: null,
    invitation_mail_ready: false,
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

export function workspaceBudgetDefault(
  overrides: Partial<WorkspaceBudgetDefault> = {},
): WorkspaceBudgetDefault {
  return {
    id: "66666666-6666-6666-6666-666666666666",
    workspace_id: "44444444-4444-4444-4444-444444444444",
    provider_key_id: null,
    name: "Default member budget",
    max_budget: 50.0,
    budget_duration_sec: 2_592_000,
    created_at: "2026-01-01T00:00:00+00:00",
    updated_at: "2026-01-01T00:00:00+00:00",
    ...overrides,
  }
}
