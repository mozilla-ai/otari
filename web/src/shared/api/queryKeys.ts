/**
 * Every query key the dashboard uses, and the cadences and options that ride
 * with them.
 *
 * One module, deliberately, and the one rule this split must not break. The
 * keys are not per-domain facts: 109 invalidations across the hooks reach 39 of
 * them, because a write in one domain changes a read in another (an alias shows
 * up as a model, a role change invalidates a roster but not the context that
 * gated it). A key declared next to its own hooks would either be duplicated or
 * be imported back across domains, which is the circular import the no-barrel
 * rule exists to prevent. The comments below are the reasoning for which keys
 * are deliberately kept apart; read them before adding one.
 */

export const MODELS = "models"
export const PRICING = "pricing"
export const SETTINGS = "settings"
export const MAIL_SETTINGS = "mail-settings"
export const MAINTENANCE_MODE = "maintenance-mode"
// One indexed single-row read, and only while the settings page is mounted.
export const MAINTENANCE_MODE_POLL_MS = 30_000
export const TOOL_SETTINGS = "tool-settings"
export const TOOLS = "tools"
export const SEARCH_TOOLS = "search-tools"
export const SEARCH_PROVIDERS = "search-providers"
export const ALIASES = "aliases"
export const ROUTING_POLICIES = "routing-policies"
// The tenant-scoped sibling of ROUTING_POLICIES. Its own key: the two lists
// answer different endpoints for different callers, and an operator's policy
// write invalidates the deployment-wide one it changed.
export const ORGANIZATION_ROUTING_POLICIES = "organization-routing-policies"
export const ORGANIZATION_ALIASES = "organization-aliases"
export const ROUTER_STATUS = "router-status"
// Deliberately not nested under MODELS: pricing mutations invalidate that key,
// and a price change cannot alter which models a provider serves. Sharing the
// key would fire a live provider call on every save.
export const DISCOVERABLE = "discoverable"
export const PROVIDERS = "providers"
export const PROVIDER_HEALTH = "provider-health"
export const STORED_PROVIDERS = "stored-providers"
export const METADATA = "model-metadata"
export const BUILD = "build"
export const HEALTH = "health"
export const KEYS = "keys"
export const BUDGETS = "budgets"
export const SCOPED_BUDGETS = "scoped-budgets"
export const USERS = "users"
export const USAGE = "usage"
export const ORGANIZATIONS = "organizations"
// Deliberately its own key rather than a child of ORGANIZATIONS: switching
// organizations invalidates both, but a role change invalidates only the roster,
// and nesting would re-read the context (and every page gated on it) as well.
export const ORGANIZATION_MEMBERS = "organization-members"
export const DEPLOYMENT_ADMIN = "deployment-admin"
// Its own key rather than a child of ORGANIZATIONS, for the reason the members
// key is: the organization context is read on nearly every page, and a rate
// edit should not make all of them refetch.
export const ORGANIZATION_PRICING = "organization-pricing"
// The organization's own budgets and ceilings, keyed apart from BUDGETS and
// SCOPED_BUDGETS above: those are the deployment-wide reads, which answer 403
// to a tenant, so one key for both would serve a cached operator answer to an
// admin and invalidate reads neither caller can make.
export const ORGANIZATION_BUDGETS = "organization-budgets"
export const ORGANIZATION_SPEND_CEILINGS = "organization-spend-ceilings"
export const ORGANIZATION_GUARDRAILS = "organization-guardrails"
// The organization's own upstream provider credentials. Its own key for the
// reason the two above have one: this is read by one page, and a credential
// edit has no business refetching the organization context every page reads.
export const ORGANIZATION_PROVIDER_KEYS = "organization-provider-keys"
// The organization's email-domain claims. Its own key for the same reason:
// one page reads it, and claiming a domain has no bearing on anything else.
export const ORGANIZATION_DOMAINS = "organization-domains"
export const WORKSPACES = "workspaces"
// The first-request setup guide's state. Its own key rather than a child of
// WORKSPACES: the guide polls while it is on screen, and nesting it would make
// every one of those ticks invalidate (or be invalidated by) the workspace list
// and its rosters.
export const ACTIVATION = "workspace-activation"
// The signed-in identity's own passkeys. Its own key and not a child of any
// organization key: a passkey belongs to a person, not to the organization they
// happen to be acting in, and switching organizations does not change the list.
export const PASSKEYS = "passkeys"

// How often an open tab asks whether the app it is running is still the one the
// gateway serves. Cheap (a hash of one small file) and only while the tab is
// open, so a minute keeps a deploy from going unnoticed for long.
export const BUILD_POLL_MS = 60_000
// How often the hybrid landing page re-asks whether this gateway is up and can
// still reach its control plane. That pair is the only thing on that page which
// changes, and it is the reason to leave the page open, so it ticks faster than
// the build check. The gateway bounds its own upstream probe (`resolve_timeout_ms`),
// so a stalled control plane answers "no" rather than piling up requests.
export const HEALTH_POLL_MS = 15_000

// The four queries below are backed by provider or models.dev fan-out
// gateway-side. That is cached and refreshed in the background now, so they are
// normally fast, but they are the ones that go slow when a provider does. The
// global default retries a failed query twice (see provider.tsx), which would
// turn one slow failure into three sequential ones and hold a browser
// connection slot for the whole time; on HTTP/1.1 (6 sockets per origin) enough
// of those queue every other request behind them, including the POST an
// operator just clicked. Failing once and showing the error is the honest
// behavior, and it frees the socket.
export const NO_RETRY = { retry: false } as const
