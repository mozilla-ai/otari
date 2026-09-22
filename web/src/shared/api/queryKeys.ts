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
// The same selectors as MODELS, folded by model and priced for the viewer. Its
// own key so the two reads can be cached apart, and every pricing mutation
// invalidates both: a rate change moves a row in each.
export const CATALOG = "catalog"
export const OVERVIEW = "overview"
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
// The guardrails service's own profile list, keyed apart from TOOL_SETTINGS
// even though `guardrails_url` is where it is read from: the catalog is the
// remote service's answer, so a settings save that changes that URL invalidates
// it, while every other tool-settings write must not re-dial the sidecar.
export const GUARDRAIL_PROFILES = "guardrail-profiles"
// The guardrails this gateway can build itself. Kept apart from
// GUARDRAIL_PROFILES for the opposite reason: this one is a property of the
// installed library and moves only on a redeploy, so pointing `guardrails_url`
// at another service must not invalidate it.
export const BUILTIN_GUARDRAIL_CATALOG = "builtin-guardrail-catalog"
// Both carry the surface they were read from and the workspace they were scoped
// to as trailing key segments, so the deployment-wide list and its tenant-scoped
// sibling share a head that one invalidation covers. See `useRoutingScope`.
export const ALIASES = "aliases"
export const ROUTING_POLICIES = "routing-policies"
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
// The caller's standing in the organization they are acting in. Composed here
// rather than spelled at the hook, because two things outside that hook address
// this one read: switching organization writes the context it was answered with
// straight into it, and the spend-ceilings read binds its role gate to whatever
// object is cached under it.
export const ORGANIZATION_CONTEXT = [ORGANIZATIONS, "context"] as const
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
// What each guardrail is, apart from where it runs. No mandate field changes
// when a definition does, so a definition write leaves the mandates alone.
export const ORGANIZATION_GUARDRAIL_DEFINITIONS =
  "organization-guardrail-definitions"
// The organization's own upstream provider credentials. Its own key for the
// reason the two above have one: this is read by one page, and a credential
// edit has no business refetching the organization context every page reads.
export const ORGANIZATION_PROVIDER_KEYS = "organization-provider-keys"
// The models an organization offers on one of its provider keys. Its own root
// key rather than a child of ORGANIZATION_PROVIDER_KEYS: making a key default or
// archiving one does not move a model row, and nesting would refetch every open
// panel on each of those writes. Scoped per key below the root, because a page
// can have one key's panel open at a time and the others must not refetch.
export const ORGANIZATION_PROVIDER_MODELS = "organization-provider-models"
// What a provider says it serves on a stored credential. The DISCOVERABLE rule
// one scope down: answering dials the upstream, so it must not share a head with
// anything a price or a toggle invalidates, or every save would re-dial.
export const ORGANIZATION_PROVIDER_AVAILABLE_MODELS =
  "organization-provider-available-models"
// The organization's email-domain claims. Its own key for the same reason:
// one page reads it, and claiming a domain has no bearing on anything else.
export const ORGANIZATION_DOMAINS = "organization-domains"
export const WORKSPACES = "workspaces"
// The first-request setup guide's state. Its own key rather than a child of
// WORKSPACES: the guide polls while it is on screen, and nesting it would make
// every one of those ticks invalidate (or be invalidated by) the workspace list
// and its rosters.
export const ACTIVATION = "workspace-activation"
// The Playground's own reads, all four of them under one key because they are
// one page's state and nothing outside that page reads or writes them: the
// caller's retention consent, their saved transcripts, their rated comparisons,
// their pinned models, and the tools menu's availability. Scoped per workspace
// below the root where the answer is per workspace (consent is not: it is about
// what this deployment stores about a person, so it is asked once).
//
// Deliberately not nested under WORKSPACES, which the workspace tool-policy
// reads are: those are the same rows the Tools pages edit, so a save there has
// to invalidate them, while nothing an operator edits changes a transcript
// somebody saved. Nesting would make every workspace write refetch the page's
// whole history.
export const PLAYGROUND = "playground"

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
