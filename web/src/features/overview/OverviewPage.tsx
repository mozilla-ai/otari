import { Button } from "@heroui/react"
import type { LinkProps } from "@tanstack/react-router"
import {
  Link,
  Navigate,
  Outlet,
  useLocation,
  useNavigate,
  useSearch,
} from "@tanstack/react-router"
import type { ReactNode } from "react"
import { useEffect, useMemo, useState } from "react"
import { EmptyState } from "@/design-system/feedback/EmptyState"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { PageError } from "@/design-system/feedback/PageError"
import { PageLoading } from "@/design-system/feedback/PageLoading"
import { useIsMobile } from "@/design-system/hooks/useIsMobile"
import { Dot } from "@/design-system/indicators/Dot"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { Section } from "@/design-system/layout/Section"
import { tabClassName } from "@/design-system/navigation/TabRow"
import { ObserveFilterBar } from "@/features/observe/ObserveFilterBar"
import { ObserveScopeCaption } from "@/features/observe/ObserveScope"
import {
  carrySearch,
  OBSERVE_PATHS,
  type ObserveView,
  viewForPath,
} from "@/features/observe/observeSearch"
import { SetupGuide } from "@/features/onboarding/SetupGuide"
import { useCurrentOwner } from "@/features/organization/attribution"
import { CreateOrganizationForm } from "@/features/organization/CreateOrganizationForm"
import { canManage } from "@/features/organization/roles"
import {
  allocationStrip,
  type BudgetHealth,
  errorRateHealth,
  providerHealthStatus,
} from "@/features/overview/overview"
import { useHostedProviders } from "@/shared/api/hostedProviders"
import { useModels } from "@/shared/api/models"
import {
  useDeploymentOperator,
  useOrganizationContext,
  usePendingOrganizationInvitations,
} from "@/shared/api/organizations"
import { useOverviewSummary } from "@/shared/api/overview"
import { useProviderHealth, useProviders } from "@/shared/api/providers"
import {
  NO_BREAKDOWNS,
  useUsageLogs,
  useUsageSummary,
} from "@/shared/api/usage"
import { formatPct } from "@/shared/helpers/format"
import type { DashboardSearch } from "@/shared/helpers/search"
import { localMidnightIso } from "@/shared/helpers/timeRange"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"
import { useDeployment, useSurfaces } from "@/shared/hooks/useDeployment"

// The operator's current local date, as a stable key. Used to hold the window
// bounds steady within a day (so query keys don't churn every render) while still
// letting them advance across midnight.
export function localDayKey(): string {
  const d = new Date()
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`
}

// Local midnight, the window the attention strip judges today's error rate
// over. Held stable within a local day so the query key doesn't churn, but
// advanced when the tab is refocused on a new day, so a tab left open overnight
// does not keep judging yesterday's "today".
function useLocalMidnight(): string {
  const [dayKey, setDayKey] = useState(localDayKey)
  useEffect(() => {
    const check = () => {
      if (document.visibilityState === "visible") {
        const next = localDayKey()
        setDayKey((prev) => (prev === next ? prev : next))
      }
    }
    document.addEventListener("visibilitychange", check)
    window.addEventListener("focus", check)
    return () => {
      document.removeEventListener("visibilitychange", check)
      window.removeEventListener("focus", check)
    }
  }, [])
  // dayKey drives the refresh: same day -> same bound; new day -> re-derive.
  // Nothing inside reads it, so the rule sees it as surplus; it is the whole
  // point of the memo.
  // biome-ignore lint/correctness/useExhaustiveDependencies: dayKey is the cache key, not an input
  return useMemo(() => localMidnightIso(), [dayKey])
}

/**
 * Today's error rate as the attention strip judges it, and the query its link
 * opens the request log on.
 *
 * The strip is about what the caller is responsible for: a manager answers for
 * the organization (or deployment), so it reads everyone's requests; a member
 * answers for their own. Neither follows the Mine | Everyone switch, and a
 * member whose usage identity is unknown is asked nothing rather than judged
 * on the workspace.
 */
function useTodayErrors(managesSpend: boolean) {
  const since = useLocalMidnight()
  const { selected: workspace } = useSelectedWorkspace()
  const owner = useCurrentOwner(!managesSpend)
  const filters = useMemo(
    () => ({
      workspace_id: workspace?.workspace_id,
      start_date: since,
      user_id: managesSpend ? undefined : [owner.ownerId],
    }),
    [workspace, since, managesSpend, owner.ownerId],
  )
  // The strip reads only `totals`, so the window opts out of every breakdown
  // rather than making the server run a GROUP BY per dimension.
  const query = useUsageSummary(
    filters,
    "hour",
    NO_BREAKDOWNS,
    managesSpend || owner.ownerId !== "",
  )
  const search: DashboardSearch = {
    status: "error",
    start_date: since,
    scope: managesSpend ? "everyone" : "mine",
  }
  return { query, search }
}

/**
 * Whether anything has been served in the selected workspace, for the
 * getting-started band: imported OTLP usage lands in the usage tables through
 * a budget-exempt key and no provider config, so "nothing to route to" alone
 * does not mean "nothing has happened". Asked only while that is in doubt.
 */
function useHasAnyUsage(enabled: boolean) {
  const { selected: workspace } = useSelectedWorkspace()
  const filters = useMemo(
    () => ({ workspace_id: workspace?.workspace_id }),
    [workspace],
  )
  const logs = useUsageLogs(filters, 0, 1, enabled)
  return {
    isKnown: logs.isSuccess,
    hasUsage: (logs.data?.length ?? 0) > 0,
  }
}

/**
 * The landing route: what needs attention, then usage, requests and agent
 * sessions over one window and one set of filters.
 *
 * A layout with one child route per view (`/`, `/requests`, `/sessions`), so a
 * view is linkable, and each tab's link carries the shared filters (see
 * `carrySearch`), so switching view never resets what was picked.
 *
 * Who the caller is decides the status bands above the tabs, and nothing else.
 * The operator's read deployment-wide endpoints (provider health, budgets),
 * which since otari-ai#1880 answer 403 to anyone who is not an operator of the
 * deployment, so they live in a component that is not rendered for anyone
 * else. Decided from the organization context, which is where `useUsageScope`
 * and the nav rail take the same answer from, so nothing on screen can hold a
 * different one (otari-ai#1936).
 *
 * An errored context lands on the operator's bands, as it does for the usage
 * hooks below them and for every other operator gate (`useDeploymentOperator`):
 * their panels report their own refusals, where the reverse would hide the
 * deployment's state from the one caller who owns it. An errored context is an
 * answer, not a wait.
 *
 * The one failure that is not sent there is a caller with no organization,
 * who has no deployment to operate and no organization to report on.
 */
export function OverviewIndex() {
  const { answer, isSettled } = useDeploymentOperator()
  const hasNoOrganization = answer === "no-organization"
  // Somebody who signed up or signed in before accepting the invitation that
  // created their account: the inbox is the page that can fix it, so they are
  // sent there rather than left to find it in the switcher.
  const invitations = usePendingOrganizationInvitations(hasNoOrganization)

  if (!isSettled || (hasNoOrganization && invitations.isPending)) {
    return <PageLoading />
  }
  if (hasNoOrganization && invitations.isError) {
    // The overview below cannot serve this identity either, so the one way
    // forward is offered rather than a page of failed reads.
    return (
      <PageError error={invitations.error}>
        Your invitations could not be loaded.{" "}
        <Link to="/invitations" className="text-link hover:text-link-hover">
          Open your invitations
        </Link>{" "}
        to try again.
      </PageError>
    )
  }
  if (hasNoOrganization) {
    return (invitations.data ?? []).length > 0 ? (
      <Navigate to="/invitations" replace />
    ) : (
      <NoOrganizationOverview />
    )
  }
  return (
    <OverviewLayout
      status={
        answer === "not-operator" ? <OrganizationStatus /> : <OperatorStatus />
      }
    />
  )
}

/**
 * The landing page for a signed-in caller who belongs to no organization and
 * has no invitation waiting.
 *
 * Creating an organization is offered only where signup is open: where it is
 * closed the server refuses a caller with no membership, so the way in is an
 * invitation from an owner.
 */
function NoOrganizationOverview() {
  const { open_signup } = useDeployment()
  const [creating, setCreating] = useState(false)
  // Keys the form so each open starts from an empty draft.
  const [creatingCount, setCreatingCount] = useState(0)

  return (
    <div className="flex flex-col">
      <PageIntro title="Overview" />
      <EmptyState
        title="You're not a member of an organization yet"
        description={
          open_signup
            ? "Ask an owner of an organization to invite you, or create one of your own. Invitations you receive are listed under Invitations."
            : "Ask an owner of an organization to invite you. Invitations you receive are listed under Invitations."
        }
        actionLabel={open_signup ? "Create organization" : undefined}
        onAction={() => {
          setCreatingCount((count) => count + 1)
          setCreating(true)
        }}
      >
        <Link
          to="/invitations"
          className="inline-flex min-h-11 items-center self-start text-sm text-muted underline underline-offset-2 hover:text-foreground"
        >
          View invitations <span aria-hidden>→</span>
        </Link>
      </EmptyState>
      {open_signup ? (
        <CreateOrganizationForm
          key={creatingCount}
          isOpen={creating}
          onClose={() => setCreating(false)}
        />
      ) : null}
    </div>
  )
}

const TABS: { view: ObserveView; label: string; surface: string }[] = [
  { view: "usage", label: "Usage", surface: "usage" },
  { view: "requests", label: "Requests", surface: "usage" },
  { view: "sessions", label: "Sessions", surface: "traces" },
]

const hasValue = (value: DashboardSearch[string]) =>
  Array.isArray(value) ? value.length > 0 : Boolean(value)

function OverviewLayout({ status }: { status: ReactNode }) {
  const hasSurface = useSurfaces()
  const isMobile = useIsMobile()
  const current = viewForPath(useLocation({ select: (loc) => loc.pathname }))
  const search = useSearch({ strict: false }) as DashboardSearch
  const tabs = TABS.filter((tab) => hasSurface(tab.surface))
  const isServed = tabs.some((tab) => tab.view === current)

  // Where only traces are recorded, a view this deployment does not serve
  // opens on the first one it does.
  const landing = tabs[0]
  if (landing && !isServed) {
    return (
      <Navigate
        to={OBSERVE_PATHS[landing.view]}
        search={carrySearch(current, landing.view, search)}
        replace
      />
    )
  }

  // On a phone an open request or session is a full-width panel, so what it
  // covers leaves the tab order and the accessibility tree.
  const isCovered =
    isMobile && (hasValue(search.request) || hasValue(search.peek))

  return (
    <div className="flex flex-col">
      <PageIntro title="Overview">
        What needs attention, then spend, requests and agent sessions over one
        window and one set of filters. Click a bar or a row to narrow it; the
        filters follow you between tabs.
      </PageIntro>
      <div inert={isCovered}>
        {/* Each band rules its own top; the last one closes the group, so the
            controls below sit under one rule. */}
        <div className="flex flex-col pb-4 empty:hidden [&>section:last-of-type]:border-b">
          {status}
        </div>
        {isServed ? (
          <>
            <ObserveScopeCaption />
            {tabs.length > 1 ? (
              <nav aria-label="Overview views" className="pb-5">
                <div className="inline-flex max-w-full items-center gap-1 overflow-x-auto">
                  {tabs.map((tab) => {
                    const isActive = tab.view === current
                    return (
                      // A link wearing the `Tab` classes: each view is a
                      // destination with its own URL, not a switch inside one
                      // page.
                      <Link
                        key={tab.view}
                        to={OBSERVE_PATHS[tab.view]}
                        search={carrySearch(current, tab.view, search)}
                        aria-current={isActive ? "page" : undefined}
                        className={tabClassName(isActive)}
                      >
                        {tab.label}
                      </Link>
                    )
                  })}
                </div>
              </nav>
            ) : null}
            <ObserveFilterBar
              view={current}
              usageAvailable={hasSurface("usage")}
            />
          </>
        ) : null}
      </div>
      {isServed ? <Outlet /> : null}
    </div>
  )
}

/**
 * The operator's status bands: getting started on an empty gateway, the setup
 * guide once a provider exists, and what needs attention across the
 * deployment.
 *
 * The provider lists make the empty gateway a useful getting-started page: the
 * `config.yml` instances and the deployment's hosted providers, either of which
 * serves a request. Both queries are operator-gated, which is why this renders
 * only once the gate in `OverviewIndex` has answered.
 */
function OperatorStatus() {
  const providers = useProviders()
  const hosted = useHostedProviders()
  // A failed providers query leaves the setup state unknown, so it stays
  // neutral (no getting-started band) and the error is reported instead of
  // swallowed. A failed hosted read counts none rather than blocking the page;
  // the configured list is the one whose failure is reported.
  const providerCount = providers.isSuccess
    ? providers.data.providers.length + (hosted.data?.data?.length ?? 0)
    : 0
  const needsSetup = providers.isSuccess && providerCount === 0
  const usage = useHasAnyUsage(needsSetup)
  const { selected: workspace } = useSelectedWorkspace()
  const health = useProviderHealth()
  // The budget scan the gateway judges, scoped the way the Budgets page is.
  const summary = useOverviewSummary(workspace?.workspace_id)
  const today = useTodayErrors(true)
  const err = errorRateHealth(today.query.data?.totals)

  const budget = allocationStrip(summary.data?.budgets, {
    none: "No budgets configured",
    noneCapped: "No capped budgets",
  })

  return (
    <>
      {needsSetup && usage.isKnown && !usage.hasUsage ? (
        <GetStartedStrip />
      ) : null}
      {/* The step after that one: a provider exists, so the guide can hand out
          a key and watch for the first request. It decides for itself whether
          to render, including holding back while there is no provider, which
          is when the band above is the right guide instead. */}
      <SetupGuide canServeRequests={providerCount > 0} />
      {/* The query that decides whether this is a getting-started page, so its
          failure is the most load-bearing thing to tell the operator about. */}
      <ErrorBanner error={providers.error} />
      <AttentionStrip
        providerHealth={providerHealthStatus(health.data)}
        healthy={health.data?.healthy ?? 0}
        degraded={health.data?.degraded ?? 0}
        total={health.data?.total ?? 0}
        budget={budget}
        errStatus={err.status}
        errRate={err.rate}
        errorSearch={today.search}
        // The strip evaluates health, budgets, and error rate only after all
        // three load successfully, avoiding transient or false alerts.
        isReady={health.isSuccess && summary.isSuccess && today.query.isSuccess}
        hasFailed={health.isError || summary.isError || today.query.isError}
      />
    </>
  )
}

/**
 * The status bands for a caller who does not operate the deployment.
 *
 * Provider health and the deployment's budgets are not this caller's to see,
 * so the strip judges the organization's spend ceilings (for an owner or
 * admin, the line the server draws on the ceilings route) and today's error
 * rate. There is no `useDeploymentOperator` anywhere below this line: this
 * *is* the non-operator branch, so asking again is how otari-ai#2080 happened.
 */
function OrganizationStatus() {
  const context = useOrganizationContext()
  const managesSpend = canManage(context.data)
  const { selected: workspace } = useSelectedWorkspace()
  const scope = workspace?.workspace_id
  // What this caller could route a request to, which is the setup guide's gate
  // here: Hosted providers, the operator's answer to the same question, refuses
  // this caller. The catalog is filtered to the selectors this caller may name,
  // so an empty one means there is nothing to send a request to yet. Not asked
  // until a workspace is selected, since the guide is about one.
  const models = useModels(scope !== undefined)
  const hasNoModels = models.isSuccess && models.data.data.length === 0
  const usage = useHasAnyUsage(hasNoModels)
  // Withheld by the server from a member, so a member is not asked.
  const summary = useOverviewSummary(scope, managesSpend)
  const today = useTodayErrors(managesSpend)
  const err = errorRateHealth(today.query.data?.totals)

  const ceilingHealth = allocationStrip(
    managesSpend ? summary.data?.ceilings : null,
    {
      none: "No spend ceilings configured",
      noneCapped: "No ceiling caps spend",
    },
  )

  return (
    <>
      {/* Offered to whoever manages the workspace, which on a multi-tenant
          deployment is this caller. It decides for itself whether to render. */}
      <SetupGuide canServeRequests={(models.data?.data.length ?? 0) > 0} />
      {/* The guide holds back until a request could succeed; without this a
          new account with nothing to route to has no next step. Waits for the
          catalog and the log so it never flashes in on a set-up account. */}
      {hasNoModels && usage.isKnown && !usage.hasUsage ? (
        <GetStartedStrip
          description={
            managesSpend
              ? "Add a provider key to start serving models, then create an API key and send your first request."
              : "Ask an owner or admin of this organization to add a provider key. Once models are available, you can create an API key and send a request."
          }
          action={managesSpend ? "organization-provider-key" : null}
        />
      ) : null}
      <AttentionStrip
        providerHealth="neutral"
        healthy={0}
        degraded={0}
        total={0}
        budget={ceilingHealth}
        budgetNoun="spend ceiling"
        errStatus={err.status}
        errRate={err.rate}
        errorSearch={today.search}
        isReady={today.query.isSuccess && (!managesSpend || summary.isSuccess)}
        hasFailed={today.query.isError || (managesSpend && summary.isError)}
      />
    </>
  )
}

function GetStartedStrip({
  description = "Add a provider to begin serving models. Once it is configured, this page will show your gateway’s traffic, spend, and health.",
  action = "provider",
}: {
  description?: string
  /** The button, or none for a caller who may not act on it. */
  action?: "provider" | "organization-provider-key" | null
}) {
  const navigate = useNavigate()

  return (
    <Section
      className="border-t border-border py-5"
      contentClassName="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between"
    >
      <div className="flex items-start gap-3">
        <Dot className="mt-2.5 bg-accent" />
        <div>
          <h2 className="text-display-sub">Get started with Otari</h2>
          <p className="mt-1 text-sm text-muted">{description}</p>
        </div>
      </div>
      {action === null ? null : (
        <Button
          variant="primary"
          className="shrink-0"
          onPress={() =>
            navigate({
              to:
                // Only an operator is offered "provider" (`OverviewIndex` hands
                // anyone else to `OrganizationStatus`), so it goes to the
                // deployment's own keys.
                action === "provider"
                  ? "/admin/hosted-providers"
                  : "/organization/provider-keys",
            })
          }
        >
          {action === "provider"
            ? "Add your first provider"
            : "Add a provider key"}
        </Button>
      )}
    </Section>
  )
}

// A neutral, hue-free line for a failed status source.
function NeutralStrip({ text }: { text: string }) {
  return (
    <Section
      role="status"
      className="border-t border-border py-3"
      contentClassName="text-sm text-muted"
    >
      {text}
    </Section>
  )
}

function AttentionStrip({
  providerHealth,
  healthy,
  degraded,
  total,
  budget,
  budgetNoun = "budget",
  errStatus,
  errRate,
  errorSearch,
  isReady,
  hasFailed,
}: {
  providerHealth: "ok" | "warn" | "alert" | "neutral"
  healthy: number
  degraded: number
  total: number
  budget: BudgetHealth
  /** What a capped row is called in the message: the operator's budgets, a tenant's spend ceilings. */
  budgetNoun?: string
  errStatus: "ok" | "warn" | "alert" | "neutral"
  errRate: number | null
  /** The request log's query for the failures the rate counts. */
  errorSearch: DashboardSearch
  isReady: boolean
  hasFailed: boolean
}) {
  // A failed source deserves a visible status message; while loading, wait for
  // actionable information instead of reserving space for a transient banner.
  if (hasFailed) {
    return <NeutralStrip text="Some status data could not be loaded." />
  }
  if (!isReady) {
    return null
  }

  const problems: {
    text: string
    to: LinkProps["to"]
    search?: LinkProps["search"]
  }[] = []
  if ((providerHealth === "warn" || providerHealth === "alert") && total > 0) {
    // A degraded provider answers no model listing but is not down, so it is
    // reported separately rather than folded into "unreachable" (issue #447).
    const down = total - healthy - degraded
    if (down > 0) {
      problems.push({
        text: `${down} provider${down === 1 ? "" : "s"} unreachable`,
        to: "/admin/hosted-providers",
      })
    }
    if (degraded > 0) {
      problems.push({
        text: `${degraded} provider${degraded === 1 ? "" : "s"} without model discovery`,
        to: "/admin/hosted-providers",
      })
    }
  }
  if (budget.overCount > 0) {
    problems.push({
      text: `${budget.overCount} ${budgetNoun}${budget.overCount === 1 ? "" : "s"} over limit`,
      to: "/budgets",
    })
  } else if (budget.nearCount > 0) {
    problems.push({
      text: `${budget.nearCount} ${budgetNoun}${budget.nearCount === 1 ? "" : "s"} near limit`,
      to: "/budgets",
    })
  }
  if (errStatus === "alert" && errRate !== null) {
    problems.push({
      text: `error rate ${formatPct(errRate)} today`,
      to: OBSERVE_PATHS.requests,
      search: errorSearch,
    })
  }

  if (problems.length === 0) {
    return null
  }

  // No fill and no radius: the strip is a band of the page under a rule, and
  // the square danger dot is what carries the urgency.
  return (
    <Section
      role="alert"
      className="border-t border-border py-3"
      contentClassName="flex flex-col gap-2 text-sm sm:flex-row sm:flex-wrap sm:items-center"
    >
      <span className="flex items-center gap-2 font-semibold text-foreground">
        <Dot className="bg-danger" />
        Needs attention
      </span>
      {problems.map((p, i) => (
        <span key={p.text} className="flex items-center gap-2 text-muted">
          {i > 0 ? (
            <span aria-hidden className="text-subtle">
              ·
            </span>
          ) : null}
          <Link
            to={p.to}
            search={p.search}
            // Thicken the underline on hover rather than lightening the text:
            // the color here is already the one tuned to clear AA.
            className="underline underline-offset-2 hover:decoration-2"
          >
            {p.text}
          </Link>
        </span>
      ))}
    </Section>
  )
}
