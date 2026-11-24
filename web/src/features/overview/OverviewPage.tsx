import {
  Link,
  Navigate,
  Outlet,
  useLocation,
  useSearch,
} from "@tanstack/react-router"
import { useState } from "react"
import { EmptyState } from "@/design-system/feedback/EmptyState"
import { PageError } from "@/design-system/feedback/PageError"
import { PageLoading } from "@/design-system/feedback/PageLoading"
import { useIsMobile } from "@/design-system/hooks/useIsMobile"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { ObserveFilterBar } from "@/features/observe/ObserveFilterBar"
import {
  carrySearch,
  OBSERVE_PATHS,
  type ObserveView,
  viewForPath,
} from "@/features/observe/observeSearch"
import { SetupGuide } from "@/features/onboarding/SetupGuide"
import { CreateOrganizationForm } from "@/features/organization/CreateOrganizationForm"
import { useHostedProviders } from "@/shared/api/hostedProviders"
import { useModels } from "@/shared/api/models"
import {
  useDeploymentOperator,
  usePendingOrganizationInvitations,
} from "@/shared/api/organizations"
import { useProviders } from "@/shared/api/providers"
import { ViewTabs } from "@/shared/components/ViewTabs"
import type { DashboardSearch } from "@/shared/helpers/search"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"
import { useDeployment, useSurfaces } from "@/shared/hooks/useDeployment"

/**
 * The pathless `_overview` layout over `/`, `/sessions`, `/requests` and
 * `/router-savings`.
 *
 * `/` is Home (`features/home/Home.tsx`, its index route), and the other three
 * are the views of Sessions under one title and tab row. A
 * caller with no organization gets the page that sends them to their
 * invitations instead.
 */
export function OverviewIndex() {
  const pathname = useLocation({ select: (location) => location.pathname })
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
    // Home cannot serve this identity either, so the one way forward is
    // offered rather than a page of failed reads.
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
    <>
      <FirstRequestGuide isOperator={answer !== "not-operator"} />
      {/* While a navigation away is pending this layout still renders, with
          the destination's path. Redirecting it to a tab would cancel the
          navigation. */}
      {pathname === "/" ? (
        <Outlet />
      ) : isSessionsPath(pathname) ? (
        <SessionsLayout />
      ) : null}
    </>
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
      <PageIntro title="Home" />
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

/**
 * The first-request guide, over Home and Sessions. It decides for itself
 * whether to show (a workspace that has never served a request, for someone
 * who can make it serve one), and holds back while there is nothing to call.
 * What counts as something to call depends on who asks: for the operator a
 * provider, configured or hosted, since a hosted one can serve a model its
 * upstream does not list; for anyone else a model in the catalog.
 */
function FirstRequestGuide({ isOperator }: { isOperator: boolean }) {
  const { selected } = useSelectedWorkspace()
  const scope = selected?.workspace_id
  const models = useModels(!isOperator && scope !== undefined, scope)
  const providers = useProviders(isOperator)
  const hosted = useHostedProviders(isOperator)
  const canServeRequests = isOperator
    ? (providers.data?.providers.length ?? 0) +
        (hosted.data?.data?.length ?? 0) >
      0
    : (models.data?.data.length ?? 0) > 0
  return <SetupGuide canServeRequests={canServeRequests} />
}

const SAVINGS_PATH = "/router-savings"

/**
 * A view of Sessions. `view` is what the shared filter bar reads the tab as;
 * a tab without one has controls of its own and carries no filters across.
 */
interface SessionsTab {
  path: string
  label: string
  surface: string
  view?: ObserveView
}

const TABS: SessionsTab[] = [
  {
    path: OBSERVE_PATHS.sessions,
    label: "Sessions",
    surface: "traces",
    view: "sessions",
  },
  {
    path: OBSERVE_PATHS.requests,
    label: "Requests",
    surface: "usage",
    view: "requests",
  },
  // What the subagent router saved, over a window of its own.
  { path: SAVINGS_PATH, label: "Router savings", surface: "subagent_savings" },
]

const hasValue = (value: DashboardSearch[string]) =>
  Array.isArray(value) ? value.length > 0 : Boolean(value)

const isUnder = (pathname: string, path: string) =>
  pathname === path || pathname.startsWith(`${path}/`)

const isSessionsPath = (pathname: string) =>
  TABS.some((tab) => isUnder(pathname, tab.path))

/** Sessions' title and tab row over its views, each its own route. */
function SessionsLayout() {
  const hasSurface = useSurfaces()
  const isMobile = useIsMobile()
  const pathname = useLocation({ select: (loc) => loc.pathname })
  const search = useSearch({ strict: false }) as DashboardSearch
  const tabs = TABS.filter((tab) => hasSurface(tab.surface))
  const current = tabs.find((tab) => isUnder(pathname, tab.path))
  // The filters travel between the two views that share the filter bar.
  const searchFor = (tab: SessionsTab) =>
    current?.view && tab.view
      ? carrySearch(current.view, tab.view, search)
      : undefined
  // A view this deployment does not serve opens on the first one it does. A
  // tab with no view has no filters or window to carry, so the landing view
  // opens on its own defaults.
  const landing = tabs[0]
  if (landing && !current) {
    const left = TABS.find((tab) => isUnder(pathname, tab.path))
    return (
      <Navigate
        to={landing.path}
        search={
          landing.view && left?.view
            ? carrySearch(viewForPath(pathname), landing.view, search)
            : {}
        }
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
      <PageIntro title="Sessions" />
      <div inert={isCovered}>
        {tabs.length > 1 ? (
          <div className="pb-5">
            <ViewTabs
              label="Sessions views"
              tabs={tabs.map((tab) => ({
                to: tab.path,
                label: tab.label,
                search: searchFor(tab),
              }))}
              current={current?.path ?? ""}
            />
          </div>
        ) : null}
        {current?.view ? (
          <ObserveFilterBar
            view={current.view}
            usageAvailable={hasSurface("usage")}
          />
        ) : null}
      </div>
      <Outlet />
    </div>
  )
}
