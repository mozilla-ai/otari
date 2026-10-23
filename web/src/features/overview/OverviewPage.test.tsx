import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { useLocation } from "@tanstack/react-router"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactElement } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { AllocationHealth, OverviewSummary, UsageSummary } from "@/client"
import { localDayKey, OverviewIndex } from "@/features/overview/OverviewPage"
import { API_ROOT, defaultQueryRetry } from "@/shared/api/client"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  bootstrap,
  organizationContext,
  organizationMember,
  pendingOrganizationInvitation,
  SURFACES,
  usageTotals,
  workspaceActivation,
} from "@/tests/fixtures"
import { withRouter } from "@/tests/router"

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

function summary(
  totals: Partial<UsageSummary["totals"]>,
  series: UsageSummary["series"] = [],
): UsageSummary {
  return {
    start_date: "2026-06-22T00:00:00Z",
    end_date: "2026-07-22T00:00:00Z",
    bucket: "day",
    totals: usageTotals({
      cost: 0,
      prompt_tokens: 0,
      completion_tokens: 0,
      total_tokens: 0,
      cache_read_tokens: 0,
      cache_write_tokens: 0,
      request_count: 0,
      error_count: 0,
      avg_latency_ms: null,
      ...totals,
    }),
    by_model: [],
    by_user: [],
    by_api_key: [],
    by_source: [],
    by_source_label: [],
    by_endpoint: [],
    by_provider: [],
    by_tool: [],
    by_tag: [],
    errors_by_status_code: [],
    series,
  }
}

/** Everything withheld by default, which is what a member's overview looks like. */
function overviewSummary(over: Partial<OverviewSummary> = {}): OverviewSummary {
  return {
    active_keys: 0,
    active_members: 0,
    budgets: null,
    ceilings: null,
    ...over,
  }
}

/** One strip's scan, as the gateway answers it. */
function strip(over: Partial<AllocationHealth> = {}): AllocationHealth {
  return {
    over_count: 0,
    near_count: 0,
    capped_count: 1,
    total_count: 1,
    worst: {
      budget_id: "11111111-2222-3333-4444-555555555555",
      name: "Monthly",
      spent: 50,
      allocated: 100,
      scope_type: null,
      scope_id: null,
    },
    ...over,
  }
}

interface Bodies {
  /** What GET /hosted-providers lists. */
  hostedProviders?: unknown[]
  today?: Partial<UsageSummary["totals"]>
  /** Today's hourly series, which the sparklines draw. */
  series?: UsageSummary["series"]
  health?: unknown
  /** The overview's own summary: the two rail counts and the budget strips. */
  overview?: Partial<OverviewSummary>
  /** A summary per workspace id, for the counts that follow the switcher. */
  overviewByWorkspace?: Record<string, Partial<OverviewSummary>>
  budgets?: unknown
  /** The organization's spend ceilings, which is the tenant's budget signal. */
  ceilings?: unknown
  keys?: unknown
  users?: unknown
  logs?: unknown
  providers?: unknown
  /** One workspace's roster, keyed by workspace id. */
  workspaceMembers?: Record<string, unknown[]>
  context?: Parameters<typeof organizationContext>[0]
  /** Where the selected workspace stands on its first request. */
  activation?: unknown
  /** Model ids the caller's catalog reports, which is what the guide gates on. */
  models?: string[]
  /** Answers the catalog with a server error instead of a list. */
  modelsFail?: boolean
}

/** The catalog shape `${API_ROOT}/models` answers, which is the setup guide's gate. */
function modelCatalog(ids: string[]) {
  return {
    object: "list",
    data: ids.map((id) => ({
      id,
      object: "model",
      created: 0,
      owned_by: "openai",
    })),
  }
}

// Order matters: /v1/usage/summary is matched BEFORE the bare /v1/usage logs
// endpoint, and /v1/providers/health returns an OBJECT (not the [] fallback) so
// the health/status logic reads real counts. The summary mock is param-aware so
// the Today (bucket=hour) and Last-30d (bucket=day) tiles render distinct values.
function mockApi(b: Bodies) {
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input)
    // The shell reads this before it paints, and the usage hooks wait for it:
    // it is what tells them whether this caller reads the deployment-wide
    // routes or the organization-scoped ones (otari#837). Answered first, and
    // on an exact match, so it cannot shadow /v1/organizations/me/usage.
    if (url.endsWith(`${API_ROOT}/organizations/me`)) {
      return jsonResponse(organizationContext(b.context))
    }
    // The rail's roster is per workspace, so the id in the path picks the
    // answer. A `Paged` envelope and not a bare array: this one goes through
    // `fetchAllPaged`, which reads `body.data` and pages until a short one.
    const roster = url.match(/\/api\/v1\/workspaces\/([^/?]+)\/members/)
    if (roster) {
      return jsonResponse({ data: b.workspaceMembers?.[roster[1]] ?? [] })
    }
    if (url.includes("/activation")) {
      return jsonResponse(b.activation ?? workspaceActivation())
    }
    if (url.includes(`${API_ROOT}/models`)) {
      return jsonResponse(modelCatalog(b.models ?? ["openai:gpt-4o-mini"]))
    }
    if (url.includes(`${API_ROOT}/usage/summary`)) {
      return jsonResponse(summary(b.today ?? {}, b.series))
    }
    if (url.includes(`${API_ROOT}/overview`)) {
      // The workspace travels in the query, because the gateway is what scopes
      // the counts now. Keyed on it here so a test can show the rail moving.
      const scoped = new URL(url, "http://localhost").searchParams.get(
        "workspace_id",
      )
      return jsonResponse(
        overviewSummary(
          (scoped ? b.overviewByWorkspace?.[scoped] : undefined) ?? b.overview,
        ),
      )
    }
    if (url.includes(`${API_ROOT}/providers/health`)) {
      return jsonResponse(
        b.health ?? { providers: [], healthy: 0, total: 0, checked_at: null },
      )
    }
    if (url.includes(`${API_ROOT}/hosted-providers`)) {
      const hosted = b.hostedProviders ?? []
      return jsonResponse({ data: hosted, count: hosted.length })
    }
    if (url.includes(`${API_ROOT}/budgets`))
      return jsonResponse(b.budgets ?? [])
    if (url.includes(`${API_ROOT}/keys`)) return jsonResponse(b.keys ?? [])
    if (url.includes(`${API_ROOT}/users`)) return jsonResponse(b.users ?? [])
    if (url.includes(`${API_ROOT}/providers`))
      return jsonResponse({
        providers: b.providers ?? [{ provider: "openai" }],
      })
    if (url.includes(`${API_ROOT}/usage`)) return jsonResponse(b.logs ?? [])
    return jsonResponse([])
  })
}

/** Reports where a navigation landed, for the routes `renderPage` mounts. */
function LocationProbe() {
  const loc = useLocation()
  return <div data-testid="loc">{loc.pathname}</div>
}

// The fixture's roster plus traces, so every tab of the Overview is served.
const ALL_SURFACES = [...SURFACES, "traces"]

function renderPage(ui: ReactElement, initial = "/overview") {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <DeploymentProvider value={bootstrap({ surfaces: ALL_SURFACES })}>
        {ui}
      </DeploymentProvider>
    </QueryClientProvider>,
    {
      wrapper: withRouter({
        url: initial,
        routes: [
          { path: "/admin/hosted-providers", element: <LocationProbe /> },
          { path: "/organization/provider-keys", element: <LocationProbe /> },
        ],
      }),
    },
  )
}

/**
 * The same page with a workspace actually selected.
 *
 * `renderPage` deliberately mounts no switcher, so `useSelectedWorkspace`
 * answers NO_WORKSPACE there and every scoped query is disabled. The rail's
 * roster is one of those, so a test about it has to seat the provider, which
 * seeds itself from `workspace_memberships` on the organization context.
 */
function renderPageInWorkspace(ui: ReactElement, selected: string) {
  window.localStorage.setItem("otari.dashboard.selectedWorkspace", selected)
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <DeploymentProvider value={bootstrap()}>
        <SelectedWorkspaceProvider>{ui}</SelectedWorkspaceProvider>
      </DeploymentProvider>
    </QueryClientProvider>,
    {
      wrapper: withRouter({
        url: "/overview",
        routes: [
          { path: "/admin/hosted-providers", element: <LocationProbe /> },
          { path: "/organization/provider-keys", element: <LocationProbe /> },
        ],
      }),
    },
  )
}

const WORKSPACE_A = "44444444-4444-4444-4444-444444444444"
const WORKSPACE_B = "55555555-5555-5555-5555-555555555555"

const TWO_WORKSPACES = {
  workspace_memberships: [
    { workspace_id: WORKSPACE_A, name: "Default Workspace", role: "owner" },
    { workspace_id: WORKSPACE_B, name: "Staging", role: "member" },
  ],
}

/** The operator's provider-health read, which only the operator's strip makes. */
const asksHealth = (requested: string[]) =>
  requested.some((url) => url.includes(`${API_ROOT}/providers/health`))

describe("the Overview's layout", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  const tabs = () =>
    within(screen.getByRole("navigation", { name: "Overview views" }))

  const hrefOf = (name: string) =>
    new URL(
      tabs().getByRole("link", { name }).getAttribute("href") ?? "",
      "http://localhost",
    )

  it("marks the tab the path renders", async () => {
    mockApi({})
    renderPage(<OverviewIndex />, "/requests")
    await screen.findByRole("navigation", { name: "Overview views" })

    expect(tabs().getByRole("link", { name: "Requests" })).toHaveAttribute(
      "aria-current",
      "page",
    )
    expect(tabs().getByRole("link", { name: "Usage" })).not.toHaveAttribute(
      "aria-current",
    )
  })

  it("carries the shared filters to every other tab and drops the rest", async () => {
    mockApi({})
    renderPage(
      <OverviewIndex />,
      "/requests?range=7d&user_id=u1&user_id=u2&status=error&page=3",
    )
    await screen.findByRole("navigation", { name: "Overview views" })

    const sessions = hrefOf("Sessions")
    expect(sessions.pathname).toBe("/sessions")
    expect(sessions.searchParams.get("range")).toBe("7d")
    expect(sessions.searchParams.getAll("user_id")).toEqual(["u1", "u2"])
    expect(sessions.searchParams.get("status")).toBe("error")
    expect(sessions.searchParams.has("page")).toBe(false)
    expect(hrefOf("Usage").pathname).toBe("/")
  })

  it("keeps the window when the tabs default to different ones", async () => {
    // Usage opens on 30 days and Requests on 24 hours, so an unset range has
    // to travel as the one the source tab was showing.
    mockApi({})
    renderPage(<OverviewIndex />, "/")
    await screen.findByRole("navigation", { name: "Overview views" })

    expect(hrefOf("Requests").searchParams.get("range")).toBe("30d")
  })

  it("puts the scope switch ahead of the tabs", async () => {
    mockApi({})
    renderPage(<OverviewIndex />, "/")

    const mine = await screen.findByRole("button", { name: "Mine" })
    const views = screen.getByRole("navigation", { name: "Overview views" })
    expect(
      mine.compareDocumentPosition(views) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy()
  })

  it("puts what needs attention above the tabs", async () => {
    mockApi({
      health: { providers: [], healthy: 2, total: 3, checked_at: null },
    })
    renderPage(<OverviewIndex />, "/sessions")

    const alert = await screen.findByRole("alert")
    const views = screen.getByRole("navigation", { name: "Overview views" })
    expect(
      alert.compareDocumentPosition(views) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy()
  })
})

describe("the Overview's tabs by deployment", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  function renderWith(surfaces: string[], url: string) {
    mockApi({})
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    })
    return render(
      <QueryClientProvider client={client}>
        <DeploymentProvider value={bootstrap({ surfaces })}>
          <OverviewIndex />
        </DeploymentProvider>
      </QueryClientProvider>,
      {
        wrapper: withRouter({
          url,
          routes: [{ path: "/sessions", element: <LocationProbe /> }],
        }),
      },
    )
  }

  it("offers no Sessions tab where traces are not recorded", async () => {
    renderWith(["usage"], "/")

    const tabs = within(
      await screen.findByRole("navigation", { name: "Overview views" }),
    )
    expect(tabs.queryByRole("link", { name: "Sessions" })).toBeNull()
    expect(tabs.getByRole("link", { name: "Requests" })).toBeVisible()
  })

  it("lands a tab it does not serve on Sessions where only traces are recorded", async () => {
    renderWith(["traces"], "/")

    expect(await screen.findByTestId("loc")).toHaveTextContent("/sessions")
  })
})

describe("the Overview on a phone", () => {
  const realMatchMedia = window.matchMedia

  afterEach(() => {
    window.matchMedia = realMatchMedia
    vi.restoreAllMocks()
  })

  const covered = () =>
    screen
      .getByRole("button", { name: "Everyone", hidden: true })
      .closest("[inert]")

  async function renderOnPhone(url: string) {
    window.matchMedia = vi.fn(() => ({
      matches: true,
      media: "",
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
    })) as unknown as typeof window.matchMedia
    mockApi({})
    renderPage(<OverviewIndex />, url)
    await screen.findByRole("button", { name: "Everyone", hidden: true })
  }

  it("takes the controls behind an open request out of reach", async () => {
    await renderOnPhone("/requests?request=r1")

    const region = covered()
    expect(region).not.toBeNull()
    expect(region).toContainElement(
      screen.getByRole("navigation", { name: "Overview views", hidden: true }),
    )
  })

  it("does the same behind an open session", async () => {
    await renderOnPhone("/sessions?peek=t1")

    expect(covered()).not.toBeNull()
  })

  it("leaves the controls alone with no panel open", async () => {
    await renderOnPhone("/requests")

    expect(covered()).toBeNull()
  })
})

describe("the operator's status bands", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("uses a zero-padded, one-based local calendar date as its refresh key", () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date(2026, 0, 5, 12))
    expect(localDayKey()).toBe("2026-01-05")
    vi.useRealTimers()
  })

  it("summarizes provider health and budgets in the attention strip", async () => {
    mockApi({
      health: {
        providers: [],
        healthy: 2,
        total: 3,
        checked_at: "2026-07-22T00:00:00Z",
      },
      overview: { budgets: strip({ over_count: 1 }) },
    })
    renderPage(<OverviewIndex />)

    expect(
      await screen.findByText("1 provider unreachable"),
    ).toBeInTheDocument()
    expect(screen.getByText("1 budget over limit")).toBeInTheDocument()
  })

  it("links each attention problem at the view that explains it", async () => {
    mockApi({
      // 20% errors, past the alert threshold, so the error-rate problem appears
      // alongside the provider one and the strip carries both link shapes.
      today: { request_count: 100, error_count: 20 },
      health: {
        providers: [],
        healthy: 2,
        total: 3,
        checked_at: "2026-07-22T00:00:00Z",
      },
    })
    renderPage(<OverviewIndex />)

    expect(
      await screen.findByRole("link", { name: "1 provider unreachable" }),
    ).toHaveAttribute("href", "/admin/hosted-providers")
    // The one place the strip hands the request log a query, so a link that
    // dropped it would land on an unfiltered log and still look like it worked.
    const errors = new URL(
      screen
        .getByRole("link", { name: "error rate 20.0% today" })
        .getAttribute("href") ?? "",
      "http://localhost",
    )
    expect(errors.pathname).toBe("/requests")
    expect(errors.searchParams.get("status")).toBe("error")
    expect(errors.searchParams.get("start_date")).toBeTruthy()
    expect(errors.searchParams.get("scope")).toBe("everyone")
  })

  it("is absent when nothing needs attention", async () => {
    const fetchMock = mockApi({
      health: { providers: [], healthy: 3, total: 3, checked_at: null },
    })
    renderPage(<OverviewIndex />)

    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some(([input]) =>
          String(input).includes(`${API_ROOT}/providers/health`),
        ),
      ).toBe(true),
    )
    await new Promise((resolve) => setTimeout(resolve, 50))
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
    expect(screen.queryByRole("status")).not.toBeInTheDocument()
  })

  it("says so, without judging, when a status source fails", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input)
      if (url.endsWith(`${API_ROOT}/organizations/me`)) {
        return jsonResponse(organizationContext())
      }
      if (url.includes(`${API_ROOT}/overview`)) {
        return jsonResponse({ detail: "budgets exploded" }, 500)
      }
      if (url.includes(`${API_ROOT}/usage/summary`)) {
        return jsonResponse(summary({ request_count: 100 }))
      }
      if (url.includes(`${API_ROOT}/providers/health`)) {
        return jsonResponse({
          providers: [],
          healthy: 1,
          total: 1,
          checked_at: null,
        })
      }
      if (url.includes(`${API_ROOT}/providers`)) {
        return jsonResponse({ providers: [{ provider: "openai" }] })
      }
      return jsonResponse([])
    })
    renderPage(<OverviewIndex />)

    expect(await screen.findByText(/could not be loaded/)).toBeInTheDocument()
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
  })

  it("waits for its sources rather than showing a transient strip", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) =>
      String(input).endsWith(`${API_ROOT}/organizations/me`)
        ? jsonResponse(organizationContext())
        : new Promise<Response>(() => {}),
    )
    renderPage(<OverviewIndex />)
    expect(
      await screen.findByRole("heading", { name: "Overview" }),
    ).toBeInTheDocument()
    expect(screen.queryByRole("status")).not.toBeInTheDocument()
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
  })

  it("shows a getting-started band and links to providers on a fresh gateway", async () => {
    // The deployment's own credentials: only an operator is sent to Hosted
    // providers, since the tenant band links to the organization's keys.
    mockApi({ providers: [] })
    const user = userEvent.setup()
    renderPage(<OverviewIndex />)

    expect(
      await screen.findByText("Get started with Otari"),
    ).toBeInTheDocument()
    await user.click(
      screen.getByRole("button", { name: "Add your first provider" }),
    )
    expect(await screen.findByTestId("loc")).toHaveTextContent(
      "/admin/hosted-providers",
    )
  })

  it("counts a hosted provider as a configured one", async () => {
    // Hosted providers are not in GET /providers, which lists config.yml
    // instances only, so a gateway whose one provider is hosted is not empty.
    const fetchMock = mockApi({
      providers: [],
      hostedProviders: [{ name: "openai" }],
    })
    renderPage(<OverviewIndex />)

    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some(([input]) =>
          String(input).includes(`${API_ROOT}/hosted-providers`),
        ),
      ).toBe(true),
    )
    await new Promise((resolve) => setTimeout(resolve, 50))
    expect(screen.queryByText("Get started with Otari")).toBeNull()
  })

  it("is not a fresh gateway when it already has usage", async () => {
    // Imported OTLP usage needs no provider config.
    mockApi({ providers: [], logs: [{ id: "1" }] })
    renderPage(<OverviewIndex />)

    await screen.findByRole("navigation", { name: "Overview views" })
    await new Promise((resolve) => setTimeout(resolve, 50))
    expect(screen.queryByText("Get started with Otari")).toBeNull()
  })

  it("reports a failed provider query instead of calling the gateway fresh", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input)
      if (url.endsWith(`${API_ROOT}/organizations/me`)) {
        return jsonResponse(organizationContext())
      }
      if (url.includes(`${API_ROOT}/providers/health`)) {
        return jsonResponse({
          providers: [],
          healthy: 1,
          total: 1,
          checked_at: null,
        })
      }
      if (url.includes(`${API_ROOT}/usage/summary`))
        return jsonResponse(summary({}))
      if (url.includes(`${API_ROOT}/providers`))
        return jsonResponse({ detail: "providers exploded" }, 500)
      return jsonResponse([])
    })
    renderPage(<OverviewIndex />)

    expect(await screen.findByText(/providers exploded/)).toBeInTheDocument()
    expect(screen.queryByText("Get started with Otari")).not.toBeInTheDocument()
  })
})

describe("OverviewIndex operator-ness", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  function mockOperator(context: "ok" | "fails") {
    const requested: string[] = []
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input)
      requested.push(url)
      if (url.endsWith(`${API_ROOT}/organizations/me`)) {
        return context === "ok"
          ? jsonResponse(organizationContext())
          : jsonResponse({ detail: "context exploded" }, 500)
      }
      if (url.includes(`${API_ROOT}/usage/summary`)) {
        return jsonResponse(summary({ cost: 200, request_count: 2000 }))
      }
      if (url.includes(`${API_ROOT}/providers/health`)) {
        return jsonResponse({
          providers: [],
          healthy: 1,
          total: 1,
          checked_at: null,
        })
      }
      if (url.includes(`${API_ROOT}/providers`)) {
        return jsonResponse({ providers: [{ provider: "openai" }] })
      }
      return jsonResponse([])
    })
    return requested
  }

  it("shows the operator's strip off the context, without asking /v1/admin/access", async () => {
    const requested = mockOperator("ok")
    renderPage(<OverviewIndex />)

    await waitFor(() => expect(asksHealth(requested)).toBe(true))
    expect(
      requested.some((url) => url.endsWith(`${API_ROOT}/admin/access`)),
    ).toBe(false)
  })

  it("lands a failed context on the operator's strip, where its usage reads too", async () => {
    // Every operator gate fails open on a failed read (otari#876), and the page
    // and the usage hooks below it take that answer from one place, so the
    // operator panels never sit above views serving the caller's organization.
    const requested = mockOperator("fails")
    renderPage(<OverviewIndex />)

    await waitFor(() => expect(asksHealth(requested)).toBe(true))
    expect(
      requested.some((url) =>
        url.includes(`${API_ROOT}/organizations/me/usage`),
      ),
    ).toBe(false)
    // And it settled there: a page reading the context's transient pending
    // state would flip back to its spinner and ask again without end.
    expect(
      requested.filter((url) => url.endsWith(`${API_ROOT}/organizations/me`))
        .length,
    ).toBeLessThan(4)
  })
})

// The wire for a caller who does not operate the deployment: the organization
// context says so, and the usage hooks therefore read
// /api/v1/organizations/me/usage (otari#837). Everything else is an endpoint this
// caller must never be asked to read, so it refuses the way the server would; a
// leak fails the assertions loudly instead of rendering a plausible tile. Every
// URL is recorded so the tests can say so, /api/v1/admin/access included: this
// page decides from the context alone, so asking it at all is the bug
// otari-ai#1936 is about.
function mockScopedApi(b: Bodies): string[] {
  const requested: string[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input)
    requested.push(url)
    if (url.endsWith(`${API_ROOT}/organizations/me`)) {
      return jsonResponse(
        organizationContext({ deployment_operator: false, ...b.context }),
      )
    }
    // The caller's usage identity, which a member's attention strip is judged on.
    if (url.includes(`${API_ROOT}/organizations/me/members`)) {
      const member = organizationMember({ attribution_user_id: "me-id" })
      return jsonResponse({ data: [member], total: 1 })
    }
    if (url.includes(`${API_ROOT}/organizations/me/usage/summary`)) {
      return jsonResponse(summary(b.today ?? {}, b.series))
    }
    if (url.includes(`${API_ROOT}/organizations/me/usage`)) {
      return jsonResponse(b.logs ?? [])
    }
    // The one tenant surface the rest of the page reads. The ceiling strip and
    // both rail counts come from here now, scoped and judged by the gateway,
    // which is also what withholds the ceilings from a plain member.
    if (url.includes(`${API_ROOT}/overview`)) {
      const scoped = new URL(url, "http://localhost").searchParams.get(
        "workspace_id",
      )
      return jsonResponse(
        overviewSummary(
          (scoped ? b.overviewByWorkspace?.[scoped] : undefined) ?? b.overview,
        ),
      )
    }
    // The two reads the setup guide adds to this page. Both are open to any
    // signed-in caller: the catalog is scoped to the caller's own providers
    // rather than operator-gated, and the activation read answers every member
    // who can see the workspace, reporting per caller whether the guide is on
    // offer.
    if (url.includes("/activation")) {
      return jsonResponse(b.activation ?? workspaceActivation())
    }
    if (url.includes(`${API_ROOT}/models`)) {
      if (b.modelsFail) return jsonResponse({ detail: "catalog down" }, 500)
      return jsonResponse(modelCatalog(b.models ?? ["openai:gpt-4o-mini"]))
    }
    return jsonResponse({ detail: "forbidden" }, 403)
  })
  return requested
}

describe("the Overview for a caller who does not operate the deployment", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("reads only the organization-scoped surface", async () => {
    const requested = mockScopedApi({
      today: { cost: 200, request_count: 2000 },
    })
    renderPage(<OverviewIndex />)
    await screen.findByRole("navigation", { name: "Overview views" })
    await waitFor(() =>
      expect(
        requested.some((url) => url.includes("/organizations/me/usage")),
      ).toBe(true),
    )

    // Nothing left the scoped surface: beyond the context, every request this
    // page made is under /v1/organizations/me. A bare /v1/usage, /v1/budgets or
    // /v1/providers read here would be a deployment-wide one the server
    // refuses, so the page must not even attempt it, and neither must it ask
    // the gate. /v1/overview and the catalog are scoped by the identity asking
    // rather than by their prefix.
    expect(
      requested.some((url) => url.endsWith(`${API_ROOT}/admin/access`)),
    ).toBe(false)
    expect(asksHealth(requested)).toBe(false)
    const scoped = requested.filter(
      (url) =>
        !url.endsWith(`${API_ROOT}/organizations/me`) &&
        !url.includes(`${API_ROOT}/overview`) &&
        !url.includes(`${API_ROOT}/models`),
    )
    expect(scoped.length).toBeGreaterThan(0)
    for (const url of scoped) {
      expect(url).toContain(`${API_ROOT}/organizations/me/`)
    }
  })

  it("shows no deployment-wide panel", async () => {
    mockScopedApi({ today: { cost: 200, request_count: 2000 } })
    renderPage(<OverviewIndex />)
    await screen.findByRole("navigation", { name: "Overview views" })
    await new Promise((resolve) => setTimeout(resolve, 50))

    // Provider health and the deployment's budgets are operator surfaces, so
    // nothing here reports on them or on failing to read them (otari-ai#2085).
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
    expect(
      screen.queryByText("Some status data could not be loaded."),
    ).not.toBeInTheDocument()
  })

  it("reports a failed error-rate read instead of judging it", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input)
      if (url.endsWith(`${API_ROOT}/organizations/me`)) {
        return jsonResponse(organizationContext({ deployment_operator: false }))
      }
      if (url.includes(`${API_ROOT}/organizations/me/usage/summary`)) {
        return jsonResponse({ detail: "usage exploded" }, 500)
      }
      if (url.includes(`${API_ROOT}/organizations/me/usage`)) {
        return jsonResponse([])
      }
      return jsonResponse({ detail: "forbidden" }, 403)
    })
    renderPage(<OverviewIndex />)

    expect(
      await screen.findByText("Some status data could not be loaded."),
    ).toBeInTheDocument()
  })
})

describe("the tenant Overview's attention strip", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  const overCeiling = strip({
    over_count: 1,
    worst: {
      budget_id: "eeeeeeee-1111-2222-3333-444444444444",
      name: "Org cap",
      spent: 300,
      allocated: 250,
      scope_type: null,
      scope_id: null,
    },
  })

  it("shows a member only the error rate, linked at today's failed requests", async () => {
    mockScopedApi({
      context: { role: "member" },
      today: { request_count: 100, error_count: 20 },
      overview: { ceilings: overCeiling },
    })
    renderPage(<OverviewIndex />)

    const link = await screen.findByRole("link", { name: /^error rate 20/ })
    expect(link).toHaveAttribute(
      "href",
      expect.stringMatching(/^\/requests\?status=error&start_date=/),
    )
    expect(screen.getByRole("alert")).toBeInTheDocument()
    expect(screen.queryByText(/spend ceiling/)).not.toBeInTheDocument()
  })

  it("adds a ceiling at its limit for an owner or admin", async () => {
    mockScopedApi({
      context: { role: "admin" },
      today: { request_count: 100, error_count: 20 },
      overview: { ceilings: overCeiling },
    })
    renderPage(<OverviewIndex />)

    expect(
      await screen.findByRole("link", { name: "1 spend ceiling over limit" }),
    ).toHaveAttribute("href", "/budgets")
    expect(
      screen.getByRole("link", { name: /^error rate/ }),
    ).toBeInTheDocument()
  })

  it("names a ceiling near its limit", async () => {
    mockScopedApi({
      context: { role: "admin" },
      overview: { ceilings: strip({ near_count: 2, capped_count: 2 }) },
    })
    renderPage(<OverviewIndex />)

    expect(
      await screen.findByRole("link", { name: "2 spend ceilings near limit" }),
    ).toBeInTheDocument()
  })

  it("is absent when nothing needs attention", async () => {
    mockScopedApi({
      context: { role: "admin" },
      today: { request_count: 100, error_count: 1 },
      overview: { ceilings: strip() },
    })
    renderPage(<OverviewIndex />)

    await screen.findByRole("navigation", { name: "Overview views" })
    await new Promise((resolve) => setTimeout(resolve, 50))
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
  })
})

// Who may be offered the first-request guide is the server's answer:
// `WorkspaceActivationService._is_eligible` ends in
// `has_workspace_management_access` and is reported as `experience_eligible` per
// caller. Operating the deployment is no part of that answer, so the guide
// belongs on whichever Overview the caller landed on (otari-ai#2080).
describe("the setup guide on either Overview", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("offers it to a caller who does not operate the deployment", async () => {
    const requested = mockScopedApi({ context: TWO_WORKSPACES })
    renderPageInWorkspace(<OverviewIndex />, WORKSPACE_A)

    expect(
      await screen.findByRole("heading", { name: "Send your first request" }),
    ).toBeInTheDocument()
    // And its gate came from the catalog, which this caller may read, and not
    // from the operator-gated provider list.
    expect(requested.some((url) => url.includes(`${API_ROOT}/models`))).toBe(
      true,
    )
    expect(requested.some((url) => url.endsWith(`${API_ROOT}/providers`))).toBe(
      false,
    )
  })

  it("still offers it to an operator", async () => {
    mockApi({ context: TWO_WORKSPACES })
    renderPageInWorkspace(<OverviewIndex />, WORKSPACE_A)

    expect(
      await screen.findByRole("heading", { name: "Send your first request" }),
    ).toBeInTheDocument()
  })

  it("shows nothing to a caller the server reports ineligible", async () => {
    const requested = mockScopedApi({
      context: TWO_WORKSPACES,
      activation: workspaceActivation({ experience_eligible: false }),
    })
    renderPageInWorkspace(<OverviewIndex />, WORKSPACE_A)

    // Anchored on the answer this test varies. Without it the negative passes on
    // its first tick, before the activation read has been made at all, and would
    // read the same for an eligible caller.
    await waitFor(() => {
      expect(requested.some((url) => url.includes("/activation"))).toBe(true)
    })
    await new Promise((resolve) => setTimeout(resolve, 50))
    expect(
      screen.queryByRole("heading", { name: "Send your first request" }),
    ).not.toBeInTheDocument()
  })

  it("holds back, without asking, while the caller can route nowhere", async () => {
    // An empty catalog is the tenant's form of "no provider configured": a key
    // handed out here would be for a call that must fail, so the guide is not
    // offered and the activation read is never made.
    const requested = mockScopedApi({ context: TWO_WORKSPACES, models: [] })
    renderPageInWorkspace(<OverviewIndex />, WORKSPACE_A)

    // Settled, not merely issued: the band the empty catalog decides is the
    // answer having landed rather than being on its way.
    expect(
      await screen.findByText("Get started with Otari"),
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("heading", { name: "Send your first request" }),
    ).not.toBeInTheDocument()
    expect(requested.some((url) => url.includes("/activation"))).toBe(false)
  })
})

describe("the first-steps strip on the tenant Overview", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  const GET_STARTED = "Get started with Otari"

  it("points an admin of a brand-new organization at adding a provider key", async () => {
    mockScopedApi({ context: TWO_WORKSPACES, models: [] })
    const user = userEvent.setup()
    renderPageInWorkspace(<OverviewIndex />, WORKSPACE_A)

    expect(await screen.findByText(GET_STARTED)).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Add a provider key" }))
    expect(await screen.findByTestId("loc")).toHaveTextContent(
      "/organization/provider-keys",
    )
  })

  it("tells a member who to ask, and offers no button they cannot use", async () => {
    mockScopedApi({
      context: { ...TWO_WORKSPACES, role: "member" },
      models: [],
    })
    renderPageInWorkspace(<OverviewIndex />, WORKSPACE_A)

    expect(await screen.findByText(GET_STARTED)).toBeInTheDocument()
    expect(screen.getByText(/Ask an owner or admin/)).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Add a provider key" }),
    ).not.toBeInTheDocument()
  })

  it("is absent once there is something to route to, where the guide takes over", async () => {
    mockScopedApi({ context: TWO_WORKSPACES })
    renderPageInWorkspace(<OverviewIndex />, WORKSPACE_A)

    await screen.findByRole("heading", { name: "Send your first request" })
    expect(screen.queryByText(GET_STARTED)).not.toBeInTheDocument()
  })

  it("is absent when the account already has traffic", async () => {
    mockScopedApi({
      context: TWO_WORKSPACES,
      models: [],
      logs: [
        {
          id: "1",
          user_id: null,
          api_key_id: null,
          timestamp: "2026-07-22T00:00:00Z",
          model: "gpt-5.6",
          provider: "openai",
          endpoint: "/v1/chat/completions",
          prompt_tokens: 10,
          completion_tokens: 5,
          total_tokens: 15,
          cache_read_tokens: 0,
          cache_write_tokens: 0,
          cost: 0.5,
          status: "success",
          error_message: null,
          latency_ms: 120,
        },
      ],
    })
    renderPageInWorkspace(<OverviewIndex />, WORKSPACE_A)

    await screen.findByRole("navigation", { name: "Overview views" })
    await new Promise((resolve) => setTimeout(resolve, 50))
    expect(screen.queryByText(GET_STARTED)).not.toBeInTheDocument()
  })

  it("stays out of the way when the catalog cannot be read", async () => {
    const requested = mockScopedApi({
      context: TWO_WORKSPACES,
      modelsFail: true,
    })
    renderPageInWorkspace(<OverviewIndex />, WORKSPACE_A)

    await waitFor(() => {
      expect(requested.some((url) => url.includes(`${API_ROOT}/models`))).toBe(
        true,
      )
    })
    await new Promise((resolve) => setTimeout(resolve, 50))
    expect(screen.queryByText(GET_STARTED)).not.toBeInTheDocument()
  })
})

describe("OverviewIndex for an identity with no organization", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  function mockWithoutOrganization(
    invitations: ReturnType<typeof pendingOrganizationInvitation>[] | "fails",
  ) {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input)
      if (url.endsWith(`${API_ROOT}/organizations/me`)) {
        return jsonResponse({ detail: "Organization not found" }, 404)
      }
      if (url.includes(`${API_ROOT}/organizations/me/pending-memberships`)) {
        if (invitations === "fails") {
          // A 404 is the one failure the hook does not retry, which keeps this fast.
          return jsonResponse({ detail: "Not Found" }, 404)
        }
        return jsonResponse({ data: invitations, count: invitations.length })
      }
      if (url.includes(`${API_ROOT}/usage/summary`)) {
        return jsonResponse(summary({}))
      }
      if (url.includes(`${API_ROOT}/providers/health`)) {
        return jsonResponse({
          providers: [],
          healthy: 1,
          total: 1,
          checked_at: null,
        })
      }
      if (url.includes(`${API_ROOT}/providers`)) {
        return jsonResponse({ providers: [{ provider: "openai" }] })
      }
      return jsonResponse([])
    })
  }

  function renderHome({
    openSignup = false,
    client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    }),
  } = {}) {
    return render(
      <QueryClientProvider client={client}>
        <DeploymentProvider value={bootstrap({ open_signup: openSignup })}>
          <OverviewIndex />
        </DeploymentProvider>
      </QueryClientProvider>,
      {
        wrapper: withRouter({
          url: "/",
          routes: [{ path: "/invitations", element: <LocationProbe /> }],
        }),
      },
    )
  }

  const EMPTY_STATE = "You're not a member of an organization yet"

  it("sends an invitee who signed up before accepting to their invitations", async () => {
    mockWithoutOrganization([pendingOrganizationInvitation()])

    renderHome()

    expect(await screen.findByTestId("loc")).toHaveTextContent("/invitations")
  })

  it("offers the invitations page when they cannot be loaded", async () => {
    mockWithoutOrganization("fails")

    renderHome()

    expect(await screen.findByRole("alert")).toBeInTheDocument()
    expect(
      screen.getByRole("link", { name: "Open your invitations" }),
    ).toHaveAttribute("href", "/invitations")
    expect(
      screen.queryByRole("navigation", { name: "Overview views" }),
    ).not.toBeInTheDocument()
  })

  it("says how to get in when nothing is waiting, rather than the operator page", async () => {
    mockWithoutOrganization([])

    renderHome()

    expect(await screen.findByText(EMPTY_STATE)).toBeInTheDocument()
    // The deployment-wide page fails open on an unreadable context, and a 404
    // is not one: it is the server saying there is no organization.
    expect(
      screen.queryByRole("navigation", { name: "Overview views" }),
    ).toBeNull()
    expect(
      screen.getByRole("link", { name: /View invitations/ }),
    ).toHaveAttribute("href", "/invitations")
    // Signup is closed, so the server would refuse to create one.
    expect(
      screen.queryByRole("button", { name: "Create organization" }),
    ).toBeNull()
  })

  it("offers to create an organization where signup is open", async () => {
    mockWithoutOrganization([])

    renderHome({ openSignup: true })

    expect(await screen.findByText(EMPTY_STATE)).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Create organization" }),
    ).toBeInTheDocument()
  })

  it("does not retry the 404 that says there is no organization", async () => {
    mockWithoutOrganization([])
    // The app's own default, which would retry a 404 twice.
    const client = new QueryClient({
      defaultOptions: { queries: { retry: defaultQueryRetry } },
    })

    renderHome({ client })

    expect(await screen.findByText(EMPTY_STATE)).toBeInTheDocument()
    const contextReads = vi
      .mocked(globalThis.fetch)
      .mock.calls.filter(([input]) =>
        String(input).endsWith(`${API_ROOT}/organizations/me`),
      )
    expect(contextReads).toHaveLength(1)
  })
})
