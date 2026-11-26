import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { useLocation } from "@tanstack/react-router"
import { render, screen, waitFor, within } from "@testing-library/react"
import type { ReactElement } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { OverviewIndex } from "@/features/overview/OverviewPage"
import { API_ROOT, defaultQueryRetry } from "@/shared/api/client"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  bootstrap,
  organizationContext,
  pendingOrganizationInvitation,
  SURFACES,
  workspaceActivation,
} from "@/tests/fixtures"
import { withRouter } from "@/tests/router"

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

interface Bodies {
  /** What GET /hosted-providers lists. */
  hostedProviders?: unknown[]
  providers?: unknown[]
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

function mockApi(b: Bodies = {}) {
  const requested: string[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input)
    requested.push(url)
    // Answered first, and on an exact match, so it cannot shadow
    // /v1/organizations/me/usage.
    if (url.endsWith(`${API_ROOT}/organizations/me`)) {
      return jsonResponse(organizationContext(b.context))
    }
    if (url.includes("/activation")) {
      return jsonResponse(b.activation ?? workspaceActivation())
    }
    if (url.includes(`${API_ROOT}/models`)) {
      if (b.modelsFail) return jsonResponse({ detail: "catalog down" }, 500)
      return jsonResponse(modelCatalog(b.models ?? ["openai:gpt-4o-mini"]))
    }
    if (url.includes(`${API_ROOT}/hosted-providers`)) {
      const hosted = b.hostedProviders ?? []
      return jsonResponse({ data: hosted, count: hosted.length })
    }
    if (url.endsWith(`${API_ROOT}/providers`)) {
      return jsonResponse({
        providers: b.providers ?? [{ provider: "openai" }],
      })
    }
    return jsonResponse([])
  })
  return requested
}

/** The wire for a caller who does not operate the deployment. */
function mockScopedApi(b: Bodies = {}) {
  return mockApi({
    ...b,
    context: { deployment_operator: false, ...b.context },
  })
}

/** Reports where a navigation landed, for the routes `renderPage` mounts. */
function LocationProbe() {
  const loc = useLocation()
  return (
    <>
      <div data-testid="loc">{loc.pathname}</div>
      <div data-testid="search">{loc.searchStr}</div>
    </>
  )
}

// The fixture's roster plus traces, so every tab of Sessions is served.
const ALL_SURFACES = [...SURFACES, "traces"]

function renderPage(ui: ReactElement, initial = "/sessions") {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <DeploymentProvider value={bootstrap({ surfaces: ALL_SURFACES })}>
        {ui}
      </DeploymentProvider>
    </QueryClientProvider>,
    { wrapper: withRouter({ url: initial }) },
  )
}

/**
 * The same page with a workspace actually selected.
 *
 * `renderPage` deliberately mounts no switcher, so `useSelectedWorkspace`
 * answers NO_WORKSPACE there and every scoped query is disabled. The guide's
 * catalog read is one of those, so a test about it has to seat the provider,
 * which seeds itself from `workspace_memberships` on the organization context.
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
    { wrapper: withRouter({ url: "/requests" }) },
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

const SESSIONS_VIEWS = { name: "Sessions views" }

describe("the Sessions layout", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  const tabs = () => within(screen.getByRole("navigation", SESSIONS_VIEWS))

  const hrefOf = (name: string) =>
    new URL(
      tabs().getByRole("link", { name }).getAttribute("href") ?? "",
      "http://localhost",
    )

  it("titles the page Sessions over its views", async () => {
    mockApi()
    renderPage(<OverviewIndex />, "/requests")

    expect(
      await screen.findByRole("heading", { name: "Sessions" }),
    ).toBeInTheDocument()
    expect(
      tabs()
        .getAllByRole("link")
        .map((link) => link.textContent),
    ).toEqual(["Sessions", "Requests", "Router savings"])
  })

  it("opens Router savings as a view with controls of its own", async () => {
    mockApi()
    renderPage(<OverviewIndex />, "/router-savings")

    expect(
      await screen.findByRole("heading", { name: "Sessions" }),
    ).toBeInTheDocument()
    expect(
      tabs().getByRole("link", { name: "Router savings" }),
    ).toHaveAttribute("aria-current", "page")
    // The tab carries no session filters across, in either direction.
    expect(hrefOf("Router savings").search).toBe("")
    expect(hrefOf("Requests").search).toBe("")
  })

  it("leaves Router savings out where the deployment does not serve it", async () => {
    mockApi()
    render(
      <QueryClientProvider
        client={
          new QueryClient({ defaultOptions: { queries: { retry: false } } })
        }
      >
        <DeploymentProvider
          value={bootstrap({
            surfaces: ALL_SURFACES.filter(
              (surface) => surface !== "subagent_savings",
            ),
          })}
        >
          <OverviewIndex />
        </DeploymentProvider>
      </QueryClientProvider>,
      { wrapper: withRouter({ url: "/sessions" }) },
    )

    await screen.findByRole("navigation", SESSIONS_VIEWS)
    expect(
      tabs().queryByRole("link", { name: "Router savings" }),
    ).not.toBeInTheDocument()
  })

  it("marks the tab the path renders", async () => {
    mockApi()
    renderPage(<OverviewIndex />, "/requests")
    await screen.findByRole("navigation", SESSIONS_VIEWS)

    expect(tabs().getByRole("link", { name: "Requests" })).toHaveAttribute(
      "aria-current",
      "page",
    )
    expect(tabs().getByRole("link", { name: "Sessions" })).not.toHaveAttribute(
      "aria-current",
    )
  })

  it("carries the shared filters to the other tab and drops the rest", async () => {
    mockApi()
    renderPage(
      <OverviewIndex />,
      "/requests?range=7d&user_id=u1&user_id=u2&status=error&page=3",
    )
    await screen.findByRole("navigation", SESSIONS_VIEWS)

    const sessions = hrefOf("Sessions")
    expect(sessions.pathname).toBe("/sessions")
    expect(sessions.searchParams.get("range")).toBe("7d")
    expect(sessions.searchParams.getAll("user_id")).toEqual(["u1", "u2"])
    expect(sessions.searchParams.get("status")).toBe("error")
    expect(sessions.searchParams.has("page")).toBe(false)
  })

  it("puts the tabs ahead of the filters", async () => {
    mockApi()
    renderPage(<OverviewIndex />, "/sessions")

    const views = await screen.findByRole("navigation", SESSIONS_VIEWS)
    expect(
      views.compareDocumentPosition(screen.getByText("Time range")) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy()
  })

  it("leaves the Sessions title and tabs off Home", async () => {
    mockApi()
    renderPage(<OverviewIndex />, "/")

    await waitFor(() =>
      expect(globalThis.fetch).toHaveBeenCalledWith(
        expect.stringContaining(`${API_ROOT}/organizations/me`),
        expect.anything(),
      ),
    )
    await new Promise((resolve) => setTimeout(resolve, 50))
    expect(screen.queryByRole("navigation", SESSIONS_VIEWS)).toBeNull()
    expect(screen.queryByRole("heading", { name: "Sessions" })).toBeNull()
  })
})

describe("the Sessions tabs by deployment", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  function renderWith(surfaces: string[], url: string, target: string) {
    mockApi()
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
          routes: [{ path: target, element: <LocationProbe /> }],
        }),
      },
    )
  }

  it("lands Requests on Sessions where only traces are recorded", async () => {
    renderWith(["traces"], "/requests", "/sessions")

    expect(await screen.findByTestId("loc")).toHaveTextContent("/sessions")
  })

  it("lands Sessions on Requests where only usage is recorded", async () => {
    renderWith(["usage"], "/sessions", "/requests")

    expect(await screen.findByTestId("loc")).toHaveTextContent("/requests")
  })

  it("lands Router savings on the first view's own defaults where it is not served", async () => {
    renderWith(["usage"], "/router-savings?period=30d&who=me", "/requests")

    expect(await screen.findByTestId("loc")).toHaveTextContent("/requests")
    // Usage's 30-day default is not pinned on a view that opens on 24 hours.
    expect(screen.getByTestId("search")).toBeEmptyDOMElement()
  })
})

describe("Sessions on a phone", () => {
  const realMatchMedia = window.matchMedia

  afterEach(() => {
    window.matchMedia = realMatchMedia
    vi.restoreAllMocks()
  })

  const covered = () =>
    screen
      .getByRole("navigation", { ...SESSIONS_VIEWS, hidden: true })
      .closest("[inert]")

  async function renderOnPhone(url: string) {
    window.matchMedia = vi.fn(() => ({
      matches: true,
      media: "",
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
    })) as unknown as typeof window.matchMedia
    mockApi()
    renderPage(<OverviewIndex />, url)
    await screen.findByRole("navigation", { ...SESSIONS_VIEWS, hidden: true })
  }

  it("takes the controls behind an open request out of reach", async () => {
    await renderOnPhone("/requests?request=r1")

    const region = covered()
    expect(region).not.toBeNull()
    expect(region).toContainElement(
      screen.getByRole("navigation", { ...SESSIONS_VIEWS, hidden: true }),
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

describe("OverviewIndex operator-ness", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("reads the operator's provider list off the context, without asking /v1/admin/access", async () => {
    const requested = mockApi()
    renderPage(<OverviewIndex />)

    await waitFor(() =>
      expect(
        requested.some((url) => url.endsWith(`${API_ROOT}/providers`)),
      ).toBe(true),
    )
    expect(
      requested.some((url) => url.endsWith(`${API_ROOT}/admin/access`)),
    ).toBe(false)
  })

  it("lands a failed context on the operator's reads, and settles there", async () => {
    // Every operator gate fails open on a failed read (otari#876), so the
    // guide takes the operator's gate rather than the catalog's.
    const requested: string[] = []
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input)
      requested.push(url)
      if (url.endsWith(`${API_ROOT}/organizations/me`)) {
        return jsonResponse({ detail: "context exploded" }, 500)
      }
      if (url.endsWith(`${API_ROOT}/providers`)) {
        return jsonResponse({ providers: [{ provider: "openai" }] })
      }
      return jsonResponse([])
    })
    renderPage(<OverviewIndex />)

    await waitFor(() =>
      expect(
        requested.some((url) => url.endsWith(`${API_ROOT}/providers`)),
      ).toBe(true),
    )
    // And it settled there: a page reading the context's transient pending
    // state would flip back to its spinner and ask again without end.
    expect(
      requested.filter((url) => url.endsWith(`${API_ROOT}/organizations/me`))
        .length,
    ).toBeLessThan(4)
  })
})

// Who may be offered the first-request guide is the server's answer:
// `WorkspaceActivationService._is_eligible` ends in
// `has_workspace_management_access` and is reported as `experience_eligible` per
// caller. Operating the deployment is no part of that answer, so the guide
// belongs over Home and Sessions whichever caller landed (otari-ai#2080).
describe("the setup guide over Home and Sessions", () => {
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

    // Settled, not merely issued: the catalog having answered empty.
    await waitFor(() => {
      expect(requested.some((url) => url.includes(`${API_ROOT}/models`))).toBe(
        true,
      )
    })
    await new Promise((resolve) => setTimeout(resolve, 50))
    expect(
      screen.queryByRole("heading", { name: "Send your first request" }),
    ).not.toBeInTheDocument()
    expect(requested.some((url) => url.includes("/activation"))).toBe(false)
  })

  it("holds back for an operator with no provider, configured or hosted", async () => {
    const requested = mockApi({
      context: TWO_WORKSPACES,
      providers: [],
      hostedProviders: [],
    })
    renderPageInWorkspace(<OverviewIndex />, WORKSPACE_A)

    await waitFor(() => {
      expect(
        requested.some((url) => url.includes(`${API_ROOT}/hosted-providers`)),
      ).toBe(true)
    })
    await new Promise((resolve) => setTimeout(resolve, 50))
    expect(
      screen.queryByRole("heading", { name: "Send your first request" }),
    ).not.toBeInTheDocument()
    expect(requested.some((url) => url.includes("/activation"))).toBe(false)
  })

  it("counts a hosted provider as one an operator can serve from", async () => {
    // Hosted providers are not in GET /providers, which lists config.yml
    // instances only, so a gateway whose one provider is hosted is not empty.
    mockApi({
      context: TWO_WORKSPACES,
      providers: [],
      hostedProviders: [{ name: "openai" }],
    })
    renderPageInWorkspace(<OverviewIndex />, WORKSPACE_A)

    expect(
      await screen.findByRole("heading", { name: "Send your first request" }),
    ).toBeInTheDocument()
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
    expect(
      screen.queryByRole("heading", { name: "Send your first request" }),
    ).not.toBeInTheDocument()
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
      screen.queryByRole("navigation", { name: "Sessions views" }),
    ).not.toBeInTheDocument()
  })

  it("says how to get in when nothing is waiting, rather than Home", async () => {
    mockWithoutOrganization([])

    renderHome()

    expect(await screen.findByText(EMPTY_STATE)).toBeInTheDocument()
    expect(screen.getByRole("heading", { name: "Home" })).toBeInTheDocument()
    // The deployment-wide page fails open on an unreadable context, and a 404
    // is not one: it is the server saying there is no organization.
    expect(
      screen.queryByRole("navigation", { name: "Sessions views" }),
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
