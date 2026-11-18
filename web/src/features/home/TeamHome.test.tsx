import { Outlet, useLocation } from "@tanstack/react-router"
import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
import { TeamHome } from "@/features/home/TeamHome"
import { API_ROOT } from "@/shared/api/client"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  bootstrap,
  organizationContext,
  organizationMember,
  SURFACES,
  usageTotals,
} from "@/tests/fixtures"
import { AppProviders as Provider } from "@/tests/providers"
import { renderWithRouter } from "@/tests/router"
import { findRowMenuItem } from "@/tests/rowMenu"
import { selectTrigger } from "@/tests/select"

const ADA = organizationMember({
  organization_member_id: "m-ada",
  user_id: "u-ada",
  attribution_user_id: "u-ada",
  full_name: "Ada Lovelace",
})
const WEI = organizationMember({
  organization_member_id: "m-wei",
  user_id: "u-wei",
  attribution_user_id: "u-wei",
  full_name: "Wei Zhang",
})

describe("Home, Where it went", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("shows no columns until the roster says which cut it opens on", async () => {
    let answerRoster: ((response: Response) => void) | undefined
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const path = String(input).slice(API_ROOT.length).split("?")[0]
      if (path === "/organizations/me") {
        return Response.json(
          organizationContext({ role: "owner", deployment_operator: false }),
        )
      }
      if (path === "/organizations/me/members") {
        return new Promise<Response>((resolve) => {
          answerRoster = resolve
        })
      }
      // Every other read is still on its way.
      return new Promise<Response>(() => {})
    })
    await renderWithRouter(<TeamHome />, {
      shell: (
        <Provider>
          <DeploymentProvider value={bootstrap()}>
            <SelectedWorkspaceProvider>
              <Outlet />
            </SelectedWorkspaceProvider>
          </DeploymentProvider>
        </Provider>
      ),
    })

    const section = await screen.findByRole("region", {
      name: "Where it went",
    })
    await waitFor(() => expect(answerRoster).toBeDefined())
    expect(within(section).getByText("Loading…")).toBeVisible()
    expect(within(section).queryAllByRole("columnheader")).toHaveLength(0)

    answerRoster?.(Response.json({ data: [ADA, WEI], count: 2 }))
    const table = await within(section).findByRole("grid", {
      name: "Spend by people",
    })
    expect(
      within(table).getByRole("columnheader", { name: "Person" }),
    ).toBeVisible()
  })
})

describe("Home, People active", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  const placed = (id: string, workspaceId: string, name: string) =>
    organizationMember({
      organization_member_id: `m-${id}`,
      user_id: `u-${id}`,
      attribution_user_id: `u-${id}`,
      full_name: name,
      workspaces: [
        {
          workspace_id: workspaceId,
          workspace_member_id: `wm-${id}`,
          workspace_name: workspaceId,
          role: "member",
        },
      ],
    })

  /** Every read answered; returns the trace count requests it saw. */
  function mockApi() {
    const counts: URL[] = []
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = new URL(String(input), "http://test")
      const path = url.pathname.replace(
        new URL(API_ROOT, "http://test").pathname,
        "",
      )
      if (path === "/organizations/me") {
        return Response.json(
          organizationContext({
            role: "owner",
            deployment_operator: false,
            // The sidebar has a workspace selected, which the page's own
            // "All workspaces" must not narrow to.
            workspace_memberships: [
              { workspace_id: "ws-a", name: "Platform", role: "owner" },
            ],
          }),
        )
      }
      if (path === "/organizations/me/members") {
        return Response.json({
          data: [
            placed("ada", "ws-a", "Ada Lovelace"),
            placed("lin", "ws-a", "Lin Chen"),
            placed("wei", "ws-b", "Wei Zhang"),
          ],
          count: 3,
        })
      }
      if (path === "/organizations/me/usage/summary") {
        return Response.json({
          totals: usageTotals({ cost: 0, request_count: 0 }),
          series: [],
          by_user: [],
          by_api_key: [],
          by_model: [],
          errors_by_status_code: [],
        })
      }
      if (path.endsWith("/traces/count")) {
        counts.push(url)
        return Response.json({ count: 3 })
      }
      if (path === "/organizations/me/spend-ceilings") {
        return Response.json({ data: [], count: 0 })
      }
      if (path === "/workspaces") return Response.json({ data: [], count: 0 })
      return Response.json([])
    })
    return counts
  }

  async function renderHome(url: string, { hasTraces = true } = {}) {
    await renderWithRouter(<TeamHome />, {
      url,
      shell: (
        <Provider>
          <DeploymentProvider
            value={bootstrap(
              hasTraces ? { surfaces: [...SURFACES, "traces"] } : {},
            )}
          >
            <SelectedWorkspaceProvider>
              <Outlet />
            </SelectedWorkspaceProvider>
          </DeploymentProvider>
        </Provider>
      ),
    })
  }

  it("counts the people of the workspace the page shows, not the organization", async () => {
    mockApi()
    await renderHome("/?workspace=ws-a")
    // Two of the three members are in ws-a, and nobody spent anything.
    expect(await screen.findByText("0 of 2")).toBeVisible()
  })

  it("links the sessions where the deployment records them", async () => {
    mockApi()
    await renderHome("/?workspace=ws-a")
    expect(await screen.findByText("0 of 2")).toBeVisible()
    expect(
      await screen.findByRole("link", { name: "See the sessions" }),
    ).toBeVisible()
    expect(screen.getByText("A row opens its sessions.")).toBeVisible()
  })

  it("offers no sessions where the deployment records none", async () => {
    const counts = mockApi()
    await renderHome("/?workspace=ws-a", { hasTraces: false })
    expect(await screen.findByText("0 of 2")).toBeVisible()
    expect(screen.getByText("in this workspace")).toBeVisible()
    expect(screen.queryByRole("link", { name: "See the sessions" })).toBeNull()
    expect(screen.queryByText("A row opens its sessions.")).toBeNull()
    expect(counts).toHaveLength(0)
  })

  it("counts every workspace's sessions when the page shows them all", async () => {
    const counts = mockApi()
    await renderHome("/")
    await waitFor(() => expect(counts.length).toBeGreaterThan(0))
    for (const url of counts) {
      expect(url.searchParams.get("workspace_id")).toBeNull()
    }
  })
})

describe("Home, by provider", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  const PROVIDERS = [
    { key: "openai", label: null, cost: 6, tokens: 600, requests: 4 },
    { key: "anthropic", label: null, cost: 3, tokens: 300, requests: 2 },
    // Requests that recorded no provider.
    { key: null, label: null, cost: 1, tokens: 100, requests: 3 },
  ]

  /** An owner's view, with one or two workspaces. */
  function mockApi({ workspaces = 1 }: { workspaces?: number } = {}) {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = new URL(String(input), "http://test")
      const path = url.pathname.replace(
        new URL(API_ROOT, "http://test").pathname,
        "",
      )
      if (path === "/organizations/me") {
        return Response.json(
          organizationContext({ role: "owner", deployment_operator: false }),
        )
      }
      if (path === "/organizations/me/members") {
        return Response.json({ data: [ADA, WEI], count: 2 })
      }
      if (path === "/organizations/me/usage/summary") {
        const provider = url.searchParams.get("provider")
        if (provider) {
          // A provider row's top model.
          return Response.json({
            totals: usageTotals({ cost: 6, request_count: 4 }),
            series: [],
            by_model: [
              {
                key: `${provider}-flagship`,
                label: null,
                cost: 5,
                tokens: 500,
                requests: 3,
                is_other: false,
              },
            ],
          })
        }
        return Response.json({
          totals: usageTotals({ cost: 10, request_count: 9 }),
          series: [],
          by_user: [],
          by_api_key: [],
          by_model: [],
          by_provider: PROVIDERS.map((row) => ({ ...row, is_other: false })),
          errors_by_status_code: [],
        })
      }
      if (path === "/organizations/me/spend-ceilings") {
        return Response.json({ data: [], count: 0 })
      }
      if (path === "/workspaces") {
        const all = [
          { id: "ws-a", name: "Platform" },
          { id: "ws-b", name: "Data" },
        ].slice(0, workspaces)
        return Response.json({ data: all, count: all.length })
      }
      return Response.json([])
    })
  }

  function RequestsProbe() {
    const location = useLocation()
    return (
      <div data-testid="requests">
        {location.pathname}
        {location.searchStr}
      </div>
    )
  }

  async function renderHome(url = "/?by=providers") {
    await renderWithRouter(<TeamHome />, {
      url,
      routes: [{ path: "/requests", element: <RequestsProbe /> }],
      shell: (
        <Provider>
          <DeploymentProvider value={bootstrap()}>
            <SelectedWorkspaceProvider>
              <Outlet />
            </SelectedWorkspaceProvider>
          </DeploymentProvider>
        </Provider>
      ),
    })
  }

  async function providerTable() {
    const table = await screen.findByRole("grid", {
      name: "Spend by providers",
    })
    await within(table).findByText("OpenAI")
    return table
  }

  it("offers Providers beside the other cuts", async () => {
    mockApi()
    await renderHome("/")
    const cuts = await screen.findByRole("radiogroup", {
      name: "Break down by",
    })
    expect(
      within(cuts)
        .getAllByRole("radio")
        .map((radio) => radio.closest("label")?.textContent),
    ).toEqual(["People", "API keys", "Models", "Providers"])
  })

  it("lists each provider's spend and share, the unrecorded ones as one row", async () => {
    mockApi()
    await renderHome()
    const table = await providerTable()
    const names = within(table)
      .getAllByRole("rowheader")
      .map((cell) => cell.textContent)
    expect(names).toEqual([
      "OpenAI",
      "Anthropic",
      "Unknown providerNot recorded",
    ])
    const openai = within(table).getByRole("row", { name: /OpenAI/ })
    expect(within(openai).getByText("$6.00")).toBeVisible()
    expect(within(openai).getByText("60%")).toBeVisible()
    expect(
      await within(openai).findByText("openai-flagship"),
    ).toBeInTheDocument()
    const unknown = within(table).getByRole("row", {
      name: /Unknown provider/,
    })
    expect(within(unknown).getByText("10%")).toBeVisible()
    // A provider's limit lives on a stored key, so the cut shows none.
    expect(
      within(table).queryByRole("columnheader", { name: "Limit" }),
    ).toBeNull()
    expect(screen.getByText("A row opens its requests.")).toBeVisible()
  })

  it("opens a provider's requests over the window", async () => {
    mockApi()
    const user = userEvent.setup()
    await renderHome()
    const table = await providerTable()
    await user.click(within(table).getByText("OpenAI"))
    const landed = await screen.findByTestId("requests")
    const [path, query] = (landed.textContent ?? "").split("?")
    expect(path).toBe("/requests")
    const search = new URLSearchParams(query)
    expect(search.get("provider")).toBe("openai")
    expect(search.get("start_date")).toBeTruthy()
  })

  it("offers nothing to open on the row that recorded no provider", async () => {
    mockApi()
    const user = userEvent.setup()
    await renderHome()
    const table = await providerTable()
    const unknown = within(table).getByRole("row", {
      name: /Unknown provider/,
    })
    expect(await findRowMenuItem(user, /See its requests/, unknown)).toBeNull()
    const openai = within(table).getByRole("row", { name: /OpenAI/ })
    expect(
      await findRowMenuItem(user, /See its requests/, openai),
    ).not.toBeNull()
  })

  it("picks the cut from a list once there are more than four", async () => {
    mockApi({ workspaces: 2 })
    await renderHome()
    await providerTable()
    await waitFor(() =>
      expect(
        screen.queryByRole("radiogroup", { name: "Break down by" }),
      ).toBeNull(),
    )
    expect(selectTrigger("Break down by")).toHaveTextContent("Providers")
  })
})
