import { Outlet, useLocation, useRouterState } from "@tanstack/react-router"
import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
import { TeamHome } from "@/features/home/TeamHome"
import { API_ROOT } from "@/shared/api/client"
import { formatUsd } from "@/shared/helpers/format"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  bootstrap,
  organizationContext,
  organizationMember,
  SURFACES,
  usageTotals,
  workspace,
} from "@/tests/fixtures"
import { AppProviders as Provider } from "@/tests/providers"
import { LocationProbe, renderWithRouter } from "@/tests/router"
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

describe("Home, Where it went by model", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  /** Two providers' ids for one model, as the gateway marks them. */
  const BY_MODEL = [
    {
      key: "accounts/fireworks/models/deepseek-v4p1-flash-1017",
      label: "DeepSeek V4.1 Flash 1017",
      model_group: "deepseekv41flash1017",
      cost: 149.33,
      tokens: 8_590_300_000,
      requests: 900,
      is_other: false,
    },
    {
      key: "moonshotai/Kimi-K3.1-Pro",
      label: "Kimi K3.1 Pro",
      model_group: "kimik31pro",
      cost: 70.28,
      tokens: 314_900_000,
      requests: 200,
      is_other: false,
    },
    {
      key: "deepseek-ai/DeepSeek-V4.1-Flash-1017",
      label: "DeepSeek V4.1 Flash 1017",
      model_group: "deepseekv41flash1017",
      cost: 50.47,
      tokens: 2_973_500_000,
      requests: 300,
      is_other: false,
    },
  ]

  function SessionsProbe() {
    const search = useRouterState({
      select: (state) => state.location.searchStr,
    })
    return <output aria-label="Sessions search">{search}</output>
  }

  it("lists one model once, by name, and opens the sessions of every id it ran as", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const path = String(input).slice(API_ROOT.length).split("?")[0]
      if (path === "/organizations/me") {
        return Response.json(
          organizationContext({ role: "owner", deployment_operator: false }),
        )
      }
      if (path === "/organizations/me/members") {
        return Response.json({ data: [ADA, WEI], count: 2 })
      }
      if (path === "/organizations/me/usage/summary") {
        return Response.json({
          totals: usageTotals({ cost: 270.08, request_count: 1400 }),
          series: [],
          by_user: [],
          by_api_key: [],
          by_model: BY_MODEL,
          errors_by_status_code: [],
        })
      }
      if (path === "/organizations/me/spend-ceilings") {
        return Response.json({ data: [], count: 0 })
      }
      if (path.endsWith("/traces/count")) return Response.json({ count: 3 })
      if (path === "/workspaces") return Response.json({ data: [], count: 0 })
      return Response.json([])
    })
    await renderWithRouter(<TeamHome />, {
      url: "/?by=models",
      routes: [{ path: "/sessions", element: <SessionsProbe /> }],
      shell: (
        <Provider>
          <DeploymentProvider
            value={bootstrap({ surfaces: [...SURFACES, "traces"] })}
          >
            <SelectedWorkspaceProvider>
              <Outlet />
            </SelectedWorkspaceProvider>
          </DeploymentProvider>
        </Provider>
      ),
    })

    const table = await screen.findByRole("grid", { name: "Spend by models" })
    await within(table).findByText("Kimi K3.1 Pro")
    const rows = within(table).getAllByRole("row").slice(1)
    expect(
      rows.map((row) => within(row).getByRole("rowheader").textContent),
    ).toEqual(["DeepSeek V4.1 Flash 1017", "Kimi K3.1 Pro"])
    expect(within(rows[0]).getByText("$199.80")).toBeVisible()

    await userEvent.setup().click(within(rows[0]).getByRole("rowheader"))
    const search = new URLSearchParams(
      (await screen.findByRole("status", { name: "Sessions search" }))
        .textContent ?? "",
    )
    expect(search.getAll("model")).toEqual([
      "accounts/fireworks/models/deepseek-v4p1-flash-1017",
      "deepseek-ai/DeepSeek-V4.1-Flash-1017",
    ])
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

  it("points an empty window at a longer one, but not from the longest", async () => {
    mockApi()
    await renderHome("/?period=7d&by=models")
    expect(
      await screen.findByText(
        "Nothing was spent in this window. Pick a longer one above.",
      ),
    ).toBeVisible()
  })

  it("offers no longer window from 6M", async () => {
    mockApi()
    await renderHome("/?period=6m&by=models")
    expect(
      await screen.findByText("Nothing was spent in this window."),
    ).toBeVisible()
    expect(screen.queryByText(/Pick a longer one/)).toBeNull()
  })

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
        if (provider && url.searchParams.get("dimensions") === "none") {
          // Which workspace a provider row opens in: Data spent the most.
          const workspace = url.searchParams.get("workspace_id")
          return Response.json({
            totals: usageTotals({ cost: workspace === "ws-b" ? 5 : 1 }),
            series: [],
          })
        }
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

  async function renderHome(url = "/?by=providers") {
    await renderWithRouter(<TeamHome />, {
      url,
      routes: [{ path: "/requests", element: <LocationProbe /> }],
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
    const landed = await screen.findByTestId("location")
    const [path, query] = (landed.textContent ?? "").split("?")
    expect(path).toBe("/requests")
    const search = new URLSearchParams(query)
    expect(search.get("provider")).toBe("openai")
    expect(search.get("start_date")).toBeTruthy()
  })

  it("opens a provider's requests in the workspace that spent the most on it", async () => {
    mockApi({ workspaces: 2 })
    const user = userEvent.setup()
    await renderHome("/?by=providers")
    const table = await providerTable()
    await user.click(within(table).getByText("OpenAI"))
    const landed = await screen.findByTestId("location")
    expect(landed.textContent).toContain("provider=openai")
    // Each workspace's spend on the provider was weighed, not just the one
    // the sidebar had selected.
    const weighed = vi
      .mocked(globalThis.fetch)
      .mock.calls.map(([input]) => new URL(String(input), "http://test"))
      .filter(
        (url) =>
          url.searchParams.get("provider") === "openai" &&
          url.searchParams.get("dimensions") === "none",
      )
      .map((url) => url.searchParams.get("workspace_id"))
    expect(weighed.sort()).toEqual(["ws-a", "ws-b"])
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

  it("picks the cut from a list on a phone once there are more than three", async () => {
    const original = window.matchMedia
    window.matchMedia = ((query: string) => ({
      matches: query === "(max-width: 767px)",
      media: query,
      onchange: null,
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
      addListener: () => undefined,
      removeListener: () => undefined,
      dispatchEvent: () => false,
    })) as typeof window.matchMedia
    try {
      mockApi()
      await renderHome()
      await waitFor(() =>
        expect(selectTrigger("Break down by")).toBeInTheDocument(),
      )
      expect(
        screen.queryByRole("radiogroup", { name: "Break down by" }),
      ).toBeNull()
    } finally {
      window.matchMedia = original
    }
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

describe("Home, the window", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  function LocationProbe() {
    const { searchStr } = useLocation()
    return <output aria-label="Search">{searchStr}</output>
  }

  const spent = (key: string, label: string) => ({
    key,
    label,
    cost: 2,
    requests: 3,
    tokens: 900,
    is_other: false,
  })

  /** Every read answered with some spend; returns every URL the page asked. */
  function mockApi({
    errors = 0,
    hosted = 0,
    cached = 0,
  }: {
    errors?: number
    hosted?: number
    /** Input tokens read from cache, of 1,000 over two buckets. */
    cached?: number
  } = {}) {
    const asked: URL[] = []
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = new URL(String(input), "http://test")
      asked.push(url)
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
        return Response.json({
          totals: usageTotals({
            cost: 4,
            request_count: 6,
            error_count: errors,
            deployment_paid_cost: hosted,
          }),
          series: cached
            ? [0, 1].map((day) => ({
                bucket_start: new Date(Date.now() - day * 3600e3).toISOString(),
                cache_read_tokens: cached / 2,
                cache_write_tokens: 0,
                cost: 2,
                errors: 0,
                input_tokens: 500,
                output_tokens: 100,
                requests: 3,
                tokens: 600,
              }))
            : [],
          by_user: [spent("u-ada", "Ada"), spent("u-wei", "Wei")],
          by_api_key: [spent("k-1", "laptop")],
          by_model: [spent("gpt-5", "gpt-5")],
          errors_by_status_code: [],
        })
      }
      if (path.endsWith("/traces/count")) return Response.json({ count: 3 })
      if (path === "/organizations/me/spend-ceilings") {
        return Response.json({ data: [], count: 0 })
      }
      if (path === "/workspaces") {
        return Response.json({
          data: [
            workspace({ id: "ws-a", name: "Platform" }),
            workspace({ id: "ws-b", name: "Research" }),
          ],
          count: 2,
        })
      }
      return Response.json([])
    })
    return asked
  }

  async function renderHome(url: string) {
    await renderWithRouter(<TeamHome />, {
      url,
      shell: (
        <Provider>
          <DeploymentProvider
            value={bootstrap({ surfaces: [...SURFACES, "traces"] })}
          >
            <SelectedWorkspaceProvider>
              <Outlet />
              <LocationProbe />
            </SelectedWorkspaceProvider>
          </DeploymentProvider>
        </Provider>
      ),
    })
  }

  /** The reads over a window: usage's `start_date`, sessions' `start`. */
  const windowed = (asked: URL[]) =>
    asked.filter(
      (url) =>
        url.searchParams.has("start_date") || url.searchParams.has("start"),
    )

  it("splits the spend by who paid once a hosted provider served some", async () => {
    mockApi({ hosted: 1 })
    await renderHome("/?period=today")
    expect(
      await screen.findByText(
        `${formatUsd(1)} hosted · ${formatUsd(3)} own keys`,
      ),
    ).toBeVisible()
  })

  it("keeps the request count on Tokens once the split and the cache share both take a subline", async () => {
    mockApi({ hosted: 1, cached: 250 })
    await renderHome("/?period=today")
    expect(
      await screen.findByText("6 requests · 25.0% of input from cache"),
    ).toBeVisible()
  })

  it("counts one failed request in the singular", async () => {
    mockApi({ errors: 1 })
    await renderHome("/?period=today")
    expect(await screen.findByText("1 failed request")).toBeVisible()
  })
  it("ends every read of yesterday at midnight, so none counts today", async () => {
    const asked = mockApi()
    await renderHome("/?period=yesterday")
    expect(await screen.findByText("Spent yesterday")).toBeVisible()
    // The top model of each person on screen is read over the window too.
    await waitFor(() =>
      expect(
        asked.some(
          (url) =>
            url.searchParams.get("user_id") === "u-ada" &&
            url.searchParams.get("dimensions") === "model",
        ),
      ).toBe(true),
    )
    await waitFor(() =>
      expect(asked.some((url) => url.pathname.endsWith("/traces/count"))).toBe(
        true,
      ),
    )
    const reads = windowed(asked)
    expect(reads.length).toBeGreaterThan(3)
    for (const url of reads) {
      expect(
        url.searchParams.get("end_date") ?? url.searchParams.get("end"),
        String(url),
      ).toBeTruthy()
    }
    // One day reads by the hour.
    const main = reads.find((url) =>
      url.searchParams.getAll("dimensions").includes("user"),
    )
    expect(main?.searchParams.get("bucket")).toBe("hour")
    const sessions = screen.getByRole("link", { name: "See the sessions" })
    expect(sessions.getAttribute("href")).toContain("end_date=")
  })

  it("ends each workspace's read of yesterday too", async () => {
    const asked = mockApi()
    await renderHome("/?period=yesterday&by=workspaces")
    await screen.findByRole("grid", { name: "Spend by workspaces" })
    await waitFor(() =>
      expect(
        asked.filter((url) => url.searchParams.get("workspace_id")),
      ).toHaveLength(2),
    )
    for (const url of windowed(asked)) {
      expect(
        url.searchParams.get("end_date") ?? url.searchParams.get("end"),
        String(url),
      ).toBeTruthy()
    }
  })

  it("opens on 30 days, and moves to today by the hour in the URL", async () => {
    const user = userEvent.setup()
    const asked = mockApi()
    await renderHome("/")
    const picker = await screen.findByRole("radiogroup", { name: "Window" })
    expect(
      within(picker)
        .getAllByRole("radio")
        .map((radio) => radio.closest("label")?.textContent),
    ).toEqual(["Today", "Yesterday", "7D", "30D", "3M", "6M"])
    expect(within(picker).getByRole("radio", { name: "30D" })).toBeChecked()
    expect(await screen.findByText("Spent, 30 days")).toBeVisible()
    // An open window names no end; only the window before it has one.
    await waitFor(() =>
      expect(
        asked.some((url) =>
          url.searchParams.getAll("dimensions").includes("user"),
        ),
      ).toBe(true),
    )
    for (const url of windowed(asked)) {
      const isPrevious = url.searchParams.getAll("dimensions").includes("none")
      expect(url.searchParams.has("end_date"), String(url)).toBe(isPrevious)
    }

    await user.click(within(picker).getByRole("radio", { name: "Today" }))
    expect(screen.getByRole("status", { name: "Search" })).toHaveTextContent(
      "period=today",
    )
    expect(await screen.findByText("Spent today")).toBeVisible()
    await waitFor(() =>
      expect(
        asked.some(
          (url) =>
            url.searchParams.get("bucket") === "hour" &&
            url.searchParams.getAll("dimensions").includes("user"),
        ),
      ).toBe(true),
    )
  })

  it("reads an old 90-day link as 3M, and a retired or unknown one as 30D", async () => {
    mockApi()
    await renderHome("/?period=90d")
    const picker = await screen.findByRole("radiogroup", { name: "Window" })
    expect(within(picker).getByRole("radio", { name: "3M" })).toBeChecked()
    expect(await screen.findByText("Spent, 3 months")).toBeVisible()
  })

  it.each(["month", "custom", "bogus"])(
    "falls back to 30D for period=%s",
    async (period) => {
      mockApi()
      await renderHome(`/?period=${period}`)
      const picker = await screen.findByRole("radiogroup", { name: "Window" })
      expect(within(picker).getByRole("radio", { name: "30D" })).toBeChecked()
    },
  )
})
