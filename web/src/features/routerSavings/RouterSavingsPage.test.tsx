import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { SubagentSavings } from "@/client"
import { RouterSavingsPage } from "@/features/routerSavings/RouterSavingsPage"
import { API_ROOT } from "@/shared/api/client"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  bootstrap,
  organizationContext,
  organizationMember,
  savingsAmounts,
  subagentSavings,
} from "@/tests/fixtures"
import { renderWithRouter } from "@/tests/router"

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

/** Answers the savings read with `savings`, and returns every URL asked for. */
function mockApi(
  savings: SubagentSavings | "refused" = subagentSavings(),
  members = [organizationMember()],
  memberships: { workspace_id: string; name: string; role: string }[] = [],
) {
  const urls: string[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input)
    urls.push(url)
    if (url.includes(`${API_ROOT}/organizations/me/subagent-savings`)) {
      return savings === "refused"
        ? jsonResponse({ detail: "The savings could not be read" }, 500)
        : jsonResponse(savings)
    }
    if (url.includes(`${API_ROOT}/organizations/me/members`)) {
      return jsonResponse({ data: members, count: members.length })
    }
    if (url.endsWith(`${API_ROOT}/organizations/me`)) {
      return jsonResponse(
        organizationContext({ workspace_memberships: memberships }),
      )
    }
    return jsonResponse({ detail: `unmocked ${url}` }, 500)
  })
  return urls
}

function renderPage({ withProvider = false } = {}) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return renderWithRouter(
    <DeploymentProvider value={bootstrap()}>
      <QueryClientProvider client={client}>
        {withProvider ? (
          <SelectedWorkspaceProvider>
            <RouterSavingsPage />
          </SelectedWorkspaceProvider>
        ) : (
          <RouterSavingsPage />
        )}
      </QueryClientProvider>
    </DeploymentProvider>,
    { url: "/router-savings" },
  )
}

const savingsRequests = (urls: string[]) =>
  urls.filter((url) => url.includes("/subagent-savings"))

afterEach(() => {
  vi.restoreAllMocks()
  window.localStorage.clear()
})

describe("RouterSavingsPage", () => {
  it("shows what was saved, what it would have cost and what was spent", async () => {
    mockApi()
    await renderPage()

    expect(await screen.findByText("$5.50")).toBeInTheDocument()
    expect(screen.getByText("$0.10 on the router")).toBeInTheDocument()
    expect(screen.getByText("$8.10")).toBeInTheDocument()
    // The subagents' own tokens and the recommendations.
    expect(screen.getByText("$2.60")).toBeInTheDocument()
    expect(
      screen.getByText(
        "Of 6 recommendations: 1 measured, 3 estimated, 1 not reported, 1 not priced.",
      ),
    ).toBeInTheDocument()
  })

  it("shows a subagent type that lost money as a loss", async () => {
    mockApi()
    await renderPage()

    const table = await screen.findByRole("grid", {
      name: "Savings by subagent type",
    })
    const plan = within(table).getByRole("row", { name: /Plan/ })
    expect(within(plan).getByText("-$0.21")).toBeInTheDocument()
    const explore = within(table).getByRole("row", { name: /Explore/ })
    expect(within(explore).getByText("Mixed")).toBeInTheDocument()
  })

  it("says a swap with no price is not priced instead of showing zero", async () => {
    mockApi()
    await renderPage()

    const table = await screen.findByRole("grid", { name: "Savings by model" })
    const unpriced = within(table).getByRole("row", {
      name: /instead of opus/,
    })
    expect(within(unpriced).getAllByText("Not priced").length).toBeGreaterThan(
      0,
    )
    expect(within(unpriced).queryByText("$0.00")).not.toBeInTheDocument()
  })

  it("marks a net loss on the headline", async () => {
    mockApi(
      subagentSavings({
        totals: {
          gross_saved: -0.2,
          net_saved: -0.3,
          estimated: 1,
          measured: 0,
        },
      }),
    )
    await renderPage()

    expect(await screen.findByText("-$0.30")).toBeInTheDocument()
    expect(screen.getByText("LOSS")).toBeInTheDocument()
  })

  it("says so when no router is switched on", async () => {
    mockApi(
      subagentSavings({
        recommender_available: false,
        totals: savingsAmounts(),
        series: [],
        by_agent_type: [],
        by_swap: [],
      }),
    )
    await renderPage()

    expect(
      await screen.findByRole("heading", {
        name: "No subagent router is switched on",
      }),
    ).toBeInTheDocument()
  })

  it("says how to start when a router is on and nothing was asked", async () => {
    mockApi(
      subagentSavings({
        totals: { ...savingsAmounts(), no_outcome: 0, unpriced: 0 },
        series: [],
        by_agent_type: [],
        by_swap: [],
      }),
    )
    await renderPage()

    expect(
      await screen.findByRole("heading", { name: /No recommendations/ }),
    ).toBeInTheDocument()
    expect(screen.getByText(/otari-router plugin/)).toBeInTheDocument()
  })

  it("says why there is no saving when no subagent has reported", async () => {
    mockApi(
      subagentSavings({
        totals: {
          ...savingsAmounts({
            decisions: 3,
            router_cost: 0.03,
            net_saved: -0.03,
          }),
          no_outcome: 3,
          unpriced: 0,
        },
        series: [],
        by_agent_type: [
          {
            ...savingsAmounts({
              decisions: 3,
              router_cost: 0.03,
              net_saved: -0.03,
            }),
            agent_type: "Explore",
          },
        ],
        by_swap: [],
      }),
    )
    await renderPage()

    expect(
      await screen.findByText(/reports from version\s+0\.2\.0/),
    ).toBeInTheDocument()
    const table = screen.getByRole("grid", {
      name: "Savings by subagent type",
    })
    expect(within(table).getAllByText("Not reported").length).toBeGreaterThan(0)
    expect(within(table).queryByText("Not priced")).not.toBeInTheDocument()
    // No priced outcome, so no saving is claimed, though the router was paid.
    expect(screen.getByText("No priced outcome yet")).toBeInTheDocument()
    expect(screen.getByText("$0.03")).toBeInTheDocument()
  })

  it("reads the caller's own subagents under Just me", async () => {
    const urls = mockApi()
    await renderPage()
    await screen.findByText("$5.50")
    expect(savingsRequests(urls).at(-1)).not.toContain("user_id=")

    await userEvent.click(screen.getByRole("radio", { name: "Just me" }))

    await waitFor(() =>
      expect(savingsRequests(urls).at(-1)).toContain(
        `user_id=${organizationMember().attribution_user_id}`,
      ),
    )
  })

  it("does not show everyone's figures under Just me while the caller's own load", async () => {
    let answerMine: (response: Response) => void = () => {}
    const mine = new Promise<Response>((resolve) => {
      answerMine = resolve
    })
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input)
      if (url.includes("/subagent-savings")) {
        return url.includes("user_id=") ? mine : jsonResponse(subagentSavings())
      }
      if (url.includes(`${API_ROOT}/organizations/me/members`)) {
        return jsonResponse({ data: [organizationMember()], count: 1 })
      }
      return jsonResponse(organizationContext())
    })
    await renderPage()
    await screen.findByText("$5.50")

    await userEvent.click(screen.getByRole("radio", { name: "Just me" }))

    await waitFor(() =>
      expect(screen.queryByText("$5.50")).not.toBeInTheDocument(),
    )
    answerMine(jsonResponse(subagentSavings({ totals: { net_saved: 1.25 } })))
    expect(await screen.findByText("$1.25")).toBeInTheDocument()
    expect(screen.queryByText("$5.50")).not.toBeInTheDocument()
  })

  it("does not show everyone's figures as the caller's own when they have no id", async () => {
    const urls = mockApi(subagentSavings(), [])
    await renderPage()
    await screen.findByText("$5.50")

    await userEvent.click(screen.getByRole("radio", { name: "Just me" }))

    expect(
      await screen.findByRole("heading", {
        name: "Nothing is recorded under your name",
      }),
    ).toBeInTheDocument()
    expect(screen.queryByText("$5.50")).not.toBeInTheDocument()
    expect(
      savingsRequests(urls).every((url) => !url.includes("user_id=")),
    ).toBe(true)
  })

  it("says the caller's own usage could not be read instead of calling it empty", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input)
      if (url.includes("/subagent-savings")) {
        return jsonResponse(subagentSavings())
      }
      if (url.includes(`${API_ROOT}/organizations/me/members`)) {
        return jsonResponse({ detail: "The roster could not be read" }, 500)
      }
      return jsonResponse(organizationContext())
    })
    await renderPage()
    await screen.findByText("$5.50")

    await userEvent.click(screen.getByRole("radio", { name: "Just me" }))

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Your own usage could not be read",
    )
    expect(
      screen.queryByRole("heading", {
        name: "Nothing is recorded under your name",
      }),
    ).not.toBeInTheDocument()
  })

  it("says the workspaces could not be read instead of loading forever", async () => {
    const urls: string[] = []
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      urls.push(String(input))
      return jsonResponse({ detail: "The organization could not be read" }, 500)
    })
    await renderPage({ withProvider: true })

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Your workspaces could not be read",
    )
    expect(screen.queryByText("Loading…")).not.toBeInTheDocument()
    expect(savingsRequests(urls)).toEqual([])
  })

  it("reads the selected workspace's subagents and no others", async () => {
    window.localStorage.setItem(
      "otari.dashboard.selectedWorkspace",
      "ws-staging",
    )
    const urls = mockApi(
      subagentSavings(),
      [organizationMember()],
      [
        {
          workspace_id: "ws-default",
          name: "Default Workspace",
          role: "owner",
        },
        { workspace_id: "ws-staging", name: "Staging", role: "member" },
      ],
    )
    await renderPage({ withProvider: true })

    expect(await screen.findByText("$5.50")).toBeInTheDocument()
    expect(savingsRequests(urls)).not.toHaveLength(0)
    expect(
      savingsRequests(urls).every((url) =>
        url.includes("workspace_id=ws-staging"),
      ),
    ).toBe(true)
  })

  it("shows a read that failed", async () => {
    mockApi("refused")
    await renderPage()

    expect(
      await screen.findByText(/The savings could not be read/),
    ).toBeInTheDocument()
  })
})
