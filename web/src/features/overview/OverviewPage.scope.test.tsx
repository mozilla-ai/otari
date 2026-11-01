import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { OverviewIndex } from "@/features/overview/OverviewPage"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  bootstrap,
  organizationContext,
  organizationMember,
  usageTotals,
  workspaceActivation,
} from "@/tests/fixtures"
import { withRouter } from "@/tests/router"

afterEach(() => {
  vi.restoreAllMocks()
})

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })

const summaryOf = (cost: number, errors = 1) => ({
  start_date: "",
  end_date: "",
  bucket: "hour",
  totals: usageTotals({
    cost,
    request_count: 40,
    error_count: errors,
    prompt_tokens: 0,
    completion_tokens: 0,
    total_tokens: 0,
    cache_read_tokens: 0,
    cache_write_tokens: 0,
    avg_latency_ms: null,
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
  series: [],
})

/** Everyone spent $100 today; the caller ("me-id") spent $7 of it. */
function mockGateway(
  context: Parameters<typeof organizationContext>[0],
  withIdentity = true,
  errors: { mine: number; everyone: number } = { mine: 1, everyone: 1 },
) {
  const requested: URL[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = new URL(String(input), "http://localhost")
    requested.push(url)
    const path = url.pathname.replace(/^.*\/api\/v1/, "")
    if (path === "/organizations/me") {
      return json(
        organizationContext({
          workspace_memberships: [
            {
              workspace_id: "ws-1",
              name: "Production",
              role: "owner",
            },
          ],
          ...context,
        }),
      )
    }
    if (path === "/organizations/me/members") {
      const member = organizationMember({
        attribution_user_id: withIdentity ? "me-id" : null,
      })
      return json({ data: [member], total: 1 })
    }
    if (path.endsWith("/usage/summary")) {
      return json(
        url.searchParams.getAll("user_id").includes("me-id")
          ? summaryOf(7, errors.mine)
          : summaryOf(100, errors.everyone),
      )
    }
    if (path.endsWith("/usage")) return json([])
    if (path === "/overview") {
      return json({
        active_keys: 0,
        active_members: 0,
        budgets: null,
        ceilings: null,
      })
    }
    if (path.includes("/activation")) return json(workspaceActivation())
    if (path === "/models") {
      // Something to route to, so the getting-started probe is never asked.
      return json({
        object: "list",
        data: [{ id: "m", object: "model", created: 0, owned_by: "openai" }],
      })
    }
    if (path === "/providers/health") {
      return json({ providers: [], healthy: 0, total: 0, checked_at: null })
    }
    if (path === "/providers")
      return json({ providers: [{ provider: "openai" }] })
    if (path.endsWith("/hosted-providers")) return json({ data: [], count: 0 })
    return json([])
  })
  return requested
}

function renderOverview(url = "/overview") {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <DeploymentProvider value={bootstrap()}>
        <SelectedWorkspaceProvider>
          <OverviewIndex />
        </SelectedWorkspaceProvider>
      </DeploymentProvider>
    </QueryClientProvider>,
    { wrapper: withRouter({ url }) },
  )
}

// By text rather than role: the setup guide's sheet may be open over the page,
// which hides the rest of it from the accessibility tree.
const errorLink = async () =>
  new URL(
    (await screen.findByText(/^error rate/))
      .closest("a")
      ?.getAttribute("href") ?? "",
    "http://localhost",
  )

const summaryReads = (requested: URL[]) =>
  requested.filter((u) => u.pathname.endsWith("/usage/summary"))

describe("Overview scope", () => {
  it("opens a member on everyone's requests, narrowed by the User filter", async () => {
    const requested = mockGateway({
      deployment_operator: false,
      role: "member",
    })
    renderOverview()

    await waitFor(() =>
      expect(
        summaryReads(requested).some((u) => !u.searchParams.has("user_id")),
      ).toBe(true),
    )
    expect(screen.queryByText("Mine")).toBeNull()
  })

  it("reads a legacy Mine link as the caller, with a chip to drop it", async () => {
    const requested = mockGateway({
      deployment_operator: false,
      role: "admin",
    })
    renderOverview("/overview?scope=mine")

    const remove = await screen.findByRole("button", {
      name: "Remove User filter",
      hidden: true,
    })
    await waitFor(() =>
      expect(
        summaryReads(requested).some((u) =>
          u.searchParams.getAll("user_id").includes("me-id"),
        ),
      ).toBe(true),
    )
    const user = userEvent.setup()
    await user.click(remove)
    await waitFor(() =>
      expect(
        screen.queryByRole("button", {
          name: "Remove User filter",
          hidden: true,
        }),
      ).toBeNull(),
    )
  })

  it("judges a manager's attention strip on everyone even under Mine", async () => {
    mockGateway({ deployment_operator: false, role: "admin" }, true, {
      mine: 0,
      everyone: 20,
    })
    renderOverview("/overview?scope=mine")

    expect(await screen.findByText(/error rate 50/)).toBeVisible()
    // And it opens the log on the requests it counted.
    const link = await errorLink()
    expect(link.searchParams.has("user_id")).toBe(false)
    expect(link.searchParams.has("scope")).toBe(false)
  })

  it("judges an operator's attention strip on everyone", async () => {
    mockGateway({ deployment_operator: true, role: "owner" }, true, {
      mine: 0,
      everyone: 20,
    })
    renderOverview("/overview?scope=mine")

    expect(await screen.findByText(/error rate 50/)).toBeVisible()
  })

  it("judges a member's attention strip on their own usage", async () => {
    const requested = mockGateway(
      { deployment_operator: false, role: "member" },
      true,
      { mine: 0, everyone: 20 },
    )
    renderOverview()

    await waitFor(() =>
      expect(
        summaryReads(requested).some((u) =>
          u.searchParams.getAll("user_id").includes("me-id"),
        ),
      ).toBe(true),
    )
    await waitFor(() => expect(screen.queryByRole("alert")).toBeNull())
    expect(screen.queryByText(/error rate 50/)).toBeNull()
  })

  it("links a member's own failures filtered to them", async () => {
    mockGateway({ deployment_operator: false, role: "member" }, true, {
      mine: 20,
      everyone: 0,
    })
    renderOverview()

    const link = await errorLink()
    expect(link.searchParams.getAll("user_id")).toEqual(["me-id"])
    expect(link.searchParams.has("scope")).toBe(false)
  })
})
