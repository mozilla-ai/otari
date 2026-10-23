import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
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

describe("Overview scope", () => {
  it("opens a member on their own requests, and asks for no one else's", async () => {
    const requested = mockGateway({
      deployment_operator: false,
      role: "member",
    })
    renderOverview()

    expect(await screen.findByText("Your requests in Production")).toBeVisible()
    await waitFor(() =>
      expect(requested.some((u) => u.pathname.endsWith("/usage/summary"))).toBe(
        true,
      ),
    )
    // Not one read named anything but the caller, the first included: nothing
    // is asked before the scope resolves.
    for (const url of requested.filter((u) => u.pathname.includes("/usage"))) {
      expect(url.searchParams.getAll("user_id")).toEqual(["me-id"])
    }
  })

  it("opens a manager on everyone", async () => {
    mockGateway({ deployment_operator: false, role: "admin" })
    renderOverview()

    expect(await screen.findByText("Everyone in Production")).toBeVisible()
  })

  it("judges a manager's attention strip on everyone even under Mine", async () => {
    mockGateway({ deployment_operator: false, role: "admin" }, true, {
      mine: 0,
      everyone: 20,
    })
    renderOverview("/overview?scope=mine")

    expect(await screen.findByText(/error rate 50/)).toBeVisible()
    // And it opens the log on the requests it counted.
    expect((await errorLink()).searchParams.get("scope")).toBe("everyone")
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
    mockGateway({ deployment_operator: false, role: "member" }, true, {
      mine: 0,
      everyone: 20,
    })
    renderOverview()

    await screen.findByText("Your requests in Production")
    await waitFor(() => expect(screen.queryByRole("alert")).toBeNull())
    expect(screen.queryByText(/error rate 50/)).toBeNull()
  })

  it("judges a member's attention strip on their own usage under Everyone", async () => {
    mockGateway({ deployment_operator: false, role: "member" }, true, {
      mine: 0,
      everyone: 20,
    })
    renderOverview("/overview?scope=everyone")

    await screen.findByText("Everyone in Production")
    await waitFor(() => expect(screen.queryByRole("alert")).toBeNull())
    expect(screen.queryByText(/error rate 50/)).toBeNull()
  })

  it("links a member's own failures with their scope", async () => {
    mockGateway({ deployment_operator: false, role: "member" }, true, {
      mine: 20,
      everyone: 0,
    })
    renderOverview()

    expect((await errorLink()).searchParams.get("scope")).toBe("mine")
  })

  it("leaves Mine off for a caller with no usage identity", async () => {
    const requested = mockGateway({ deployment_operator: true }, false)
    renderOverview()

    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Mine", hidden: true }),
      ).toHaveAttribute("aria-disabled", "true"),
    )
    expect(requested.filter((u) => u.searchParams.has("user_id"))).toEqual([])
  })
})
