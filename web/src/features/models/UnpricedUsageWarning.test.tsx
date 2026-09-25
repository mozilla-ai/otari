import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type {
  OrganizationContext,
  PricingResponse,
  UsageGroupRow,
  UsageSummary,
} from "@/client"
import { UnpricedUsageWarning } from "@/features/models/UnpricedUsageWarning"
import * as apiClient from "@/shared/api/client"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import {
  organizationContext,
  pricingResponse,
  usageTotals,
} from "@/tests/fixtures"
import { withRouter } from "@/tests/router"

function modelRow(key: string, requests: number): UsageGroupRow {
  return { key, cost: 0, tokens: 0, requests, is_other: false }
}

function summary(requests: number, byModel: UsageGroupRow[]): UsageSummary {
  return {
    start_date: "2026-09-23T00:00:00Z",
    end_date: "2026-09-24T00:00:00Z",
    bucket: "hour",
    totals: usageTotals({ request_count: requests }),
    by_model: byModel,
    by_user: [],
    by_api_key: [],
    by_source: [],
    by_source_label: [],
    by_endpoint: [],
    by_provider: [],
    by_tool: [],
    errors_by_status_code: [],
    series: [],
  }
}

function mockApi(
  answer: UsageSummary,
  context: OrganizationContext = organizationContext(),
  prices: PricingResponse[] = [],
) {
  return vi
    .spyOn(apiClient, "apiFetch")
    .mockImplementation(async (path: string) => {
      if (path === "/organizations/me") return context as never
      if (path.startsWith("/usage/summary?")) return answer as never
      if (path.startsWith("/pricing?")) return prices as never
      throw new Error(`unexpected request: ${path}`)
    })
}

function summaryParams(spy: ReturnType<typeof mockApi>): URLSearchParams {
  const path = spy.mock.calls
    .map(([path]) => String(path))
    .find((path) => path.startsWith("/usage/summary?"))
  return new URLSearchParams(String(path).split("?")[1])
}

function renderBanner() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  render(
    <QueryClientProvider client={client}>
      <SelectedWorkspaceProvider>
        <UnpricedUsageWarning />
      </SelectedWorkspaceProvider>
    </QueryClientProvider>,
    { wrapper: withRouter() },
  )
  return client
}

describe("UnpricedUsageWarning", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("names the unpriced models and links to their requests", async () => {
    const spy = mockApi(
      summary(19, [
        modelRow("gemini-3.8-flash", 4),
        modelRow("gemini-3.7-flash", 15),
      ]),
    )
    renderBanner()

    expect(
      await screen.findByText(
        "19 requests in the last 24 hours had no model price",
      ),
    ).toBeInTheDocument()
    // Busiest first, so the model costing the most unbilled traffic leads.
    const names = screen.getAllByText(/^gemini-/).map((el) => el.textContent)
    expect(names).toEqual(["gemini-3.7-flash", "gemini-3.8-flash"])
    // The link lands on the same rows the count is taken from.
    expect(
      screen.getByRole("link", { name: "View unpriced requests" }),
    ).toHaveAttribute(
      "href",
      "/activity?status=success&priced=false&range=24h&source=gateway",
    )

    const params = summaryParams(spy)
    expect(params.get("status")).toBe("success")
    expect(params.get("priced")).toBe("false")
    expect(params.get("source")).toBe("gateway")
    expect(params.getAll("dimensions")).toEqual(["model"])
    const since = new Date(String(params.get("start_date"))).getTime()
    expect(Date.now() - since).toBeLessThanOrEqual(86_400_000 + 5_000)
  })

  it("scopes the count to the selected workspace, as Activity does", async () => {
    // Activity narrows its rows to the shell's selected workspace, so a
    // deployment-wide count would link to a view that cannot show them.
    const spy = mockApi(
      summary(3, [modelRow("gemini-3.7-flash", 3)]),
      organizationContext({
        workspace_memberships: [
          { workspace_id: "ws-research", name: "Research", role: "owner" },
        ],
      }),
    )
    renderBanner()

    expect(await screen.findByText(/in Research/)).toBeInTheDocument()
    expect(summaryParams(spy).get("workspace_id")).toBe("ws-research")
  })

  it("asks deployment-wide when no workspace is selected", async () => {
    const spy = mockApi(summary(3, [modelRow("gemini-3.7-flash", 3)]))
    renderBanner()

    await screen.findByText(/had no model price/)
    expect(summaryParams(spy).has("workspace_id")).toBe(false)
  })

  it("folds models past the first three", async () => {
    mockApi(
      summary(5, [
        modelRow("a", 1),
        modelRow("b", 1),
        modelRow("c", 1),
        modelRow("d", 1),
        modelRow("e", 1),
      ]),
    )
    renderBanner()

    expect(await screen.findByText(/and 2 more/)).toBeInTheDocument()
  })

  it("singularizes a lone request", async () => {
    mockApi(summary(1, [modelRow("gemini-3.7-flash", 1)]))
    renderBanner()

    expect(
      await screen.findByText(
        "1 request in the last 24 hours had no model price",
      ),
    ).toBeInTheDocument()
  })

  it("stays quiet while every request was priced", async () => {
    const spy = mockApi(summary(0, []))
    const client = renderBanner()

    // Settled, not merely started, so the absence below is the answer.
    await waitFor(() => {
      expect(summaryParams(spy).get("priced")).toBe("false")
      expect(client.isFetching()).toBe(0)
    })
    expect(screen.queryByText(/had no model price/)).not.toBeInTheDocument()
  })

  it("asks nothing of the deployment-wide usage read for a tenant", async () => {
    const spy = mockApi(
      summary(19, [modelRow("gemini-3.7-flash", 19)]),
      organizationContext({ deployment_operator: false }),
    )
    const client = renderBanner()

    await waitFor(() =>
      expect(client.getQueryState(["organizations", "context"])?.status).toBe(
        "success",
      ),
    )
    expect(
      spy.mock.calls.some(([path]) => String(path).includes("/summary")),
    ).toBe(false)
    expect(screen.queryByText(/had no model price/)).not.toBeInTheDocument()
  })

  it("drops a model priced since its requests were logged", async () => {
    // Pricing a model leaves its logged rows uncosted, so the summary still
    // counts them; the banner must not keep asking for that price.
    mockApi(
      summary(19, [
        modelRow("gemini-3.8-flash", 4),
        modelRow("gemini-3.7-flash", 15),
      ]),
      organizationContext(),
      [pricingResponse({ model_key: "gemini:gemini-3.7-flash" })],
    )
    renderBanner()

    expect(
      await screen.findByText(
        "4 requests in the last 24 hours had no model price",
      ),
    ).toBeInTheDocument()
    expect(screen.queryByText("gemini-3.7-flash")).not.toBeInTheDocument()
  })

  it("goes away once every model it named is priced", async () => {
    const spy = mockApi(
      summary(15, [modelRow("gemini-3.7-flash", 15)]),
      organizationContext(),
      [pricingResponse({ model_key: "gemini:gemini-3.7-flash" })],
    )
    const client = renderBanner()

    await waitFor(() => {
      expect(
        spy.mock.calls.some(([path]) => String(path).startsWith("/pricing?")),
      ).toBe(true)
      expect(client.isFetching()).toBe(0)
    })
    expect(screen.queryByText(/had no model price/)).not.toBeInTheDocument()
  })

  it("can be dismissed", async () => {
    mockApi(summary(19, [modelRow("gemini-3.7-flash", 19)]))
    const user = userEvent.setup()
    renderBanner()

    await user.click(await screen.findByRole("button", { name: "Dismiss" }))
    expect(screen.queryByText(/had no model price/)).not.toBeInTheDocument()
  })
})
