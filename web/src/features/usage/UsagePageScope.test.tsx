/**
 * Which usage surface the Usage page reads, and for whom (otari#837).
 *
 * A file of its own rather than a describe inside `UsagePage.test.tsx`: a case
 * there drives a router navigation, and the tree it leaves behind lands a stray
 * deployment-wide read in whichever fetch spy is installed next. The claim here
 * is universally quantified ("every usage read this page made was scoped"), so
 * it needs a spy nothing else can write to.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import type { ReactElement } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"
import type { UsageSummary } from "@/client"
import { UsagePage } from "@/features/usage/UsagePage"
import { organizationContext, usageTotals } from "@/tests/fixtures"
import { withRouter } from "@/tests/router"

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

function summary(): UsageSummary {
  return {
    start_date: "2026-06-21T00:00:00Z",
    end_date: "2026-07-21T00:00:00Z",
    bucket: "day",
    totals: usageTotals({ cost: 1240.5, request_count: 84_000 }),
    by_model: [],
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

// Answers on the path's tail, so one mock serves `/v1/usage/*` and
// `/v1/organizations/me/usage/*` alike and which was asked for is what the
// assertions read back off the spy.
function mockApi(deploymentOperator: boolean) {
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input)
    if (url.endsWith("/v1/organizations/me")) {
      return jsonResponse(
        organizationContext({ deployment_operator: deploymentOperator }),
      )
    }
    if (url.includes("/usage/summary")) return jsonResponse(summary())
    if (url.includes("/usage/series")) {
      return jsonResponse({
        start_date: "2026-06-21T00:00:00Z",
        end_date: "2026-07-21T00:00:00Z",
        bucket: "day",
        group_by: "model",
        groups: [],
        points: [],
      })
    }
    return jsonResponse([])
  })
}

function renderPage(ui: ReactElement) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>{ui}</QueryClientProvider>,
    { wrapper: withRouter({ url: "/usage" }) },
  )
}

function usageReads(fetchMock: ReturnType<typeof mockApi>): string[] {
  return fetchMock.mock.calls
    .map(([url]) => String(url))
    .filter((url) => url.includes("/usage/"))
}

describe("UsagePage scope", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("reads the organization-scoped routes for a tenant who does not operate the deployment", async () => {
    const fetchMock = mockApi(false)
    renderPage(<UsagePage />)
    await screen.findByText("$1,240.50")

    const reads = usageReads(fetchMock)
    expect(reads).not.toHaveLength(0)
    // Every one of them, not merely one: the tiles and the stacked chart come
    // from two endpoints, and a scope on only one would put another tenant's
    // totals above this tenant's chart.
    for (const url of reads) {
      expect(url).toContain("/v1/organizations/me/usage/")
    }
  })

  it("reads the deployment-wide routes for an operator, so the case above is not vacuous", async () => {
    const fetchMock = mockApi(true)
    renderPage(<UsagePage />)
    await screen.findByText("$1,240.50")

    const reads = usageReads(fetchMock)
    expect(reads).not.toHaveLength(0)
    for (const url of reads) {
      expect(url).not.toContain("/v1/organizations/me/usage/")
    }
  })
})
