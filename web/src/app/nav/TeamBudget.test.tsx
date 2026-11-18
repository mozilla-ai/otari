import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import type { OrganizationSpendCeiling } from "@/client"
import { API_ROOT } from "@/shared/api/client"
import { organizationContext, organizationSpendCeiling } from "@/tests/fixtures"
import { renderWithRouter } from "@/tests/router"
import { TeamBudget } from "./TeamBudget"

function json(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    headers: { "Content-Type": "application/json" },
  })
}

/** A period that has not ended, so its spend is what the ceiling says. */
const OPEN = {
  period_start: "2026-11-01T00:00:00Z",
  period_end: "2999-01-01T00:00:00Z",
}

async function renderBudget(ceilings: OrganizationSpendCeiling[]) {
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const path = String(input).replace(API_ROOT, "").split("?")[0]
    if (path === "/organizations/me") {
      return json(organizationContext({ deployment_operator: false }))
    }
    if (path === "/organizations/me/spend-ceilings") {
      return json({ data: ceilings, count: ceilings.length })
    }
    return json([])
  })
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  await renderWithRouter(
    <QueryClientProvider client={client}>
      <TeamBudget />
    </QueryClientProvider>,
  )
}

describe("TeamBudget", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("shows a $0 team budget as used up rather than hiding it", async () => {
    await renderBudget([
      organizationSpendCeiling({ max_budget: 0, current_spend: 0, ...OPEN }),
    ])
    expect(await screen.findByText("Team budget")).toBeVisible()
    expect(screen.getByText("100%")).toBeVisible()
  })

  it("counts a period that ended with no request since as nothing spent, as Home does", async () => {
    // November's $97, still on the ceiling on December 1 before the first
    // request of the month resets it.
    await renderBudget([
      organizationSpendCeiling({
        max_budget: 100,
        current_spend: 97,
        period_start: "2026-10-01T00:00:00Z",
        period_end: "2026-11-01T00:00:00Z",
      }),
    ])
    expect(await screen.findByText("Team budget")).toBeVisible()
    expect(screen.getByText("0%")).toBeVisible()
    expect(screen.getByText("$0.00")).toBeVisible()
  })

  it("is the organization's limit over every provider, not one on a single provider", async () => {
    await renderBudget([
      organizationSpendCeiling({
        id: "c-one-provider",
        provider_key_id: "pk-openai",
        max_budget: 10,
        current_spend: 9,
        ...OPEN,
      }),
      organizationSpendCeiling({
        id: "c-team",
        max_budget: 100,
        current_spend: 25,
        ...OPEN,
      }),
    ])
    expect(await screen.findByText("Team budget")).toBeVisible()
    expect(screen.getByText("25%")).toBeVisible()
    expect(screen.queryByText("90%")).toBeNull()
  })
})
