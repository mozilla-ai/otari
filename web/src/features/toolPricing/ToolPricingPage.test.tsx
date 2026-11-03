import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { DeploymentBootstrap, PricingResponse } from "@/client"
import { ToolPricingPage } from "@/features/toolPricing/ToolPricingPage"
import { API_ROOT } from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  bootstrap,
  organizationContext,
  pricingResponse,
} from "@/tests/fixtures"
import { renderWithRouter } from "@/tests/router"

interface Request {
  url: string
  method: string
  body: unknown
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

/** A per-call rate as the API stores it, per million requests. */
const buildPerCallRate = (
  modelKey: string,
  dollars: number,
  overrides: Partial<PricingResponse> = {},
) =>
  pricingResponse({
    model_key: modelKey,
    input_price_per_million: dollars * 1_000_000,
    unit: "requests",
    effective_at: "2026-10-01T09:30:00Z",
    ...overrides,
  })

function mockApi(
  opts: {
    operator?: boolean
    /** The rates in force, keyed `kind:model_key`; anything else answers 404. */
    rates?: Record<string, PricingResponse>
    /** Keys, as `kind:model_key`, whose read answers 500. */
    failingReads?: string[]
    /** Answers a price write with this refusal. */
    writeRefusal?: string
    requirePricing?: boolean
  } = {},
) {
  const rates = opts.rates ?? {}
  const requests: Request[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
    const body = init?.body ? JSON.parse(String(init.body)) : undefined
    requests.push({ url, method, body })

    if (url.endsWith(`${API_ROOT}/organizations/me`)) {
      return jsonResponse(
        organizationContext({ deployment_operator: opts.operator ?? true }),
      )
    }
    if (url.includes(`${API_ROOT}/settings`)) {
      return jsonResponse({ require_pricing: opts.requirePricing ?? false })
    }
    if (url.includes(`${API_ROOT}/pricing`)) {
      if (method === "POST") {
        return opts.writeRefusal
          ? jsonResponse({ detail: opts.writeRefusal }, 422)
          : jsonResponse(body, 201)
      }
      const parsed = new URL(url, "http://localhost")
      const key = decodeURIComponent(
        parsed.pathname.slice(`${API_ROOT}/pricing/`.length),
      )
      const id = `${parsed.searchParams.get("kind")}:${key}`
      if (opts.failingReads?.includes(id)) {
        return jsonResponse({ detail: "Database unavailable" }, 500)
      }
      const rate = rates[id]
      return rate
        ? jsonResponse(rate)
        : jsonResponse({ detail: "Not found" }, 404)
    }
    return jsonResponse({})
  })
  return requests
}

function renderPage(deployment: DeploymentBootstrap = bootstrap()) {
  return renderWithRouter(
    <DeploymentProvider value={deployment}>
      <QueryClientProvider
        client={
          new QueryClient({ defaultOptions: { queries: { retry: false } } })
        }
      >
        <ToolPricingPage service="code_execution" />
      </QueryClientProvider>
    </DeploymentProvider>,
    { url: "/admin/code-execution" },
  )
}

const findRow = async (name: string) => {
  const table = await screen.findByRole("grid", { name: /pricing/ })
  const [cell] = await within(table).findAllByText(name)
  const row = cell.closest('[role="row"]')
  if (!(row instanceof HTMLElement)) throw new Error(`no row for ${name}`)
  return row
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe("ToolPricingPage", () => {
  it.each(["1e3", "0x10", "abc"])(
    "refuses %s as a price rather than letting Number read it",
    async (raw) => {
      const requests = mockApi()
      const user = userEvent.setup()
      await renderPage()

      await user.click(
        await screen.findByRole("button", {
          name: "Edit the price of Code execution",
        }),
      )
      await user.type(await screen.findByLabelText(/Price per call, USD/), raw)

      expect(
        await screen.findByText("An amount in dollars, such as 0.01."),
      ).toBeInTheDocument()
      expect(screen.getByRole("button", { name: "Save" })).toBeDisabled()
      expect(requests.some((request) => request.method === "POST")).toBe(false)
    },
  )

  it("drops a refused save's error once the price is edited", async () => {
    mockApi({ writeRefusal: "Price rejected by the server." })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      await screen.findByRole("button", {
        name: "Edit the price of Code execution",
      }),
    )
    const price = await screen.findByLabelText(/Price per call, USD/)
    await user.type(price, "0.02")
    await user.click(screen.getByRole("button", { name: "Save" }))
    expect(
      await screen.findByText("Price rejected by the server."),
    ).toBeInTheDocument()

    await user.type(price, "5")
    expect(screen.queryByText("Price rejected by the server.")).toBeNull()
  })

  it("warns that an unpriced tool is refused while require_pricing is on", async () => {
    mockApi({ requirePricing: true })
    await renderPage()

    expect(
      await screen.findByText(/a request that would run an unpriced tool/),
    ).toBeInTheDocument()
  })

  it("prices code execution under its one gateway row", async () => {
    mockApi({
      rates: {
        "model:otari:code_execution": buildPerCallRate(
          "otari:code_execution",
          0.02,
        ),
      },
    })
    await renderPage()

    const row = await findRow("Code execution")
    expect(within(row).getByText("otari:code_execution")).toBeInTheDocument()
    expect(await within(row).findByText("$0.02")).toBeInTheDocument()
  })

  it("refuses a caller who does not operate the deployment without asking for a rate", async () => {
    const requests = mockApi({ operator: false })
    await renderPage()

    expect(
      await screen.findByText("Code execution pricing is not available to you"),
    ).toBeInTheDocument()
    expect(
      requests.some((request) => request.url.includes(`${API_ROOT}/pricing`)),
    ).toBe(false)
  })
})
