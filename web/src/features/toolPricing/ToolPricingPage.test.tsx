import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type {
  DeploymentBootstrap,
  PricingResponse,
  SearchToolsResponse,
} from "@/client"
import { ToolPricingPage } from "@/features/toolPricing/ToolPricingPage"
import type { ToolPricingService } from "@/features/toolPricing/toolRates"
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

const SEARCH_TOOLS_ROWS: SearchToolsResponse = {
  stored: [
    {
      name: "exa",
      kind: "search",
      provider: "exa",
      options: {},
      decryptable: true,
      shadows_config: false,
    },
  ],
  config: [
    {
      name: "tavily-prod",
      kind: "search",
      provider: "tavily",
      has_api_key: true,
      shadowed: false,
    },
  ],
}

const FETCH_TOOLS: SearchToolsResponse = {
  stored: [],
  config: [
    {
      name: "builtin_fetch",
      kind: "fetch",
      provider: "builtin",
      has_api_key: false,
      shadowed: false,
    },
  ],
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
    if (url.includes(`${API_ROOT}/search-tools`)) {
      return jsonResponse(
        url.includes("kind=fetch") ? FETCH_TOOLS : SEARCH_TOOLS_ROWS,
      )
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

function renderPage(
  service: ToolPricingService,
  deployment: DeploymentBootstrap = bootstrap(),
  client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  }),
) {
  return renderWithRouter(
    <DeploymentProvider value={deployment}>
      <QueryClientProvider client={client}>
        <ToolPricingPage service={service} />
      </QueryClientProvider>
    </DeploymentProvider>,
    {
      url:
        service === "web_search"
          ? "/admin/web-search"
          : "/admin/code-execution",
    },
  )
}

const findRow = async (name: string) => {
  const table = await screen.findByRole("grid", { name: /pricing/ })
  // An instance is named twice in its row, as the tool and as the key it is
  // priced under, so the first match is enough to find the row.
  const [cell] = await within(table).findAllByText(name)
  const row = cell.closest('[role="row"]')
  if (!(row instanceof HTMLElement)) throw new Error(`no row for ${name}`)
  return row
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe("ToolPricingPage", () => {
  it("prices each web tool and instance, saying where an instance's rate comes from", async () => {
    mockApi({
      rates: {
        "model:otari:web_search": buildPerCallRate("otari:web_search", 0.01),
        "search:exa": buildPerCallRate("exa", 0.005, { kind: "search" }),
      },
    })
    await renderPage("web_search")

    const exa = await findRow("exa")
    expect(await within(exa).findByText("$0.005")).toBeInTheDocument()
    expect(within(exa).getByText("Search instance")).toBeInTheDocument()

    // No rate of its own, so its searches bill at the gateway's search rate.
    const tavily = await findRow("tavily-prod")
    expect(
      await within(tavily).findByText("From otari:web_search"),
    ).toBeInTheDocument()
    expect(within(tavily).getByText("config.yml")).toBeInTheDocument()

    // Neither its own rate nor a fetch rate to fall back to.
    const fetcher = await findRow("builtin_fetch")
    expect(await within(fetcher).findByText("Unpriced")).toBeInTheDocument()
  })

  it("saves an instance's rate as a per-call row of its own kind", async () => {
    const requests = mockApi()
    const user = userEvent.setup()
    await renderPage("web_search")

    await user.click(
      await within(await findRow("exa")).findByRole("button", {
        name: "Edit the price of exa",
      }),
    )
    const price = await screen.findByLabelText(/Price per call, USD/)
    await user.type(price, "0.07")
    await user.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() =>
      expect(
        requests.find(
          (request) =>
            request.method === "POST" && request.url.endsWith("/pricing"),
        )?.body,
      ).toEqual({
        kind: "search",
        model_key: "exa",
        // Rounded: 0.07 * 1e6 is 70000.00000000001 in binary floating point.
        input_price_per_million: 70000,
        output_price_per_million: 0,
        unit: "requests",
      }),
    )
  })

  it.each(["1e3", "0x10", "abc"])(
    "refuses %s as a price rather than letting Number read it",
    async (raw) => {
      const requests = mockApi()
      const user = userEvent.setup()
      await renderPage("code_execution")

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

  it("calls an instance unreadable, not unpriced, when the rate it inherits could not be read", async () => {
    mockApi({ failingReads: ["model:otari:web_search"], requirePricing: true })
    await renderPage("web_search")

    const exa = await findRow("exa")
    expect(await within(exa).findByText("Could not read")).toBeInTheDocument()
    expect(within(exa).queryByText("Unpriced")).toBeNull()
  })

  it("drops a refused save's error once the price is edited", async () => {
    mockApi({ writeRefusal: "Price rejected by the server." })
    const user = userEvent.setup()
    await renderPage("code_execution")

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

  it("leaves a rate config.yml sets to config.yml", async () => {
    mockApi({
      rates: {
        "search:tavily-prod": buildPerCallRate("tavily-prod", 0.008, {
          kind: "search",
          origin: "config",
        }),
      },
    })
    await renderPage("web_search")

    const tavily = await findRow("tavily-prod")
    expect(
      await within(tavily).findByText("Edit in config.yml"),
    ).toBeInTheDocument()
    expect(within(tavily).queryByRole("button", { name: /Edit/ })).toBeNull()
  })

  it("warns that an unpriced tool is refused while require_pricing is on", async () => {
    mockApi({ requirePricing: true })
    await renderPage("code_execution")

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
    await renderPage("code_execution")

    const row = await findRow("Code execution")
    expect(within(row).getByText("otari:code_execution")).toBeInTheDocument()
    expect(await within(row).findByText("$0.02")).toBeInTheDocument()
  })

  it("lists no search instance on code execution, even with them cached from web search", async () => {
    mockApi()
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    })
    // Visit Web search pricing first, which leaves its instances cached.
    const first = await renderPage("web_search", bootstrap(), client)
    await findRow("exa")
    first.unmount()
    await renderPage("code_execution", bootstrap(), client)

    const table = await screen.findByRole("grid", { name: /pricing/ })
    await within(table).findByText("otari:code_execution")
    expect(within(table).queryByText("exa")).toBeNull()
    expect(within(table).queryByText("builtin_fetch")).toBeNull()
  })

  it("refuses a caller who does not operate the deployment without asking for a rate", async () => {
    const requests = mockApi({ operator: false })
    await renderPage("web_search")

    expect(
      await screen.findByText("Web search pricing is not available to you"),
    ).toBeInTheDocument()
    expect(
      requests.some((request) => request.url.includes(`${API_ROOT}/pricing`)),
    ).toBe(false)
  })
})
