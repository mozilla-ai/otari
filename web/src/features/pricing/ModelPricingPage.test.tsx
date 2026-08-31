import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactElement } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import type {
  GatewaySettings,
  OrganizationContext,
  PricingResponse,
} from "@/client"
import { ModelPricingPage } from "@/features/pricing/ModelPricingPage"
import { organizationContext } from "@/tests/fixtures"
import { withRouter } from "@/tests/router"

const SETTINGS: GatewaySettings = {
  mode: "standalone",
  version: "1.2.3",
  model_discovery: true,
  default_pricing: true,
  require_pricing: false,
  master_key_source: "generated",
  secret_key_configured: true,
  config: [],
}

const PRICE_REFRESH = {
  fetched_at: "2026-07-22T00:00:00Z",
  added_count: 1,
  changed_count: 2,
  removed_count: 0,
  protected_model_count: 1,
  changes: [
    { model_key: "openai:gpt-4o-mini", change: "changed" },
    { model_key: "openai:gpt-5", change: "added" },
  ],
  changes_truncated: false,
}

function price(overrides: Partial<PricingResponse> = {}): PricingResponse {
  return {
    model_key: "openai:gpt-5",
    input_price_per_million: 1.25,
    output_price_per_million: 10,
    cache_read_price_per_million: null,
    cache_write_price_per_million: null,
    cache_write_1h_price_per_million: null,
    pricing_tiers: [],
    effective_at: "2026-01-01T00:00:00Z",
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    ...overrides,
  }
}

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  })
}

// The caller's standing, which decides how much of this page renders. An
// operator by default, matching the fixture and the deployment most of these
// tests describe; the admin cases below say otherwise.
function mockApi(
  options: {
    settings?: Partial<GatewaySettings>
    pricing?: PricingResponse[]
    context?: OrganizationContext
  } = {},
) {
  const settings = { ...SETTINGS, ...options.settings }
  const pricing = options.pricing ?? [price()]
  const context = options.context ?? organizationContext()
  return vi
    .spyOn(globalThis, "fetch")
    .mockImplementation(async (input, init) => {
      const url = String(input)
      const method = (init?.method ?? "GET").toUpperCase()
      if (url.includes("/v1/pricing/refresh/confirm") && method === "POST") {
        return jsonResponse({ applied: true })
      }
      if (url.includes("/v1/pricing/refresh/reject") && method === "POST") {
        return new Response(null, { status: 204 })
      }
      if (url.includes("/v1/pricing/refresh") && method === "POST") {
        return jsonResponse(PRICE_REFRESH)
      }
      // Before the bare `/v1/pricing` arm below and before the context one: the
      // organization's own overrides are a different surface from the catalog,
      // and they answer the paged tenancy shape rather than a list.
      if (url.includes("/v1/organizations/me/pricing")) {
        return jsonResponse({ data: [], count: 0 })
      }
      if (url.includes("/v1/settings")) return jsonResponse(settings)
      if (url.includes("/v1/pricing")) return jsonResponse(pricing)
      if (url.includes("/v1/organizations/me")) return jsonResponse(context)
      return jsonResponse([])
    })
}

function renderPage(ui: ReactElement) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>{ui}</QueryClientProvider>,
    { wrapper: withRouter({ url: "/organization/pricing" }) },
  )
}

describe("ModelPricingPage", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("lists the price in force per model, not the whole history", async () => {
    // /v1/pricing returns one row per effective_at, so a repriced model appears
    // more than once. Only the newest row whose date has passed is what a request
    // is metered at, and showing both would read as two prices for one model.
    mockApi({
      pricing: [
        price({
          input_price_per_million: 1,
          effective_at: "2026-01-01T00:00:00Z",
        }),
        price({
          input_price_per_million: 3,
          effective_at: "2026-02-01T00:00:00Z",
        }),
      ],
    })
    renderPage(<ModelPricingPage />)

    const table = await screen.findByRole("grid", { name: "Model prices" })
    const rows = within(table).getAllByRole("row")
    // One header row and one model row.
    expect(rows).toHaveLength(2)
    expect(within(table).getByText("$3.00")).toBeInTheDocument()
    expect(within(table).queryByText("$1.00")).toBeNull()
  })

  it("keeps a sub-cent rate legible instead of rounding it to nothing", async () => {
    // Two cents to the dollar is not enough precision for a per-million rate: at
    // two digits a cache-read price of 0.0025 renders as "$0.00", which is what
    // this table's em dash means (no cache-read rate at all). `formatCost` is
    // what Models formats the same stored numbers with, so the two pages cannot
    // print different figures for one price.
    mockApi({
      pricing: [
        price({
          input_price_per_million: 0.0025,
          cache_read_price_per_million: 0.075,
        }),
      ],
    })
    renderPage(<ModelPricingPage />)

    const table = await screen.findByRole("grid", { name: "Model prices" })
    expect(within(table).getByText("$0.0025")).toBeInTheDocument()
    expect(within(table).getByText("$0.08")).toBeInTheDocument()
    expect(within(table).queryByText("$0.00")).toBeNull()
  })

  it("says an unset cache rate is unset rather than free", async () => {
    mockApi({ pricing: [price({ cache_read_price_per_million: null })] })
    renderPage(<ModelPricingPage />)

    const table = await screen.findByRole("grid", { name: "Model prices" })
    // A model with no cache-read rate is not one that reads cache for free, and
    // "$0.00" would say the second.
    expect(within(table).queryByText("$0.00")).toBeNull()
    expect(within(table).getAllByText("—").length).toBeGreaterThan(0)
  })

  it("says what an unpriced model costs when defaults are on", async () => {
    mockApi({ settings: { default_pricing: true } })
    renderPage(<ModelPricingPage />)

    expect(await screen.findByText(/Default pricing is on/)).toBeInTheDocument()
  })

  it("warns that the table is everything billable when defaults are off", async () => {
    // The consequential case: with defaults off, a model absent from this table
    // is either refused or served for free, and which one is a separate switch.
    mockApi({ settings: { default_pricing: false, require_pricing: true } })
    renderPage(<ModelPricingPage />)

    expect(
      await screen.findByText(/Default pricing is off/),
    ).toBeInTheDocument()
    expect(screen.getByText(/HTTP 402/)).toBeInTheDocument()
  })

  it("says a request is served for free when nothing requires a price", async () => {
    mockApi({ settings: { default_pricing: false, require_pricing: false } })
    renderPage(<ModelPricingPage />)

    expect(await screen.findByText(/metered at zero/)).toBeInTheDocument()
  })

  it("offers to price a model elsewhere when none is priced yet", async () => {
    mockApi({ pricing: [] })
    renderPage(<ModelPricingPage />)

    expect(
      await screen.findByText(/No model carries a stored price yet/),
    ).toBeInTheDocument()
  })

  it("reports a failed price read rather than an empty catalog", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input)
      if (url.includes("/v1/settings")) return jsonResponse(SETTINGS)
      return new Response(JSON.stringify({ detail: "nope" }), { status: 500 })
    })
    renderPage(<ModelPricingPage />)

    // An empty table after a failed read says "nothing is priced", which is the
    // opposite of what a 500 means.
    expect(await screen.findByRole("alert")).toBeInTheDocument()
  })

  it("reviews and accepts persisted default price updates", async () => {
    const fetchMock = mockApi()
    const user = userEvent.setup()

    renderPage(<ModelPricingPage />)
    await screen.findByText("Default pricing catalog")

    await user.click(
      screen.getByRole("button", { name: "Check for price updates" }),
    )

    expect(
      await screen.findByRole("alertdialog", {
        name: "Review default price updates",
      }),
    ).toBeInTheDocument()
    expect(
      screen.getByText(/1 added, 2 changed, and 0 removed/),
    ).toBeInTheDocument()
    expect(screen.getByText("openai:gpt-4o-mini: changed")).toBeInTheDocument()

    await user.click(
      screen.getByRole("button", { name: "Accept price updates" }),
    )

    await waitFor(() =>
      expect(
        screen.queryByRole("alertdialog", {
          name: "Review default price updates",
        }),
      ).not.toBeInTheDocument(),
    )
    expect(
      fetchMock.mock.calls.some(
        ([url, init]) =>
          String(url).endsWith("/v1/pricing/refresh/confirm") &&
          init?.method === "POST",
      ),
    ).toBe(true)
  })

  it("rejects a reviewed default price update", async () => {
    const fetchMock = mockApi()
    const user = userEvent.setup()

    renderPage(<ModelPricingPage />)
    await screen.findByText("Default pricing catalog")
    await user.click(
      screen.getByRole("button", { name: "Check for price updates" }),
    )
    await screen.findByRole("alertdialog", {
      name: "Review default price updates",
    })

    await user.click(screen.getByRole("button", { name: "Reject changes" }))

    await waitFor(() =>
      expect(
        screen.queryByRole("alertdialog", {
          name: "Review default price updates",
        }),
      ).not.toBeInTheDocument(),
    )
    expect(
      fetchMock.mock.calls.some(
        ([url, init]) =>
          String(url).endsWith("/v1/pricing/refresh/reject") &&
          init?.method === "POST",
      ),
    ).toBe(true)
  })

  it("gives an organization admin the prices and its own overrides, not the catalog", async () => {
    // The roles matrix puts Model pricing at Edit for an admin (otari-ai#1943),
    // and the page is two halves: the organization's rate overrides, which the
    // server already lets an owner or admin write, and the deployment's catalog
    // controls, which it does not. So an admin gets the first and the price
    // table, and the second is withheld rather than rendered as a refusal.
    const fetchMock = mockApi({
      // An admin, not the owner the fixture defaults to: the matrix row is
      // about the admin, and `canManage` is what the overrides card asks.
      context: organizationContext({
        role: "admin",
        deployment_operator: false,
      }),
    })
    renderPage(<ModelPricingPage />)

    expect(await screen.findByText("Rate overrides")).toBeInTheDocument()
    await screen.findByRole("grid", { name: "Model prices" })
    expect(screen.queryByText("Default pricing catalog")).toBeNull()
    expect(
      screen.queryByRole("button", { name: "Check for price updates" }),
    ).toBeNull()
    expect(screen.queryByText(/Default pricing is/)).toBeNull()

    // Withheld at the request, not only in the markup. Both reads are
    // `require_deployment_operator`, so firing them would put a 403 banner on a
    // page that is the admin's to use (the shape otari#838 removed elsewhere).
    const asked = fetchMock.mock.calls.map(([url]) => String(url))
    expect(asked.some((url) => url.includes("/v1/settings"))).toBe(false)
    expect(asked.some((url) => url.includes("/v1/pricing/refresh"))).toBe(false)
  })

  it("does not point an admin at an editor they would be refused", async () => {
    // Since otari#867 a non-operator reads Models with every pricing affordance
    // gone, and setting a catalog rate is operator-only, so the empty table says
    // nothing is priced without offering to price one.
    mockApi({
      pricing: [],
      context: organizationContext({ deployment_operator: false }),
    })
    renderPage(<ModelPricingPage />)

    expect(
      await screen.findByText("No model carries a stored price yet."),
    ).toBeInTheDocument()
    expect(screen.queryByText(/Price one from the Models page/)).toBeNull()
    expect(screen.queryByText(/A rate is edited beside the model/)).toBeNull()
  })

  it("keeps the catalog controls for a deployment operator", async () => {
    // The other side of the split, pinned so a future change cannot quietly take
    // the catalog away from the caller it belongs to.
    mockApi()
    renderPage(<ModelPricingPage />)

    expect(
      await screen.findByText("Default pricing catalog"),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Check for price updates" }),
    ).toBeInTheDocument()
  })
})
