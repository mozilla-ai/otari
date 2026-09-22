import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactElement } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import type {
  CatalogModelDetail,
  CatalogModelSummary,
  CatalogOffering,
  CatalogResponse,
  OrganizationContext,
} from "@/client"
import { ModelCatalogPage } from "@/features/models/ModelCatalogPage"
import { API_ROOT } from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap, organizationContext } from "@/tests/fixtures"
import { withRouter } from "@/tests/router"

const CAPABILITIES = {
  reasoning: true,
  tool_call: true,
  structured_output: false,
  attachment: false,
  temperature: true,
}

const GLM: CatalogModelSummary = {
  id: "z-ai/glm-5.3",
  name: "GLM-5.3",
  vendor: "Z.ai",
  description: "Z.ai's flagship.",
  family: "glm",
  capabilities: CAPABILITIES,
  input_modalities: ["text"],
  output_modalities: ["text"],
  context_window: 200_000,
  max_output_tokens: 128_000,
  release_date: "2026-07-01",
  knowledge_cutoff: null,
  open_weights: true,
  deprecated: false,
  offering_count: 2,
  provider_count: 2,
  providers: ["fireworks", "nebius"],
  selector: "z-ai/glm-5.3",
  resolves_to: "nebius:zai-org/GLM-5.3",
  selectors: [
    "fireworks:accounts/fireworks/models/glm-5p3",
    "nebius:zai-org/GLM-5.3",
  ],
  price_sources: ["defaults", "deployment"],
  unpriced_count: 0,
  discovered: true,
  min_input_price_per_million: 0.5,
  min_output_price_per_million: 2,
}

const KIMI: CatalogModelSummary = {
  ...GLM,
  id: "moonshotai/kimi-k2.6",
  name: "Kimi K2.6",
  vendor: "Moonshot AI",
  description: null,
  family: null,
  capabilities: { ...CAPABILITIES, reasoning: false },
  context_window: 262_144,
  open_weights: false,
  offering_count: 1,
  provider_count: 1,
  providers: ["nebius"],
  selector: null,
  resolves_to: null,
  selectors: ["nebius:moonshotai/Kimi-K2.6"],
  price_sources: ["defaults"],
  unpriced_count: 0,
  discovered: true,
  min_input_price_per_million: 0.6,
  min_output_price_per_million: 2.4,
}

function offering(overrides: Partial<CatalogOffering>): CatalogOffering {
  return {
    selector: "nebius:zai-org/GLM-5.3",
    short_selector: "nebius:z-ai/glm-5.3",
    provider: "nebius",
    provider_type: "nebius",
    credential: "deployment",
    discovered: true,
    context_window: 200_000,
    max_output_tokens: 128_000,
    quantization: null,
    pricing: {
      input_price_per_million: 0.5,
      output_price_per_million: 2,
      cache_read_price_per_million: 0.05,
      cache_write_price_per_million: null,
      cache_write_1h_price_per_million: null,
      pricing_tiers: [],
      unit: "tokens",
    },
    price_source: "deployment",
    price_reference: "nebius:zai-org/GLM-5.3",
    ...overrides,
  }
}

const GLM_DETAIL: CatalogModelDetail = {
  ...GLM,
  default_pricing: true,
  offerings: [
    offering({}),
    offering({
      selector: "fireworks:accounts/fireworks/models/glm-5p3",
      short_selector: "fireworks:z-ai/glm-5.3",
      provider: "fireworks",
      provider_type: "fireworks",
      context_window: 131_072,
      max_output_tokens: 16_384,
      pricing: {
        input_price_per_million: 0.075,
        output_price_per_million: 2.5,
        cache_read_price_per_million: null,
        cache_write_price_per_million: null,
        cache_write_1h_price_per_million: null,
        pricing_tiers: [],
        unit: "tokens",
      },
      price_source: "defaults",
      price_reference: "fireworks:glm-5p3",
    }),
  ],
  also_available_from: [{ provider_type: "groq", name: "Groq" }],
}

const CATALOG: CatalogResponse = {
  default_pricing: true,
  defaults_as_of: null,
  metadata_available: true,
  count: 2,
  models: [GLM, KIMI],
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

function mockApi(
  options: { catalog?: CatalogResponse; context?: OrganizationContext } = {},
) {
  const catalog = options.catalog ?? CATALOG
  const context = options.context ?? organizationContext()
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input)
    if (url.includes(`${API_ROOT}/catalog/models/z-ai/glm-5.3`)) {
      return jsonResponse(GLM_DETAIL)
    }
    if (url.includes(`${API_ROOT}/catalog/models/`)) {
      return jsonResponse({ detail: "Model 'nope' not found" }, 404)
    }
    if (url.includes(`${API_ROOT}/catalog/models`)) return jsonResponse(catalog)
    if (url.includes(`${API_ROOT}/organizations/me`))
      return jsonResponse(context)
    return jsonResponse([])
  })
}

function renderPage(ui: ReactElement, url = "/models") {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <DeploymentProvider value={bootstrap()}>
      <QueryClientProvider client={client}>{ui}</QueryClientProvider>
    </DeploymentProvider>,
    {
      wrapper: withRouter({
        url,
        routes: [
          {
            path: "/models/$vendor/$model",
            element: <span>opened a model</span>,
          },
        ],
      }),
    },
  )
}

describe("ModelCatalogPage", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    localStorage.clear()
  })

  it("lists one card per model with the cheapest offering's price and a link to its page", async () => {
    mockApi()
    renderPage(<ModelCatalogPage />)

    const list = await screen.findByRole("list", { name: "Models" })
    const glm = within(list).getByRole("link", { name: "Z.ai: GLM-5.3" })
    expect(glm).toHaveAttribute("href", "/models/z-ai/glm-5.3")
    const card = glm.closest("article") as HTMLElement
    // Two providers folded into one card, priced from the cheaper.
    expect(within(card).getByText("2 providers")).toBeInTheDocument()
    expect(within(card).getByText("$0.50/M input tokens")).toBeInTheDocument()
    expect(within(card).getByText("$2.00/M output tokens")).toBeInTheDocument()
    expect(within(card).getByText("200K context")).toBeInTheDocument()
    expect(within(card).getByText("Z.ai's flagship.")).toBeInTheDocument()
    expect(screen.getByText(/2 models across 2 providers/)).toBeInTheDocument()
  })

  it("marks the maker on a card and in the table", async () => {
    // The wiring, not the resolution: `brandMarks` pins which glyph a slug
    // gets, and this pins that the page asks. Z.ai has a mark, so a card's
    // meta line carries an <svg> beside "by Z.ai" and so does the table's
    // sub-line under the same model.
    mockApi()
    renderPage(<ModelCatalogPage />)

    const list = await screen.findByRole("list", { name: "Models" })
    const byLine = within(list).getByText(/by Z\.ai/)
    // The geometry is on a lazy chunk, so the mark lands a microtask after the
    // row. `waitFor` is the await; the reserved box is what renders until then.
    await waitFor(() => expect(byLine.querySelector("svg")).not.toBeNull())

    await userEvent.click(screen.getByRole("radio", { name: "Table" }))
    // `closest("span")` would return the sub-line itself; the mark is its
    // sibling, so the assertion is on the parent.
    const subLine = await screen.findByText(/Z\.ai · 2 providers/)
    await waitFor(() =>
      expect(subLine.parentElement?.querySelector("svg")).not.toBeNull(),
    )
  })

  it("marks the makers and the providers in the rail", async () => {
    mockApi()
    renderPage(<ModelCatalogPage />)
    const user = userEvent.setup()
    await screen.findByRole("list", { name: "Models" })

    // Both groups fold until something in them is chosen, and a hidden row is
    // not in the accessibility tree.
    await user.click(screen.getByRole("button", { name: "Vendors" }))
    await user.click(screen.getByRole("button", { name: "Providers" }))

    // Both rail lists resolve at least one mark on this seed, so both reserve
    // the slot and every row in them carries one.
    const vendors = screen.getByRole("checkbox", { name: /Z\.ai/ })
    const provider = screen.getByRole("checkbox", { name: /Fireworks AI/ })
    await waitFor(() => {
      expect(vendors.closest("label")?.querySelector("svg")).not.toBeNull()
      expect(provider.closest("label")?.querySelector("svg")).not.toBeNull()
    })
  })

  it("opens a model from the description inside its card", async () => {
    mockApi()
    renderPage(<ModelCatalogPage />)
    const user = userEvent.setup()
    const description = await screen.findByText("Z.ai's flagship.")
    expect(description.closest("a")).toHaveAccessibleName("Z.ai: GLM-5.3")
    await user.click(description)
    expect(await screen.findByText("opened a model")).toBeInTheDocument()
  })

  it.each(["List", "Table"])(
    "remembers %s after leaving and remounting the catalog",
    async (view) => {
      mockApi()
      const user = userEvent.setup()
      const first = renderPage(<ModelCatalogPage />)
      await user.click(await screen.findByRole("radio", { name: "Table" }))
      if (view === "List")
        await user.click(screen.getByRole("radio", { name: "List" }))
      first.unmount()
      renderPage(<ModelCatalogPage />)
      expect(await screen.findByRole("radio", { name: view })).toBeChecked()
      expect(
        await screen.findByRole(view === "Table" ? "grid" : "list", {
          name: "Models",
        }),
      ).toBeInTheDocument()
    },
  )

  it("keeps the view switch usable when browser storage is blocked", async () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new DOMException("Storage blocked", "SecurityError")
    })
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new DOMException("Storage blocked", "SecurityError")
    })
    mockApi()
    renderPage(<ModelCatalogPage />)
    const user = userEvent.setup()
    await screen.findByRole("list", { name: "Models" })
    await user.click(screen.getByRole("radio", { name: "Table" }))
    expect(
      await screen.findByRole("grid", { name: "Models" }),
    ).toBeInTheDocument()
  })

  it("falls back to list for an invalid saved view", async () => {
    localStorage.setItem("otari.dashboard.modelsView", "invalid")
    mockApi()
    renderPage(<ModelCatalogPage />)
    expect(
      await screen.findByRole("list", { name: "Models" }),
    ).toBeInTheDocument()
  })

  it("narrows the list to the provider the Providers page linked with", async () => {
    mockApi()
    renderPage(<ModelCatalogPage />, "/models?provider=fireworks")

    const list = await screen.findByRole("list", { name: "Models" })
    expect(
      within(list).getByRole("link", { name: "Z.ai: GLM-5.3" }),
    ).toBeInTheDocument()
    expect(within(list).queryByText(/Kimi K2.6/)).toBeNull()
    // The rail says one provider is in force.
    // The rail names the vendor and filters on the instance id (otari#990).
    expect(screen.getByRole("checkbox", { name: "Fireworks AI" })).toBeChecked()
  })

  it("searches by vendor", async () => {
    mockApi()
    renderPage(<ModelCatalogPage />)
    const user = userEvent.setup()

    await screen.findByRole("list", { name: "Models" })
    await user.type(
      screen.getByRole("searchbox", { name: "Search models" }),
      "moonshot",
    )

    const list = screen.getByRole("list", { name: "Models" })
    expect(within(list).queryByText(/GLM-5.3/)).toBeNull()
    expect(within(list).getByText(/Kimi K2.6/)).toBeInTheDocument()
  })

  it("narrows by a checkbox in the rail and says how many are in force", async () => {
    mockApi()
    renderPage(<ModelCatalogPage />)
    const user = userEvent.setup()

    await screen.findByRole("list", { name: "Models" })
    // Capabilities is folded until something in it is chosen.
    await user.click(screen.getByRole("button", { name: "Capabilities" }))
    await user.click(screen.getByRole("checkbox", { name: "Reasoning" }))

    const list = screen.getByRole("list", { name: "Models" })
    expect(within(list).queryByText(/Kimi K2.6/)).toBeNull()
    expect(within(list).getByText(/GLM-5.3/)).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: /^Capabilities\s*1$/ }),
    ).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Clear" }))
    expect(within(list).getByText(/Kimi K2.6/)).toBeInTheDocument()
  })

  it("narrows by what a model produces", async () => {
    mockApi({
      catalog: {
        ...CATALOG,
        models: [GLM, { ...KIMI, output_modalities: ["image"] }],
      },
    })
    renderPage(<ModelCatalogPage />)
    const user = userEvent.setup()

    await screen.findByRole("list", { name: "Models" })
    await user.click(screen.getByRole("button", { name: "Output modalities" }))
    // Two groups offer "Image"; the second is the output one.
    const image = screen.getAllByRole("checkbox", { name: "Image" })[1]
    await user.click(image as HTMLElement)

    const list = screen.getByRole("list", { name: "Models" })
    expect(within(list).queryByText(/GLM-5.3/)).toBeNull()
    expect(within(list).getByText(/Kimi K2.6/)).toBeInTheDocument()
  })

  it("offers the same rows as a table", async () => {
    mockApi()
    renderPage(<ModelCatalogPage />)
    const user = userEvent.setup()

    await screen.findByRole("list", { name: "Models" })
    await user.click(screen.getByRole("radio", { name: "Table" }))

    const grid = await screen.findByRole("grid", { name: "Models" })
    expect(within(grid).getByText("GLM-5.3")).toBeInTheDocument()
    expect(within(grid).getByText("from $0.50")).toBeInTheDocument()
    const row = within(grid).getByRole("row", { name: /GLM-5.3/ })
    await user.click(within(row).getByRole("gridcell", { name: "200K" }))
    expect(await screen.findByText("opened a model")).toBeInTheDocument()
  })

  it("opens a model when its card is pressed", async () => {
    mockApi()
    renderPage(<ModelCatalogPage />)
    const user = userEvent.setup()

    await user.click(
      await screen.findByRole("link", { name: "Moonshot AI: Kimi K2.6" }),
    )

    expect(await screen.findByText("opened a model")).toBeInTheDocument()
  })

  it("asks for the whole catalog, not the endpoint's default window", async () => {
    const fetchMock = mockApi()
    renderPage(<ModelCatalogPage />)

    await screen.findByRole("list", { name: "Models" })

    const listCall = fetchMock.mock.calls
      .map(([input]) => String(input))
      .find((url) => url.includes(`${API_ROOT}/catalog/models?`))
    expect(listCall).toContain("limit=1000")
  })

  it("says so when the catalog is larger than the page holds", async () => {
    mockApi({ catalog: { ...CATALOG, count: 126 } })
    renderPage(<ModelCatalogPage />)

    // Every facet on this page is computed from what arrived, so the count the
    // reader is shown has to be the one they can actually filter.
    expect(await screen.findByText(/holds the first 2/)).toBeInTheDocument()
    expect(screen.getByText(/remaining/)).toHaveTextContent("124")
  })

  it("draws no truncation notice when the whole catalog arrived", async () => {
    mockApi()
    renderPage(<ModelCatalogPage />)

    await screen.findByRole("list", { name: "Models" })
    expect(screen.queryByText(/holds the first/)).not.toBeInTheDocument()
  })
})
