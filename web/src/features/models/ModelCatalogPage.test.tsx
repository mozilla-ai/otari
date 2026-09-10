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
  id: "glm-5-3",
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
  id: "kimi-k2-6",
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
  offerings: [
    offering({}),
    offering({
      selector: "fireworks:accounts/fireworks/models/glm-5p3",
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
    if (url.includes("/v1/catalog/models/glm-5-3")) {
      return jsonResponse(GLM_DETAIL)
    }
    if (url.includes("/v1/catalog/models/")) {
      return jsonResponse({ detail: "Model 'nope' not found" }, 404)
    }
    if (url.includes("/v1/catalog/models")) return jsonResponse(catalog)
    if (url.includes("/v1/organizations/me")) return jsonResponse(context)
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
            path: "/models/$modelId",
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
  })

  it("lists one card per model with the cheapest offering's price and a link to its page", async () => {
    mockApi()
    renderPage(<ModelCatalogPage />)

    const list = await screen.findByRole("list", { name: "Models" })
    const glm = within(list).getByRole("link", { name: "Z.ai: GLM-5.3" })
    expect(glm).toHaveAttribute("href", "/models/glm-5-3")
    const card = glm.closest("article") as HTMLElement
    // Two providers folded into one card, priced from the cheaper.
    expect(within(card).getByText("2 providers")).toBeInTheDocument()
    expect(within(card).getByText("$0.50/M input tokens")).toBeInTheDocument()
    expect(within(card).getByText("$2.00/M output tokens")).toBeInTheDocument()
    expect(within(card).getByText("200K context")).toBeInTheDocument()
    expect(within(card).getByText("Z.ai's flagship.")).toBeInTheDocument()
    expect(screen.getByText(/2 models across 2 providers/)).toBeInTheDocument()
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
    expect(screen.getByRole("checkbox", { name: "fireworks" })).toBeChecked()
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

  it("re-reads the list at a comparison size", async () => {
    const fetchMock = mockApi()
    renderPage(<ModelCatalogPage />)
    const user = userEvent.setup()

    await screen.findByRole("list", { name: "Models" })
    await user.click(screen.getByRole("button", { name: /Compare prices at/ }))
    await user.click(screen.getByRole("radio", { name: "Compare at 200K" }))

    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some(([url]) =>
          String(url).includes("/v1/catalog/models?at_context=200000"),
        ),
      ).toBe(true),
    )
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
})
