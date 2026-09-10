import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, within } from "@testing-library/react"
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
  description: "Z.ai's flagship.",
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

  it("lists one row per model with the cheapest offering's price", async () => {
    mockApi()
    renderPage(<ModelCatalogPage />)

    // The grid renders its loading row before the rows, so wait on a row.
    const glm = (await screen.findByText("GLM-5.3")).closest(
      "tr",
    ) as HTMLElement
    // Two providers folded into one row, priced from the cheaper.
    expect(within(glm).getByText("from $0.50")).toBeInTheDocument()
    expect(
      within(glm).getByText(/Z\.ai · 2 providers · up to 200K/),
    ).toBeInTheDocument()
    expect(screen.getByText(/2 models across 2 providers/)).toBeInTheDocument()
    // Nothing selected: the detail column says so instead of guessing.
    expect(
      screen.getByText(/Select a model to compare the providers/),
    ).toBeInTheDocument()
  })

  it("narrows the list to the provider the Providers page linked with", async () => {
    mockApi()
    renderPage(<ModelCatalogPage />, "/models?provider=fireworks")

    const table = await screen.findByRole("grid", { name: "Models" })
    expect(await within(table).findByText("GLM-5.3")).toBeInTheDocument()
    expect(within(table).queryByText("Kimi K2.6")).toBeNull()
  })

  it("searches by vendor", async () => {
    mockApi()
    renderPage(<ModelCatalogPage />)
    const user = userEvent.setup()

    await screen.findByText("GLM-5.3")
    await user.type(
      screen.getByRole("searchbox", { name: "Search models" }),
      "moonshot",
    )

    const table = screen.getByRole("grid", { name: "Models" })
    expect(within(table).queryByText("GLM-5.3")).toBeNull()
    expect(within(table).getByText("Kimi K2.6")).toBeInTheDocument()
  })

  it("shows the selected model's offerings, cheapest first, with each price's source", async () => {
    mockApi()
    renderPage(<ModelCatalogPage modelId="glm-5-3" />, "/models/glm-5-3")

    const panel = await screen.findByRole("complementary", {
      name: "Model details",
    })
    expect(
      await within(panel).findByRole("heading", { name: "GLM-5.3" }),
    ).toBeInTheDocument()
    expect(within(panel).getByText("Z.ai's flagship.")).toBeInTheDocument()
    // Derived limits carry their rule: the largest any offering serves.
    expect(within(panel).getByText("up to 200K")).toBeInTheDocument()

    const offerings = within(panel).getByRole("grid", {
      name: "Offerings of GLM-5.3",
    })
    const rows = within(offerings).getAllByRole("row").slice(1)
    expect(rows).toHaveLength(2)
    expect(
      within(rows[0] as HTMLElement).getByText("nebius:zai-org/GLM-5.3"),
    ).toBeInTheDocument()
    expect(
      within(rows[0] as HTMLElement).getByText("CUSTOM"),
    ).toBeInTheDocument()
    // A sub-cent rate keeps its digits (otari#700).
    expect(
      within(rows[1] as HTMLElement).getByText("$0.075"),
    ).toBeInTheDocument()
    expect(
      within(rows[1] as HTMLElement).getByText("DEFAULT"),
    ).toBeInTheDocument()
    // The provider models.dev knows and this deployment does not.
    expect(within(panel).getByText(/Also served by Groq/)).toBeInTheDocument()
    // The runnable request names the cheapest offering.
    const curl = within(panel).getByLabelText("cURL") as HTMLTextAreaElement
    expect(curl.value).toContain("nebius:zai-org/GLM-5.3")
  })

  it("links an operator to Model pricing to edit a rate, and nobody else", async () => {
    mockApi()
    renderPage(<ModelCatalogPage modelId="glm-5-3" />, "/models/glm-5-3")

    const panel = await screen.findByRole("complementary", {
      name: "Model details",
    })
    const links = await within(panel).findAllByRole("link", {
      name: "Edit rate",
    })
    expect(links).toHaveLength(2)
    expect(links[0]).toHaveAttribute(
      "href",
      "/organization/pricing?model=nebius%3Azai-org%2FGLM-5.3",
    )
    // Nothing on this page writes a price.
    expect(within(panel).queryByRole("button", { name: /price/i })).toBeNull()
  })

  it("points an organization admin at its own rate override, not the deployment's price", async () => {
    mockApi({
      context: organizationContext({
        role: "admin",
        deployment_operator: false,
      }),
    })
    renderPage(<ModelCatalogPage modelId="glm-5-3" />, "/models/glm-5-3")

    const panel = await screen.findByRole("complementary", {
      name: "Model details",
    })
    const links = await within(panel).findAllByRole("link", {
      name: "Set your rate",
    })
    expect(links[0]).toHaveAttribute(
      "href",
      "/organization/pricing?override=nebius%3Azai-org%2FGLM-5.3",
    )
    expect(within(panel).queryByRole("link", { name: "Edit rate" })).toBeNull()
  })

  it("shows what the organization was charged for an offering beside its sticker price", async () => {
    mockApi()
    GLM_DETAIL.offerings[0] = offering({
      usage_30d: {
        requests: 42,
        total_tokens: 1_000_000,
        cache_read_tokens: 250_000,
        spend_usd: 0.4,
        cache_hit_rate: 0.25,
        effective_price_per_million: 0.4,
      },
    })
    try {
      renderPage(<ModelCatalogPage modelId="glm-5-3" />, "/models/glm-5-3")

      const panel = await screen.findByRole("complementary", {
        name: "Model details",
      })
      const grid = await within(panel).findByRole("grid", {
        name: "Offerings of GLM-5.3",
      })
      expect(
        within(grid).getByRole("columnheader", { name: "Yours, 30d" }),
      ).toBeInTheDocument()
      expect(within(grid).getByText("42 req · cache 25%")).toBeInTheDocument()
    } finally {
      GLM_DETAIL.offerings[0] = offering({})
    }
  })

  it("marks a metered rate that differs from the provider's list price", async () => {
    mockApi()
    GLM_DETAIL.offerings[0] = offering({
      metadata_input_price_per_million: 1,
      metadata_output_price_per_million: 2.01,
    })
    try {
      renderPage(<ModelCatalogPage modelId="glm-5-3" />, "/models/glm-5-3")

      const panel = await screen.findByRole("complementary", {
        name: "Model details",
      })
      await within(panel).findByRole("grid", { name: "Offerings of GLM-5.3" })
      // Input is metered at half the list price and says so; output is within
      // rounding of it and does not.
      expect(within(panel).getByText("list $1.00")).toBeInTheDocument()
      expect(within(panel).queryByText("list $2.01")).toBeNull()
      expect(
        within(panel).getByText(
          /differs from the price the provider publishes/,
        ),
      ).toBeInTheDocument()
    } finally {
      GLM_DETAIL.offerings[0] = offering({})
    }
  })

  it("keeps the catalog read-only for a member", async () => {
    mockApi({ context: organizationContext({ deployment_operator: false }) })
    renderPage(<ModelCatalogPage modelId="glm-5-3" />, "/models/glm-5-3")

    const panel = await screen.findByRole("complementary", {
      name: "Model details",
    })
    await within(panel).findByRole("grid", { name: "Offerings of GLM-5.3" })
    expect(within(panel).queryByRole("link", { name: "Edit rate" })).toBeNull()
    expect(
      within(panel).queryByRole("link", { name: "Add a provider" }),
    ).toBeNull()
  })

  it("opens a model when its row is pressed", async () => {
    mockApi()
    renderPage(<ModelCatalogPage />)
    const user = userEvent.setup()

    await user.click(await screen.findByText("Kimi K2.6"))

    expect(await screen.findByText("opened a model")).toBeInTheDocument()
  })

  it("reports a model that does not exist rather than an empty panel", async () => {
    mockApi()
    renderPage(<ModelCatalogPage modelId="nope" />, "/models/nope")

    expect(await screen.findByRole("alert")).toHaveTextContent(/not found/)
  })

  it("says why an offering is unpriced when defaults are off", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input)
      if (url.includes("/v1/catalog/models/glm-5-3")) {
        return jsonResponse({
          ...GLM_DETAIL,
          offerings: [
            offering({
              pricing: null,
              price_source: null,
              price_reference: null,
            }),
          ],
        })
      }
      if (url.includes("/v1/catalog/models")) {
        return jsonResponse({ ...CATALOG, default_pricing: false })
      }
      if (url.includes("/v1/organizations/me"))
        return jsonResponse(organizationContext())
      return jsonResponse([])
    })
    renderPage(<ModelCatalogPage modelId="glm-5-3" />, "/models/glm-5-3")

    expect(
      await screen.findByText(
        /Default pricing is off, so an offering with no stored rate/,
      ),
    ).toBeInTheDocument()
  })
})
