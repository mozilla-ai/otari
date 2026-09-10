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
import { ModelDetailPage } from "@/features/models/ModelDetailPage"
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
    short_selector: "nebius:glm-5.3",
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
      short_selector: "fireworks:glm-5p3",
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
    if (url.includes("/v1/catalog/models/z-ai/glm-5.3")) {
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

function renderPage(ui: ReactElement, url = "/models/z-ai/glm-5.3") {
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

describe("ModelDetailPage", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("shows the model's facts and its offerings, cheapest first, with each price's source", async () => {
    mockApi()
    renderPage(<ModelDetailPage modelId="z-ai/glm-5.3" />)

    expect(
      await screen.findByRole("heading", { name: "Z.ai: GLM-5.3" }),
    ).toBeInTheDocument()
    expect(screen.getByText("Z.ai's flagship.")).toBeInTheDocument()
    // Derived limits carry their rule: the largest any offering serves.
    expect(screen.getByText("up to 200K")).toBeInTheDocument()
    expect(screen.getByRole("link", { name: "← All models" })).toHaveAttribute(
      "href",
      "/models",
    )

    const offerings = screen.getByRole("grid", { name: "Offerings of GLM-5.3" })
    const rows = within(offerings).getAllByRole("row").slice(1)
    expect(rows).toHaveLength(2)
    // Cheapest first: Fireworks' sub-cent rate keeps its digits (otari#700).
    expect(
      within(rows[0] as HTMLElement).getByText("$0.075"),
    ).toBeInTheDocument()
    expect(
      within(rows[0] as HTMLElement).getByText("DEFAULT"),
    ).toBeInTheDocument()
    expect(
      within(rows[1] as HTMLElement).getByText("nebius"),
    ).toBeInTheDocument()
    expect(
      within(rows[1] as HTMLElement).getByText("CUSTOM"),
    ).toBeInTheDocument()
    // The selector is not a lane; it opens under the row a reader picks.
    expect(within(offerings).queryByText("nebius:zai-org/GLM-5.3")).toBeNull()
    // The provider models.dev knows and this deployment does not.
    expect(screen.getByText(/Also served by Groq/)).toBeInTheDocument()
  })

  it("opens an offering's selector and request under its row", async () => {
    mockApi()
    renderPage(<ModelDetailPage modelId="z-ai/glm-5.3" />)
    const user = userEvent.setup()

    const grid = await screen.findByRole("grid", {
      name: "Offerings of GLM-5.3",
    })
    await user.click(within(grid).getByText("nebius"))

    // The short spelling leads, the full selector is still named, and the
    // request sends the short one.
    expect(await within(grid).findByText("nebius:glm-5.3")).toBeInTheDocument()
    expect(within(grid).getByText("nebius:zai-org/GLM-5.3")).toBeInTheDocument()
    expect(within(grid).getByText("zai-org/GLM-5.3")).toBeInTheDocument()
    const curl = within(grid).getByLabelText("cURL") as HTMLTextAreaElement
    expect(curl.value).toContain('"model": "nebius:glm-5.3"')

    await user.click(within(grid).getByRole("button", { name: "Close" }))
    expect(within(grid).queryByText("nebius:zai-org/GLM-5.3")).toBeNull()
  })

  it("links an operator to Model pricing to edit a rate, and nobody else", async () => {
    mockApi()
    renderPage(<ModelDetailPage modelId="z-ai/glm-5.3" />)

    const links = await screen.findAllByRole("link", { name: "Edit rate" })
    expect(links.map((link) => link.getAttribute("href"))).toEqual([
      "/organization/pricing?model=fireworks%3Aaccounts%2Ffireworks%2Fmodels%2Fglm-5p3",
      "/organization/pricing?model=nebius%3Azai-org%2FGLM-5.3",
    ])
    // Nothing on this page writes a price.
    expect(
      screen.queryByRole("button", { name: /set price|edit price/i }),
    ).toBeNull()
  })

  it("points an organization admin at its own rate override, not the deployment's price", async () => {
    mockApi({
      context: organizationContext({
        role: "admin",
        deployment_operator: false,
      }),
    })
    renderPage(<ModelDetailPage modelId="z-ai/glm-5.3" />)

    const links = await screen.findAllByRole("link", { name: "Set your rate" })
    expect(links.map((link) => link.getAttribute("href"))).toContain(
      "/organization/pricing?override=nebius%3Azai-org%2FGLM-5.3",
    )
    expect(screen.queryByRole("link", { name: "Edit rate" })).toBeNull()
  })

  it("shows what the organization was charged for an offering", async () => {
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
      renderPage(<ModelDetailPage modelId="z-ai/glm-5.3" />)

      const grid = await screen.findByRole("grid", {
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
      renderPage(<ModelDetailPage modelId="z-ai/glm-5.3" />)

      await screen.findByRole("grid", { name: "Offerings of GLM-5.3" })
      // Input is metered at half the list price and says so; output is within
      // rounding of it and does not.
      expect(screen.getByText("list $1.00")).toBeInTheDocument()
      expect(screen.queryByText("list $2.01")).toBeNull()
      expect(
        screen.getByText(/differs from the price the provider publishes/),
      ).toBeInTheDocument()
    } finally {
      GLM_DETAIL.offerings[0] = offering({})
    }
  })

  it("keeps the page read-only for a member", async () => {
    mockApi({ context: organizationContext({ deployment_operator: false }) })
    renderPage(<ModelDetailPage modelId="z-ai/glm-5.3" />)

    await screen.findByRole("grid", { name: "Offerings of GLM-5.3" })
    expect(screen.queryByRole("link", { name: "Edit rate" })).toBeNull()
    expect(screen.queryByRole("link", { name: "Add a provider" })).toBeNull()
    expect(screen.queryByRole("link", { name: "Model pricing" })).toBeNull()
  })

  it("reports a model that does not exist rather than an empty page", async () => {
    mockApi()
    renderPage(<ModelDetailPage modelId="nope" />, "/models/nope")

    expect(await screen.findByRole("alert")).toHaveTextContent(/not found/)
  })

  it("says why an offering is unpriced when defaults are off", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input)
      if (url.includes("/v1/catalog/models/z-ai/glm-5.3")) {
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
    renderPage(<ModelDetailPage modelId="z-ai/glm-5.3" />)

    expect(
      await screen.findByText(
        /Default pricing is off, so an offering with no stored rate/,
      ),
    ).toBeInTheDocument()
  })
})
