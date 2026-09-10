import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type {
  CatalogModelDetail,
  CatalogModelSummary,
  CatalogResponse,
} from "@/client"
import { PublicCatalogPage } from "@/features/models/PublicCatalogPage"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap } from "@/tests/fixtures"

const GLM: CatalogModelSummary = {
  id: "glm-5-3",
  name: "GLM-5.3",
  vendor: "Z.ai",
  family: "glm",
  capabilities: {
    reasoning: true,
    tool_call: true,
    structured_output: false,
    attachment: false,
    temperature: true,
  },
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
  selector: "glm-5-3",
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

const GLM_DETAIL: CatalogModelDetail = {
  ...GLM,
  description: "Z.ai's flagship.",
  offerings: [
    {
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
        cache_read_price_per_million: null,
        cache_write_price_per_million: null,
        cache_write_1h_price_per_million: null,
        pricing_tiers: [],
        unit: "tokens",
      },
      price_source: "defaults",
      price_reference: "nebius:GLM-5.3",
    },
    {
      selector: "fireworks:accounts/fireworks/models/glm-5p3",
      short_selector: "fireworks:glm-5p3",
      provider: "fireworks",
      provider_type: "fireworks",
      credential: "deployment",
      discovered: true,
      context_window: 131_072,
      max_output_tokens: 16_384,
      quantization: null,
      pricing: null,
      price_source: null,
      price_reference: null,
    },
  ],
  also_available_from: [],
}

const CATALOG: CatalogResponse = {
  default_pricing: true,
  defaults_as_of: null,
  metadata_available: true,
  models: [GLM],
}

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  })
}

function mockApi() {
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input)
    if (url.includes("/v1/catalog/models/glm-5-3")) {
      return jsonResponse(GLM_DETAIL)
    }
    if (url.includes("/v1/catalog/models")) return jsonResponse(CATALOG)
    return jsonResponse({ detail: "unexpected" })
  })
}

// No router on purpose: the page renders ahead of the session, where the app
// has not mounted one, so a component that reached for a router hook here
// would be the bug the test exists to catch.
function renderPage(modelId?: string) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <DeploymentProvider value={bootstrap({ public_catalog: true })}>
      <QueryClientProvider client={client}>
        <PublicCatalogPage modelId={modelId} />
      </QueryClientProvider>
    </DeploymentProvider>,
  )
}

describe("PublicCatalogPage", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.location.hash = ""
  })

  it("lists the catalog for a visitor with a way to sign in and nothing to edit", async () => {
    const fetchMock = mockApi()
    renderPage()

    const list = await screen.findByRole("list", { name: "Models" })
    // A plain hash link, since there is no router ahead of the session.
    expect(
      within(list).getByRole("link", { name: "Z.ai: GLM-5.3" }),
    ).toHaveAttribute("href", "#/models/glm-5-3")
    expect(screen.getByRole("link", { name: "Sign in" })).toHaveAttribute(
      "href",
      "#/",
    )
    expect(screen.getByText(/deployment's list rates/)).toBeInTheDocument()
    // A visitor has no organization to ask about, and asking would be a 401.
    expect(
      fetchMock.mock.calls.some(([url]) =>
        String(url).includes("/v1/organizations/me"),
      ),
    ).toBe(false)
  })

  it("shows a model's offerings without the links that need a session", async () => {
    mockApi()
    renderPage("glm-5-3")

    const grid = await screen.findByRole("grid", {
      name: "Offerings of GLM-5.3",
    })
    expect(within(grid).getAllByRole("row")).toHaveLength(3)
    expect(screen.queryByRole("link", { name: "Edit rate" })).toBeNull()
    expect(screen.queryByRole("link", { name: "Set your rate" })).toBeNull()
    expect(screen.queryByRole("link", { name: /route across/ })).toBeNull()
    expect(screen.queryByText(/Your effective price/)).toBeNull()
    expect(screen.getByRole("link", { name: "← All models" })).toHaveAttribute(
      "href",
      "#/models",
    )
  })

  it("tells a visitor to sign in before the request to copy", async () => {
    mockApi()
    renderPage("glm-5-3")
    const user = userEvent.setup()

    await user.click(
      await screen.findByRole("button", { name: "Use this model" }),
    )

    const dialog = await screen.findByRole("dialog", { name: "Use this model" })
    expect(
      within(dialog).getByRole("link", { name: "Sign in" }),
    ).toHaveAttribute("href", "#/")
  })
})
