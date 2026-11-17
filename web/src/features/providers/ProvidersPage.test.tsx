import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, within } from "@testing-library/react"
import type { ReactElement } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import type {
  CatalogProviderDetail,
  CatalogProviderSummary,
  CatalogProvidersResponse,
} from "@/client"
import { ProviderDetailPage } from "@/features/providers/ProviderDetailPage"
import { ProvidersPage } from "@/features/providers/ProvidersPage"
import { PublicProvidersPage } from "@/features/providers/PublicProvidersPage"
import { API_ROOT } from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap } from "@/tests/fixtures"
import { renderWithRouter } from "@/tests/router"

const FIREWORKS: CatalogProviderSummary = {
  name: "fireworks",
  provider: "fireworks",
  display_name: "Fireworks AI",
  model_count: 2,
  terms_url: "https://fireworks.example/terms",
  privacy_url: "https://fireworks.example/privacy",
}

const OPENAI_EU: CatalogProviderSummary = {
  name: "openai-eu",
  provider: "openai",
  display_name: "openai-eu",
  model_count: 1,
  terms_url: null,
  privacy_url: null,
}

const LIST: CatalogProvidersResponse = { providers: [FIREWORKS, OPENAI_EU] }

const FIREWORKS_DETAIL: CatalogProviderDetail = {
  ...FIREWORKS,
  models: [
    {
      id: "z-ai/glm-5.3",
      name: "GLM-5.3",
      vendor: "Z.ai",
      price_unit: "token",
      input_price: 0.5,
      output_price: 2,
      context_length: 200_000,
    },
    {
      id: "acme/mystery",
      name: "Mystery",
      vendor: null,
      price_unit: null,
      input_price: null,
      output_price: null,
      context_length: null,
    },
    {
      id: "acme/painter",
      name: "Painter",
      vendor: "Acme",
      price_unit: "image",
      input_price: null,
      output_price: 40_000,
      context_length: null,
    },
  ],
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

function mockApi() {
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input)
    if (url.endsWith(`${API_ROOT}/catalog/providers`)) return jsonResponse(LIST)
    if (url.endsWith(`${API_ROOT}/catalog/providers/fireworks`)) {
      return jsonResponse(FIREWORKS_DETAIL)
    }
    if (url.includes(`${API_ROOT}/catalog/providers/`)) {
      return jsonResponse({ detail: "Provider not found" }, 404)
    }
    return jsonResponse({ detail: "unexpected" }, 500)
  })
}

function providers(ui: ReactElement) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return (
    <DeploymentProvider value={bootstrap({ public_catalog: true })}>
      <QueryClientProvider client={client}>{ui}</QueryClientProvider>
    </DeploymentProvider>
  )
}

describe("ProvidersPage", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.location.hash = ""
  })

  it("lists each provider with its model count and links to its page", async () => {
    mockApi()
    await renderWithRouter(providers(<ProvidersPage />), { url: "/providers" })

    const list = await screen.findByRole("list", { name: "Providers" })
    const rows = within(list).getAllByRole("listitem")
    expect(rows).toHaveLength(2)
    const fireworks = rows[0]
    expect(
      within(fireworks).getByRole("link", { name: "Fireworks AI" }),
    ).toHaveAttribute("href", "/providers/fireworks")
    expect(within(fireworks).getByText("2 models")).toBeInTheDocument()

    const terms = within(fireworks).getByRole("link", {
      name: /Terms of service/,
    })
    expect(terms).toHaveAttribute("href", "https://fireworks.example/terms")
    expect(terms).toHaveAttribute("target", "_blank")
    expect(terms).toHaveAttribute("rel", "noopener noreferrer nofollow")
    expect(
      within(fireworks).getByRole("link", { name: /Privacy policy/ }),
    ).toHaveAttribute("href", "https://fireworks.example/privacy")
  })

  it("shows no policy links for a provider without them", async () => {
    mockApi()
    await renderWithRouter(providers(<ProvidersPage />), { url: "/providers" })

    const list = await screen.findByRole("list", { name: "Providers" })
    const openai = within(list).getAllByRole("listitem")[1]
    expect(within(openai).getByText("1 model")).toBeInTheDocument()
    expect(within(openai).getAllByRole("link")).toHaveLength(1)
    expect(within(openai).queryByText("Terms of service")).toBeNull()
    expect(within(openai).queryByText("Privacy policy")).toBeNull()
  })
})

describe("ProviderDetailPage", () => {
  afterEach(() => vi.restoreAllMocks())

  it("lists the provider's models, each linking to the model page", async () => {
    mockApi()
    await renderWithRouter(providers(<ProviderDetailPage name="fireworks" />), {
      url: "/providers/fireworks",
    })

    expect(
      await screen.findByRole("heading", { name: "Fireworks AI", level: 1 }),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("link", { name: /Terms of service/ }),
    ).toHaveAttribute("href", "https://fireworks.example/terms")
    const table = screen.getByRole("grid", { name: "Models from Fireworks AI" })
    expect(
      within(table).getByRole("link", { name: "GLM-5.3" }),
    ).toHaveAttribute("href", "/models/z-ai/glm-5.3")
    expect(within(table).getByText("Z.ai")).toBeInTheDocument()
    expect(within(table).getByText("$0.50 / 1M")).toBeInTheDocument()
    expect(within(table).getByText("$2.00 / 1M")).toBeInTheDocument()
    expect(within(table).getByText("200K")).toBeInTheDocument()
    expect(within(table).getByText(/\$0\.04.*image/)).toBeInTheDocument()
    expect(screen.getByRole("link", { name: "← Providers" })).toHaveAttribute(
      "href",
      "/providers",
    )
  })

  it("says so when the provider does not exist", async () => {
    mockApi()
    await renderWithRouter(providers(<ProviderDetailPage name="nope" />), {
      url: "/providers/nope",
    })

    expect(
      await screen.findByRole("heading", { name: "Provider not found" }),
    ).toBeInTheDocument()
    expect(screen.getByText(/No provider named nope/)).toBeInTheDocument()
  })
})

// No router on purpose: the public pages render ahead of the session, where
// the app has not mounted one.
describe("PublicProvidersPage", () => {
  afterEach(() => vi.restoreAllMocks())

  it("links providers and the bar by hash for a visitor", async () => {
    mockApi()
    render(providers(<PublicProvidersPage />))

    const list = await screen.findByRole("list", { name: "Providers" })
    expect(
      within(list).getByRole("link", { name: "Fireworks AI" }),
    ).toHaveAttribute("href", "#/providers/fireworks")
    const site = screen.getByRole("navigation", { name: "Site" })
    expect(
      within(site).getAllByRole("link", { name: "Providers" })[0],
    ).toHaveAttribute("aria-current", "page")
    expect(
      within(site).getAllByRole("link", { name: "Models" })[0],
    ).toHaveAttribute("href", "#/models")
  })

  it("links a provider's models to the public catalog", async () => {
    mockApi()
    render(providers(<PublicProvidersPage name="fireworks" />))

    const table = await screen.findByRole("grid", {
      name: "Models from Fireworks AI",
    })
    expect(
      within(table).getByRole("link", { name: "GLM-5.3" }),
    ).toHaveAttribute("href", "#/models/z-ai/glm-5.3")
    expect(screen.getByRole("link", { name: "← Providers" })).toHaveAttribute(
      "href",
      "#/providers",
    )
  })
})
