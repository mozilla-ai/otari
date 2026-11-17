import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactElement } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import type {
  CatalogProviderDetail,
  CatalogProviderSummary,
  CatalogProvidersResponse,
} from "@/client"
import { PublicProvidersPage } from "@/features/providers/PublicProvidersPage"
import { API_ROOT } from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap } from "@/tests/fixtures"

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

const LIST: CatalogProvidersResponse = {
  count: 2,
  providers: [FIREWORKS, OPENAI_EU],
}

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

function mockApi({
  list = () => LIST,
  detail = () => FIREWORKS_DETAIL,
  listStatus = 200,
}: {
  list?: (params: URLSearchParams) => CatalogProvidersResponse
  detail?: (params: URLSearchParams) => CatalogProviderDetail
  listStatus?: number
} = {}) {
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = new URL(String(input), "http://localhost")
    const path = url.pathname
    if (path.endsWith(`${API_ROOT}/catalog/providers`)) {
      return listStatus === 200
        ? jsonResponse(list(url.searchParams))
        : jsonResponse({ detail: "The catalog is unavailable" }, listStatus)
    }
    if (path.endsWith(`${API_ROOT}/catalog/providers/fireworks`)) {
      return jsonResponse(detail(url.searchParams))
    }
    if (path.includes(`${API_ROOT}/catalog/providers/`)) {
      return jsonResponse({ detail: "Provider not found" }, 404)
    }
    return jsonResponse({ detail: "unexpected" }, 500)
  })
}

function provider(index: number): CatalogProviderSummary {
  return {
    ...OPENAI_EU,
    name: `provider-${index}`,
    display_name: `Provider ${index}`,
  }
}

function requestsTo(fetchMock: ReturnType<typeof mockApi>, path: string) {
  return fetchMock.mock.calls
    .map(([input]) => new URL(String(input), "http://localhost"))
    .filter((url) => url.pathname.endsWith(path))
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

// No router on purpose: the public pages render ahead of the session, where
// the app has not mounted one.
describe("ProvidersView", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.history.replaceState(null, "", "/")
  })

  it("lists each provider with its model count and links to its page", async () => {
    mockApi()
    render(providers(<PublicProvidersPage />))

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
    render(providers(<PublicProvidersPage />))

    const list = await screen.findByRole("list", { name: "Providers" })
    const openai = within(list).getAllByRole("listitem")[1]
    expect(within(openai).getByText("1 model")).toBeInTheDocument()
    expect(within(openai).getAllByRole("link")).toHaveLength(1)
    expect(within(openai).queryByText("Terms of service")).toBeNull()
    expect(within(openai).queryByText("Privacy policy")).toBeNull()
  })

  it("says so when the providers cannot be read", async () => {
    mockApi({ listStatus: 503 })
    render(providers(<PublicProvidersPage />))

    expect(
      await screen.findByText(/The catalog is unavailable/),
    ).toBeInTheDocument()
    expect(screen.queryByText("Loading providers…")).toBeNull()
  })

  it("asks for one page and carries no pager while everything fits on it", async () => {
    const fetchMock = mockApi()
    render(providers(<PublicProvidersPage />))

    await screen.findByRole("list", { name: "Providers" })
    const [request] = requestsTo(fetchMock, "/catalog/providers")
    expect(request?.searchParams.get("skip")).toBe("0")
    expect(request?.searchParams.get("limit")).toBe("25")
    expect(request?.searchParams.has("workspace_id")).toBe(false)
    expect(screen.queryByRole("button", { name: /next/i })).toBeNull()
  })

  it("pages through more providers than fit on one page", async () => {
    const fetchMock = mockApi({
      list: (params) => ({
        count: 26,
        providers:
          params.get("skip") === "25"
            ? [provider(26)]
            : Array.from({ length: 25 }, (_, index) => provider(index + 1)),
      }),
    })
    render(providers(<PublicProvidersPage />))
    const user = userEvent.setup()

    await screen.findByRole("link", { name: "Provider 1" })
    await user.click(screen.getByRole("button", { name: /next/i }))

    expect(
      await screen.findByRole("link", { name: "Provider 26" }),
    ).toBeInTheDocument()
    expect(screen.queryByRole("link", { name: "Provider 1" })).toBeNull()
    await waitFor(() =>
      expect(
        requestsTo(fetchMock, "/catalog/providers").some(
          (url) => url.searchParams.get("skip") === "25",
        ),
      ).toBe(true),
    )
    for (const request of requestsTo(fetchMock, "/catalog/providers")) {
      expect(request.searchParams.has("workspace_id")).toBe(false)
    }
  })
})

describe("ProviderDetailView", () => {
  afterEach(() => vi.restoreAllMocks())

  it("lists the provider's models, each linking to the model page", async () => {
    mockApi()
    render(providers(<PublicProvidersPage name="fireworks" />))

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
    render(providers(<PublicProvidersPage name="nope" />))

    expect(
      await screen.findByRole("heading", { name: "Provider not found" }),
    ).toBeInTheDocument()
    expect(screen.getByText(/No provider named nope/)).toBeInTheDocument()
  })

  it("pages through more models than fit on one page", async () => {
    const model = (index: number) => ({
      ...FIREWORKS_DETAIL.models[0]!,
      id: `acme/model-${index}`,
      name: `Model ${index}`,
    })
    const fetchMock = mockApi({
      detail: (params) => ({
        ...FIREWORKS,
        model_count: 26,
        models:
          params.get("skip") === "25"
            ? [model(26)]
            : Array.from({ length: 25 }, (_, index) => model(index + 1)),
      }),
    })
    render(providers(<PublicProvidersPage name="fireworks" />))
    const user = userEvent.setup()

    const table = await screen.findByRole("grid", {
      name: "Models from Fireworks AI",
    })
    expect(within(table).getByRole("link", { name: "Model 1" })).toBeVisible()
    await user.click(screen.getByRole("button", { name: /next/i }))

    expect(
      await within(table).findByRole("link", { name: "Model 26" }),
    ).toBeInTheDocument()
    const pages = requestsTo(fetchMock, "/catalog/providers/fireworks")
    expect(pages.at(-1)?.searchParams.get("skip")).toBe("25")
    expect(pages.at(-1)?.searchParams.get("limit")).toBe("25")
  })
})

describe("PublicProvidersPage", () => {
  afterEach(() => vi.restoreAllMocks())

  it("marks Providers current in the bar and links Models by path", async () => {
    mockApi()
    render(providers(<PublicProvidersPage />))

    await screen.findByRole("list", { name: "Providers" })
    const site = screen.getByRole("navigation", { name: "Site" })
    expect(
      within(site).getAllByRole("link", { name: "Providers" })[0],
    ).toHaveAttribute("aria-current", "page")
    expect(
      within(site).getAllByRole("link", { name: "Models" })[0],
    ).toHaveAttribute("href", "/models")
  })
})
