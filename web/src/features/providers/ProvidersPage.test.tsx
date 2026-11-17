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
import { ProviderDetailPage } from "@/features/providers/ProviderDetailPage"
import { ProvidersPage } from "@/features/providers/ProvidersPage"
import { PublicProvidersPage } from "@/features/providers/PublicProvidersPage"
import { API_ROOT } from "@/shared/api/client"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap, organizationContext } from "@/tests/fixtures"
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
  workspaceId,
}: {
  list?: (params: URLSearchParams) => CatalogProvidersResponse
  detail?: (params: URLSearchParams) => CatalogProviderDetail
  listStatus?: number
  workspaceId?: string
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
    if (path.includes(`${API_ROOT}/organizations/me`)) {
      return jsonResponse(
        organizationContext({
          workspace_memberships: workspaceId
            ? [{ workspace_id: workspaceId, name: "Research", role: "owner" }]
            : [],
        }),
      )
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

describe("ProvidersPage", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.history.replaceState(null, "", "/")
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

  it("says so when the providers cannot be read", async () => {
    mockApi({ listStatus: 503 })
    await renderWithRouter(providers(<ProvidersPage />), { url: "/providers" })

    expect(
      await screen.findByText(/The catalog is unavailable/),
    ).toBeInTheDocument()
    expect(screen.queryByText("Loading providers…")).toBeNull()
  })

  it("asks for one page and carries no pager while everything fits on it", async () => {
    const fetchMock = mockApi()
    await renderWithRouter(providers(<ProvidersPage />), { url: "/providers" })

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
    await renderWithRouter(providers(<ProvidersPage />), { url: "/providers" })
    const user = userEvent.setup()

    await screen.findByRole("link", { name: "Provider 1" })
    expect(screen.getByText(/26 providers serve/)).toBeInTheDocument()
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
  })

  it("lists what the selected workspace may call", async () => {
    const workspaceId = "44444444-4444-4444-4444-444444444444"
    const fetchMock = mockApi({ workspaceId })
    await renderWithRouter(
      providers(
        <SelectedWorkspaceProvider>
          <ProvidersPage />
        </SelectedWorkspaceProvider>,
      ),
      { url: "/providers" },
    )

    await screen.findByRole("list", { name: "Providers" })
    const requests = requestsTo(fetchMock, "/catalog/providers")
    // The first read included: the page waits for the selection rather than
    // showing the whole organization's providers first.
    expect(requests.length).toBeGreaterThan(0)
    for (const request of requests) {
      expect(request.searchParams.get("workspace_id")).toBe(workspaceId)
    }
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
    await renderWithRouter(providers(<ProviderDetailPage name="fireworks" />), {
      url: "/providers/fireworks",
    })
    const user = userEvent.setup()

    const table = await screen.findByRole("grid", {
      name: "Models from Fireworks AI",
    })
    expect(within(table).getByRole("link", { name: "Model 1" })).toBeVisible()
    expect(screen.getByText(/26 models/)).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: /next/i }))

    expect(
      await within(table).findByRole("link", { name: "Model 26" }),
    ).toBeInTheDocument()
    const pages = requestsTo(fetchMock, "/catalog/providers/fireworks")
    expect(pages.at(-1)?.searchParams.get("skip")).toBe("25")
    expect(pages.at(-1)?.searchParams.get("limit")).toBe("25")
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
    ).toHaveAttribute("href", "/providers/fireworks")
    const site = screen.getByRole("navigation", { name: "Site" })
    expect(
      within(site).getAllByRole("link", { name: "Providers" })[0],
    ).toHaveAttribute("aria-current", "page")
    expect(
      within(site).getAllByRole("link", { name: "Models" })[0],
    ).toHaveAttribute("href", "/models")
  })

  it("links a provider's models to the public catalog", async () => {
    mockApi()
    render(providers(<PublicProvidersPage name="fireworks" />))

    const table = await screen.findByRole("grid", {
      name: "Models from Fireworks AI",
    })
    expect(
      within(table).getByRole("link", { name: "GLM-5.3" }),
    ).toHaveAttribute("href", "/models/z-ai/glm-5.3")
    expect(screen.getByRole("link", { name: "← Providers" })).toHaveAttribute(
      "href",
      "/providers",
    )
  })

  it("pages a visitor's list in memory and names no workspace", async () => {
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
    const requests = requestsTo(fetchMock, "/catalog/providers")
    expect(requests.at(-1)?.searchParams.get("skip")).toBe("25")
    for (const request of requests) {
      expect(request.searchParams.has("workspace_id")).toBe(false)
    }
  })
})
