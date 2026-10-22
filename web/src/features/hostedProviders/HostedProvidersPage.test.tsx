import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type {
  DeploymentProviderMigrationRow,
  HostedCatalogPriceUpdate,
  HostedCatalogRefresh,
  HostedModel,
  HostedProvider,
  HostedProviderTestResult,
  ProviderHealth,
  ProviderInfo,
} from "@/client"
import { HostedProvidersPage } from "@/features/hostedProviders/HostedProvidersPage"
import { API_ROOT } from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap } from "@/tests/fixtures"
import { renderWithRouter } from "@/tests/router"

interface Request {
  url: string
  method: string
  body: unknown
}

function hostedProvider(
  overrides: Partial<HostedProvider> = {},
): HostedProvider {
  return {
    id: "11111111-1111-4111-8111-111111111111",
    // Named after its implementation unless the case names it otherwise.
    name: overrides.provider ?? "openai",
    provider: "openai",
    source: "database",
    shadows_config: false,
    has_api_key: true,
    decryptable: true,
    api_key_last4: "abcd",
    api_base: null,
    client_args: null,
    enabled: true,
    restricted_to_organization_id: null,
    serves_unlisted_models: false,
    created_at: "2026-10-01T00:00:00Z",
    updated_at: null,
    ...overrides,
  }
}

function hostedModel(overrides: Partial<HostedModel> = {}): HostedModel {
  return {
    id: "m1",
    provider: "openai",
    model: "gpt-4o",
    input_price_per_million: 2.5,
    output_price_per_million: 10,
    cache_read_price_per_million: null,
    cache_write_price_per_million: null,
    unit: "tokens",
    price_source: "defaults",
    enabled: true,
    created_at: "2026-10-01T00:00:00Z",
    updated_at: null,
    ...overrides,
  }
}

function providerInfo(instance: string): ProviderInfo {
  return {
    instance,
    provider_type: instance,
    name: instance,
    doc_url: null,
    description: null,
    env_key: null,
    pricing_urls: [],
    source: "config",
    capabilities: {
      streaming: true,
      reasoning: false,
      vision: false,
      pdf: false,
      embeddings: false,
      image_generation: false,
      audio: false,
      rerank: false,
      responses_api: false,
      moderation: false,
      list_models: true,
    },
  }
}

const EMPTY_SWEEP: HostedCatalogRefresh = {
  providers: [],
  removed: [],
  removed_count: 0,
  removed_price_rows: 0,
  removed_override_rows: 0,
  kept: [],
}

const PRICES: HostedCatalogPriceUpdate = {
  fetched_at: "2026-10-21T00:00:00Z",
  added_count: 2,
  changed_count: 1,
  removed_count: 0,
  changes: [],
  changes_truncated: false,
  digest: "d".repeat(64),
  needs_review: false,
  review_reason: null,
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

function mockApi(
  opts: {
    granted?: boolean
    /** Answers the list 404, as the router refusing a non-operator. */
    listRefused?: boolean
    /** Answers a provider PATCH with this refusal. */
    patchRefusal?: string
    providers?: HostedProvider[]
    models?: HostedModel[]
    /** What the catalog update finds and does: its removals show only with the cleanup ticked. */
    sweep?: HostedCatalogRefresh
    /** The models.dev prices the preview holds; none changed otherwise. */
    prices?: HostedCatalogPriceUpdate
    /** Answers each preview read in turn; built from `sweep` and `prices` otherwise. */
    previews?: (() => Promise<Response>)[]
    /** The stored providers the move would carry, and what each becomes. */
    migration?: DeploymentProviderMigrationRow[]
    /** The `providers:` instances GET /providers lists. */
    configured?: ProviderInfo[]
    /** What a connection test answers. */
    testResult?: HostedProviderTestResult
    /** The health records GET /providers/health answers. */
    health?: ProviderHealth[]
    /** Whether the organization context reports a usable OTARI_SECRET_KEY. */
    keyEncryption?: boolean
    /** What GET /settings answers about pricing. */
    pricing?: { require_pricing: boolean; default_pricing: boolean }
  } = {},
) {
  const granted = opts.granted ?? true
  const providers = opts.providers ?? [hostedProvider()]
  const models = opts.models ?? [hostedModel()]
  const sweep = opts.sweep ?? EMPTY_SWEEP
  const previews = [...(opts.previews ?? [])]
  const requests: Request[] = []

  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
    const body = init?.body ? JSON.parse(String(init.body)) : undefined
    requests.push({ url, method, body })
    const hosted = `${API_ROOT}/hosted-providers`

    if (url.includes(`${API_ROOT}/admin/access`)) {
      return jsonResponse({ granted })
    }
    if (url.includes(`${API_ROOT}/organizations/me`)) {
      return jsonResponse({
        provider_key_encryption_available: opts.keyEncryption ?? true,
      })
    }
    if (url.includes(`${API_ROOT}/settings`)) {
      return jsonResponse(opts.pricing ?? {})
    }
    if (url.includes(`${API_ROOT}/providers/catalog/`)) {
      const id = url.split("/").pop() ?? ""
      return jsonResponse({
        id,
        name: id,
        env_key: null,
        default_api_base: null,
        requires_api_key: id !== "ollama",
        env_key_present: false,
      })
    }
    if (url.includes(`${API_ROOT}/providers/catalog`)) {
      return jsonResponse([
        { id: "anthropic", name: "Anthropic" },
        { id: "openai", name: "OpenAI" },
      ])
    }
    if (url.includes(`${API_ROOT}/providers/health`)) {
      const health = opts.health ?? []
      return jsonResponse({
        providers: health,
        healthy: health.filter((row) => row.ok).length,
        degraded: 0,
        total: health.length,
        checked_at: null,
      })
    }
    if (url.includes(`${API_ROOT}/providers`)) {
      return jsonResponse({ providers: opts.configured ?? [] })
    }
    if (url.includes(`${hosted}/test`)) {
      return jsonResponse(
        opts.testResult ?? {
          ok: true,
          model_count: 3,
          error: null,
          discovery_unsupported: false,
        },
      )
    }
    if (url.includes(`${hosted}/migration`)) {
      const rows = opts.migration ?? []
      return jsonResponse({
        rows:
          method === "POST"
            ? rows.map((row) => ({
                ...row,
                outcome: row.outcome === "migrate" ? "migrated" : row.outcome,
                models: 1,
                source_removed: row.outcome === "migrate",
              }))
            : rows,
      })
    }
    if (url.includes(`${hosted}/catalog-update/preview`)) {
      const next = previews.shift()
      if (next) return next()
      const cleanup = (body as { cleanup: boolean }).cleanup
      return jsonResponse({
        prices: opts.prices ?? null,
        price_error: null,
        catalog: cleanup
          ? sweep
          : { ...EMPTY_SWEEP, providers: sweep.providers },
      })
    }
    if (url.includes(`${hosted}/catalog-update`)) {
      const sent = body as { digest: string | null; cleanup: boolean }
      return jsonResponse({
        prices_applied: sent.digest !== null,
        catalog: sent.cleanup
          ? sweep
          : { ...EMPTY_SWEEP, providers: sweep.providers },
      })
    }
    if (url.includes("/models")) {
      if (method === "PATCH") return jsonResponse({ ...models[0], ...body })
      return jsonResponse({ data: models, count: models.length })
    }
    if (url.includes(hosted)) {
      if (method === "GET") {
        if (opts.listRefused) return jsonResponse({ detail: "Not Found" }, 404)
        return jsonResponse({ data: providers, count: providers.length })
      }
      if (method === "DELETE") {
        return jsonResponse({ message: "Hosted provider deleted" })
      }
      if (method === "PATCH" && opts.patchRefusal) {
        return jsonResponse({ detail: opts.patchRefusal }, 400)
      }
      return jsonResponse({ ...providers[0], ...body }, 201)
    }
    return jsonResponse({})
  })

  return requests
}

function renderPage() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return renderWithRouter(
    <DeploymentProvider value={bootstrap()}>
      <QueryClientProvider client={client}>
        <HostedProvidersPage />
      </QueryClientProvider>
    </DeploymentProvider>,
    { url: "/admin/hosted-providers" },
  )
}

const hostedRequests = (requests: Request[], method?: string) =>
  requests.filter(
    (request) =>
      request.url.includes("/hosted-providers") &&
      (method === undefined || request.method === method),
  )

afterEach(() => {
  vi.restoreAllMocks()
})

describe("HostedProvidersPage", () => {
  it("lists each provider by the tail of its key and who it serves", async () => {
    mockApi({
      providers: [
        hostedProvider(),
        hostedProvider({
          id: "22222222-2222-4222-8222-222222222222",
          provider: "mistral",
          api_key_last4: "wxyz",
          api_base: "https://eu.example.com/v1",
          restricted_to_organization_id: "33333333-3333-4333-8333-333333333333",
        }),
      ],
    })
    await renderPage()

    const table = await screen.findByRole("grid", { name: "Hosted providers" })
    expect(await within(table).findByText("••••abcd")).toBeInTheDocument()
    expect(within(table).getByText("Every organization")).toBeInTheDocument()
    expect(within(table).getByText("One organization")).toBeInTheDocument()
    expect(
      within(table).getByText("33333333-3333-4333-8333-333333333333"),
    ).toBeInTheDocument()
    expect(
      within(table).getByText("https://eu.example.com/v1"),
    ).toBeInTheDocument()
  })

  it("refuses a caller who does not operate the deployment without asking for the list", async () => {
    const requests = mockApi({ granted: false })
    await renderPage()

    expect(
      await screen.findByText("Hosted providers is not available to you"),
    ).toBeInTheDocument()
    expect(hostedRequests(requests)).toEqual([])
    expect(screen.queryByRole("button", { name: "Add provider" })).toBeNull()
    expect(screen.queryByRole("button", { name: "Update catalog" })).toBeNull()
  })

  it("refuses an operator the list itself 404s, without offering a write", async () => {
    mockApi({ listRefused: true })
    await renderPage()

    expect(
      await screen.findByText("Hosted providers is not available to you"),
    ).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Add provider" })).toBeNull()
    expect(screen.queryByRole("button", { name: "Update catalog" })).toBeNull()
  })

  it("walks the first provider in when none is configured", async () => {
    mockApi({ providers: [] })
    const user = userEvent.setup()
    await renderPage()

    expect(await screen.findByText("Welcome to Otari")).toBeInTheDocument()
    // The panel owns the empty state, so no empty table sits under it.
    expect(screen.queryByRole("grid", { name: "Hosted providers" })).toBeNull()
    expect(screen.queryByText(/Enable default pricing/)).toBeNull()

    await user.click(
      screen.getByRole("button", { name: "Add your first provider" }),
    )
    expect(
      await screen.findByRole("dialog", { name: "Add hosted provider" }),
    ).toBeInTheDocument()
  })

  it("offers default pricing on first run while pricing is required and unset", async () => {
    const requests = mockApi({
      providers: [],
      pricing: { require_pricing: true, default_pricing: false },
    })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      await screen.findByRole("button", { name: "Enable default pricing" }),
    )
    await waitFor(() =>
      expect(
        requests.filter(
          (request) =>
            request.url.endsWith("/settings") && request.method === "PATCH",
        ),
      ).toEqual([expect.objectContaining({ body: { default_pricing: true } })]),
    )
  })

  it("disables adding a provider while OTARI_SECRET_KEY is not set", async () => {
    mockApi({ providers: [], keyEncryption: false })
    await renderPage()

    expect(
      await screen.findByText(/is not set, so provider keys/),
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Add provider" })).toBeDisabled()
    expect(
      screen.getByRole("button", { name: "Add your first provider" }),
    ).toBeDisabled()
  })

  it("says nothing about OTARI_SECRET_KEY while it is set", async () => {
    mockApi()
    await renderPage()

    await screen.findByRole("grid", { name: "Hosted providers" })
    expect(screen.queryByText(/OTARI_SECRET_KEY/)).toBeNull()
    expect(screen.getByRole("button", { name: "Add provider" })).toBeEnabled()
  })

  it("links each provider to the models it serves", async () => {
    mockApi({
      providers: [hostedProvider({ name: "openai-eu" })],
    })
    await renderPage()

    expect(
      await screen.findByRole("link", { name: "openai-eu" }),
    ).toHaveAttribute("href", "/models?provider=openai-eu")
  })

  it("tests a stored provider from its row, on what it stores", async () => {
    const requests = mockApi({
      providers: [hostedProvider({ name: "openai-eu" })],
    })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      await screen.findByRole("button", { name: "Test openai-eu" }),
    )

    expect(
      await screen.findByText(/Connected\. 3 models available\./),
    ).toBeInTheDocument()
    expect(hostedRequests(requests, "POST")).toEqual([
      expect.objectContaining({
        url: expect.stringContaining("/hosted-providers/test"),
        body: { provider: "openai", name: "openai-eu" },
      }),
    ])
  })

  it("offers no row test for a key no secret reads, or for a config.yml provider", async () => {
    mockApi({
      providers: [
        hostedProvider({ decryptable: false }),
        hostedProvider({ id: "c", provider: "mistral", source: "config" }),
      ],
    })
    await renderPage()

    expect(
      await screen.findByRole("button", { name: "Test openai" }),
    ).toBeDisabled()
    expect(screen.queryByRole("button", { name: "Test mistral" })).toBeNull()
  })

  it("summarizes reachability and re-checks every provider on demand", async () => {
    const requests = mockApi({
      health: [
        {
          instance: "openai",
          ok: true,
          model_count: 3,
          error: null,
          checked_at: "2026-10-19T00:00:00Z",
          discovery_unsupported: false,
        },
      ],
    })
    const user = userEvent.setup()
    await renderPage()

    expect(
      await screen.findByText("1 of 1 provider reachable"),
    ).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Re-check all" }))
    await waitFor(() =>
      expect(
        requests.some((request) =>
          request.url.endsWith("/providers/health?refresh=true"),
        ),
      ).toBe(true),
    )
  })

  it("adds a provider that serves models not on its list", async () => {
    const requests = mockApi({ providers: [] })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      (await screen.findAllByRole("button", { name: "Add provider" }))[0],
    )
    const dialog = await screen.findByRole("dialog")
    await user.click(within(dialog).getByRole("combobox", { name: "Provider" }))
    await user.click(await screen.findByRole("option", { name: "Anthropic" }))
    await user.type(within(dialog).getByLabelText("API key"), "sk-ant-secret")
    await user.click(
      within(dialog).getByRole("checkbox", {
        name: "Serve models not on its list",
      }),
    )
    await user.click(
      within(dialog).getByRole("button", { name: "Add provider" }),
    )

    await waitFor(() =>
      expect(hostedRequests(requests, "POST")).toHaveLength(1),
    )
    expect(hostedRequests(requests, "POST")[0].body).toMatchObject({
      serves_unlisted_models: true,
    })
  })

  it("switches serving models not on its list from the edit dialog", async () => {
    const requests = mockApi({
      providers: [hostedProvider({ serves_unlisted_models: true })],
    })
    const user = userEvent.setup()
    await renderPage()

    await user.click(await screen.findByRole("button", { name: "Edit openai" }))
    const dialog = await screen.findByRole("dialog")
    const box = within(dialog).getByRole("checkbox", {
      name: "Serve models not on its list",
    })
    expect(box).toBeChecked()
    await user.click(box)
    await user.click(within(dialog).getByRole("button", { name: "Save" }))

    await waitFor(() =>
      expect(hostedRequests(requests, "PATCH")).toHaveLength(1),
    )
    expect(hostedRequests(requests, "PATCH")[0].body).toEqual({
      api_base: "",
      serves_unlisted_models: false,
    })
  })

  it("adds a provider with its key, leaving a blank endpoint out", async () => {
    const requests = mockApi({ providers: [] })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      (await screen.findAllByRole("button", { name: "Add provider" }))[0],
    )
    const dialog = await screen.findByRole("dialog")
    await user.click(within(dialog).getByRole("combobox", { name: "Provider" }))
    await user.click(await screen.findByRole("option", { name: "Anthropic" }))
    await user.type(within(dialog).getByLabelText("API key"), "sk-ant-secret")
    await user.click(
      within(dialog).getByRole("button", { name: "Add provider" }),
    )

    await waitFor(() =>
      expect(hostedRequests(requests, "POST")).toHaveLength(1),
    )
    expect(hostedRequests(requests, "POST")[0].body).toEqual({
      provider: "anthropic",
      api_key: "sk-ant-secret",
      serves_unlisted_models: false,
    })
  })

  it("edits a provider without resending its key or its restriction", async () => {
    const requests = mockApi({
      providers: [
        hostedProvider({
          api_base: "https://proxy.example.com/v1",
          restricted_to_organization_id: "33333333-3333-4333-8333-333333333333",
        }),
      ],
    })
    const user = userEvent.setup()
    await renderPage()

    await user.click(await screen.findByRole("button", { name: "Edit openai" }))
    const dialog = await screen.findByRole("dialog")
    const base = within(dialog).getByRole("textbox", { name: "API base URL" })
    await user.clear(base)
    await user.type(base, "https://proxy.example.com/v2")
    await user.click(within(dialog).getByRole("button", { name: "Save" }))

    await waitFor(() =>
      expect(hostedRequests(requests, "PATCH")).toHaveLength(1),
    )
    const patch = hostedRequests(requests, "PATCH")[0]
    expect(patch.url).toContain("/hosted-providers/openai")
    // The API showed no arguments, so none are sent: null would clear any it
    // could not read back.
    expect(patch.body).toEqual({ api_base: "https://proxy.example.com/v2" })
  })

  it("asks for the key again before pointing a provider at another host", async () => {
    const requests = mockApi()
    const user = userEvent.setup()
    await renderPage()

    await user.click(await screen.findByRole("button", { name: "Edit openai" }))
    const dialog = await screen.findByRole("dialog")
    await user.type(
      within(dialog).getByRole("textbox", { name: "API base URL" }),
      "https://proxy.example.com/v1",
    )
    expect(
      within(dialog).getByText(
        "This endpoint is on another host, scheme or port, so enter the key again.",
      ),
    ).toBeInTheDocument()
    expect(within(dialog).getByRole("button", { name: "Save" })).toBeDisabled()

    await user.type(within(dialog).getByLabelText(/API key/), "sk-new")
    await user.click(within(dialog).getByRole("button", { name: "Save" }))
    await waitFor(() =>
      expect(hostedRequests(requests, "PATCH")).toHaveLength(1),
    )
    expect(hostedRequests(requests, "PATCH")[0].body).toMatchObject({
      api_key: "sk-new",
      api_base: "https://proxy.example.com/v1",
    })
  })

  it("keeps the edit open on the server's refusal, naming it", async () => {
    mockApi({
      patchRefusal:
        "Pointing a hosted provider at another origin requires entering its key again",
    })
    const user = userEvent.setup()
    await renderPage()

    await user.click(await screen.findByRole("button", { name: "Edit openai" }))
    const dialog = await screen.findByRole("dialog")
    await user.click(within(dialog).getByRole("button", { name: "Save" }))

    expect(
      await within(dialog).findByText(
        /Pointing a hosted provider at another origin requires entering its key again/,
      ),
    ).toBeInTheDocument()
    expect(screen.getByRole("dialog")).toBeInTheDocument()
  })

  it("switches a provider off from its row", async () => {
    const requests = mockApi()
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      await screen.findByRole("switch", { name: "Serve traffic on openai" }),
    )

    await waitFor(() =>
      expect(hostedRequests(requests, "PATCH")).toHaveLength(1),
    )
    expect(hostedRequests(requests, "PATCH")[0].body).toEqual({
      enabled: false,
    })
  })

  it("removes a provider only once the confirm is pressed", async () => {
    const requests = mockApi()
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      await screen.findByRole("button", { name: "Remove openai" }),
    )
    const dialog = await screen.findByRole("alertdialog")
    expect(hostedRequests(requests, "DELETE")).toEqual([])
    await user.click(
      within(dialog).getByRole("button", { name: "Remove provider" }),
    )

    await waitFor(() =>
      expect(hostedRequests(requests, "DELETE")).toHaveLength(1),
    )
    expect(hostedRequests(requests, "DELETE")[0].url).toContain(
      "/hosted-providers/openai",
    )
  })

  it("opens a provider's models with their price source and a serving switch", async () => {
    const requests = mockApi({
      models: [
        hostedModel(),
        hostedModel({
          id: "m2",
          model: "o9-preview",
          input_price_per_million: null,
          output_price_per_million: null,
          unit: null,
          price_source: null,
          enabled: false,
        }),
      ],
    })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      await screen.findByRole("button", { name: "Models on openai" }),
    )
    const panel = await screen.findByRole("region", {
      name: "Models on openai",
    })
    expect(await within(panel).findByText("gpt-4o")).toBeInTheDocument()
    expect(within(panel).getByText("Default")).toBeInTheDocument()
    expect(within(panel).getByText("Unpriced")).toBeInTheDocument()

    await user.click(
      within(panel).getByRole("switch", { name: "Serve gpt-4o" }),
    )
    await waitFor(() =>
      expect(
        hostedRequests(requests, "PATCH").filter((request) =>
          request.url.includes("/hosted-providers/openai/models/m1"),
        ),
      ).toHaveLength(1),
    )
  })

  it("lists a provider defined in config.yml read-only, with its models still managed here", async () => {
    const requests = mockApi({
      providers: [
        hostedProvider({
          id: null,
          provider: "anthropic",
          source: "config",
          created_at: null,
        }),
      ],
    })
    const user = userEvent.setup()
    await renderPage()

    const table = await screen.findByRole("grid", { name: "Hosted providers" })
    expect(await within(table).findByText("config.yml")).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Edit anthropic" })).toBeNull()
    expect(
      screen.queryByRole("button", { name: "Remove anthropic" }),
    ).toBeNull()
    expect(
      screen.getByRole("switch", { name: "Serve traffic on anthropic" }),
    ).toBeDisabled()

    await user.click(
      screen.getByRole("button", { name: "Models on anthropic" }),
    )
    const panel = await screen.findByRole("region", {
      name: "Models on anthropic",
    })
    expect(
      within(panel).getByText(/defined in config\.yml/),
    ).toBeInTheDocument()
    await user.click(
      await within(panel).findByRole("switch", { name: "Serve gpt-4o" }),
    )
    await waitFor(() =>
      expect(
        hostedRequests(requests, "PATCH").filter((request) =>
          request.url.includes("/hosted-providers/anthropic/models/m1"),
        ),
      ).toHaveLength(1),
    )
  })

  it("marks a stored provider that overrides config.yml, and says what removing it restores", async () => {
    mockApi({ providers: [hostedProvider({ shadows_config: true })] })
    const user = userEvent.setup()
    await renderPage()

    expect(await screen.findByText("Overrides config.yml")).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Remove openai" }))
    const dialog = await screen.findByRole("alertdialog")
    expect(
      within(dialog).getByText(/defined in config\.yml serves again/),
    ).toBeInTheDocument()
  })

  it("warns that a provider config.yml already defines would be overridden", async () => {
    mockApi({
      providers: [
        hostedProvider({ id: null, provider: "anthropic", source: "config" }),
      ],
    })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      (await screen.findAllByRole("button", { name: "Add provider" }))[0],
    )
    const dialog = await screen.findByRole("dialog")
    expect(
      within(dialog).queryByText(/already defined in config\.yml/),
    ).toBeNull()
    await user.click(within(dialog).getByRole("combobox", { name: "Provider" }))
    await user.click(await screen.findByRole("option", { name: "Anthropic" }))

    expect(
      within(dialog).getByText(/already defined in config\.yml/),
    ).toBeInTheDocument()
  })

  const updates = (requests: Request[]) =>
    hostedRequests(requests, "POST").filter((request) =>
      request.url.endsWith("/catalog-update"),
    )
  const previewsOf = (requests: Request[]) =>
    hostedRequests(requests, "POST").filter((request) =>
      request.url.endsWith("/catalog-update/preview"),
    )

  it("applies the previewed prices by their digest, then says what it did", async () => {
    const requests = mockApi({
      prices: PRICES,
      sweep: {
        ...EMPTY_SWEEP,
        providers: [
          {
            provider: "openai",
            added: ["gpt-5-1017"],
            repriced: [],
            error: null,
            discovery_unsupported: false,
            credential_unreadable: false,
            source: "database",
          },
        ],
      },
    })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      await screen.findByRole("button", { name: "Update catalog" }),
    )
    const dialog = await screen.findByRole("alertdialog")
    expect(
      await within(dialog).findByText(
        "models.dev has 2 new, 1 changed prices to apply.",
      ),
    ).toBeInTheDocument()
    expect(
      within(dialog).getByText(
        "1 new model to offer on openai, each served once something prices it.",
      ),
    ).toBeInTheDocument()
    expect(updates(requests)).toEqual([])

    await user.click(
      within(dialog).getByRole("button", { name: "Update catalog" }),
    )
    await waitFor(() => expect(updates(requests)).toHaveLength(1))
    expect(updates(requests)[0].body).toEqual({
      digest: PRICES.digest,
      cleanup: false,
    })
    expect(
      await screen.findByText(
        "Applied the latest models.dev prices. Offered 1 new model.",
      ),
    ).toBeInTheDocument()
  })

  it("names what the cleanup would remove only once it is ticked, before removing anything", async () => {
    const requests = mockApi({
      sweep: {
        ...EMPTY_SWEEP,
        removed: ["mistral:old-model"],
        removed_count: 1,
        removed_price_rows: 2,
        removed_override_rows: 0,
      },
    })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      await screen.findByRole("button", { name: "Update catalog" }),
    )
    const dialog = await screen.findByRole("alertdialog")
    expect(
      await within(dialog).findByText("models.dev prices are unchanged."),
    ).toBeInTheDocument()
    expect(within(dialog).queryByText("mistral:old-model")).toBeNull()

    await user.click(
      within(dialog).getByRole("checkbox", {
        name: "Also remove the prices of models no provider serves",
      }),
    )
    expect(
      await within(dialog).findByText(
        "1 model will be removed, taking 2 stored rates and 0 organization overrides with it.",
      ),
    ).toBeInTheDocument()
    expect(within(dialog).getByText("mistral:old-model")).toBeInTheDocument()
    expect(updates(requests)).toEqual([])

    await user.click(
      within(dialog).getByRole("button", { name: "Update and remove" }),
    )
    await waitFor(() => expect(updates(requests)).toHaveLength(1))
    expect(updates(requests)[0].body).toEqual({ digest: null, cleanup: true })
    expect(
      await screen.findByText("Removed 1 model nothing here serves."),
    ).toBeInTheDocument()
  })

  it("asks for a fresh preview on every opening and holds the confirm until it lands", async () => {
    let answerSecond: (response: Response) => void = () => {}
    const requests = mockApi({
      previews: [
        async () =>
          jsonResponse({
            prices: PRICES,
            price_error: null,
            catalog: EMPTY_SWEEP,
          }),
        () =>
          new Promise<Response>((resolve) => {
            answerSecond = resolve
          }),
      ],
    })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      await screen.findByRole("button", { name: "Update catalog" }),
    )
    let dialog = await screen.findByRole("alertdialog")
    expect(
      await within(dialog).findByText(
        "models.dev has 2 new, 1 changed prices to apply.",
      ),
    ).toBeInTheDocument()
    await user.click(within(dialog).getByRole("button", { name: "Cancel" }))
    await waitFor(() => expect(screen.queryByRole("alertdialog")).toBeNull())

    await user.click(screen.getByRole("button", { name: "Update catalog" }))
    dialog = await screen.findByRole("alertdialog")
    // The last opening's preview is gone, and nothing can be confirmed yet.
    expect(
      within(dialog).getByText(
        "Asking models.dev and every provider what changed…",
      ),
    ).toBeInTheDocument()
    expect(
      within(dialog).queryByText(
        "models.dev has 2 new, 1 changed prices to apply.",
      ),
    ).toBeNull()
    const confirm = within(dialog).getByRole("button", {
      name: "Update catalog",
    })
    expect(confirm).toBeDisabled()
    await user.click(confirm)

    answerSecond(
      jsonResponse({ prices: null, price_error: null, catalog: EMPTY_SWEEP }),
    )
    await waitFor(() => expect(confirm).toBeEnabled())
    expect(previewsOf(requests)).toHaveLength(2)
    expect(updates(requests)).toEqual([])
  })

  it("never confirms an update whose preview failed", async () => {
    const requests = mockApi({
      previews: [
        async () => jsonResponse({ detail: "Upstream unreachable" }, 502),
      ],
    })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      await screen.findByRole("button", { name: "Update catalog" }),
    )
    const dialog = await screen.findByRole("alertdialog")
    expect(
      await within(dialog).findByText("Upstream unreachable"),
    ).toBeInTheDocument()
    const confirm = within(dialog).getByRole("button", {
      name: "Update catalog",
    })
    expect(confirm).toBeDisabled()
    await user.click(confirm)
    expect(updates(requests)).toEqual([])
  })
})

describe("HostedProvidersPage, named providers and the move", () => {
  it("discards a stored provider that cannot move, once confirmed", async () => {
    const requests = mockApi({
      migration: [
        {
          instance: "groq",
          provider: "groq",
          api_base: null,
          outcome: "conflict",
          detail: "A hosted provider already has this name.",
          models: null,
          source_removed: false,
          notes: [],
        },
        {
          instance: "openai-eu",
          provider: "openai",
          api_base: null,
          outcome: "migrate",
          detail: null,
          models: null,
          source_removed: false,
          notes: [],
        },
      ],
    })
    const user = userEvent.setup()
    await renderPage()

    // A row that can still move is moved, not discarded.
    await screen.findByRole("button", { name: "Discard groq" })
    expect(
      screen.queryByRole("button", { name: "Discard openai-eu" }),
    ).toBeNull()

    await user.click(screen.getByRole("button", { name: "Discard groq" }))
    const dialog = await screen.findByRole("alertdialog")
    expect(dialog).toHaveTextContent("Discard groq?")
    const previewReads = () =>
      requests.filter(
        (request) =>
          request.method === "GET" &&
          request.url.endsWith("/hosted-providers/migration"),
      ).length
    const readsBefore = previewReads()
    await user.click(within(dialog).getByRole("button", { name: "Discard" }))

    await waitFor(() =>
      expect(
        requests.some(
          (request) =>
            request.method === "DELETE" &&
            request.url.endsWith("/hosted-providers/migration/groq"),
        ),
      ).toBe(true),
    )
    await waitFor(() => expect(previewReads()).toBeGreaterThan(readsBefore))
  })

  it("keeps offering the move after one that left a provider behind", async () => {
    mockApi({
      migration: [
        {
          instance: "groq",
          provider: "groq",
          api_base: null,
          outcome: "failed",
          detail: "The endpoint was refused.",
          models: null,
          source_removed: false,
          notes: [],
        },
        {
          instance: "openai-eu",
          provider: "openai",
          api_base: null,
          outcome: "migrate",
          detail: null,
          models: null,
          source_removed: false,
          notes: [],
        },
      ],
    })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      await screen.findByRole("button", { name: "Review the move" }),
    )
    await user.click(
      within(await screen.findByRole("alertdialog")).getByRole("button", {
        name: "Move 1 provider",
      }),
    )

    expect(
      await screen.findByText(/Moved 1 of 2 stored providers/),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Review the move" }),
    ).toBeInTheDocument()
  })

  it("reports a row copied but still served from its original entry apart from a completed move", async () => {
    mockApi({
      migration: [
        {
          instance: "quiet",
          provider: "openai",
          api_base: null,
          outcome: "migrate",
          detail: null,
          models: null,
          source_removed: false,
          notes: [],
        },
      ],
    })
    const fetchMock = vi.mocked(globalThis.fetch)
    const answer = fetchMock.getMockImplementation()
    fetchMock.mockImplementation(async (input, init) => {
      if (
        String(input).includes("/hosted-providers/migration") &&
        init?.method === "POST"
      ) {
        return jsonResponse({
          rows: [
            {
              instance: "quiet",
              provider: "openai",
              api_base: null,
              outcome: "migrated",
              detail: null,
              models: 0,
              source_removed: false,
              notes: [],
            },
          ],
        })
      }
      return answer ? answer(input, init) : jsonResponse({})
    })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      await screen.findByRole("button", { name: "Review the move" }),
    )
    await user.click(
      within(await screen.findByRole("alertdialog")).getByRole("button", {
        name: "Move 1 provider",
      }),
    )

    expect(
      await screen.findByText(/Moved 0 of 1 stored providers\. 1 copied/),
    ).toBeInTheDocument()
    expect(screen.getByText(/^Copied, original entry kept/)).toBeInTheDocument()
  })

  it("shows inside the dialog why a failed re-read holds the move", async () => {
    mockApi({
      migration: [
        {
          instance: "openai-eu",
          provider: "openai",
          api_base: null,
          outcome: "migrate",
          detail: null,
          models: null,
          source_removed: false,
          notes: [],
        },
      ],
    })
    const user = userEvent.setup()
    await renderPage()
    await screen.findByRole("button", { name: "Review the move" })
    const fetchMock = vi.mocked(globalThis.fetch)
    const answer = fetchMock.getMockImplementation()
    fetchMock.mockImplementation(async (input, init) => {
      if (String(input).includes("/hosted-providers/migration")) {
        return jsonResponse({ detail: "Preview went away" }, 400)
      }
      return answer ? answer(input, init) : jsonResponse({})
    })

    await user.click(screen.getByRole("button", { name: "Review the move" }))

    const dialog = await screen.findByRole("alertdialog")
    expect(
      await within(dialog).findByText("Preview went away"),
    ).toBeInTheDocument()
    expect(
      within(dialog).getByRole("button", { name: "Move 1 provider" }),
    ).toBeDisabled()
  })

  it("says the move could not be previewed rather than hiding it", async () => {
    mockApi()
    const fetchMock = vi.mocked(globalThis.fetch)
    const answer = fetchMock.getMockImplementation()
    fetchMock.mockImplementation(async (input, init) => {
      if (String(input).includes("/hosted-providers/migration")) {
        return jsonResponse({ detail: "Preview unavailable" }, 500)
      }
      return answer ? answer(input, init) : jsonResponse({})
    })
    await renderPage()

    expect(await screen.findByText("Preview unavailable")).toBeInTheDocument()
  })

  it("shows no status for a hosted provider a configured instance of its name is dialed under", async () => {
    const reachable = (instance: string): ProviderHealth => ({
      instance,
      ok: true,
      model_count: 1,
      error: null,
      checked_at: "2026-10-19T00:00:00Z",
      discovery_unsupported: false,
    })
    mockApi({
      providers: [
        hostedProvider({ name: "openai" }),
        hostedProvider({ id: "m", name: "mistral", provider: "mistral" }),
      ],
      configured: [providerInfo("openai")],
      health: [reachable("openai"), reachable("mistral")],
    })
    await renderPage()

    const table = await screen.findByRole("grid", { name: "Hosted providers" })
    const rows = (name: string) =>
      within(table)
        .getAllByRole("rowheader", { name: new RegExp(`^${name}`) })
        .map((header) => header.parentElement as HTMLElement)
    await within(table).findByText("Edit in config.yml")
    const [configuredOpenai, hostedOpenai] = [...rows("openai")].sort(
      (a, b) =>
        Number(within(b).queryByText("Edit in config.yml") !== null) -
        Number(within(a).queryByText("Edit in config.yml") !== null),
    )
    // The configured instance's record is the one dialed under "openai".
    expect(
      await within(configuredOpenai).findByText("Reachable"),
    ).toBeInTheDocument()
    expect(within(hostedOpenai).queryByText("Reachable")).toBeNull()
    expect(
      await within(rows("mistral")[0]).findByText("Reachable"),
    ).toBeInTheDocument()
  })

  it("names a provider of its own and tests it before saving", async () => {
    const requests = mockApi({ providers: [] })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      (await screen.findAllByRole("button", { name: "Add provider" }))[0],
    )
    const dialog = await screen.findByRole("dialog")
    await user.click(within(dialog).getByRole("combobox", { name: "Provider" }))
    await user.click(await screen.findByRole("option", { name: "OpenAI" }))
    await user.type(within(dialog).getByLabelText("Name"), "openai-eu")
    await user.type(
      within(dialog).getByLabelText("API base URL"),
      "https://eu.example.com/v1",
    )
    // An endpoint of its own may go without a key.
    expect(
      within(dialog).getByRole("button", { name: "Add provider" }),
    ).toBeEnabled()

    await user.click(
      within(dialog).getByRole("button", { name: "Test connection" }),
    )
    expect(
      await within(dialog).findByText(/Connected\. 3 models available\./),
    ).toBeInTheDocument()
    const [test] = hostedRequests(requests, "POST").filter((request) =>
      request.url.endsWith("/hosted-providers/test"),
    )
    expect(test.body).toEqual({
      provider: "openai",
      name: "openai-eu",
      api_base: "https://eu.example.com/v1",
    })
    expect(hostedRequests(requests, "POST")).toHaveLength(1)

    await user.click(
      within(dialog).getByRole("button", { name: "Add provider" }),
    )
    await waitFor(() =>
      expect(
        hostedRequests(requests, "POST").filter((request) =>
          request.url.endsWith("/hosted-providers"),
        ),
      ).toHaveLength(1),
    )
  })

  it("says the client options are encrypted at rest, as the hosted store keeps them", async () => {
    mockApi({ providers: [] })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      (await screen.findAllByRole("button", { name: "Add provider" }))[0],
    )
    const dialog = await screen.findByRole("dialog")

    expect(
      within(dialog).getByText(/Encrypted at rest, and an option/),
    ).toBeInTheDocument()
    expect(within(dialog).queryByText(/nothing here is encrypted/)).toBeNull()
  })

  it("holds the save until a key is typed for a provider that needs one", async () => {
    mockApi({ providers: [] })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      (await screen.findAllByRole("button", { name: "Add provider" }))[0],
    )
    const dialog = await screen.findByRole("dialog")
    await user.click(within(dialog).getByRole("combobox", { name: "Provider" }))
    await user.click(await screen.findByRole("option", { name: "OpenAI" }))

    await waitFor(() =>
      expect(
        within(dialog).getByRole("button", { name: "Add provider" }),
      ).toBeDisabled(),
    )
  })

  it("tests an edit on the stored key without asking for it", async () => {
    const requests = mockApi({
      providers: [
        hostedProvider({
          name: "openai-eu",
          api_base: "https://eu.example.com/v1",
        }),
      ],
    })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      await screen.findByRole("button", { name: "Edit openai-eu" }),
    )
    const dialog = await screen.findByRole("dialog")
    await user.click(
      within(dialog).getByRole("button", { name: "Test connection" }),
    )

    await within(dialog).findByText(/Connected\./)
    const [test] = hostedRequests(requests, "POST")
    expect(test.url).toContain("/hosted-providers/test")
    expect(test.body).toEqual({
      provider: "openai",
      name: "openai-eu",
      api_base: "https://eu.example.com/v1",
    })
  })

  it("shows a provider's name beside its implementation, and a key no secret reads", async () => {
    mockApi({
      providers: [
        hostedProvider({
          name: "openai-eu",
          decryptable: false,
        }),
      ],
    })
    await renderPage()

    const table = await screen.findByRole("grid", { name: "Hosted providers" })
    expect(await within(table).findByText("openai-eu")).toBeInTheDocument()
    expect(within(table).getByText("openai")).toBeInTheDocument()
    expect(within(table).getByText("Key unreadable")).toBeInTheDocument()
  })

  it("lists the configured providers read-only in the same table", async () => {
    mockApi({
      providers: [hostedProvider({ name: "openai" })],
      configured: [providerInfo("anthropic")],
    })
    await renderPage()

    const table = await screen.findByRole("grid", { name: "Hosted providers" })
    const header = await within(table).findByRole("rowheader", {
      name: /^anthropic/,
    })
    const row = header.parentElement as HTMLElement
    expect(within(row).getByText("config.yml")).toBeInTheDocument()
    expect(within(row).getByText("Edit in config.yml")).toBeInTheDocument()
    expect(within(row).queryByRole("button")).toBeNull()
    expect(within(row).queryByRole("switch")).toBeNull()
    expect(
      within(table).getByRole("button", { name: "Edit openai" }),
    ).toBeInTheDocument()
  })

  it("moves the stored providers once the preview is reviewed", async () => {
    const requests = mockApi({
      migration: [
        {
          instance: "openai-eu",
          provider: "openai",
          api_base: "https://eu.example.com/v1",
          outcome: "migrate",
          detail: null,
          models: null,
          source_removed: false,
          notes: [],
        },
        {
          instance: "groq",
          provider: "groq",
          api_base: null,
          outcome: "conflict",
          detail: "A hosted provider named 'groq' already exists.",
          models: null,
          source_removed: false,
          notes: [],
        },
      ],
    })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      await screen.findByRole("button", { name: "Review the move" }),
    )
    const dialog = await screen.findByRole("alertdialog")
    expect(within(dialog).getByText(/already exists/)).toBeInTheDocument()
    await user.click(
      within(dialog).getByRole("button", { name: "Move 1 provider" }),
    )

    expect(
      await screen.findByText(/Moved 1 of 2 stored providers/),
    ).toBeInTheDocument()
    const moves = hostedRequests(requests, "POST").filter((request) =>
      request.url.endsWith("/hosted-providers/migration"),
    )
    // Only what the preview offered to move, so nothing unreviewed moves.
    expect(moves.map((request) => request.body)).toEqual([
      { rows: [{ instance: "openai-eu", updated_at: null }] },
    ])
  })
})
