import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type {
  DeploymentProviderMigrationRow,
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

function providerInfo(
  instance: string,
  source: ProviderInfo["source"],
): ProviderInfo {
  return {
    instance,
    provider_type: instance,
    name: instance,
    doc_url: null,
    description: null,
    env_key: null,
    pricing_urls: [],
    source,
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
    sweep?: HostedCatalogRefresh
    /** Answers each preview read in turn; the sweep's own body otherwise. */
    previews?: (() => Promise<Response>)[]
    /** The deployment providers the move would carry, and what each becomes. */
    migration?: DeploymentProviderMigrationRow[]
    /** The `providers:` instances GET /providers lists. */
    configured?: ProviderInfo[]
    /** What a connection test answers. */
    testResult?: HostedProviderTestResult
    /** The health records GET /providers/health answers. */
    health?: ProviderHealth[]
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
    if (url.includes(`${hosted}/refresh/preview`)) {
      const next = previews.shift()
      if (next) return next()
    }
    if (url.includes(`${hosted}/refresh`)) {
      return jsonResponse(sweep)
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

function renderPage(surfaces?: string[]) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return renderWithRouter(
    <DeploymentProvider value={bootstrap(surfaces ? { surfaces } : {})}>
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
    expect(
      screen.queryByRole("button", { name: "Refresh and purge" }),
    ).toBeNull()
  })

  it("refuses an operator the list itself 404s, without offering a write", async () => {
    mockApi({ listRefused: true })
    await renderPage()

    expect(
      await screen.findByText("Hosted providers is not available to you"),
    ).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Add provider" })).toBeNull()
    expect(
      screen.queryByRole("button", { name: "Refresh and purge" }),
    ).toBeNull()
  })

  it("says the page is not served where the surface is not published, asking nothing", async () => {
    const requests = mockApi()
    await renderPage([])

    expect(
      await screen.findByText("Hosted providers is not available here"),
    ).toBeInTheDocument()
    expect(requests).toEqual([])
  })

  it("offers to add the first provider when none is configured", async () => {
    mockApi({ providers: [] })
    await renderPage()

    expect(
      await screen.findByText("No hosted providers configured"),
    ).toBeInTheDocument()
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

  it("names what a sweep would remove before it removes anything", async () => {
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
      await screen.findByRole("button", { name: "Refresh and purge" }),
    )
    const dialog = await screen.findByRole("alertdialog")
    expect(
      await within(dialog).findByText(
        "1 model will be removed, taking 2 stored rates and 0 organization overrides with it.",
      ),
    ).toBeInTheDocument()
    expect(within(dialog).getByText("mistral:old-model")).toBeInTheDocument()
    expect(hostedRequests(requests, "POST")).toEqual([])

    await user.click(
      within(dialog).getByRole("button", { name: "Refresh and purge" }),
    )
    await waitFor(() =>
      expect(hostedRequests(requests, "POST")).toHaveLength(1),
    )
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
            ...EMPTY_SWEEP,
            removed: ["mistral:old-model"],
            removed_count: 1,
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
      await screen.findByRole("button", { name: "Refresh and purge" }),
    )
    let dialog = await screen.findByRole("alertdialog")
    expect(
      await within(dialog).findByText("mistral:old-model"),
    ).toBeInTheDocument()
    await user.click(within(dialog).getByRole("button", { name: "Cancel" }))
    await waitFor(() => expect(screen.queryByRole("alertdialog")).toBeNull())

    await user.click(screen.getByRole("button", { name: "Refresh and purge" }))
    dialog = await screen.findByRole("alertdialog")
    // The last opening's list is gone, and nothing can be confirmed yet.
    expect(
      within(dialog).getByText("Working out what would be removed…"),
    ).toBeInTheDocument()
    expect(within(dialog).queryByText("mistral:old-model")).toBeNull()
    const confirm = within(dialog).getByRole("button", {
      name: "Refresh and purge",
    })
    expect(confirm).toBeDisabled()
    await user.click(confirm)

    answerSecond(jsonResponse(EMPTY_SWEEP))
    await waitFor(() => expect(confirm).toBeEnabled())
    expect(
      hostedRequests(requests).filter((request) =>
        request.url.endsWith("/refresh/preview"),
      ),
    ).toHaveLength(2)
    expect(hostedRequests(requests, "POST")).toEqual([])
  })

  it("never confirms a sweep whose preview failed", async () => {
    const requests = mockApi({
      previews: [
        async () => jsonResponse({ detail: "Upstream unreachable" }, 502),
      ],
    })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      await screen.findByRole("button", { name: "Refresh and purge" }),
    )
    const dialog = await screen.findByRole("alertdialog")
    expect(
      await within(dialog).findByText("Upstream unreachable"),
    ).toBeInTheDocument()
    const confirm = within(dialog).getByRole("button", {
      name: "Refresh and purge",
    })
    expect(confirm).toBeDisabled()
    await user.click(confirm)
    expect(hostedRequests(requests, "POST")).toEqual([])
  })
})

describe("HostedProvidersPage, named providers and the move", () => {
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
      await screen.findByText(/Moved 1 of 2 deployment providers/),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Review the move" }),
    ).toBeInTheDocument()
  })

  it("reports a row copied but still served from the old page apart from a completed move", async () => {
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
      await screen.findByText(/Moved 0 of 1 deployment providers\. 1 copied/),
    ).toBeInTheDocument()
    expect(
      screen.getByText(
        /^Copied, still served from the Deployment providers page/,
      ),
    ).toBeInTheDocument()
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
      configured: [providerInfo("openai", "config")],
      health: [reachable("openai"), reachable("mistral")],
    })
    await renderPage()

    const table = await screen.findByRole("grid", { name: "Hosted providers" })
    await screen.findByRole("grid", { name: "Configured providers" })
    const row = (name: string) =>
      within(table).getByRole("rowheader", { name: new RegExp(`^${name}`) })
        .parentElement as HTMLElement
    // The configured instance's record is the one dialed under "openai".
    expect(within(row("openai")).queryByText("Reachable")).toBeNull()
    expect(
      await within(row("mistral")).findByText("Reachable"),
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

  it("lists the configured providers read-only beside the hosted ones", async () => {
    mockApi({
      configured: [
        providerInfo("anthropic", "config"),
        providerInfo("groq", "dashboard"),
      ],
    })
    await renderPage()

    const table = await screen.findByRole("grid", {
      name: "Configured providers",
    })
    expect(
      within(table).getByRole("rowheader", { name: "anthropic" }),
    ).toBeInTheDocument()
    // Stored through the deployment providers page: the move covers it instead.
    expect(within(table).queryByText("groq")).toBeNull()
  })

  it("moves the deployment providers once the preview is reviewed", async () => {
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
      await screen.findByText(/Moved 1 of 2 deployment providers/),
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
