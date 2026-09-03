import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactElement } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"
import type {
  GatewaySettings,
  KnownProvider,
  ProviderHealth,
  ProviderHealthResponse,
  ProviderInfo,
  StoredProvider,
  TestProviderResult,
} from "@/client"
import { ProvidersPage } from "@/features/providers/ProvidersPage"
import { PROVIDER_HEALTH_REFRESH_MS } from "@/shared/api/hooks"
import { withRouter } from "@/tests/router"

const CAPS = {
  streaming: false,
  reasoning: false,
  vision: false,
  pdf: false,
  embeddings: false,
  image_generation: false,
  audio: false,
  rerank: false,
  responses_api: false,
  moderation: false,
  list_models: false,
}

function providerInfo(
  instance: string,
  envKey: string | null = null,
): ProviderInfo {
  return {
    instance,
    provider_type: instance,
    name: instance,
    doc_url: null,
    description: null,
    env_key: envKey,
    pricing_urls: [],
    capabilities: CAPS,
  }
}

function storedProvider(
  instance: string,
  last4: string | null,
  decryptable = true,
  clientArgs: Record<string, unknown> = {},
): StoredProvider {
  return {
    instance,
    provider_type: null,
    api_base: null,
    last4,
    client_args: clientArgs,
    created_at: null,
    updated_at: "2026-01-01T00:00:00+00:00",
    decryptable,
  }
}

const SETTINGS: GatewaySettings = {
  mode: "standalone",
  version: "1.0.0",
  model_discovery: true,
  default_pricing: true,
  require_pricing: false,
  master_key_source: "configured",
  secret_key_configured: true,
  config: [],
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

// Build a health response, defaulting every provider in `meta` to reachable so
// tests that don't care about health still get a well-formed payload.
function healthResponse(providers: ProviderHealth[]): ProviderHealthResponse {
  // Mirror the backend: the summary checked_at is the most recent per-provider
  // checked_at, or null when no provider has ever been checked.
  const checkedAts = providers
    .map((p) => p.checked_at)
    .filter((t): t is string => t !== null)
  return {
    providers,
    healthy: providers.filter((p) => p.ok).length,
    degraded: providers.filter((p) => !p.ok && p.discovery_unsupported).length,
    total: providers.length,
    checked_at: checkedAts.length > 0 ? checkedAts.sort().at(-1)! : null,
  }
}

interface MockOpts {
  meta?: ProviderInfo[]
  stored?: StoredProvider[]
  settings?: GatewaySettings
  testResult?: TestProviderResult
  catalog?: KnownProvider[]
  // Per-provider health; defaults to every `meta` provider reachable. `healthRefresh`
  // is served for the forced-refresh (refresh=true) request, if given.
  health?: ProviderHealth[]
  healthRefresh?: ProviderHealth[]
  // When set, GET /v1/settings blocks on this promise before responding, so a
  // test can resolve it to simulate settings landing after the page has already
  // painted (and the operator has interacted with it).
  settingsGate?: Promise<unknown>
  // Force GET /v1/settings to fail, so the fail-closed gate can be exercised.
  settingsError?: boolean
  // When set, POST .../test blocks on this promise, so a test can hold a
  // connection test in flight while the page is used.
  testGate?: Promise<unknown>
  // Scripts successive POST .../test calls, so a test can hold the first one in
  // flight and let a later one answer first. Falls back to testGate/testResult
  // once the script runs out.
  testCalls?: { gate?: Promise<unknown>; result: TestProviderResult }[]
}

function mockApi(opts: MockOpts = {}) {
  let storedList = [...(opts.stored ?? [])]
  let settings = { ...(opts.settings ?? SETTINGS) }
  const meta = opts.meta ?? []
  const testResult = opts.testResult ?? {
    ok: true,
    model_count: 3,
    error: null,
    discovery_unsupported: false,
  }
  const catalog = opts.catalog ?? []
  const health =
    opts.health ??
    meta.map((info) => ({
      instance: info.instance,
      ok: true,
      model_count: 3,
      error: null,
      checked_at: null,
      discovery_unsupported: false,
    }))
  const healthRefresh = opts.healthRefresh ?? health
  let testCallCount = 0

  return vi
    .spyOn(globalThis, "fetch")
    .mockImplementation(async (input, init) => {
      const url = String(input)
      const method = (init?.method ?? "GET").toUpperCase()

      if (url.includes("/v1/provider-credentials")) {
        if (url.endsWith("/test") && method === "POST") {
          const scripted = opts.testCalls?.[testCallCount]
          testCallCount += 1
          if (scripted) {
            if (scripted.gate) await scripted.gate
            return jsonResponse(scripted.result)
          }
          if (opts.testGate) await opts.testGate
          return jsonResponse(testResult)
        }
        if (method === "POST") {
          const body = JSON.parse(String(init?.body)) as {
            instance: string
            api_key?: string | null
            client_args?: Record<string, unknown> | null
          }
          // Mirror the backend, which normalises a null client_args to {} (the
          // column is non-null).
          const row = storedProvider(
            body.instance,
            body.api_key ? body.api_key.slice(-4) : null,
            true,
            body.client_args ?? {},
          )
          storedList = [...storedList, row]
          return jsonResponse(row, 201)
        }
        if (method === "PATCH") {
          const instance = decodeURIComponent(url.split("/").pop() ?? "")
          const body = JSON.parse(String(init?.body)) as {
            provider_type?: string | null
            api_base?: string | null
            api_key?: string | null
            client_args?: Record<string, unknown> | null
            expected_updated_at?: string | null
          }
          const existing = storedList.find((p) => p.instance === instance)
          if (!existing)
            return jsonResponse(
              { detail: `Unknown provider: ${instance}` },
              404,
            )
          // Mirror the backend's optimistic-concurrency check: a non-null
          // expected_updated_at that does not match the stored updated_at is a 412
          // (see routes/providers.py, update_stored_provider).
          if (
            body.expected_updated_at != null &&
            body.expected_updated_at !== existing.updated_at
          ) {
            return jsonResponse(
              {
                detail:
                  "This provider was modified since you loaded it; reload and retry.",
              },
              412,
            )
          }
          // Mirror the backend, which keys off model_fields_set: an omitted field is
          // kept, a field sent as null is cleared (see routes/providers.py, UNSET).
          const row: StoredProvider = {
            ...existing,
            provider_type:
              "provider_type" in body
                ? (body.provider_type ?? null)
                : existing.provider_type,
            api_base:
              "api_base" in body ? (body.api_base ?? null) : existing.api_base,
            last4:
              "api_key" in body
                ? body.api_key
                  ? body.api_key.slice(-4)
                  : null
                : existing.last4,
            client_args:
              "client_args" in body
                ? (body.client_args ?? {})
                : existing.client_args,
            updated_at: "2026-01-02T00:00:00+00:00",
          }
          storedList = storedList.map((p) =>
            p.instance === instance ? row : p,
          )
          return jsonResponse(row)
        }
        if (method === "DELETE") {
          const instance = decodeURIComponent(url.split("/").pop() ?? "")
          storedList = storedList.filter((p) => p.instance !== instance)
          return new Response(null, { status: 204 })
        }
        return jsonResponse(storedList)
      }
      if (url.includes("/v1/providers/catalog/")) {
        // Detail endpoint: autofill hints for one selected provider.
        const id = decodeURIComponent(
          url.split("/v1/providers/catalog/")[1].split("?")[0],
        )
        const detail = catalog.find((p) => p.id === id)
        return detail
          ? jsonResponse(detail)
          : jsonResponse({ detail: `Unknown provider: ${id}` }, 404)
      }
      if (url.includes("/v1/providers/catalog")) {
        // List endpoint: id + display name only.
        return jsonResponse(catalog.map((p) => ({ id: p.id, name: p.name })))
      }
      if (url.includes("/v1/providers/health")) {
        return jsonResponse(
          healthResponse(url.includes("refresh=true") ? healthRefresh : health),
        )
      }
      if (url.includes("/v1/providers")) {
        return jsonResponse({ providers: meta })
      }
      if (url.includes("/v1/settings")) {
        if (method === "PATCH") {
          settings = { ...settings, ...JSON.parse(String(init?.body)) }
        }
        if (method === "GET") {
          if (opts.settingsGate) await opts.settingsGate
          if (opts.settingsError) return jsonResponse({ detail: "boom" }, 500)
        }
        return jsonResponse(settings)
      }
      return jsonResponse([])
    })
}

function renderPage(
  ui: ReactElement,
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } }),
) {
  return render(
    <QueryClientProvider client={client}>{ui}</QueryClientProvider>,
    { wrapper: withRouter() },
  )
}

function healthRequestCount(fetchMock: ReturnType<typeof mockApi>): number {
  return fetchMock.mock.calls.filter(([url]) =>
    String(url).includes("/v1/providers/health"),
  ).length
}

describe("ProvidersPage", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("lists config and stored providers with provenance and redacted keys", async () => {
    mockApi({
      meta: [
        providerInfo("openai", "OPENAI_API_KEY"),
        providerInfo("anthropic"),
      ],
      stored: [storedProvider("anthropic", "4242")],
    })

    renderPage(<ProvidersPage />)

    // Key off cells unique to each row (the instance name appears in two columns).
    const storedRow = (await screen.findByText("••••4242")).closest("tr")!
    expect(within(storedRow).getByText("STORED")).toBeInTheDocument()

    const configRow = screen.getByText("OPENAI_API_KEY").closest("tr")!
    expect(within(configRow).getByText("CONFIG")).toBeInTheDocument()
    // The plaintext key is never shown, only the last 4.
    expect(document.body.textContent).not.toContain("sk-")
  })

  it("adds a custom provider and posts a write-only key, never rendering it", async () => {
    const fetchMock = mockApi({ meta: [], stored: [] })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await user.click(
      await screen.findByRole("button", { name: "Add your first provider" }),
    )
    await user.click(screen.getByRole("button", { name: "Custom endpoint" }))
    await user.type(screen.getByLabelText("Name"), "my-llm")
    await user.type(screen.getByLabelText("API base"), "http://box:8000/v1")
    const apiKey = screen.getByLabelText("API key (optional)")
    expect(apiKey).toHaveAttribute("type", "password")
    await user.type(apiKey, "sk-live-9999")
    await user.click(screen.getByRole("button", { name: "Add provider" }))

    const post = fetchMock.mock.calls.find(
      ([u, init]) =>
        String(u).endsWith("/v1/provider-credentials") &&
        (init?.method ?? "") === "POST",
    )
    expect(post).toBeDefined()
    expect(JSON.parse(String(post?.[1]?.body))).toMatchObject({
      instance: "my-llm",
      provider_type: "openai-compatible",
      api_base: "http://box:8000/v1",
      api_key: "sk-live-9999",
    })

    // After the round trip the row shows the redacted key, never the plaintext.
    expect(await screen.findByText("••••9999")).toBeInTheDocument()
    expect(document.body.textContent).not.toContain("sk-live-9999")
  })

  it("offers the known-provider picker with an Advanced disclosure", async () => {
    mockApi({
      stored: [storedProvider("anthropic", "0000")],
      catalog: [
        {
          id: "openai",
          name: "OpenAI",
          env_key: "OPENAI_API_KEY",
          default_api_base: "https://api.openai.com/v1",
          requires_api_key: true,
          env_key_present: false,
        },
      ],
    })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await screen.findByText("••••0000")
    await user.click(screen.getByRole("button", { name: "Add provider" }))

    // Known provider is the default tab: a provider picker plus a collapsed Advanced section.
    expect(screen.getByPlaceholderText("Search providers…")).toBeInTheDocument()
    expect(
      screen.getByText("Advanced (API base, rename, client options)"),
    ).toBeInTheDocument()
    expect(screen.queryByLabelText("API base")).not.toBeInTheDocument()
    expect(
      screen.queryByLabelText("Client options (JSON)"),
    ).not.toBeInTheDocument()
  })

  it("fetches provider autofill hints lazily, only after one is selected", async () => {
    const fetchMock = mockApi({
      stored: [storedProvider("anthropic", "0000")],
      catalog: [
        {
          id: "openai",
          name: "OpenAI",
          env_key: "OPENAI_API_KEY",
          default_api_base: "https://api.openai.com/v1",
          requires_api_key: true,
          env_key_present: false,
        },
      ],
    })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await screen.findByText("••••0000")
    await user.click(screen.getByRole("button", { name: "Add provider" }))

    const detailCalls = () =>
      fetchMock.mock.calls.filter(([u]) =>
        String(u).includes("/v1/providers/catalog/openai"),
      )

    // Opening the picker lists providers (id + name) but must not import any
    // provider SDK: no per-provider detail call until one is chosen (issue #365).
    expect(detailCalls()).toHaveLength(0)

    await user.type(screen.getByPlaceholderText("Search providers…"), "OpenAI")
    await user.click(await screen.findByRole("option", { name: /OpenAI/ }))

    // Selecting the provider triggers exactly the one detail fetch it needs.
    await screen.findByText(/OpenAI's endpoint is built in/)
    expect(detailCalls().length).toBeGreaterThan(0)
  })

  it("keeps Add disabled for a key-requiring provider until a key is entered", async () => {
    mockApi({
      stored: [storedProvider("anthropic", "0000")],
      catalog: [
        {
          id: "openai",
          name: "OpenAI",
          env_key: "OPENAI_API_KEY",
          default_api_base: "https://api.openai.com/v1",
          requires_api_key: true,
          env_key_present: false,
        },
      ],
    })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await screen.findByText("••••0000")
    await user.click(screen.getByRole("button", { name: "Add provider" }))

    await user.type(screen.getByPlaceholderText("Search providers…"), "OpenAI")
    await user.click(await screen.findByRole("option", { name: /OpenAI/ }))
    // Close the combobox popover, which otherwise aria-hides the submit button.
    await user.keyboard("{Escape}")

    const submit = screen.getByRole("button", { name: "Add provider" })
    expect(submit).toBeDisabled()

    await user.type(screen.getByLabelText("API key"), "sk-live-xxxx")
    expect(submit).toBeEnabled()
  })

  it("lets a key-requiring provider submit without a key when its env var is already set", async () => {
    const fetchMock = mockApi({
      stored: [storedProvider("anthropic", "0000")],
      catalog: [
        {
          id: "openai",
          name: "OpenAI",
          env_key: "OPENAI_API_KEY",
          default_api_base: "https://api.openai.com/v1",
          requires_api_key: true,
          env_key_present: true,
        },
      ],
    })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await screen.findByText("••••0000")
    await user.click(screen.getByRole("button", { name: "Add provider" }))

    await user.type(screen.getByPlaceholderText("Search providers…"), "OpenAI")
    await user.click(await screen.findByRole("option", { name: /OpenAI/ }))
    // Close the combobox popover, which otherwise aria-hides the submit button.
    await user.keyboard("{Escape}")

    // The field is optional and the copy explains the env fallback. The hint
    // arrives once the selected provider's detail loads, so wait for it.
    await screen.findByText(/OPENAI_API_KEY is set on the server/)
    expect(screen.getByLabelText("API key (optional)")).toBeInTheDocument()

    // Submit with no key: the server stores none and any-llm reads OPENAI_API_KEY.
    const submit = screen.getByRole("button", { name: "Add provider" })
    expect(submit).toBeEnabled()
    await user.click(submit)

    const post = fetchMock.mock.calls.find(
      ([u, init]) =>
        String(u).endsWith("/v1/provider-credentials") &&
        (init?.method ?? "") === "POST",
    )
    expect(post).toBeDefined()
    expect(JSON.parse(String(post?.[1]?.body))).toMatchObject({
      instance: "openai",
      api_key: null,
    })
  })

  it("replaces the welcome onboarding with the add form", async () => {
    mockApi({ meta: [], stored: [] })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    expect(await screen.findByText("Welcome to Otari")).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Add provider" }),
    ).not.toBeInTheDocument()
    // Only the onboarding panel ("Welcome to Otari") shows: the table (and its own
    // "no rows" fallback, whose "No providers yet" text is unique to it) is
    // suppressed so the two empty states are not stacked.
    expect(screen.queryByText(/No providers yet/)).not.toBeInTheDocument()
    expect(
      screen.queryByRole("grid", { name: "Providers" }),
    ).not.toBeInTheDocument()
    await user.click(
      screen.getByRole("button", { name: "Add your first provider" }),
    )

    expect(screen.queryByText("Welcome to Otari")).not.toBeInTheDocument()
    expect(screen.getByPlaceholderText("Search providers…")).toBeInTheDocument()
  })

  it("points the onboarding quickstart at the gateway-served tutorial in a new tab", async () => {
    mockApi({ meta: [], stored: [] })
    renderPage(<ProvidersPage />)

    await screen.findByText("Welcome to Otari")
    const quickstart = screen.getByRole("link", { name: "quickstart" })
    // /welcome is a gateway-rendered page, not a client route: a router Link
    // (href "#/welcome") would hit the catch-all and redirect to the overview.
    expect(quickstart).toHaveAttribute("href", "/welcome")
    // Following it leaves the SPA, so it must not replace the dashboard tab.
    expect(quickstart).toHaveAttribute("target", "_blank")
    expect(quickstart).toHaveAttribute("rel", "noreferrer")
  })

  it("disables adding providers when OTARI_SECRET_KEY is not set", async () => {
    mockApi({
      stored: [storedProvider("openai", "1234")],
      settings: { ...SETTINGS, secret_key_configured: false },
    })
    renderPage(<ProvidersPage />)

    await screen.findByText("••••1234")
    // The button starts enabled and flips once /v1/settings resolves, so wait
    // for the settled disabled state rather than asserting on first paint.
    expect(await screen.findByText(/OTARI_SECRET_KEY/)).toBeInTheDocument()
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Add provider" }),
      ).toBeDisabled(),
    )
  })

  it("disables the first-run add button when OTARI_SECRET_KEY is not set", async () => {
    mockApi({
      meta: [],
      stored: [],
      settings: { ...SETTINGS, secret_key_configured: false },
    })
    renderPage(<ProvidersPage />)

    expect(await screen.findByText("Welcome to Otari")).toBeInTheDocument()
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Add your first provider" }),
      ).toBeDisabled(),
    )
  })

  it("fails closed and disables adding providers when settings can't be loaded", async () => {
    mockApi({ stored: [storedProvider("openai", "1234")], settingsError: true })
    renderPage(<ProvidersPage />)

    await screen.findByText("••••1234")
    // A settings error leaves the key state unknown; disable rather than let the
    // operator fill in the form and fail on submit.
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Add provider" }),
      ).toBeDisabled(),
    )
  })

  it("retracts an open add form if settings then report OTARI_SECRET_KEY is unset", async () => {
    let releaseSettings = () => {}
    const settingsGate = new Promise<void>((resolve) => {
      releaseSettings = resolve
    })
    // The onboarding gate ignores settings loading, so the first-run card (and its
    // enabled add button) is reachable before /v1/settings resolves.
    mockApi({
      meta: [],
      stored: [],
      settings: { ...SETTINGS, secret_key_configured: false },
      settingsGate,
    })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await user.click(
      await screen.findByRole("button", { name: "Add your first provider" }),
    )
    expect(screen.getByPlaceholderText("Search providers…")).toBeInTheDocument()

    // Settings land late and report the key is unavailable: the form must retract
    // so its submit can never reach the create mutation.
    releaseSettings()
    await waitFor(() =>
      expect(
        screen.queryByPlaceholderText("Search providers…"),
      ).not.toBeInTheDocument(),
    )
    expect(screen.getByText(/OTARI_SECRET_KEY/)).toBeInTheDocument()
  })

  it("hides the onboarding once a provider exists", async () => {
    mockApi({ stored: [storedProvider("openai", "1234")] })
    renderPage(<ProvidersPage />)

    await screen.findByText("••••1234")
    expect(screen.queryByText("Welcome to Otari")).not.toBeInTheDocument()
  })

  it("flags a stored provider whose key can't be decrypted", async () => {
    mockApi({ stored: [storedProvider("home-lab", "0000", false)] })
    renderPage(<ProvidersPage />)

    expect(await screen.findByText(/key unreadable/)).toBeInTheDocument()
    // Test is disabled for an unreadable key; Edit/Delete remain to recover it.
    expect(screen.getByRole("button", { name: "Test" })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Edit" })).toBeEnabled()
  })

  it("reports a successful connection test", async () => {
    mockApi({
      stored: [storedProvider("openai", "1234")],
      testResult: {
        ok: true,
        model_count: 5,
        error: null,
        discovery_unsupported: false,
      },
    })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await screen.findByText("••••1234")
    await user.click(screen.getByRole("button", { name: "Test" }))

    expect(
      await screen.findByText(/Connected\. 5 models available\./),
    ).toBeInTheDocument()
  })

  // The gateway-wide require_pricing alarm moved to the app shell; its behavior
  // (show, enable default pricing, dismiss) is covered in PricingWarning.test.tsx.

  it("shows each provider's reachability, including config-only providers", async () => {
    mockApi({
      meta: [providerInfo("openai"), providerInfo("anthropic")],
      health: [
        {
          instance: "openai",
          ok: true,
          model_count: 12,
          error: null,
          checked_at: "2026-07-21T00:00:00+00:00",
          discovery_unsupported: false,
        },
        {
          instance: "anthropic",
          ok: false,
          model_count: 0,
          error: "authentication failed: invalid key",
          checked_at: "2026-07-21T00:00:00+00:00",
          discovery_unsupported: false,
        },
      ],
    })
    renderPage(<ProvidersPage />)

    // Scope by the status pill's row (the provider name repeats in the Type cell).
    const reachableRow = (await screen.findByText("Reachable")).closest("tr")!
    expect(within(reachableRow).getAllByText("openai").length).toBeGreaterThan(
      0,
    )

    const unreachablePill = screen.getByText("Unreachable")
    const unreachableRow = unreachablePill.closest("tr")!
    expect(
      within(unreachableRow).getAllByText("anthropic").length,
    ).toBeGreaterThan(0)
    // The provider error rides along as the pill's tooltip.
    expect(unreachablePill).toHaveAttribute(
      "title",
      expect.stringContaining("authentication failed"),
    )
  })

  it("summarizes how many providers are reachable", async () => {
    mockApi({
      meta: [providerInfo("openai"), providerInfo("anthropic")],
      health: [
        {
          instance: "openai",
          ok: true,
          model_count: 3,
          error: null,
          checked_at: null,
          discovery_unsupported: false,
        },
        {
          instance: "anthropic",
          ok: false,
          model_count: 0,
          error: "down",
          checked_at: null,
          discovery_unsupported: false,
        },
      ],
    })
    renderPage(<ProvidersPage />)

    expect(
      await screen.findByText("1 of 2 providers reachable"),
    ).toBeInTheDocument()
  })

  it("warns instead of condemning a provider whose backend has no /models endpoint", async () => {
    // otari#447: a provider that answers no model listing may still serve
    // requests, so it must not read as "Unreachable" like a bad key does.
    mockApi({
      meta: [providerInfo("openai"), providerInfo("otari")],
      health: [
        {
          instance: "openai",
          ok: true,
          model_count: 3,
          error: null,
          checked_at: null,
          discovery_unsupported: false,
        },
        {
          instance: "otari",
          ok: false,
          model_count: 0,
          error: "Error code: 404 - {'detail': 'Not Found'}",
          checked_at: null,
          discovery_unsupported: true,
        },
      ],
    })
    renderPage(<ProvidersPage />)

    const pill = await screen.findByText("No model discovery")
    expect(
      within(pill.closest("tr")!).getAllByText("otari").length,
    ).toBeGreaterThan(0)
    expect(screen.queryByText("Unreachable")).not.toBeInTheDocument()
    // The provider error stays available, alongside why it is not fatal.
    expect(pill).toHaveAttribute("title", expect.stringContaining("404"))
    expect(pill).toHaveAttribute(
      "title",
      expect.stringContaining("may still work"),
    )
    // The summary calls it out separately from the reachable count.
    expect(
      await screen.findByText("1 of 2 providers reachable"),
    ).toBeInTheDocument()
    expect(screen.getByText("1 without model discovery")).toBeInTheDocument()
  })

  it("reports a test against a provider with no model listing as unverified, not failed", async () => {
    mockApi({
      stored: [storedProvider("otari", "1234")],
      testResult: {
        ok: false,
        model_count: 0,
        error: "Error code: 404",
        discovery_unsupported: true,
      },
    })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await screen.findByText("••••1234")
    await user.click(screen.getByRole("button", { name: "Test" }))

    expect(await screen.findByText(/could not be verified/)).toBeInTheDocument()
    // The provider error stays on screen: a 404 is also what a wrong api_base
    // returns, so hiding it would mask a misconfiguration behind reassurance.
    expect(screen.getByText("Error code: 404")).toBeInTheDocument()
  })

  it("sends client options entered on the custom-endpoint form", async () => {
    // otari#517: client_args is the only way to give the provider client a
    // timeout, which a slow self-hosted backend needs; it was reachable only
    // through config.yml or the raw API before.
    const fetchMock = mockApi({ meta: [], stored: [] })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await user.click(
      await screen.findByRole("button", { name: "Add your first provider" }),
    )
    await user.click(screen.getByRole("button", { name: "Custom endpoint" }))
    await user.type(screen.getByLabelText("Name"), "homelab")
    await user.type(
      screen.getByLabelText("API base"),
      "https://my-box.example.net",
    )
    await user.type(
      screen.getByLabelText("Client options (JSON)"),
      '{{"timeout": 1800}',
    )
    await user.click(screen.getByRole("button", { name: "Add provider" }))

    const post = await waitFor(() => {
      const call = fetchMock.mock.calls.find(
        ([u, init]) =>
          String(u).endsWith("/v1/provider-credentials") &&
          (init?.method ?? "") === "POST",
      )
      expect(call).toBeDefined()
      return call!
    })
    expect(JSON.parse(String(post[1]?.body))).toMatchObject({
      instance: "homelab",
      client_args: { timeout: 1800 },
    })
  })

  it("rejects client options that are not a JSON object instead of sending them", async () => {
    const fetchMock = mockApi({ meta: [], stored: [] })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await user.click(
      await screen.findByRole("button", { name: "Add your first provider" }),
    )
    await user.click(screen.getByRole("button", { name: "Custom endpoint" }))
    await user.type(screen.getByLabelText("Name"), "homelab")
    await user.type(
      screen.getByLabelText("API base"),
      "https://my-box.example.net",
    )

    const clientArgs = screen.getByLabelText("Client options (JSON)")
    await user.type(clientArgs, "timeout: 1800")
    expect(await screen.findByText("Not valid JSON.")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Add provider" })).toBeDisabled()
    // A "Test connection" would hit the provider with the same bad options.
    expect(
      screen.getByRole("button", { name: "Test connection" }),
    ).toBeDisabled()

    // Valid JSON, but not an object: the API takes a mapping of client kwargs.
    // "[[" is userEvent's escape for a literal "[".
    await user.clear(clientArgs)
    await user.type(clientArgs, "[[1800]")
    expect(
      await screen.findByText('Must be a JSON object, like {"timeout": 1800}.'),
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Add provider" })).toBeDisabled()

    expect(
      fetchMock.mock.calls.some(
        ([u, init]) =>
          String(u).endsWith("/v1/provider-credentials") &&
          (init?.method ?? "") === "POST",
      ),
    ).toBe(false)
  })

  it("holds Advanced open while invalid client options are blocking the submit", async () => {
    // Otherwise collapsing the section leaves "Add provider" disabled with the
    // reason, and the field to fix it, off screen.
    mockApi({
      stored: [],
      catalog: [
        {
          id: "openai",
          name: "OpenAI",
          env_key: "OPENAI_API_KEY",
          default_api_base: "https://api.openai.com/v1",
          requires_api_key: true,
          env_key_present: true,
        },
      ],
    })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await user.click(
      await screen.findByRole("button", { name: "Add your first provider" }),
    )
    await user.type(screen.getByPlaceholderText("Search providers…"), "OpenAI")
    await user.click(await screen.findByRole("option", { name: /OpenAI/ }))
    // Close the combobox popover, which otherwise aria-hides the rest of the form.
    await user.keyboard("{Escape}")
    await user.click(
      screen.getByRole("button", {
        name: "Advanced (API base, rename, client options)",
      }),
    )
    await user.type(screen.getByLabelText("Client options (JSON)"), "oops")
    expect(await screen.findByText("Not valid JSON.")).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Hide advanced" }))
    expect(screen.getByLabelText("Client options (JSON)")).toBeInTheDocument()
    expect(screen.getByText("Not valid JSON.")).toBeInTheDocument()

    // The hide the operator asked for takes effect once the section is no longer
    // the thing blocking the submit.
    await user.clear(screen.getByLabelText("Client options (JSON)"))
    await waitFor(() =>
      expect(
        screen.queryByLabelText("Client options (JSON)"),
      ).not.toBeInTheDocument(),
    )
  })

  it("prefills stored client options on edit and clears them when emptied", async () => {
    const fetchMock = mockApi({
      stored: [storedProvider("homelab", "1234", true, { timeout: 1800 })],
    })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await screen.findByText("••••1234")
    await user.click(screen.getByRole("button", { name: "Edit" }))
    const clientArgs = screen.getByLabelText("Client options (JSON)")
    expect(clientArgs).toHaveValue(JSON.stringify({ timeout: 1800 }, null, 2))

    // Emptying the field clears the stored options: an explicit null, not an
    // omission, which the API would read as "leave them alone".
    await user.clear(clientArgs)
    await user.click(screen.getByRole("button", { name: "Save changes" }))

    const patch = await waitFor(() => {
      const call = fetchMock.mock.calls.find(
        ([, init]) => (init?.method ?? "") === "PATCH",
      )
      expect(call).toBeDefined()
      return call!
    })
    expect(JSON.parse(String(patch[1]?.body)).client_args).toBeNull()
  })

  it("keeps a save from going out while the edited client options are invalid", async () => {
    const fetchMock = mockApi({
      stored: [storedProvider("homelab", "1234", true, { timeout: 1800 })],
    })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await screen.findByText("••••1234")
    await user.click(screen.getByRole("button", { name: "Edit" }))
    await user.type(screen.getByLabelText("Client options (JSON)"), "oops")

    expect(await screen.findByText("Not valid JSON.")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Save changes" })).toBeDisabled()
    expect(
      fetchMock.mock.calls.some(([, init]) => (init?.method ?? "") === "PATCH"),
    ).toBe(false)
  })

  it("drops a connection-test verdict once the provider is edited", async () => {
    // otari#464: the verdict describes the credentials the test ran against, so
    // leaving it under the row after a save contradicts the status pill above it
    // and sends the operator hunting for a backend bug.
    mockApi({
      stored: [storedProvider("otari", "1234")],
      testResult: {
        ok: false,
        model_count: 0,
        error: "authentication failed: invalid key",
        discovery_unsupported: false,
      },
    })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await screen.findByText("••••1234")
    await user.click(screen.getByRole("button", { name: "Test" }))
    expect(
      await screen.findByText("authentication failed: invalid key"),
    ).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Edit" }))
    await user.clear(screen.getByLabelText("API base"))
    await user.type(
      screen.getByLabelText("API base"),
      "https://api.otari.ai/v1",
    )
    await user.click(screen.getByRole("button", { name: "Save changes" }))

    await waitFor(() =>
      expect(
        screen.queryByText("authentication failed: invalid key"),
      ).not.toBeInTheDocument(),
    )
  })

  it("does not let a test still in flight write its verdict back after a save", async () => {
    // A test against a wrong api_base settles only when it times out, which is
    // when the operator is most likely to go and fix the credentials. The save
    // retires the verdict; the late result must not restore it.
    let releaseTest: () => void = () => {}
    const testGate = new Promise<void>((resolve) => {
      releaseTest = resolve
    })
    mockApi({
      stored: [storedProvider("otari", "1234")],
      testResult: {
        ok: false,
        model_count: 0,
        error: "authentication failed: invalid key",
        discovery_unsupported: false,
      },
      testGate,
    })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await screen.findByText("••••1234")
    await user.click(screen.getByRole("button", { name: "Test" }))
    expect(await screen.findByText("Testing…")).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Edit" }))
    await user.clear(screen.getByLabelText("API base"))
    await user.type(
      screen.getByLabelText("API base"),
      "https://api.otari.ai/v1",
    )
    await user.click(screen.getByRole("button", { name: "Save changes" }))
    await waitFor(() =>
      expect(screen.queryByText("Testing…")).not.toBeInTheDocument(),
    )

    releaseTest()

    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Test" })).toBeEnabled(),
    )
    expect(
      screen.queryByText("authentication failed: invalid key"),
    ).not.toBeInTheDocument()
  })

  it("does not carry a verdict over to a provider re-added under the same name", async () => {
    mockApi({
      stored: [storedProvider("otari", "1234")],
      testResult: {
        ok: false,
        model_count: 0,
        error: "authentication failed: invalid key",
        discovery_unsupported: false,
      },
    })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await screen.findByText("••••1234")
    await user.click(screen.getByRole("button", { name: "Test" }))
    expect(
      await screen.findByText("authentication failed: invalid key"),
    ).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Delete" }))
    await user.click(screen.getByRole("button", { name: "Delete" }))
    await screen.findByText("Welcome to Otari")

    await user.click(
      screen.getByRole("button", { name: "Add your first provider" }),
    )
    await user.click(screen.getByRole("button", { name: "Custom endpoint" }))
    await user.type(screen.getByLabelText("Name"), "otari")
    await user.type(
      screen.getByLabelText("API base"),
      "https://api.otari.ai/v1",
    )
    await user.click(screen.getByRole("button", { name: "Add provider" }))

    // The rebuilt row is a different provider: it must start with no verdict.
    await screen.findByRole("button", { name: "Test" })
    expect(
      screen.queryByText("authentication failed: invalid key"),
    ).not.toBeInTheDocument()
  })

  it("does not let a test still in flight write its verdict back after a delete", async () => {
    // The delete path retires the verdict through its own callback, separate from
    // the save path, and the request behind it is still hanging. A provider
    // re-added under the same name is a different provider, so the late result
    // must not surface on its row.
    let releaseTest: () => void = () => {}
    const testGate = new Promise<void>((resolve) => {
      releaseTest = resolve
    })
    mockApi({
      stored: [storedProvider("otari", "1234")],
      testResult: {
        ok: false,
        model_count: 0,
        error: "authentication failed: invalid key",
        discovery_unsupported: false,
      },
      testGate,
    })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await screen.findByText("••••1234")
    await user.click(screen.getByRole("button", { name: "Test" }))
    expect(await screen.findByText("Testing…")).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Delete" }))
    await user.click(screen.getByRole("button", { name: "Delete" }))
    await screen.findByText("Welcome to Otari")

    await user.click(
      screen.getByRole("button", { name: "Add your first provider" }),
    )
    await user.click(screen.getByRole("button", { name: "Custom endpoint" }))
    await user.type(screen.getByLabelText("Name"), "otari")
    await user.type(
      screen.getByLabelText("API base"),
      "https://api.otari.ai/v1",
    )
    await user.click(screen.getByRole("button", { name: "Add provider" }))
    await screen.findByRole("button", { name: "Test" })

    releaseTest()

    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Test" })).toBeEnabled(),
    )
    expect(
      screen.queryByText("authentication failed: invalid key"),
    ).not.toBeInTheDocument()
  })

  it("lets the retest after a save win, even if the pre-save test answers last", async () => {
    // Saving retires the pending verdict, which re-enables Test in the same
    // instant, so the operator can retest the fixed credentials while the
    // pre-save request is still hanging. Only the run id distinguishes the two:
    // the late answer belongs to credentials that no longer exist.
    let releaseStale: () => void = () => {}
    const staleGate = new Promise<void>((resolve) => {
      releaseStale = resolve
    })
    mockApi({
      stored: [storedProvider("otari", "1234")],
      testCalls: [
        {
          gate: staleGate,
          result: {
            ok: false,
            model_count: 0,
            error: "authentication failed: invalid key",
            discovery_unsupported: false,
          },
        },
        {
          result: {
            ok: true,
            model_count: 7,
            error: null,
            discovery_unsupported: false,
          },
        },
      ],
    })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await screen.findByText("••••1234")
    await user.click(screen.getByRole("button", { name: "Test" }))
    expect(await screen.findByText("Testing…")).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Edit" }))
    await user.clear(screen.getByLabelText("API base"))
    await user.type(
      screen.getByLabelText("API base"),
      "https://api.otari.ai/v1",
    )
    await user.click(screen.getByRole("button", { name: "Save changes" }))
    await waitFor(() =>
      expect(screen.queryByText("Testing…")).not.toBeInTheDocument(),
    )

    await user.click(screen.getByRole("button", { name: "Test" }))
    releaseStale()

    expect(
      await screen.findByText("Connected. 7 models available."),
    ).toBeInTheDocument()
    expect(
      screen.queryByText("authentication failed: invalid key"),
    ).not.toBeInTheDocument()
  })

  it("settles both rows when two connection tests run at once", async () => {
    // One useMutation observer serves every row, and TanStack Query detaches it
    // from the previous mutation as soon as the next mutate lands, dropping that
    // call's callbacks. Testing a second provider while the first was in flight
    // used to leave the first row spinning on "Testing…" with its Test button
    // disabled for the life of the page.
    let releaseTests: () => void = () => {}
    const testGate = new Promise<void>((resolve) => {
      releaseTests = resolve
    })
    mockApi({
      stored: [
        storedProvider("anthropic", "1111"),
        storedProvider("openai", "2222"),
      ],
      testResult: {
        ok: true,
        model_count: 4,
        error: null,
        discovery_unsupported: false,
      },
      testGate,
    })
    const user = userEvent.setup()
    renderPage(<ProvidersPage />)

    await screen.findByText("••••1111")
    expect(screen.getAllByRole("button", { name: "Test" })).toHaveLength(2)
    await user.click(screen.getAllByRole("button", { name: "Test" })[0])
    await user.click(screen.getAllByRole("button", { name: "Test" })[1])
    expect(await screen.findAllByText("Testing…")).toHaveLength(2)

    releaseTests()

    // Both verdicts land, and neither row is left stuck pending.
    await waitFor(() =>
      expect(
        screen.getAllByText("Connected. 4 models available."),
      ).toHaveLength(2),
    )
    expect(screen.queryByText("Testing…")).not.toBeInTheDocument()
    for (const button of screen.getAllByRole("button", { name: "Test" })) {
      expect(button).toBeEnabled()
    }
  })

  it("does not automatically re-check all providers within an hour", async () => {
    const fetchMock = mockApi({ meta: [providerInfo("openai")] })
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    })
    const first = renderPage(<ProvidersPage />, client)

    await screen.findByText("1 of 1 provider reachable")
    expect(healthRequestCount(fetchMock)).toBe(1)

    first.unmount()
    client.setQueryData(
      ["provider-health"],
      healthResponse([
        {
          instance: "openai",
          ok: true,
          model_count: 3,
          error: null,
          checked_at: null,
          discovery_unsupported: false,
        },
      ]),
      { updatedAt: Date.now() - (PROVIDER_HEALTH_REFRESH_MS - 5_000) },
    )
    renderPage(<ProvidersPage />, client)

    await screen.findByText("1 of 1 provider reachable")
    await waitFor(() => expect(healthRequestCount(fetchMock)).toBe(1))
  })

  it("forces a live re-check of every provider on Re-check all", async () => {
    const user = userEvent.setup()
    mockApi({
      meta: [providerInfo("openai")],
      health: [
        {
          instance: "openai",
          ok: true,
          model_count: 3,
          error: null,
          checked_at: null,
          discovery_unsupported: false,
        },
      ],
      healthRefresh: [
        {
          instance: "openai",
          ok: false,
          model_count: 0,
          error: "provider down",
          checked_at: null,
          discovery_unsupported: false,
        },
      ],
    })
    renderPage(<ProvidersPage />)

    const row = (await screen.findByText("Reachable")).closest("tr")!
    expect(within(row).getAllByText("openai").length).toBeGreaterThan(0)

    await user.click(screen.getByRole("button", { name: "Re-check all" }))

    expect(await within(row).findByText("Unreachable")).toBeInTheDocument()
    expect(
      await screen.findByText("0 of 1 provider reachable"),
    ).toBeInTheDocument()
  })

  it("links a provider name to the filtered models page", async () => {
    mockApi({
      meta: [providerInfo("openai"), providerInfo("anthropic")],
    })

    renderPage(<ProvidersPage />)

    // Clicking a provider navigates to the Models page filtered to that provider.
    const openaiLink = await screen.findByRole("link", { name: "openai" })
    expect(openaiLink).toHaveAttribute("href", "/models?provider=openai")

    const anthropicLink = screen.getByRole("link", { name: "anthropic" })
    expect(anthropicLink).toHaveAttribute("href", "/models?provider=anthropic")
  })
})
