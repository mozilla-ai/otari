import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import type { ReactElement } from "react"
import { vi } from "vitest"

import type {
  DeploymentBootstrap,
  DeploymentProviderMigrationRow,
  HostedCatalogPriceUpdate,
  HostedCatalogRefresh,
  HostedModel,
  HostedPriceReset,
  HostedProvider,
  HostedProviderRegion,
  HostedProviderTestResult,
  HostedVendor,
  PlatformConnectStatus,
  ProviderHealth,
  ProviderInfo,
} from "@/client"
import { API_ROOT } from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap } from "@/tests/fixtures"
import { renderWithRouter, type TestRouterOptions } from "@/tests/router"

/**
 * The hosted providers API as the list page and a provider's page read it,
 * answered from fixtures by a stub over `fetch`. Every request is recorded, so
 * a test can assert what was sent.
 */

export interface Request {
  url: string
  method: string
  body: unknown
}

/** The hidden input behind "Load from JSON file", which a test uploads into directly. */
export function countriesFileInput(): HTMLInputElement {
  const input = document.querySelector<HTMLInputElement>(
    'input[type="file"][accept*="json"]',
  )
  if (!input) throw new Error("no countries file input")
  return input
}

export function hostedProvider(
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
    supported_countries: null,
    zero_data_retention: false,
    models: { total: 1, serving: 1, unpriced: 0, vendors_off: 0 },
    created_at: "2026-10-01T00:00:00Z",
    updated_at: null,
    ...overrides,
  }
}

/** One region of the `openai` provider, using every value of the provider's. */
export function hostedRegion(
  overrides: Partial<HostedProviderRegion> = {},
): HostedProviderRegion {
  const slug = overrides.slug ?? "primary"
  return {
    slug,
    selector: `openai@${slug}`,
    label: slug,
    position: 0,
    enabled: true,
    data_location: null,
    has_api_key: false,
    api_key_last4: null,
    api_base: null,
    client_args: null,
    supported_countries: null,
    decryptable: true,
    ...overrides,
  }
}

export function hostedModel(overrides: Partial<HostedModel> = {}): HostedModel {
  return {
    id: "m1",
    provider: "openai",
    model: "gpt-4o",
    type: "chat",
    input_price_per_million: 2.5,
    output_price_per_million: 10,
    cache_read_price_per_million: null,
    cache_write_price_per_million: null,
    unit: "token",
    price_source: "defaults",
    vendor: "OpenAI",
    default_rates: null,
    enabled: true,
    created_at: "2026-10-01T00:00:00Z",
    updated_at: null,
    ...overrides,
  }
}

export function providerInfo(instance: string): ProviderInfo {
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

export const EMPTY_SWEEP: HostedCatalogRefresh = {
  providers: [],
  removed: [],
  removed_count: 0,
  removed_price_rows: 0,
  removed_override_rows: 0,
  kept: [],
}

export const PRICES: HostedCatalogPriceUpdate = {
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

/** The vendor counts the models list answers, as the API derives them from the rows. */
export function vendorFacets(
  models: HostedModel[],
  vendorsOff: string[],
): HostedVendor[] {
  // A vendor switched off that no model goes by is listed too, as the server lists it.
  const names = [
    ...new Set([
      ...models.map((model) => model.vendor ?? "Other"),
      ...vendorsOff,
    ]),
  ]
  return names.sort().map((vendor) => {
    const members = models.filter(
      (model) => (model.vendor ?? "Other") === vendor,
    )
    const enabled = !vendorsOff.includes(vendor)
    return {
      vendor,
      enabled,
      total: members.length,
      serving: enabled ? members.filter((model) => model.enabled).length : 0,
      unpriced: members.filter((model) => model.disabled_reason === "unpriced")
        .length,
      unavailable: members.filter(
        (model) => model.disabled_reason === "unavailable",
      ).length,
      custom_differs: members.filter((model) => model.default_rates).length,
    }
  })
}

export function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

export function mockApi(
  opts: {
    granted?: boolean
    /** Answers the list 404, as the router refusing a non-operator. */
    listRefused?: boolean
    /** Answers a provider PATCH with this refusal. */
    patchRefusal?: unknown
    /** The status that refusal carries; 400 otherwise. */
    patchRefusalStatus?: number
    providers?: HostedProvider[]
    models?: HostedModel[]
    /** The vendor switches; every vendor of `models` is on otherwise. */
    vendorsOff?: string[]
    /** The total the models list reports, above what it returns for a provider it cuts. */
    modelCount?: number
    /** Models whose default is gone by the time a reset is applied, after the preview listed them. */
    lostDefaults?: string[]
    /** What a price reset answers, dry run or not. */
    priceReset?: HostedPriceReset
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
    /** Pricing keys a DELETE /pricing answers 404 for, as nothing stored there. */
    missingPrices?: string[]
    /**
     * Answers each connect status read after a connect is started, in turn, the
     * last one repeating. Idle before one is started, and idle throughout otherwise.
     */
    connectStatuses?: PlatformConnectStatus[]
  } = {},
) {
  const granted = opts.granted ?? true
  const providers = [...(opts.providers ?? [hostedProvider()])]
  const models = opts.models ?? [hostedModel()]
  const sweep = opts.sweep ?? EMPTY_SWEEP
  const previews = [...(opts.previews ?? [])]
  const connectStatuses = [...(opts.connectStatuses ?? [{ status: "idle" }])]
  let connectStarted = false
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
    if (url.includes(`${API_ROOT}/platform/connect`)) {
      if (method === "POST") {
        connectStarted = true
        return jsonResponse({
          user_code: "WXYZ-1234",
          verification_uri_complete: "https://otari.test/connect",
          expires_in: 600,
        })
      }
      if (!connectStarted) return jsonResponse({ status: "idle" })
      return jsonResponse(
        connectStatuses.length > 1
          ? connectStatuses.shift()
          : connectStatuses[0],
      )
    }
    if (url.includes(`${hosted}/`) && url.includes("/vendors")) {
      const sent = body as { vendor: string; enabled: boolean }
      return jsonResponse(
        vendorFacets(models, opts.vendorsOff ?? []).find(
          (vendor) => vendor.vendor === sent.vendor,
        ) ?? {},
      )
    }
    if (url.includes("/models/switch")) {
      return jsonResponse({
        changed: 2,
        skipped_unpriced: 1,
        skipped_unavailable: 0,
      })
    }
    if (url.includes("/models/reset-prices")) {
      const sent = body as { dry_run: boolean; models?: string[] | null }
      const reset = opts.priceReset ?? { changes: [], skipped_no_default: [] }
      // An apply answers for the models it was sent, as the server does.
      const lost = sent.dry_run ? [] : (opts.lostDefaults ?? [])
      const changes = reset.changes.filter(
        (change) =>
          (sent.dry_run ||
            !sent.models ||
            sent.models.includes(change.model)) &&
          !lost.includes(change.model),
      )
      return jsonResponse({
        ...reset,
        changes,
        skipped_no_default: [...reset.skipped_no_default, ...lost],
        applied: !sent.dry_run && changes.length > 0,
      })
    }
    if (url.includes("/models")) {
      if (method === "PATCH") return jsonResponse({ ...models[0], ...body })
      const typeCounts: Record<string, number> = {}
      for (const model of models) {
        const kind = model.type ?? "unknown"
        typeCounts[kind] = (typeCounts[kind] ?? 0) + 1
      }
      const params = new URL(url, "http://x").searchParams
      const kind = params.get("type")
      const search = params.get("search")?.toLowerCase()
      const enabled = params.get("enabled")
      const shown = models.filter(
        (model) =>
          (!kind || (model.type ?? "unknown") === kind) &&
          (!search || model.model.toLowerCase().includes(search)) &&
          (enabled === null || String(model.enabled) === enabled),
      )
      return jsonResponse({
        data: shown,
        count: opts.modelCount ?? shown.length,
        type_counts: typeCounts,
        vendors: vendorFacets(models, opts.vendorsOff ?? []),
      })
    }
    if (url.startsWith(`${API_ROOT}/pricing/`) && method === "DELETE") {
      const key = decodeURIComponent(
        url.slice(`${API_ROOT}/pricing/`.length).split("?")[0],
      )
      return (opts.missingPrices ?? []).includes(key)
        ? jsonResponse({ detail: "Not Found" }, 404)
        : new Response(null, { status: 204 })
    }
    const regionWrite = url.match(
      /\/hosted-providers\/([^/]+)\/regions\/([^/?]+)/,
    )
    if (regionWrite && method === "PATCH") {
      // A move is applied and kept, as the server does, so the next read of
      // the list shows the new order.
      const [, name, slug] = regionWrite.map(decodeURIComponent)
      const index = providers.findIndex((row) => row.name === name)
      const regions = [...(providers[index]?.regions ?? [])].sort(
        (a, b) => a.position - b.position,
      )
      const sent = body as { position?: number }
      const from = regions.findIndex((region) => region.slug === slug)
      if (index >= 0 && from >= 0 && sent.position !== undefined) {
        const [moved] = regions.splice(from, 1)
        regions.splice(sent.position, 0, moved)
        providers[index] = {
          ...providers[index],
          regions: regions.map((region, position) => ({
            ...region,
            position,
          })),
        }
      }
      return jsonResponse(providers[index] ?? {})
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
        return jsonResponse(
          { detail: opts.patchRefusal },
          opts.patchRefusalStatus ?? 400,
        )
      }
      const sent = body as { provider?: string; name?: string }
      // A create answers the provider it stored, named as the API names it.
      const base =
        providers[0] ??
        hostedProvider({
          provider: sent.provider,
          name: sent.name ?? sent.provider,
        })
      return jsonResponse({ ...base, ...body }, 201)
    }
    return jsonResponse({})
  })

  return requests
}

/** Mount one hosted providers page at `url`, on a fresh query client. */
export function renderHosted(
  ui: ReactElement,
  url: string,
  deployment: Partial<DeploymentBootstrap> = {},
  routes: TestRouterOptions["routes"] = [],
  /** Fills the cache before the first render, as a page visited earlier would. */
  seed: (client: QueryClient) => void = () => {},
) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  seed(client)
  return renderWithRouter(
    <DeploymentProvider value={bootstrap(deployment)}>
      <QueryClientProvider client={client}>{ui}</QueryClientProvider>
    </DeploymentProvider>,
    { url, routes },
  )
}

export const hostedRequests = (requests: Request[], method?: string) =>
  requests.filter(
    (request) =>
      request.url.includes("/hosted-providers") &&
      (method === undefined || request.method === method),
  )
