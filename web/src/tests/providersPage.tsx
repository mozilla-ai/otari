/**
 * The ProvidersPage harness: fixtures, the transport mock, and the render.
 *
 * Named for the page rather than the domain because `src/tests/providers.tsx`
 * is already the app's provider tree.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render } from "@testing-library/react"
import type { ReactElement } from "react"
import { vi } from "vitest"

import type {
  GatewaySettings,
  KnownProvider,
  OrganizationContext,
  ProviderHealth,
  ProviderHealthResponse,
  ProviderInfo,
  StoredProvider,
  TestProviderResult,
} from "@/client"
import { API_ROOT } from "@/shared/api/client"
import { organizationContext } from "@/tests/fixtures"
import { withRouter } from "@/tests/router"

export const CAPS = {
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

export function providerInfo(
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

export function storedProvider(
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

export const SETTINGS: GatewaySettings = {
  mode: "standalone",
  version: "1.0.0",
  model_discovery: true,
  default_pricing: true,
  require_pricing: false,
  master_key_source: "configured",
  secret_key_configured: true,
  config: [],
}

export function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

// Build a health response, defaulting every provider in `meta` to reachable so
// tests that don't care about health still get a well-formed payload.
export function healthResponse(
  providers: ProviderHealth[],
): ProviderHealthResponse {
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

export interface MockOpts {
  meta?: ProviderInfo[]
  stored?: StoredProvider[]
  settings?: GatewaySettings
  testResult?: TestProviderResult
  catalog?: KnownProvider[]
  // Per-provider health; defaults to every `meta` provider reachable. `healthRefresh`
  // is served for the forced-refresh (refresh=true) request, if given.
  health?: ProviderHealth[]
  healthRefresh?: ProviderHealth[]
  // The caller's membership context, which is where the page reads whether the
  // deployment can encrypt a provider credential.
  context?: OrganizationContext
  // When set, GET /v1/organizations/me blocks on this promise before responding,
  // so a test can resolve it to simulate the context landing after the page has
  // already painted (and the operator has interacted with it).
  contextGate?: Promise<unknown>
  // Force GET /v1/organizations/me to fail, so the fail-closed gate can be
  // exercised.
  contextError?: boolean
  // Refuse GET /v1/settings the way the operator-only gate does for a caller who
  // is neither a superuser nor holding the master key.
  settingsRefused?: boolean
  // When set, POST .../test blocks on this promise, so a test can hold a
  // connection test in flight while the page is used.
  testGate?: Promise<unknown>
  // Scripts successive POST .../test calls, so a test can hold the first one in
  // flight and let a later one answer first. Falls back to testGate/testResult
  // once the script runs out.
  testCalls?: { gate?: Promise<unknown>; result: TestProviderResult }[]
  // When set, the create POST blocks on this promise, so a test can hold a
  // create in flight and read the submit's state while it runs.
  createGate?: Promise<unknown>
  // When set, the per-provider detail GET blocks on this promise, so a test can
  // use the form during the window between choosing a provider and its hints
  // landing.
  detailGate?: Promise<unknown>
}

export function mockApi(opts: MockOpts = {}) {
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

      if (url.includes(`${API_ROOT}/provider-credentials`)) {
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
          if (opts.createGate) await opts.createGate
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
      if (url.includes(`${API_ROOT}/providers/catalog/`)) {
        // Detail endpoint: autofill hints for one selected provider.
        const id = decodeURIComponent(
          url.split(`${API_ROOT}/providers/catalog/`)[1].split("?")[0],
        )
        if (opts.detailGate) await opts.detailGate
        const detail = catalog.find((p) => p.id === id)
        return detail
          ? jsonResponse(detail)
          : jsonResponse({ detail: `Unknown provider: ${id}` }, 404)
      }
      if (url.includes(`${API_ROOT}/providers/catalog`)) {
        // List endpoint: id + display name only.
        return jsonResponse(catalog.map((p) => ({ id: p.id, name: p.name })))
      }
      if (url.includes(`${API_ROOT}/providers/health`)) {
        return jsonResponse(
          healthResponse(url.includes("refresh=true") ? healthRefresh : health),
        )
      }
      if (url.includes(`${API_ROOT}/providers`)) {
        return jsonResponse({ providers: meta })
      }
      if (url.includes(`${API_ROOT}/settings`)) {
        if (opts.settingsRefused) {
          return jsonResponse({ detail: "Not authorized" }, 403)
        }
        if (method === "PATCH") {
          settings = { ...settings, ...JSON.parse(String(init?.body)) }
        }
        return jsonResponse(settings)
      }
      if (url.includes(`${API_ROOT}/organizations/me`)) {
        if (opts.contextGate) await opts.contextGate
        if (opts.contextError) return jsonResponse({ detail: "boom" }, 500)
        return jsonResponse(opts.context ?? organizationContext())
      }
      return jsonResponse([])
    })
}

export function renderPage(
  ui: ReactElement,
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } }),
) {
  return render(
    <QueryClientProvider client={client}>{ui}</QueryClientProvider>,
    { wrapper: withRouter() },
  )
}

export function healthRequestCount(
  fetchMock: ReturnType<typeof mockApi>,
): number {
  return fetchMock.mock.calls.filter(([url]) =>
    String(url).includes(`${API_ROOT}/providers/health`),
  ).length
}
