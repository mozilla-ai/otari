import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render } from "@testing-library/react"
import type { ReactElement } from "react"
import { vi } from "vitest"

import type { InFlightResponse, UsageEntry } from "@/client"
import { API_ROOT } from "@/shared/api/client"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { withRouter } from "@/tests/router"

export function entry(overrides: Partial<UsageEntry> = {}): UsageEntry {
  const row = {
    id: "req-1",
    user_id: "alice",
    api_key_id: "key-1",
    timestamp: new Date().toISOString(),
    model: "gpt-4o",
    provider: "openai",
    endpoint: "/v1/chat/completions",
    prompt_tokens: 1200,
    completion_tokens: 300,
    total_tokens: 1500,
    cache_read_tokens: null,
    cache_write_tokens: null,
    cache_write_1h_tokens: null,
    billing_meters: null,
    pricing_breakdown: null,
    cost: 0.0123,
    status: "success",
    error_message: null,
    status_code: null,
    latency_ms: 842,
    source: "gateway",
    source_label: null,
    counts_toward_budget: true,
    ...overrides,
  }
  return {
    ...row,
    // Mirror the server's derivation unless a test explicitly overrides it.
    bulk_editable:
      overrides.bulk_editable ??
      (!row.counts_toward_budget && row.source !== "gateway"),
  }
}

export function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

// Usage hooks wait for organization context before choosing their route scope.
export function operatorContext(): Response {
  return jsonResponse({
    organization_member_id: "om-1",
    role: "owner",
    status: "active",
    organization: {
      id: "org-1",
      name: "Acme",
      slug: "acme",
      created_by_user_id: null,
      created_at: new Date().toISOString(),
      updated_at: null,
    },
    byo_provider_keys_allowed: true,
    deployment_operator: true,
    provider_key_encryption_available: true,
    workspace_memberships: [],
  })
}

interface FetchCall {
  url: string
  method: string
  body: string | undefined
}

export function mockApi(
  opts: {
    rows?: UsageEntry[]
    // Thunks let polling tests change the response after rendering.
    total?: number | (() => number)
    groupRows?: UsageEntry[]
    users?: string[]
    inFlight?: InFlightResponse | (() => InFlightResponse)
    workspace?: string
    deploymentOperator?: boolean
  } = {},
) {
  const rows = opts.rows ?? []
  const total = () => {
    const t = opts.total ?? rows.length
    return typeof t === "function" ? t() : t
  }
  const inFlight = () => {
    const f = opts.inFlight ?? { requests: [], total: 0 }
    return typeof f === "function" ? f() : f
  }
  const calls: FetchCall[] = []

  const mock = vi
    .spyOn(globalThis, "fetch")
    .mockImplementation(async (input, init) => {
      const url = String(input)
      const method = (init?.method ?? "GET").toUpperCase()
      calls.push({
        url,
        method,
        body: typeof init?.body === "string" ? init.body : undefined,
      })

      if (url.endsWith(`${API_ROOT}/usage`) && method === "DELETE") {
        return jsonResponse({ deleted: 1 })
      }
      if (url.includes(`${API_ROOT}/usage/set-price`)) {
        return jsonResponse({ matched: 1, updated: 1, unchanged: 0 })
      }
      // Reads serve both deployment and organization scopes; writes stay global.
      if (url.includes("/usage/count")) {
        return jsonResponse({ total: total() })
      }
      if (url.includes("/usage/in-flight")) {
        return jsonResponse(inFlight())
      }
      if (url.includes("/usage/summary")) {
        const models = Array.from(new Set(rows.map((r) => r.model)))
        return jsonResponse({
          start_date: "",
          end_date: "",
          bucket: "day",
          totals: {
            cost: 0,
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            request_count: 0,
            error_count: 0,
            avg_latency_ms: null,
          },
          by_model: models.map((m) => ({
            key: m,
            cost: 0,
            tokens: 0,
            requests: 0,
            is_other: false,
          })),
          // Label-free breakdowns keep picker options and chip labels as bare IDs.
          by_user: (opts.users ?? ["alice", "bob"]).map((u) => ({
            key: u,
            cost: 0,
            tokens: 0,
            requests: 0,
            is_other: false,
          })),
          by_api_key: [],
          by_source: Array.from(new Set(rows.map((r) => r.source))).map(
            (s) => ({
              key: s,
              cost: 0,
              tokens: 0,
              requests: 0,
              is_other: false,
            }),
          ),
          series: [],
        })
      }
      if (url.includes("/usage")) {
        // Group lookups can return siblings absent from the page's own rows.
        const asked = new URL(url, "http://localhost").searchParams.getAll(
          "request_group_id",
        )
        if (asked.length) {
          const pool = opts.groupRows ?? rows
          return jsonResponse(
            pool.filter(
              (r) => r.request_group_id && asked.includes(r.request_group_id),
            ),
          )
        }
        return jsonResponse(rows)
      }
      if (url.endsWith(`${API_ROOT}/organizations/me`)) {
        return jsonResponse({
          organization_member_id: "om-1",
          role: "owner",
          status: "active",
          organization: {
            id: "org-1",
            name: "Acme",
            slug: "acme",
            created_by_user_id: null,
            created_at: new Date().toISOString(),
            updated_at: null,
          },
          byo_provider_keys_allowed: true,
          deployment_operator: opts.deploymentOperator ?? true,
          provider_key_encryption_available: true,
          workspace_memberships: opts.workspace
            ? [
                {
                  workspace_id: opts.workspace,
                  workspace_name: "Production",
                  role: "owner",
                  status: "active",
                },
              ]
            : [],
        })
      }
      return jsonResponse([])
    })

  return { mock, calls }
}

export function renderPage(ui: ReactElement, route = "/activity") {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <SelectedWorkspaceProvider>{ui}</SelectedWorkspaceProvider>
    </QueryClientProvider>,
    {
      wrapper: withRouter({ url: route }),
    },
  )
}

// Exclude supporting reads and mutations from pagination/filter assertions.
export function listCalls(calls: FetchCall[]): string[] {
  return calls
    .filter(
      (c) =>
        c.method === "GET" &&
        c.url.includes(`${API_ROOT}/usage`) &&
        !c.url.includes("/count") &&
        !c.url.includes("/summary") &&
        !c.url.includes("/in-flight") &&
        !c.url.includes("/set-price"),
    )
    .map((c) => c.url)
}

export function countCalls(calls: FetchCall[]): string[] {
  return calls
    .filter(
      (c) => c.method === "GET" && c.url.includes(`${API_ROOT}/usage/count`),
    )
    .map((c) => c.url)
}
