import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { useLocation } from "@tanstack/react-router"
import { render } from "@testing-library/react"
import type { ReactElement } from "react"
import { vi } from "vitest"

import type { UsageSummary } from "@/client"
import { API_ROOT } from "@/shared/api/client"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  bootstrap,
  organizationContext,
  seriesPoint,
  usageTotals,
} from "@/tests/fixtures"
import { withRouter } from "@/tests/router"

export function summary(overrides: Partial<UsageSummary> = {}): UsageSummary {
  return {
    start_date: "2026-06-21T00:00:00Z",
    end_date: "2026-07-21T00:00:00Z",
    bucket: "day",
    totals: usageTotals({
      cost: 1240.5,
      prompt_tokens: 8_000_000,
      completion_tokens: 4_400_000,
      total_tokens: 12_400_000,
      // The tokens tile reads the billed total, not total_tokens.
      billed_input_tokens: 8_000_000,
      billed_output_tokens: 4_400_000,
      request_count: 84_000,
      error_count: 1_764,
      avg_latency_ms: 820,
    }),
    by_model: [
      {
        key: "gpt-5.6",
        cost: 820,
        tokens: 8_000_000,
        requests: 42_000,
        is_other: false,
      },
      {
        key: "claude-sonnet-5",
        cost: 310,
        tokens: 3_000_000,
        requests: 28_000,
        is_other: false,
      },
      {
        key: null,
        cost: 110.5,
        tokens: 1_400_000,
        requests: 14_000,
        is_other: true,
      },
    ],
    by_user: [
      {
        key: "alice",
        label: "Alice",
        cost: 900.5,
        tokens: 8_000_000,
        requests: 50_000,
        is_other: false,
      },
      {
        key: "bob",
        label: "Bob",
        cost: 340,
        tokens: 4_400_000,
        requests: 34_000,
        is_other: false,
      },
    ],
    // `label` is the server-resolved key name; the picker reads it from here
    // rather than from a full /v1/keys listing.
    by_api_key: [
      {
        key: "key-1",
        label: "ci-bot",
        cost: 500,
        tokens: 5_000_000,
        requests: 30_000,
        is_other: false,
      },
    ],
    by_source: [
      {
        key: "gateway",
        cost: 1_000,
        tokens: 9_000_000,
        requests: 60_100,
        is_other: false,
      },
      {
        key: "claude_code",
        cost: 240.5,
        tokens: 3_400_000,
        requests: 23_900,
        is_other: false,
      },
    ],
    by_source_label: [
      {
        key: "project:otari",
        cost: 700,
        tokens: 6_000_000,
        requests: 30_100,
        is_other: false,
      },
      {
        key: "project:docs",
        cost: 200,
        tokens: 2_000_000,
        requests: 9_200,
        is_other: false,
      },
      // Gateway traffic carries no session label: a real group with a null key,
      // not the synthesized fold.
      {
        key: null,
        cost: 340.5,
        tokens: 4_400_000,
        requests: 44_700,
        is_other: false,
      },
    ],
    by_endpoint: [
      {
        key: "/v1/chat/completions",
        cost: 900,
        tokens: 8_000_000,
        requests: 50_100,
        is_other: false,
      },
      {
        key: "/v1/messages",
        cost: 340.5,
        tokens: 4_400_000,
        requests: 33_900,
        is_other: false,
      },
    ],
    by_provider: [
      {
        key: "openai",
        cost: 880,
        tokens: 7_000_000,
        requests: 45_100,
        is_other: false,
      },
      {
        key: "anthropic",
        cost: 360.5,
        tokens: 5_400_000,
        requests: 38_900,
        is_other: false,
      },
    ],
    by_tool: [],
    errors_by_status_code: [],
    series: [
      seriesPoint({
        bucket_start: "2026-07-19T00:00:00Z",
        cost: 400,
        tokens: 4_000_000,
        requests: 28_000,
      }),
      seriesPoint({
        bucket_start: "2026-07-20T00:00:00Z",
        cost: 840.5,
        tokens: 8_400_000,
        requests: 56_000,
      }),
    ],
    ...overrides,
  }
}

export function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

// `extra` adds a path the default stub does not answer; the scope cases need
// /v1/organizations/me, which is where the workspace switcher's memberships come
// from and therefore where a selection has to come from too.
export function mockApi(
  body: UsageSummary | null,
  extra: Record<string, unknown> = {},
  // The previous window. Without one, that query gets the current window's body
  // back, every delta is exactly zero, and a test cannot see a direction or a
  // polarity color.
  previousBody?: UsageSummary,
) {
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input)
    for (const [path, answer] of Object.entries(extra)) {
      // Matched at a path boundary rather than anywhere in the URL. /api/v1/usage
      // and /api/v1/organizations/me are both prefixes of routes this page reads
      // (/api/v1/organizations/me/usage/summary is the tenant's own), so a bare
      // `includes` would answer a summary request with an organization context.
      if (url === path || url.endsWith(path) || url.includes(`${path}?`)) {
        return jsonResponse(answer)
      }
    }
    // The shell reads this before it paints, and the usage hooks wait for it: it
    // is what tells them whether this caller reads the deployment-wide routes or
    // the organization-scoped ones (otari#837). After `extra`, so a test that
    // supplies its own context still wins, and on an exact match so it cannot
    // shadow /v1/organizations/me/usage.
    // The roster the breakdowns name people from. Answered by default and
    // empty, which is the deployment nobody has invited anyone to: the rows
    // then read the alias the summary already carries.
    if (url.includes(`${API_ROOT}/organizations/me/members`)) {
      return jsonResponse({ data: [], total: 0 })
    }
    if (url.endsWith(`${API_ROOT}/organizations/me`)) {
      return jsonResponse(organizationContext())
    }
    if (url.includes("/usage/summary")) {
      // The previous-period query is the only one on this page passing
      // NO_BREAKDOWNS, which goes on the wire as the server's `none` sentinel,
      // so that is what tells the two windows apart here.
      if (previousBody && url.includes("dimensions=none")) {
        return jsonResponse(previousBody)
      }
      return jsonResponse(body ?? summary())
    }
    if (url.includes("/usage/series")) {
      return jsonResponse({
        start_date: "2026-06-21T00:00:00Z",
        end_date: "2026-07-21T00:00:00Z",
        bucket: "day",
        group_by: "model",
        groups: [
          {
            key: "gpt-5.6",
            cost: 820,
            tokens: 8_000_000,
            requests: 42_000,
            is_other: false,
          },
          {
            key: null,
            cost: 420.5,
            tokens: 4_400_000,
            requests: 42_000,
            is_other: true,
          },
        ],
        points: [
          {
            bucket_start: "2026-07-19T00:00:00Z",
            key: "gpt-5.6",
            is_other: false,
            cost: 400,
            tokens: 4_000_000,
            requests: 28_000,
          },
          {
            bucket_start: "2026-07-20T00:00:00Z",
            key: null,
            is_other: true,
            cost: 420.5,
            tokens: 4_400_000,
            requests: 42_000,
          },
        ],
      })
    }
    if (url.includes(`${API_ROOT}/users`)) {
      return jsonResponse([
        { user_id: "alice", alias: "Alice" },
        { user_id: "bob", alias: "Bob" },
      ])
    }
    if (url.includes(`${API_ROOT}/keys`)) {
      return jsonResponse([
        {
          id: "key-1",
          key_name: "ci-bot",
          user_id: "alice",
          allowed_models: null,
        },
      ])
    }
    return jsonResponse([])
  })
}

// Surfaces the current location so a drill-down navigation can be asserted.
export function LocationProbe() {
  const loc = useLocation()
  // A status role with an accessible name so tests query the probe by role
  // rather than a test id.
  return (
    <div
      role="status"
      aria-label="Current location"
    >{`${loc.pathname}${loc.searchStr}`}</div>
  )
}

export function renderPage(
  ui: ReactElement,
  options: { scoped?: boolean } = {},
) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  // Most cases here are about the charts and read no workspace, so the provider
  // is opt-in: outside it `useSelectedWorkspace` answers with nothing, which is
  // the same shape as a caller in no workspace and keeps those requests
  // unscoped either way. `scoped` is for the case that is *about* the scope.
  const body = options.scoped ? (
    <SelectedWorkspaceProvider>{ui}</SelectedWorkspaceProvider>
  ) : (
    ui
  )
  // The breakdowns and the chart legend ask the organization roster what to call
  // each person, and that read is gated on the `organizations` surface, so the
  // page needs the deployment context the shell always gives it.
  return render(
    <DeploymentProvider value={bootstrap()}>
      <QueryClientProvider client={client}>{body}</QueryClientProvider>
    </DeploymentProvider>,
    {
      wrapper: withRouter({
        url: "/usage",
        routes: [{ path: "/activity", element: <LocationProbe /> }],
      }),
    },
  )
}
