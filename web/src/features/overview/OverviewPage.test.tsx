import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { useLocation } from "@tanstack/react-router"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactElement } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { DeploymentBootstrap, UsageSummary } from "@/client"
import {
  localDayKey,
  OverviewIndex,
  OverviewPage,
} from "@/features/overview/OverviewPage"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  bootstrap,
  HOSTED_SURFACES,
  organizationContext,
  usageTotals,
} from "@/tests/fixtures"
import { withRouter } from "@/tests/router"

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

function summary(totals: Partial<UsageSummary["totals"]>): UsageSummary {
  return {
    start_date: "2026-06-22T00:00:00Z",
    end_date: "2026-07-22T00:00:00Z",
    bucket: "day",
    totals: usageTotals({
      cost: 0,
      prompt_tokens: 0,
      completion_tokens: 0,
      total_tokens: 0,
      cache_read_tokens: 0,
      cache_write_tokens: 0,
      request_count: 0,
      error_count: 0,
      avg_latency_ms: null,
      ...totals,
    }),
    by_model: [],
    by_user: [],
    by_api_key: [],
    by_source: [],
    by_source_label: [],
    by_endpoint: [],
    by_provider: [],
    by_tool: [],
    errors_by_status_code: [],
    series: [],
  }
}

interface Bodies {
  today?: Partial<UsageSummary["totals"]>
  period?: Partial<UsageSummary["totals"]>
  prev?: Partial<UsageSummary["totals"]>
  health?: unknown
  budgets?: unknown
  keys?: unknown
  users?: unknown
  logs?: unknown
  providers?: unknown
}

// Order matters: /v1/usage/summary is matched BEFORE the bare /v1/usage logs
// endpoint, and /v1/providers/health returns an OBJECT (not the [] fallback) so
// the health/status logic reads real counts. The summary mock is param-aware so
// the Today (bucket=hour) and Last-30d (bucket=day) tiles render distinct values.
function mockApi(b: Bodies) {
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input)
    // The shell reads this before it paints, and the usage hooks wait for it:
    // it is what tells them whether this caller reads the deployment-wide
    // routes or the organization-scoped ones (otari#837). Answered first, and
    // on an exact match, so it cannot shadow /v1/organizations/me/usage.
    if (url.endsWith("/v1/organizations/me")) {
      return jsonResponse(organizationContext())
    }
    if (url.includes("/v1/usage/summary")) {
      if (url.includes("bucket=hour"))
        return jsonResponse(summary(b.today ?? {}))
      if (url.includes("end_date=")) return jsonResponse(summary(b.prev ?? {}))
      return jsonResponse(summary(b.period ?? {}))
    }
    if (url.includes("/v1/providers/health")) {
      return jsonResponse(
        b.health ?? { providers: [], healthy: 0, total: 0, checked_at: null },
      )
    }
    if (url.includes("/v1/budgets")) return jsonResponse(b.budgets ?? [])
    if (url.includes("/v1/keys")) return jsonResponse(b.keys ?? [])
    if (url.includes("/v1/users")) return jsonResponse(b.users ?? [])
    if (url.includes("/v1/providers"))
      return jsonResponse({
        providers: b.providers ?? [{ provider: "openai" }],
      })
    if (url.includes("/v1/usage")) return jsonResponse(b.logs ?? [])
    return jsonResponse([])
  })
}

function LocationProbe() {
  const loc = useLocation()
  return <div data-testid="loc">{loc.pathname}</div>
}

function renderPage(
  ui: ReactElement,
  initial = "/overview",
  deployment: DeploymentBootstrap = bootstrap(),
) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    // The page composes the setup guide, which is gated on the deployment's
    // surfaces, so the bootstrap has to be in the tree even for the tests that
    // are about a tile. The guide itself renders nothing here: with no
    // workspace selected there is nothing for it to be about.
    <QueryClientProvider client={client}>
      <DeploymentProvider value={deployment}>{ui}</DeploymentProvider>
    </QueryClientProvider>,
    {
      wrapper: withRouter({
        url: initial,
        routes: [
          { path: "/providers", element: <LocationProbe /> },
          { path: "/organization/provider-keys", element: <LocationProbe /> },
        ],
      }),
    },
  )
}

describe("OverviewPage", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    vi.useRealTimers()
  })

  it("uses a zero-padded, one-based local calendar date as its refresh key", () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date(2026, 0, 5, 12))

    expect(localDayKey()).toBe("2026-01-05")
  })

  it("renders distinct today vs 30-day spend, request volume, and error rate", async () => {
    mockApi({
      today: { cost: 5, request_count: 100, error_count: 1 },
      period: { cost: 200, request_count: 2000, error_count: 40 }, // 2% -> warn
      prev: { cost: 100, request_count: 1000, error_count: 5 },
    })
    renderPage(<OverviewPage />)

    expect(await screen.findByText("$5.00")).toBeInTheDocument()
    expect(await screen.findByText("$200.00")).toBeInTheDocument()
    expect(screen.getByText("2,000")).toBeInTheDocument()
    expect(screen.getByText("2.0%")).toBeInTheDocument()
    expect(screen.getByText("Elevated")).toBeInTheDocument() // error-rate status word (non-hue)
  })

  it("renders period-over-period change as a trend chip, not a glyph", async () => {
    mockApi({
      today: { cost: 5, request_count: 100, error_count: 1 },
      period: { cost: 200, request_count: 2000, error_count: 40 },
      prev: { cost: 100, request_count: 1600, error_count: 8 },
    })
    renderPage(<OverviewPage />)
    // Await a chip, not "$200.00": every assertion below is about text derived
    // from the *previous* window, while the value comes from the period query.
    // Awaiting the value assumes the two land in one commit, so a skew between
    // the responses would fail the sync assertions that follow.
    await screen.findByText("+100.0% vs prev")

    // The chip carries the caption the old plain-text hint carried, so the
    // comparison still says what it is being compared against. Distinct
    // percentages per tile, so each assertion is about one chip: spend doubled,
    // requests rose a quarter, and the error rate went 0.5% -> 2.0%.
    expect(screen.getByText("+100.0% vs prev")).toBeInTheDocument()
    expect(screen.getByText("+25.0% vs prev")).toBeInTheDocument()
    expect(screen.getByText("+300.0% vs prev")).toBeInTheDocument()
    // And each announces a direction, which the glyph never did: "▲" is
    // decoration a screen reader skips. TrendChip.test.tsx owns the direction
    // and polarity mapping; this only asserts the three tiles go through it.
    // Anchored and counted: unanchored, `up` and `down` also match this page's
    // own prose, so the assertion would pass with the announcement deleted.
    expect(
      screen.getAllByText(/^(no change|up|down)(, (better|worse))?$/),
    ).toHaveLength(3)
    // The hand-rolled arrow glyphs are gone from the tiles. queryAllByText for
    // the same reason as below: queryByText throws on a multiple match, so a
    // regression restoring the glyph on all three tiles would report a broken
    // test rather than a broken tile.
    expect(screen.queryAllByText(/[▲▼]/)).toHaveLength(0)
  })

  it("reads a chip against the metric's own polarity, not the direction alone", async () => {
    // Spend and the error rate both rose, and both improve by falling, so both
    // are regressions and say so: direction plus judgment, because polarity puts
    // good and bad in hue alone. Request volume rose too, but volume carries no
    // polarity, so it announces the direction and nothing more.
    mockApi({
      today: { cost: 5, request_count: 100, error_count: 1 },
      period: { cost: 200, request_count: 2000, error_count: 40 },
      prev: { cost: 100, request_count: 1600, error_count: 8 },
    })
    renderPage(<OverviewPage />)

    // The chip again, not the value: same previous-window dependency as above.
    expect(await screen.findByText("+100.0% vs prev")).toBeInTheDocument()
    expect(screen.getAllByText("up, worse")).toHaveLength(2)
    expect(screen.getAllByText("up")).toHaveLength(1)
  })

  it("reserves no trend row for a tile with no comparable previous window", async () => {
    // No previous window on the wire leaves every delta null, and TrendChip
    // renders nothing for a null fraction. The chip has to be gated on the
    // fraction rather than on the query, or StatCard reserves the aside row for
    // an element that draws nothing.
    mockApi({
      today: { cost: 5 },
      period: { cost: 200, request_count: 2000, error_count: 40 },
      prev: { cost: 0, request_count: 0, error_count: 0 },
    })
    renderPage(<OverviewPage />)
    await screen.findByText("$200.00")

    // queryAllByText, not queryByText: a regression that renders all three
    // chips makes queryByText throw on the multiple match rather than fail on
    // the assertion, which reads as a broken test instead of a broken tile.
    expect(screen.queryAllByText(/vs prev/)).toHaveLength(0)
    expect(
      screen.queryAllByText(/^(no change|up|down)(, (better|worse))?$/),
    ).toHaveLength(0)
    // And the row itself is not reserved, which is the half the assertions
    // above cannot see: TrendChip renders no text for a null fraction either
    // way, so gating the chip on `periodTotals` instead of on the fraction
    // leaves them green while the tile keeps 42px of dead space. Asserted on
    // the utility for the reason ui.test.tsx does it: jsdom performs no layout,
    // so the reservation is observable only as the class. Scoped to this tile,
    // since Budget health reserves the row off its own hint.
    const tile = screen.getByText("Spend, last 30 days").parentElement!
    expect(tile.querySelector(".min-h-10\\.5")).toBeNull()
  })

  it("renders spend and request-volume sparklines from the 30-day series", async () => {
    const series = [
      {
        bucket_start: "2026-07-20T00:00:00Z",
        cost: 10,
        tokens: 1000,
        requests: 100,
      },
      {
        bucket_start: "2026-07-21T00:00:00Z",
        cost: 20,
        tokens: 2000,
        requests: 150,
      },
      {
        bucket_start: "2026-07-22T00:00:00Z",
        cost: 15,
        tokens: 1500,
        requests: 120,
      },
    ]
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input)
      // The shell reads this before it paints, and the usage hooks wait for it:
      // it is what tells them whether this caller reads the deployment-wide
      // routes or the organization-scoped ones (otari#837). Answered first, and
      // on an exact match, so it cannot shadow /v1/organizations/me/usage.
      if (url.endsWith("/v1/organizations/me")) {
        return jsonResponse(organizationContext())
      }
      if (url.includes("/v1/usage/summary")) {
        if (url.includes("bucket=hour"))
          return jsonResponse(summary({ cost: 5 }))
        if (url.includes("end_date="))
          return jsonResponse(summary({ cost: 100 }))
        // The 30-day (day-bucket, unbounded) query carries the series the tiles chart.
        return jsonResponse({
          ...summary({ cost: 200, request_count: 2000 }),
          series,
        })
      }
      if (url.includes("/v1/providers/health")) {
        return jsonResponse({
          providers: [],
          healthy: 1,
          total: 1,
          checked_at: null,
        })
      }
      return jsonResponse([])
    })
    renderPage(<OverviewPage />)

    expect(
      await screen.findByRole("img", {
        name: "Spend trend over the last 30 days",
      }),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("img", {
        name: "Request volume trend over the last 30 days",
      }),
    ).toBeInTheDocument()
  })

  it("shows a dash for error rate when there are no requests", async () => {
    mockApi({ period: { request_count: 0, error_count: 0 } })
    renderPage(<OverviewPage />)
    // Scope to the error-rate tile: its value is a dash (not a percentage) and it
    // carries no status word when the rate is neutral.
    const label = await screen.findByText("Error rate, last 30 days")
    const tile = label.closest("div")!
    expect(within(tile).getByText("—")).toBeInTheDocument()
    expect(within(tile).queryByText("Elevated")).not.toBeInTheDocument()
    expect(within(tile).queryByText(/%/)).not.toBeInTheDocument()
  })

  it("computes budget health with cap * user_count and links to budgets", async () => {
    mockApi({
      budgets: [
        {
          budget_id: "team",
          name: "team",
          max_budget: 10,
          user_count: 2,
          total_spend: 25,
          total_reserved: 0,
        },
        {
          budget_id: "x",
          name: "x",
          max_budget: null,
          user_count: 1,
          total_spend: 9999,
          total_reserved: 0,
        },
      ],
    })
    renderPage(<OverviewPage />)
    expect(await screen.findByText("125.0%")).toBeInTheDocument() // 25 / (10*2)
    expect(screen.getByText("Over budget")).toBeInTheDocument()
  })

  it("summarizes provider health and surfaces problems in the status strip", async () => {
    mockApi({
      health: {
        providers: [],
        healthy: 2,
        total: 3,
        checked_at: "2026-07-22T00:00:00Z",
      },
      budgets: [
        {
          budget_id: "team",
          name: "team",
          max_budget: 10,
          user_count: 2,
          total_spend: 25,
          total_reserved: 0,
        },
      ],
    })
    renderPage(<OverviewPage />)
    // Provider health has no tile of its own; a degraded state surfaces only via
    // the attention strip, each problem a link.
    expect(
      await screen.findByText("1 provider unreachable"),
    ).toBeInTheDocument()
    expect(screen.getByText("1 budget over limit")).toBeInTheDocument()
  })

  it("links each attention problem at the view that explains it", async () => {
    mockApi({
      // 20% errors, past the alert threshold, so the error-rate problem appears
      // alongside the provider one and the strip carries both link shapes.
      period: { request_count: 100, error_count: 20 },
      health: {
        providers: [],
        healthy: 2,
        total: 3,
        checked_at: "2026-07-22T00:00:00Z",
      },
    })
    renderPage(<OverviewPage />)

    // A bare destination, and one that has to carry a filter with it. The second
    // is the one worth pinning: it is the only place the overview hands the
    // activity log a query, so a link that dropped it would land on an unfiltered
    // log and still look like it worked.
    expect(
      await screen.findByRole("link", { name: "1 provider unreachable" }),
    ).toHaveAttribute("href", "/providers")
    expect(screen.getByRole("link", { name: /^error rate/ })).toHaveAttribute(
      "href",
      "/activity?status=error",
    )
  })

  it("states an unreachable provider without a link on a hosted deployment", async () => {
    // Provider health reports on `config.providers`, the process-global table a
    // hosted deployment serves no page for, so both candidate destinations are
    // wrong: `/providers` is answered with "not available here", and the
    // organization page lists a different table the instance is not on.
    mockApi({
      health: {
        providers: [],
        healthy: 2,
        total: 3,
        checked_at: "2026-07-22T00:00:00Z",
      },
    })
    renderPage(
      <OverviewPage />,
      "/overview",
      bootstrap({ deployment_type: "hosted", surfaces: HOSTED_SURFACES }),
    )

    expect(
      await screen.findByText("1 provider unreachable"),
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("link", { name: "1 provider unreachable" }),
    ).toBeNull()
  })

  it("hides the status strip when nothing needs attention", async () => {
    mockApi({
      health: {
        providers: [],
        healthy: 3,
        total: 3,
        checked_at: "2026-07-22T00:00:00Z",
      },
    })
    renderPage(<OverviewPage />)

    // Wait for the tiles to resolve (spend today + last-30d both read $0.00 here),
    // then confirm no neutral status strip renders when every source is healthy.
    expect((await screen.findAllByText("$0.00")).length).toBeGreaterThan(0)
    expect(screen.queryByRole("status")).not.toBeInTheDocument()
  })

  it("renders a recent-activity row with a null-cost entry, then empty state", async () => {
    mockApi({
      logs: [
        {
          id: "1",
          user_id: null,
          api_key_id: null,
          timestamp: "2026-07-22T00:00:00Z",
          model: "gpt-5.6",
          provider: "openai",
          endpoint: "/v1/chat/completions",
          prompt_tokens: 10,
          completion_tokens: 5,
          total_tokens: 15,
          cache_read_tokens: 0,
          cache_write_tokens: 0,
          cost: null,
          status: "error",
          error_message: "boom",
          latency_ms: 120,
        },
      ],
    })
    renderPage(<OverviewPage />)
    expect(await screen.findByText("gpt-5.6")).toBeInTheDocument()
    // null cost renders as an em-dash, not a crash.
    expect(screen.getAllByText("—").length).toBeGreaterThan(0)
  })

  it("keeps the page up when one tile query fails (per-tile isolation)", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input)
      // The shell reads this before it paints, and the usage hooks wait for it:
      // it is what tells them whether this caller reads the deployment-wide
      // routes or the organization-scoped ones (otari#837). Answered first, and
      // on an exact match, so it cannot shadow /v1/organizations/me/usage.
      if (url.endsWith("/v1/organizations/me")) {
        return jsonResponse(organizationContext())
      }
      if (url.includes("/v1/budgets"))
        return jsonResponse({ detail: "boom" }, 500)
      if (url.includes("/v1/usage/summary"))
        return jsonResponse(summary({ cost: 200, request_count: 10 }))
      if (url.includes("/v1/providers/health"))
        return jsonResponse({
          providers: [],
          healthy: 1,
          total: 1,
          checked_at: null,
        })
      return jsonResponse([])
    })
    renderPage(<OverviewPage />)
    // The spend tiles still render even though budgets errored (per-tile isolation);
    // both Today and Last-30d read $200.00 with this mock, hence findAllByText.
    expect((await screen.findAllByText("$200.00")).length).toBeGreaterThan(0)
    // ...and the status strip must NOT claim all-clear while a source query failed
    // (it would contradict the error banner). It reads as a neutral load-failure line.
    expect(screen.queryByText(/All systems normal/)).not.toBeInTheDocument()
    expect(screen.getByText(/could not be loaded/)).toBeInTheDocument()
  })

  it("hides the status strip while status sources are still loading", async () => {
    // A never-resolving fetch keeps the queries pending.
    vi.spyOn(globalThis, "fetch").mockImplementation(
      () => new Promise<Response>(() => {}),
    )
    renderPage(<OverviewPage />)
    expect(await screen.findByText("Overview")).toBeInTheDocument()
    expect(screen.queryByRole("status")).not.toBeInTheDocument()
  })
})

describe("OverviewIndex routing", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("renders the overview when a provider is configured", async () => {
    mockApi({
      health: { providers: [], healthy: 1, total: 1, checked_at: null },
    })
    renderPage(<OverviewIndex />)
    expect(await screen.findByText("Overview")).toBeInTheDocument()
  })

  it("shows a getting-started overview and links to providers on a fresh gateway", async () => {
    mockApi({ providers: [] })
    const user = userEvent.setup()
    renderPage(<OverviewIndex />)

    expect(
      await screen.findByText("Get started with Otari"),
    ).toBeInTheDocument()
    expect(screen.getByText("Overview")).toBeInTheDocument()
    await user.click(
      screen.getByRole("button", { name: "Add your first provider" }),
    )
    expect(await screen.findByTestId("loc")).toHaveTextContent("/providers")
  })

  it("sends a hosted deployment to the organization's provider keys instead", async () => {
    // A hosted deployment does not report the `providers` surface at all, so
    // the shell answers that route with "not available here"; the first thing a
    // new operator clicks must not be that panel.
    mockApi({ providers: [] })
    const user = userEvent.setup()
    renderPage(
      <OverviewIndex />,
      "/overview",
      bootstrap({
        deployment_type: "hosted",
        surfaces: HOSTED_SURFACES,
      }),
    )

    await screen.findByText("Get started with Otari")
    await user.click(
      screen.getByRole("button", { name: "Add your first provider" }),
    )
    expect(await screen.findByTestId("loc")).toHaveTextContent(
      "/organization/provider-keys",
    )
  })

  it("reports a failed provider query instead of silently rendering a normal overview", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input)
      // The shell reads this before it paints, and the usage hooks wait for it:
      // it is what tells them whether this caller reads the deployment-wide
      // routes or the organization-scoped ones (otari#837). Answered first, and
      // on an exact match, so it cannot shadow /v1/organizations/me/usage.
      if (url.endsWith("/v1/organizations/me")) {
        return jsonResponse(organizationContext())
      }
      if (url.includes("/v1/providers/health")) {
        return jsonResponse({
          providers: [],
          healthy: 1,
          total: 1,
          checked_at: null,
        })
      }
      if (url.includes("/v1/usage/summary")) return jsonResponse(summary({}))
      if (url.includes("/v1/providers"))
        return jsonResponse({ detail: "providers exploded" }, 500)
      return jsonResponse([])
    })
    renderPage(<OverviewIndex />)

    // The failure is surfaced, and the setup state stays neutral: no
    // getting-started block claiming the gateway is fresh, and no silent success.
    expect(await screen.findByText(/providers exploded/)).toBeInTheDocument()
    expect(screen.queryByText("Get started with Otari")).not.toBeInTheDocument()
  })

  it("clears the provider-query banner when Refresh retries it", async () => {
    // The providers query is cached for minutes and never refetches on focus, so
    // the page's Refresh has to drive it or the banner outlives the outage.
    let failProviders = true
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input)
      // The shell reads this before it paints, and the usage hooks wait for it:
      // it is what tells them whether this caller reads the deployment-wide
      // routes or the organization-scoped ones (otari#837). Answered first, and
      // on an exact match, so it cannot shadow /v1/organizations/me/usage.
      if (url.endsWith("/v1/organizations/me")) {
        return jsonResponse(organizationContext())
      }
      if (url.includes("/v1/providers/health")) {
        return jsonResponse({
          providers: [],
          healthy: 1,
          total: 1,
          checked_at: null,
        })
      }
      if (url.includes("/v1/usage/summary")) return jsonResponse(summary({}))
      if (url.includes("/v1/providers")) {
        return failProviders
          ? jsonResponse({ detail: "providers exploded" }, 500)
          : jsonResponse({ providers: [{ provider: "openai" }] })
      }
      return jsonResponse([])
    })
    const user = userEvent.setup()
    renderPage(<OverviewIndex />)

    expect(await screen.findByText(/providers exploded/)).toBeInTheDocument()
    failProviders = false
    await user.click(screen.getByRole("button", { name: /refresh/i }))
    await waitFor(() =>
      expect(screen.queryByText(/providers exploded/)).not.toBeInTheDocument(),
    )
  })
})

// The wire for a caller who does not operate the deployment: `/v1/admin/access`
// answers an explicit no, the organization context agrees, and the usage hooks
// therefore read `/v1/organizations/me/usage` (otari#837). Everything else is
// an endpoint this caller must never be asked to read, so it refuses the way
// the server would; a leak fails the assertions loudly instead of rendering a
// plausible tile. Every URL is recorded so the tests can say so.
function mockScopedApi(b: Bodies): string[] {
  const requested: string[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input)
    requested.push(url)
    if (url.endsWith("/v1/admin/access")) {
      return jsonResponse({ granted: false })
    }
    if (url.endsWith("/v1/organizations/me")) {
      return jsonResponse(organizationContext({ deployment_operator: false }))
    }
    if (url.includes("/v1/organizations/me/usage/summary")) {
      if (url.includes("bucket=hour"))
        return jsonResponse(summary(b.today ?? {}))
      if (url.includes("end_date=")) return jsonResponse(summary(b.prev ?? {}))
      return jsonResponse(summary(b.period ?? {}))
    }
    if (url.includes("/v1/organizations/me/usage")) {
      return jsonResponse(b.logs ?? [])
    }
    return jsonResponse({ detail: "forbidden" }, 403)
  })
  return requested
}

describe("OverviewIndex for a caller who does not operate the deployment", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("lands on their own usage, not an apology card", async () => {
    mockScopedApi({
      today: { cost: 5, request_count: 100, error_count: 1 },
      period: { cost: 200, request_count: 2000, error_count: 40 },
      prev: { cost: 100, request_count: 1000, error_count: 5 },
    })
    renderPage(<OverviewIndex />)

    // The scoped summaries land as real tiles: today vs 30-day spend, volume,
    // and the window's error rate, exactly as the operator page renders them.
    expect(await screen.findByText("$5.00")).toBeInTheDocument()
    expect(screen.getByText("$200.00")).toBeInTheDocument()
    expect(screen.getByText("2,000")).toBeInTheDocument()
    expect(screen.getByText("2.0%")).toBeInTheDocument()
    // The dead-end card this page replaced (otari-ai#1929) stays gone.
    expect(
      screen.queryByText(/This overview is for deployment operators/),
    ).not.toBeInTheDocument()
  })

  it("reads only the organization-scoped surface and shows no operator tile", async () => {
    const requested = mockScopedApi({
      period: { cost: 200, request_count: 2000 },
    })
    renderPage(<OverviewIndex />)
    await screen.findByText("$200.00")

    // The deployment-wide panels stay on the operator page: no tile here reads
    // budgets, keys, or the deployment roster.
    expect(screen.queryByText("Budget health")).not.toBeInTheDocument()
    expect(screen.queryByText("Active keys")).not.toBeInTheDocument()
    expect(screen.queryByText("Active members")).not.toBeInTheDocument()
    // And nothing left the scoped surface: beyond the gate and the context,
    // every request this page made names /v1/organizations/me/usage. A bare
    // /v1/usage read here would be a cross-tenant read the server refuses, so
    // the page must not even attempt it.
    const scoped = requested.filter(
      (url) =>
        !url.endsWith("/v1/admin/access") &&
        !url.endsWith("/v1/organizations/me"),
    )
    expect(scoped.length).toBeGreaterThan(0)
    for (const url of scoped) {
      expect(url).toContain("/v1/organizations/me/usage")
    }
  })

  it("previews the caller's recent requests with a link to the full log", async () => {
    mockScopedApi({
      logs: [
        {
          id: "1",
          user_id: null,
          api_key_id: null,
          timestamp: "2026-07-22T00:00:00Z",
          model: "gpt-5.6",
          provider: "openai",
          endpoint: "/v1/chat/completions",
          prompt_tokens: 10,
          completion_tokens: 5,
          total_tokens: 15,
          cache_read_tokens: 0,
          cache_write_tokens: 0,
          cost: 0.5,
          status: "success",
          error_message: null,
          latency_ms: 120,
        },
      ],
    })
    renderPage(<OverviewIndex />)

    expect(await screen.findByText("gpt-5.6")).toBeInTheDocument()
    // Activity serves this caller from the same scoped surface, so the preview
    // may hand them off to it.
    expect(screen.getByRole("link", { name: /view all/i })).toHaveAttribute(
      "href",
      "/activity",
    )
  })

  it("reports a failed previous window instead of silently dropping the trends", async () => {
    // The previous window's only reader is the trend chips, so its failure has
    // no tile of its own to show it: without the banner it would just strip
    // every "vs prev" chip and look like there was nothing to compare.
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input)
      if (url.endsWith("/v1/admin/access")) {
        return jsonResponse({ granted: false })
      }
      if (url.endsWith("/v1/organizations/me")) {
        return jsonResponse(organizationContext({ deployment_operator: false }))
      }
      if (url.includes("/v1/organizations/me/usage/summary")) {
        if (url.includes("end_date="))
          return jsonResponse({ detail: "previous window exploded" }, 500)
        return jsonResponse(summary({ cost: 200, request_count: 2000 }))
      }
      if (url.includes("/v1/organizations/me/usage")) {
        return jsonResponse([])
      }
      return jsonResponse({ detail: "forbidden" }, 403)
    })
    renderPage(<OverviewIndex />)

    // Both current windows read the same body here, hence findAllByText.
    expect((await screen.findAllByText("$200.00")).length).toBeGreaterThan(0)
    expect(
      await screen.findByText(/previous window exploded/),
    ).toBeInTheDocument()
  })

  it("reports a failed scoped summary instead of a silent wall of dashes", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input)
      if (url.endsWith("/v1/admin/access")) {
        return jsonResponse({ granted: false })
      }
      if (url.endsWith("/v1/organizations/me")) {
        return jsonResponse(organizationContext({ deployment_operator: false }))
      }
      if (url.includes("/v1/organizations/me/usage/summary")) {
        return jsonResponse({ detail: "usage exploded" }, 500)
      }
      if (url.includes("/v1/organizations/me/usage")) {
        return jsonResponse([])
      }
      return jsonResponse({ detail: "forbidden" }, 403)
    })
    renderPage(<OverviewIndex />)

    expect(await screen.findByText(/usage exploded/)).toBeInTheDocument()
  })
})
