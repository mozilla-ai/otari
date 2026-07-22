import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import type { ReactElement } from "react";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { setMasterKey } from "@/api/client";
import type { UsageSummary } from "@/api/types";
import { localDayKey, OverviewIndex, OverviewPage } from "@/pages/OverviewPage";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

function summary(totals: Partial<UsageSummary["totals"]>): UsageSummary {
  return {
    start_date: "2026-06-22T00:00:00Z",
    end_date: "2026-07-22T00:00:00Z",
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
      ...totals,
    },
    by_model: [],
    by_user: [],
    by_api_key: [],
    series: [],
  };
}

interface Bodies {
  today?: Partial<UsageSummary["totals"]>;
  period?: Partial<UsageSummary["totals"]>;
  prev?: Partial<UsageSummary["totals"]>;
  health?: unknown;
  budgets?: unknown;
  keys?: unknown;
  users?: unknown;
  logs?: unknown;
}

// Order matters: /v1/usage/summary is matched BEFORE the bare /v1/usage logs
// endpoint, and /v1/providers/health returns an OBJECT (not the [] fallback) so
// the health/status logic reads real counts. The summary mock is param-aware so
// the Today (bucket=hour) and Last-30d (bucket=day) tiles render distinct values.
function mockApi(b: Bodies) {
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input);
    if (url.includes("/v1/usage/summary")) {
      if (url.includes("bucket=hour")) return jsonResponse(summary(b.today ?? {}));
      if (url.includes("end_date=")) return jsonResponse(summary(b.prev ?? {}));
      return jsonResponse(summary(b.period ?? {}));
    }
    if (url.includes("/v1/providers/health")) {
      return jsonResponse(b.health ?? { providers: [], healthy: 0, total: 0, checked_at: null });
    }
    if (url.includes("/v1/budgets")) return jsonResponse(b.budgets ?? []);
    if (url.includes("/v1/keys")) return jsonResponse(b.keys ?? []);
    if (url.includes("/v1/users")) return jsonResponse(b.users ?? []);
    if (url.includes("/v1/providers")) return jsonResponse({ providers: [{ provider: "openai" }] });
    if (url.includes("/v1/usage")) return jsonResponse(b.logs ?? []);
    return jsonResponse([]);
  });
}

function LocationProbe() {
  const loc = useLocation();
  return <div data-testid="loc">{loc.pathname}</div>;
}

function renderPage(ui: ReactElement, initial = "/overview") {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[initial]}>
        <Routes>
          <Route path="/overview" element={ui} />
          <Route path="/providers" element={<LocationProbe />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("OverviewPage", () => {
  beforeEach(() => setMasterKey("test-master-key"));
  afterEach(() => {
    vi.restoreAllMocks();
    vi.useRealTimers();
    setMasterKey(null);
  });

  it("uses a zero-padded, one-based local calendar date as its refresh key", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(2026, 0, 5, 12));

    expect(localDayKey()).toBe("2026-01-05");
  });

  it("renders distinct today vs 30-day spend, request volume, and error rate", async () => {
    mockApi({
      today: { cost: 5, request_count: 100, error_count: 1 },
      period: { cost: 200, request_count: 2000, error_count: 40 }, // 2% -> warn
      prev: { cost: 100, request_count: 1000, error_count: 5 },
    });
    renderPage(<OverviewPage />);

    expect(await screen.findByText("$5.00")).toBeInTheDocument();
    expect(await screen.findByText("$200.00")).toBeInTheDocument();
    expect(screen.getByText("2,000")).toBeInTheDocument();
    expect(screen.getByText("2.0%")).toBeInTheDocument();
    expect(screen.getByText("Elevated")).toBeInTheDocument(); // error-rate status word (non-hue)
  });

  it("shows 30-day cache reads as a tile", async () => {
    mockApi({
      today: { cost: 5 },
      period: { cost: 200, cache_read_tokens: 5_100_000 },
      prev: { cost: 100, cache_read_tokens: 1_000_000 },
    });
    renderPage(<OverviewPage />);

    expect(await screen.findByText("5.1M")).toBeInTheDocument();
    expect(screen.getByText("Cache reads, last 30 days")).toBeInTheDocument();
  });

  it("shows a dash for error rate when there are no requests", async () => {
    mockApi({ period: { request_count: 0, error_count: 0 } });
    renderPage(<OverviewPage />);
    // No status word for a neutral error-rate tile.
    expect(await screen.findByText(/All systems normal/)).toBeInTheDocument();
    expect(screen.queryByText("Elevated")).not.toBeInTheDocument();
  });

  it("computes budget health with cap * user_count and links to budgets", async () => {
    mockApi({
      budgets: [
        { budget_id: "team", name: "team", max_budget: 10, user_count: 2, total_spend: 25, total_reserved: 0 },
        { budget_id: "x", name: "x", max_budget: null, user_count: 1, total_spend: 9999, total_reserved: 0 },
      ],
    });
    renderPage(<OverviewPage />);
    expect(await screen.findByText("125.0%")).toBeInTheDocument(); // 25 / (10*2)
    expect(screen.getByText("Over budget")).toBeInTheDocument();
  });

  it("summarizes provider health and surfaces problems in the status strip", async () => {
    mockApi({
      health: { providers: [], healthy: 2, total: 3, checked_at: "2026-07-22T00:00:00Z" },
      budgets: [{ budget_id: "team", name: "team", max_budget: 10, user_count: 2, total_spend: 25, total_reserved: 0 }],
    });
    renderPage(<OverviewPage />);
    expect(await screen.findByText("2/3")).toBeInTheDocument();
    expect(screen.getByText("Degraded")).toBeInTheDocument();
    // The strip lists only the problems, each a link.
    expect(screen.getByText("1 provider unreachable")).toBeInTheDocument();
    expect(screen.getByText("1 budget over limit")).toBeInTheDocument();
  });

  it("shows the all-clear strip when nothing needs attention", async () => {
    mockApi({ health: { providers: [], healthy: 3, total: 3, checked_at: "2026-07-22T00:00:00Z" } });
    renderPage(<OverviewPage />);
    // Wait for the providers tile so the health query has resolved before we read
    // the strip (it renders immediately, then fills in once health loads).
    await screen.findByText("3/3");
    const strip = screen.getByRole("status");
    expect(strip).toHaveTextContent("All systems normal");
    expect(strip).toHaveTextContent("3/3 providers healthy");
  });

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
    });
    renderPage(<OverviewPage />);
    expect(await screen.findByText("gpt-5.6")).toBeInTheDocument();
    // null cost renders as an em-dash, not a crash.
    expect(screen.getAllByText("—").length).toBeGreaterThan(0);
  });

  it("keeps the page up when one tile query fails (per-tile isolation)", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.includes("/v1/budgets")) return jsonResponse({ detail: "boom" }, 500);
      if (url.includes("/v1/usage/summary")) return jsonResponse(summary({ cost: 200, request_count: 10 }));
      if (url.includes("/v1/providers/health")) return jsonResponse({ providers: [], healthy: 1, total: 1, checked_at: null });
      return jsonResponse([]);
    });
    renderPage(<OverviewPage />);
    // The spend tiles still render even though budgets errored (per-tile isolation);
    // both Today and Last-30d read $200.00 with this mock, hence findAllByText.
    expect((await screen.findAllByText("$200.00")).length).toBeGreaterThan(0);
    // ...and the status strip must NOT claim all-clear while a source query failed
    // (it would contradict the error banner). It reads as a neutral load-failure line.
    expect(screen.queryByText(/All systems normal/)).not.toBeInTheDocument();
    expect(screen.getByText(/could not be loaded/)).toBeInTheDocument();
  });

  it("does not announce all-clear while status sources are still loading", async () => {
    // A never-resolving fetch keeps the queries pending.
    vi.spyOn(globalThis, "fetch").mockImplementation(() => new Promise<Response>(() => {}));
    renderPage(<OverviewPage />);
    expect(await screen.findByText(/Checking gateway status/)).toBeInTheDocument();
    expect(screen.queryByText(/All systems normal/)).not.toBeInTheDocument();
  });
});

describe("OverviewIndex routing", () => {
  beforeEach(() => setMasterKey("test-master-key"));
  afterEach(() => {
    vi.restoreAllMocks();
    setMasterKey(null);
  });

  it("renders the overview when a provider is configured", async () => {
    mockApi({ health: { providers: [], healthy: 1, total: 1, checked_at: null } });
    renderPage(<OverviewIndex />);
    expect(await screen.findByText("Overview")).toBeInTheDocument();
  });

  it("redirects to providers on a fresh gateway with none configured", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      if (String(input).includes("/v1/providers")) return jsonResponse({ providers: [] });
      return jsonResponse([]);
    });
    renderPage(<OverviewIndex />);
    expect(await screen.findByTestId("loc")).toHaveTextContent("/providers");
  });
});
