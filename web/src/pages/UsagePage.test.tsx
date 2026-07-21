import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactElement } from "react";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { setMasterKey } from "@/api/client";
import type { UsageSummary } from "@/api/types";
import { UsagePage } from "@/pages/UsagePage";

function summary(overrides: Partial<UsageSummary> = {}): UsageSummary {
  return {
    start_date: "2026-06-21T00:00:00Z",
    end_date: "2026-07-21T00:00:00Z",
    bucket: "day",
    totals: {
      cost: 1240.5,
      prompt_tokens: 8_000_000,
      completion_tokens: 4_400_000,
      total_tokens: 12_400_000,
      cache_read_tokens: 0,
      cache_write_tokens: 0,
      request_count: 84_000,
      error_count: 1_764,
      avg_latency_ms: 820,
    },
    by_model: [
      { key: "gpt-5.6", cost: 820, tokens: 8_000_000, requests: 42_000 },
      { key: "claude-sonnet-5", cost: 310, tokens: 3_000_000, requests: 28_000 },
      { key: null, cost: 110.5, tokens: 1_400_000, requests: 14_000 },
    ],
    by_user: [
      { key: "alice", cost: 900.5, tokens: 8_000_000, requests: 50_000 },
      { key: "bob", cost: 340, tokens: 4_400_000, requests: 34_000 },
    ],
    by_api_key: [],
    series: [
      { bucket_start: "2026-07-19T00:00:00Z", cost: 400, tokens: 4_000_000, requests: 28_000 },
      { bucket_start: "2026-07-20T00:00:00Z", cost: 840.5, tokens: 8_400_000, requests: 56_000 },
    ],
    ...overrides,
  };
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

function mockApi(body: UsageSummary | null) {
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input);
    if (url.includes("/v1/usage/summary")) {
      return jsonResponse(body ?? summary());
    }
    if (url.includes("/v1/users")) {
      return jsonResponse([{ user_id: "alice", alias: "Alice" }]);
    }
    return jsonResponse([]);
  });
}

// Surfaces the current location so a drill-down navigation can be asserted.
function LocationProbe() {
  const loc = useLocation();
  return <div data-testid="loc">{`${loc.pathname}${loc.search}`}</div>;
}

function renderPage(ui: ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={["/usage"]}>
        <Routes>
          <Route path="/usage" element={ui} />
          <Route path="/activity" element={<LocationProbe />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("UsagePage", () => {
  beforeEach(() => setMasterKey("test-master-key"));
  afterEach(() => {
    vi.restoreAllMocks();
    setMasterKey(null);
  });

  it("renders totals tiles with compact currency and error rate", async () => {
    mockApi(summary());
    renderPage(<UsagePage />);

    // The total ($1,240.50) is unique to the tile: no single breakdown row equals it.
    expect(await screen.findByText("$1,240.50")).toBeInTheDocument();
    expect(screen.getByText("84,000")).toBeInTheDocument();
    expect(screen.getByText("12.4M")).toBeInTheDocument();
    // 1764 / 84000 = 2.1% errors.
    expect(screen.getByText(/2\.1% errors/)).toBeInTheDocument();
  });

  it("lists spend by model with a reconciling 'other' fold row", async () => {
    mockApi(summary());
    renderPage(<UsagePage />);

    expect(await screen.findByText("gpt-5.6")).toBeInTheDocument();
    expect(screen.getByText("claude-sonnet-5")).toBeInTheDocument();
    // The null-key fold row renders as an "Other" summary, not a blank row.
    expect(screen.getByText(/Other \(14,000 req\)/)).toBeInTheDocument();
  });

  it("drills into the Activity log filtered on the clicked model", async () => {
    const user = userEvent.setup();
    mockApi(summary());
    renderPage(<UsagePage />);

    const row = (await screen.findByText("gpt-5.6")).closest("tr")!;
    await user.click(row);

    const loc = screen.getByTestId("loc").textContent ?? "";
    expect(loc.startsWith("/activity")).toBe(true);
    expect(loc).toContain("model=gpt-5.6");
  });

  it("filters models by typeahead and commits the exact picked model", async () => {
    const fetchMock = mockApi(summary());
    const user = userEvent.setup();
    renderPage(<UsagePage />);
    await screen.findByText("gpt-5.6");

    // The model box is a typeahead sourced from the in-window models, not a
    // free-text exact-match input.
    const modelInput = screen.getByRole("combobox", { name: "Model" });
    await user.click(modelInput);
    await user.type(modelInput, "claude");
    await user.click(await screen.findByRole("option", { name: /claude-sonnet-5/ }));

    const summaryCalls = fetchMock.mock.calls
      .map(([u]) => String(u))
      .filter((u) => u.includes("/usage/summary") && !u.includes(".csv"));
    expect(summaryCalls.at(-1)).toContain("model=claude-sonnet-5");
  });

  it("switches the chart metric via the segmented toggle", async () => {
    const user = userEvent.setup();
    mockApi(summary());
    renderPage(<UsagePage />);

    await screen.findByText("gpt-5.6");
    // Default metric is Cost; the caption shows the peak in dollars.
    expect(screen.getByText(/\$840\.50 peak/)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Requests" }));
    expect(screen.getByText(/56,000 peak/)).toBeInTheDocument();
  });

  it("shows an onboarding empty state when the gateway has no usage", async () => {
    mockApi(
      summary({
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
        by_model: [],
        by_user: [],
        series: [],
      }),
    );
    renderPage(<UsagePage />);

    // Default window is 30d (a filter), so the empty state only reads as
    // "never used" after clearing filters.
    await screen.findByText("Spend by model");
    await userEvent.setup().click(screen.getByRole("button", { name: "Clear filters" }));
    expect(await screen.findByText(/No usage yet/)).toBeInTheDocument();
  });

  it("exports CSV through the authenticated blob download", async () => {
    const fetchMock = mockApi(summary());
    // jsdom defines neither, so assign (spyOn needs an existing method) before
    // the export path calls them after a successful fetch.
    const createUrl = vi.fn(() => "blob:x");
    URL.createObjectURL = createUrl as unknown as typeof URL.createObjectURL;
    URL.revokeObjectURL = vi.fn();
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});

    const user = userEvent.setup();
    renderPage(<UsagePage />);
    await screen.findByText("gpt-5.6");

    await user.click(screen.getByRole("button", { name: "Export CSV" }));

    const csvCall = fetchMock.mock.calls.map(([u]) => String(u)).find((u) => u.includes("/summary.csv"));
    expect(csvCall).toBeDefined();
    expect(createUrl).toHaveBeenCalled();
  });
});
