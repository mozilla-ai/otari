import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactElement } from "react";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

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
      { key: "gpt-5.6", cost: 820, tokens: 8_000_000, requests: 42_000, is_other: false },
      { key: "claude-sonnet-5", cost: 310, tokens: 3_000_000, requests: 28_000, is_other: false },
      { key: null, cost: 110.5, tokens: 1_400_000, requests: 14_000, is_other: true },
    ],
    by_user: [
      { key: "alice", cost: 900.5, tokens: 8_000_000, requests: 50_000, is_other: false },
      { key: "bob", cost: 340, tokens: 4_400_000, requests: 34_000, is_other: false },
    ],
    by_api_key: [],
    by_source: [
      { key: "gateway", cost: 1_000, tokens: 9_000_000, requests: 60_100, is_other: false },
      { key: "claude_code", cost: 240.5, tokens: 3_400_000, requests: 23_900, is_other: false },
    ],
    by_source_label: [
      { key: "project:otari", cost: 700, tokens: 6_000_000, requests: 30_100, is_other: false },
      { key: "project:docs", cost: 200, tokens: 2_000_000, requests: 9_200, is_other: false },
      // Gateway traffic carries no session label: a real group with a null key,
      // not the synthesized fold.
      { key: null, cost: 340.5, tokens: 4_400_000, requests: 44_700, is_other: false },
    ],
    by_endpoint: [
      { key: "/v1/chat/completions", cost: 900, tokens: 8_000_000, requests: 50_100, is_other: false },
      { key: "/v1/messages", cost: 340.5, tokens: 4_400_000, requests: 33_900, is_other: false },
    ],
    by_provider: [
      { key: "openai", cost: 880, tokens: 7_000_000, requests: 45_100, is_other: false },
      { key: "anthropic", cost: 360.5, tokens: 5_400_000, requests: 38_900, is_other: false },
    ],
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
    if (url.includes("/v1/keys")) {
      return jsonResponse([{ id: "key-1", key_name: "ci-bot", user_id: "alice", allowed_models: null }]);
    }
    return jsonResponse([]);
  });
}

// Surfaces the current location so a drill-down navigation can be asserted.
function LocationProbe() {
  const loc = useLocation();
  // A status role with an accessible name so tests query the probe by role
  // rather than a test id.
  return (
    <div role="status" aria-label="Current location">{`${loc.pathname}${loc.search}`}</div>
  );
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
  afterEach(() => {
    vi.restoreAllMocks();
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

  it("filters usage by API key", async () => {
    const user = userEvent.setup();
    const fetchMock = mockApi(summary());
    renderPage(<UsagePage />);
    await screen.findByText("$1,240.50");

    await user.click(screen.getByPlaceholderText("All keys"));
    await user.click(await screen.findByRole("option", { name: "ci-bot" }));

    const summaryCalls = fetchMock.mock.calls
      .map(([u]) => String(u))
      .filter((u) => u.includes("/v1/usage/summary"));
    expect(summaryCalls.some((u) => u.includes("api_key_id=key-1"))).toBe(true);
  });

  it("does not render the CSV export action", async () => {
    mockApi(summary());
    renderPage(<UsagePage />);

    await screen.findByText("$1,240.50");
    expect(screen.queryByRole("button", { name: "Export CSV" })).not.toBeInTheDocument();
  });

  it("re-queries with an hourly bucket when a sub-day preset is chosen from the timeline", async () => {
    const user = userEvent.setup();
    const fetchMock = mockApi(summary());
    renderPage(<UsagePage />);
    await screen.findByText("$1,240.50");
    fetchMock.mockClear();

    await user.click(screen.getByRole("button", { name: "Last hour" }));

    await vi.waitFor(() => {
      const summaryCalls = fetchMock.mock.calls.map(([u]) => String(u)).filter((u) => u.includes("/v1/usage/summary"));
      // The sub-day extent buckets hourly (both the context histogram and the tiles).
      expect(summaryCalls.some((u) => u.includes("bucket=hour"))).toBe(true);
    });
  });

  it("shows cache read and write totals as tiles", async () => {
    const base = summary();
    mockApi(summary({ totals: { ...base.totals, cache_read_tokens: 5_100_000, cache_write_tokens: 2_700_000 } }));
    renderPage(<UsagePage />);

    // Await a value (loads after the query resolves), not the static label.
    expect(await screen.findByText("5.1M")).toBeInTheDocument();
    expect(screen.getByText("2.7M")).toBeInTheDocument();
    expect(screen.getByText("Cache read")).toBeInTheDocument();
    expect(screen.getByText("Cache write")).toBeInTheDocument();
  });

  it("queries the previous period with a bounded end_date for deltas", async () => {
    const fetchMock = mockApi(summary());
    renderPage(<UsagePage />);
    await screen.findByText("gpt-5.6");

    const summaryCalls = fetchMock.mock.calls
      .map(([u]) => String(u))
      .filter((u) => u.includes("/usage/summary"));
    // The default 30d preset fires a current window (no end_date, "up to now")
    // and a previous window whose end_date is pinned so it does not overlap.
    expect(summaryCalls.some((u) => u.includes("end_date="))).toBe(true);
    expect(summaryCalls.some((u) => !u.includes("end_date="))).toBe(true);
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

    const loc = screen.getByRole("status", { name: "Current location" }).textContent ?? "";
    expect(loc.startsWith("/activity")).toBe(true);
    expect(loc).toContain("model=gpt-5.6");
  });

  it("keeps an active user filter when drilling into a model", async () => {
    const user = userEvent.setup();
    mockApi(summary());
    renderPage(<UsagePage />);
    await screen.findByText("gpt-5.6");

    // Filter by a user, then drill into a model row. The user constraint must
    // survive the navigation, not be dropped in favor of only the clicked model.
    const userInput = screen.getByRole("combobox", { name: "User" });
    await user.click(userInput);
    await user.type(userInput, "alice");
    await user.click(await screen.findByRole("option", { name: /alice/ }));

    const row = (await screen.findByText("gpt-5.6")).closest("tr")!;
    await user.click(row);

    const loc = screen.getByRole("status", { name: "Current location" }).textContent ?? "";
    expect(loc.startsWith("/activity")).toBe(true);
    expect(loc).toContain("model=gpt-5.6");
    expect(loc).toContain("user_id=alice");
  });

  it("keeps an active model filter when drilling into a user", async () => {
    const user = userEvent.setup();
    mockApi(summary());
    renderPage(<UsagePage />);
    await screen.findByText("alice");

    // Filter by a model, then drill into a user row. The model constraint must
    // survive the navigation, not be dropped in favor of only the clicked user.
    const modelInput = screen.getByRole("combobox", { name: "Model" });
    await user.click(modelInput);
    await user.type(modelInput, "gpt");
    await user.click(await screen.findByRole("option", { name: /gpt-5.6/ }));

    const row = (await screen.findByText("alice")).closest("tr")!;
    await user.click(row);

    const loc = screen.getByRole("status", { name: "Current location" }).textContent ?? "";
    expect(loc.startsWith("/activity")).toBe(true);
    expect(loc).toContain("user_id=alice");
    expect(loc).toContain("model=gpt-5.6");
  });

  it("keeps an active API key filter when drilling into a model", async () => {
    const user = userEvent.setup();
    mockApi(summary());
    renderPage(<UsagePage />);
    await screen.findByText("gpt-5.6");

    // Filter by an API key, then drill into a model row. The key constraint must
    // survive the navigation alongside the clicked model.
    await user.click(screen.getByPlaceholderText("All keys"));
    await user.click(await screen.findByRole("option", { name: "ci-bot" }));

    const row = (await screen.findByText("gpt-5.6")).closest("tr")!;
    await user.click(row);

    const loc = screen.getByRole("status", { name: "Current location" }).textContent ?? "";
    expect(loc.startsWith("/activity")).toBe(true);
    expect(loc).toContain("model=gpt-5.6");
    expect(loc).toContain("api_key_id=key-1");
  });

  it("shows the session breakdown by default, labelling unlabelled gateway traffic", async () => {
    mockApi(summary());
    renderPage(<UsagePage />);

    // Session is the default secondary dimension: it is what names the work
    // behind a bill for agent traffic.
    expect(await screen.findByText("project:otari")).toBeInTheDocument();
    expect(screen.getByText("Spend by session")).toBeInTheDocument();
    expect(screen.getByText("project:docs")).toBeInTheDocument();
    // Gateway rows carry no label. That is a real group, not the "other" fold,
    // so it must not read as unknown/missing data.
    expect(screen.getByText("(no session)")).toBeInTheDocument();
  });

  it("marks the active dimension button as pressed", async () => {
    // The picker's selected state cannot ride on the button variant alone: to
    // assistive tech that is four identically-named buttons with no indication of
    // which dimension the table below is showing.
    const user = userEvent.setup();
    mockApi(summary());
    renderPage(<UsagePage />);
    await screen.findByText("project:otari");

    expect(screen.getByRole("button", { name: "Session" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "Provider" })).toHaveAttribute("aria-pressed", "false");

    await user.click(screen.getByRole("button", { name: "Provider" }));
    expect(screen.getByRole("button", { name: "Provider" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "Session" })).toHaveAttribute("aria-pressed", "false");
  });

  it("asks the summary endpoint only for the breakdowns the page renders", async () => {
    // Each breakdown is its own GROUP BY over the window. The page renders model,
    // user, and the four picker dimensions; the previous-period and timeline-context
    // reads use only totals/series, so they must opt out of all of them.
    const fetchMock = mockApi(summary());
    renderPage(<UsagePage />);
    await screen.findByText("project:otari");

    const summaryCalls = fetchMock.mock.calls.map(([u]) => String(u)).filter((u) => u.includes("/v1/usage/summary"));
    const main = summaryCalls.find((u) => u.includes("dimensions=model") && u.includes("dimensions=user"));
    expect(main).toBeDefined();
    expect(main).toContain("dimensions=source_label");
    expect(main).toContain("dimensions=provider");
    // No table on this page breaks spend down by API key.
    expect(main).not.toContain("dimensions=api_key");
    expect(summaryCalls.some((u) => u.includes("dimensions=none"))).toBe(true);
  });

  it("switches the secondary breakdown between session, endpoint, provider, and source", async () => {
    const user = userEvent.setup();
    mockApi(summary());
    renderPage(<UsagePage />);
    await screen.findByText("project:otari");

    await user.click(screen.getByRole("button", { name: "Provider" }));
    expect(screen.getByText("Spend by provider")).toBeInTheDocument();
    expect(screen.getByText("anthropic")).toBeInTheDocument();
    expect(screen.queryByText("project:otari")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Endpoint" }));
    expect(screen.getByText("/v1/chat/completions")).toBeInTheDocument();

    // by_source is computed and shipped by the server; it now has a home in the UI.
    await user.click(screen.getByRole("button", { name: "Source" }));
    expect(screen.getByText("claude_code")).toBeInTheDocument();
  });

  it("drills into the Activity log scoped to the clicked session", async () => {
    const user = userEvent.setup();
    mockApi(summary());
    renderPage(<UsagePage />);

    const row = (await screen.findByText("project:otari")).closest("tr")!;
    await user.click(row);

    const loc = screen.getByRole("status", { name: "Current location" }).textContent ?? "";
    expect(loc.startsWith("/activity")).toBe(true);
    expect(loc).toContain("source_label=project%3Aotari");
  });

  it("drills into the Activity log scoped to the clicked provider", async () => {
    const user = userEvent.setup();
    mockApi(summary());
    renderPage(<UsagePage />);
    await screen.findByText("project:otari");

    await user.click(screen.getByRole("button", { name: "Provider" }));
    const row = screen.getByText("anthropic").closest("tr")!;
    await user.click(row);

    const loc = screen.getByRole("status", { name: "Current location" }).textContent ?? "";
    expect(loc).toContain("provider=anthropic");
  });

  it("does not drill on the unlabelled-session row, which has no id to filter on", async () => {
    const user = userEvent.setup();
    mockApi(summary());
    renderPage(<UsagePage />);

    const row = (await screen.findByText("(no session)")).closest("tr")!;
    await user.click(row);

    // Still on the Usage page: a null key cannot scope the request log.
    expect(screen.queryByRole("status", { name: "Current location" })).not.toBeInTheDocument();
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
      .filter((u) => u.includes("/usage/summary"));
    expect(summaryCalls.at(-1)).toContain("model=claude-sonnet-5");
  });

  it("renders the trend with recharts and retires the hand-rolled SVG chart", async () => {
    mockApi(summary());
    const { container } = renderPage(<UsagePage />);
    await screen.findByText("gpt-5.6");

    // The trend is now a recharts chart (labelled "<metric> per <bucket>"), and a
    // reusable sparkline rides the KPI tiles off the same bucketed series.
    expect(screen.getByRole("img", { name: "cost per day" })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /Spend trend/ })).toBeInTheDocument();
    expect(container.querySelector(".recharts-surface")).not.toBeNull();

    // The retired hand-rolled chart's fingerprints are gone: its "<metric> over
    // time" label and its fixed 720x224 viewBox.
    expect(screen.queryByRole("img", { name: /over time/ })).not.toBeInTheDocument();
    expect(container.querySelector('svg[viewBox="0 0 720 224"]')).toBeNull();
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

    // The default 30d window is the baseline (not a user-applied filter), so an
    // empty gateway reads as onboarding rather than "no rows match".
    expect(await screen.findByText(/No usage yet/)).toBeInTheDocument();
  });

  it("bulk-deletes imported rows from the Individual requests list", async () => {
    const importedRow = {
      id: "imp-1",
      user_id: "alice",
      api_key_id: null,
      timestamp: new Date().toISOString(),
      model: "claude-sonnet-5",
      provider: "anthropic",
      endpoint: "external",
      prompt_tokens: 100,
      completion_tokens: 50,
      total_tokens: 150,
      cache_read_tokens: null,
      cache_write_tokens: null,
      cache_write_1h_tokens: null,
      billing_meters: null,
      pricing_breakdown: null,
      cost: null,
      status: "success",
      error_message: null,
      latency_ms: 120,
      source: "claude_code",
      source_label: null,
      counts_toward_budget: false,
    };
    const calls: { url: string; method: string; body: string | undefined }[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      const method = (init?.method ?? "GET").toUpperCase();
      calls.push({ url, method, body: typeof init?.body === "string" ? init.body : undefined });
      if (url.endsWith("/v1/usage") && method === "DELETE") return jsonResponse({ deleted: 1 });
      if (url.includes("/v1/usage/summary")) return jsonResponse(summary());
      if (url.includes("/v1/usage/count")) return jsonResponse({ total: 1 });
      if (url.includes("/v1/users")) return jsonResponse([{ user_id: "alice", alias: "Alice" }]);
      if (url.includes("/v1/keys")) return jsonResponse([]);
      if (url.includes("/v1/usage")) return jsonResponse([importedRow]);
      return jsonResponse([]);
    });
    const user = userEvent.setup();
    renderPage(<UsagePage />);

    const row = (await screen.findByText("claude-sonnet-5")).closest("tr")!;
    await user.click(within(row).getByRole("checkbox"));

    const bar = (await screen.findByText("1 selected")).closest("div")!;
    await user.click(within(bar).getByRole("button", { name: "Delete" }));
    const dialog = await screen.findByRole("alertdialog");
    await user.click(within(dialog).getByRole("button", { name: "Delete" }));

    await vi.waitFor(() => {
      const del = calls.find((c) => c.url.endsWith("/v1/usage") && c.method === "DELETE");
      expect(del).toBeTruthy();
      expect(del!.body).toContain("imp-1");
    });
  });

  it("keeps the filter pickers behind an 'Add filter' toggle", async () => {
    mockApi(summary());
    const user = userEvent.setup();
    renderPage(<UsagePage />);
    await screen.findByText("$1,240.50");

    const toggle = screen.getByRole("button", { name: "Add filter" });
    const region = document.getElementById(toggle.getAttribute("aria-controls")!)!;
    // jsdom does not apply Tailwind's `.hidden`, so assert on the class the toggle
    // flips (display:none collapsed, flex expanded) rather than computed visibility.
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(region.className).toContain("hidden");

    await user.click(toggle);

    expect(toggle).toHaveAttribute("aria-expanded", "true");
    expect(region.className).toContain("flex");
    expect(region.className).not.toContain("hidden");
  });

  it("surfaces an active filter as a removable chip", async () => {
    mockApi(summary());
    const user = userEvent.setup();
    renderPage(<UsagePage />);
    await screen.findByText("$1,240.50");

    // No entity filters yet, so no chips.
    expect(screen.queryByRole("button", { name: /Remove .* filter/ })).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Add filter" }));
    await user.click(screen.getByPlaceholderText("All keys"));
    await user.click(await screen.findByRole("option", { name: "ci-bot" }));

    // The picked key shows as a chip with a remove control.
    expect(await screen.findByRole("button", { name: "Remove API key filter" })).toBeInTheDocument();
  });
});
