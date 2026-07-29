import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactElement } from "react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { UsageEntry } from "@/api/types";
import { ActivityPage, copyToClipboard } from "@/pages/ActivityPage";

function entry(overrides: Partial<UsageEntry> = {}): UsageEntry {
  return {
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
    latency_ms: 842,
    source: "gateway",
    source_label: null,
    counts_toward_budget: true,
    ...overrides,
  };
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

interface FetchCall {
  url: string;
  method: string;
  body: string | undefined;
}

// Mock fetch for the usage list/count/summary reads plus the delete and
// set-price mutations. Records every call so tests can assert URLs and bodies.
function mockApi(opts: { rows?: UsageEntry[]; total?: number } = {}) {
  const rows = opts.rows ?? [];
  const total = opts.total ?? rows.length;
  const calls: FetchCall[] = [];

  const mock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input);
    const method = (init?.method ?? "GET").toUpperCase();
    calls.push({ url, method, body: typeof init?.body === "string" ? init.body : undefined });

    if (url.endsWith("/v1/usage") && method === "DELETE") {
      return jsonResponse({ deleted: 1 });
    }
    if (url.includes("/v1/usage/set-price")) {
      return jsonResponse({ matched: 1, updated: 1, unchanged: 0 });
    }
    if (url.includes("/v1/usage/count")) {
      return jsonResponse({ total });
    }
    if (url.includes("/v1/usage/summary")) {
      const models = Array.from(new Set(rows.map((r) => r.model)));
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
        by_model: models.map((m) => ({ key: m, cost: 0, tokens: 0, requests: 0, is_other: false })),
        by_user: [],
        by_api_key: [],
        series: [],
      });
    }
    if (url.includes("/v1/usage")) {
      return jsonResponse(rows);
    }
    if (url.includes("/v1/users") || url.includes("/v1/keys")) {
      return jsonResponse([]);
    }
    return jsonResponse([]);
  });

  return { mock, calls };
}

function renderPage(ui: ReactElement, route = "/activity") {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[route]}>{ui}</MemoryRouter>
    </QueryClientProvider>,
  );
}

// Only the list requests (not /count, /summary, or mutations) carry the
// pagination + filter params.
function listCalls(calls: FetchCall[]): string[] {
  return calls
    .filter(
      (c) =>
        c.method === "GET" &&
        c.url.includes("/v1/usage") &&
        !c.url.includes("/count") &&
        !c.url.includes("/summary") &&
        !c.url.includes("/set-price"),
    )
    .map((c) => c.url);
}

describe("ActivityPage", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders a request row with humanized latency, tokens, and status", async () => {
    mockApi({ rows: [entry({ total_tokens: 1500, latency_ms: 842, cost: 0.0123 })] });
    renderPage(<ActivityPage />);

    const row = (await screen.findByText("gpt-4o")).closest("tr")!;
    expect(within(row).getByText("1,500")).toBeInTheDocument();
    expect(within(row).getByText("842 ms")).toBeInTheDocument();
    expect(within(row).getByText("$0.0123")).toBeInTheDocument();
    expect(within(row).getByText("success")).toBeInTheDocument();
  });

  it("shows the api key column, and an em-dash for master-key rows", async () => {
    mockApi({
      rows: [
        entry({ id: "g", model: "gateway-model", api_key_id: "key-1" }),
        entry({ id: "x", model: "imported-model", api_key_id: null }),
      ],
    });
    renderPage(<ActivityPage />);

    const importedRow = (await screen.findByText("imported-model")).closest("tr")!;
    expect(within(importedRow).getByText("—")).toBeInTheDocument();
  });

  it("sends the api key filter to the API", async () => {
    const { calls } = mockApi({ rows: [entry({ api_key_id: "key-1" })] });
    renderPage(<ActivityPage />, "/activity?api_key_id=key-1");

    await screen.findByText("gpt-4o");
    expect(listCalls(calls).some((url) => url.includes("api_key_id=key-1"))).toBe(true);
  });

  it("honors a source drill-down and shows it as a clearable chip", async () => {
    // The pricing alarm links here scoped to gateway traffic. The param has no
    // select of its own, so if the page ignored it the banner's count and this
    // list would disagree, and the scoping would be invisible.
    const { calls } = mockApi({ rows: [entry({ status: "error" })] });
    renderPage(<ActivityPage />, "/activity?status=error&range=1h&source=gateway");

    await screen.findByText("gpt-4o");
    expect(listCalls(calls).some((url) => url.includes("source=gateway"))).toBe(true);

    const user = userEvent.setup();
    const chip = screen.getByRole("button", { name: /Source/ });
    await user.click(chip);
    await waitFor(() => expect(listCalls(calls).at(-1)).not.toContain("source="));
  });

  it("renders latency over a second as seconds and null latency as an em-dash", async () => {
    mockApi({
      rows: [
        entry({ id: "a", model: "slow-model", latency_ms: 1420 }),
        entry({ id: "b", model: "batch-model", latency_ms: null }),
      ],
    });
    renderPage(<ActivityPage />);

    const slow = (await screen.findByText("slow-model")).closest("tr")!;
    expect(within(slow).getByText("1.42 s")).toBeInTheDocument();
    const batch = screen.getByText("batch-model").closest("tr")!;
    expect(within(batch).getByText("—")).toBeInTheDocument();
  });

  it("opens an error row's detail without exposing provider diagnostics", async () => {
    const user = userEvent.setup();
    mockApi({ rows: [entry({ status: "error", error_message: "provider exploded: quota exceeded" })] });
    renderPage(<ActivityPage />);

    const row = (await screen.findByText("gpt-4o")).closest("tr")!;
    expect(within(row).getByText("error")).toBeInTheDocument();

    await user.click(row);
    // Source-neutral: gateway-side rejections land in this list too, so the
    // summary must not blame the provider for every failure.
    expect(screen.getByText("This request failed. Inspect gateway logs for details.")).toBeInTheDocument();
    expect(screen.queryByText("provider exploded: quota exceeded")).not.toBeInTheDocument();
  });

  it("opens the detail inline directly under the clicked row, and Close collapses it", async () => {
    // Regression: the shared-table migration rendered the detail below the
    // whole table, so on a full page a row click looked like it did nothing.
    const user = userEvent.setup();
    mockApi({ rows: [entry({ id: "r1" }), entry({ id: "r2", model: "gpt-4o-mini" })] });
    renderPage(<ActivityPage />);

    const row = (await screen.findByText("gpt-4o")).closest("tr")!;
    await user.click(row);

    expect(screen.getByText("Request detail")).toBeInTheDocument();
    // The panel is the row's next sibling (accordion), not a card after the table.
    expect(row.nextElementSibling?.textContent).toContain("Request detail");

    await user.click(screen.getByRole("button", { name: "Close" }));
    expect(screen.queryByText("Request detail")).not.toBeInTheDocument();
  });

  it("reports copying only after the clipboard write succeeds", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    const copied = await copyToClipboard("provider exploded", { writeText });
    expect(writeText).toHaveBeenCalledWith("provider exploded");
    expect(copied).toBe(true);
  });

  it("does not report success when the clipboard write fails", async () => {
    const writeText = vi.fn().mockRejectedValue(new Error("clipboard denied"));
    const copied = await copyToClipboard("provider exploded", { writeText });
    expect(writeText).toHaveBeenCalledWith("provider exploded");
    expect(copied).toBe(false);
  });

  it("sends the status filter to the API", async () => {
    const { calls } = mockApi({ rows: [entry()] });
    const user = userEvent.setup();
    renderPage(<ActivityPage />);

    await screen.findByText("gpt-4o");
    await user.selectOptions(screen.getByLabelText("Status"), "error");

    await waitFor(() => expect(listCalls(calls).at(-1)).toContain("status=error"));
  });

  it("sends the priced filter to the API", async () => {
    const { calls } = mockApi({ rows: [entry()] });
    const user = userEvent.setup();
    renderPage(<ActivityPage />);

    await screen.findByText("gpt-4o");
    await user.selectOptions(screen.getByLabelText("Priced?"), "false");

    await waitFor(() => expect(listCalls(calls).at(-1)).toContain("priced=false"));
  });

  it("distinguishes filtered-empty from never-used", async () => {
    const user = userEvent.setup();
    mockApi({ rows: [], total: 0 });
    renderPage(<ActivityPage />);

    // The default 24h window bounds the query, so an empty result reads as filtered.
    expect(await screen.findByText("No requests match these filters.")).toBeInTheDocument();

    // The truthful "All" preset drops the window entirely (unbounded), so an empty
    // result now reads as "never used".
    await user.click(screen.getByRole("button", { name: "All" }));
    expect(await screen.findByText("No requests recorded yet.")).toBeInTheDocument();
  });

  it("keeps Next reachable when the count request fails", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.includes("/v1/usage/count")) {
        return jsonResponse({ detail: "boom" }, 500);
      }
      if (url.includes("/v1/usage/summary")) {
        return jsonResponse({ by_model: [], by_user: [], by_api_key: [], series: [] });
      }
      if (url.includes("/v1/usage")) {
        return jsonResponse(Array.from({ length: 50 }, (_, i) => entry({ id: `r${i}` })));
      }
      return jsonResponse([]);
    });
    renderPage(<ActivityPage />);

    await screen.findAllByText("gpt-4o");
    expect(screen.getByRole("button", { name: "Next page" })).toBeEnabled();
    expect(screen.getByText("1–50")).toBeInTheDocument();
    expect(screen.queryByText("0 of 0")).not.toBeInTheDocument();
  });

  it("seeds filters from the drill-down query string", async () => {
    const { calls } = mockApi({ rows: [entry()] });
    renderPage(<ActivityPage />, "/activity?model=gpt-4o&user_id=alice&status=error");

    await screen.findByText("gpt-4o");
    const latest = listCalls(calls).at(-1)!;
    expect(latest).toContain("model=gpt-4o");
    expect(latest).toContain("user_id=alice");
    expect(latest).toContain("status=error");
  });

  it("snaps URL-supplied page sizes to the nearest offered option", async () => {
    // An old size=500 bookmark must not resurrect second-long selection
    // clicks, and a hand-edited size=-5 must not reach the API as a bad limit.
    const { calls } = mockApi({ rows: [entry()] });
    renderPage(<ActivityPage />, "/activity?size=500");

    await screen.findByText("gpt-4o");
    expect(listCalls(calls).at(-1)).toContain("limit=100");

    const { calls: negativeCalls } = mockApi({ rows: [entry()] });
    renderPage(<ActivityPage />, "/activity?size=-5");
    await waitFor(() => expect(listCalls(negativeCalls).length).toBeGreaterThan(0));
    expect(listCalls(negativeCalls).at(-1)).toContain("limit=25");
  });

  it("shows the paginator range and total", async () => {
    mockApi({ rows: Array.from({ length: 50 }, (_, i) => entry({ id: `r${i}` })), total: 120 });
    renderPage(<ActivityPage />);

    expect(await screen.findByText("1–50 of 120")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Previous page" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Next page" })).toBeEnabled();
  });

  it("only lets imported rows be selected", async () => {
    mockApi({
      rows: [
        entry({ id: "gw", model: "gateway-model", counts_toward_budget: true }),
        entry({ id: "imp", model: "imported-model", counts_toward_budget: false }),
      ],
    });
    renderPage(<ActivityPage />);

    const gatewayRow = (await screen.findByText("gateway-model")).closest("tr")!;
    const importedRow = screen.getByText("imported-model").closest("tr")!;
    expect(within(gatewayRow).getByRole("checkbox")).toBeDisabled();
    expect(within(importedRow).getByRole("checkbox")).toBeEnabled();
  });

  it("deletes the selected imported rows by id", async () => {
    const user = userEvent.setup();
    const { calls } = mockApi({
      rows: [entry({ id: "imp-1", model: "imported-model", counts_toward_budget: false })],
      total: 1,
    });
    renderPage(<ActivityPage />);

    const row = (await screen.findByText("imported-model")).closest("tr")!;
    await user.click(within(row).getByRole("checkbox"));

    // Bulk bar appears with the page selection count.
    expect(await screen.findByText("1 selected")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Delete" }));

    // Confirm in the dialog.
    const dialog = await screen.findByRole("alertdialog");
    await user.click(within(dialog).getByRole("button", { name: "Delete" }));

    await waitFor(() => {
      const del = calls.find((c) => c.url.endsWith("/v1/usage") && c.method === "DELETE");
      expect(del).toBeTruthy();
      expect(del!.body).toContain("imp-1");
    });
  });

  it("sets a manual price on the selected imported rows", async () => {
    const user = userEvent.setup();
    const { calls } = mockApi({
      rows: [entry({ id: "imp-1", model: "imported-model", counts_toward_budget: false })],
      total: 1,
    });
    renderPage(<ActivityPage />);

    const row = (await screen.findByText("imported-model")).closest("tr")!;
    await user.click(within(row).getByRole("checkbox"));
    await user.click(screen.getByRole("button", { name: "Set price" }));

    const dialog = await screen.findByRole("alertdialog");
    await user.type(within(dialog).getByLabelText("Input $ / 1M"), "3");
    await user.type(within(dialog).getByLabelText("Output $ / 1M"), "15");
    await user.click(within(dialog).getByRole("button", { name: "Set price" }));

    await waitFor(() => {
      const priceCall = calls.find((c) => c.url.includes("/v1/usage/set-price") && c.method === "POST");
      expect(priceCall).toBeTruthy();
      expect(priceCall!.body).toContain("imp-1");
      expect(priceCall!.body).toContain("\"input_price_per_million\":3");
      expect(priceCall!.body).toContain("\"output_price_per_million\":15");
    });
  });

  it("keeps the filter pickers behind an 'Add filter' toggle", async () => {
    mockApi({ rows: [entry()] });
    const user = userEvent.setup();
    renderPage(<ActivityPage />);
    await screen.findByText("gpt-4o");

    // The picker row is collapsed until the operator opts to add a filter. jsdom
    // does not apply Tailwind's `.hidden`, so assert on the toggled class rather
    // than computed visibility.
    const toggle = screen.getByRole("button", { name: "Add filter" });
    const region = document.getElementById(toggle.getAttribute("aria-controls")!)!;
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(region.className).toContain("hidden");

    await user.click(toggle);

    expect(toggle).toHaveAttribute("aria-expanded", "true");
    expect(region.className).toContain("flex");
    expect(region.className).not.toContain("hidden");
  });

  it("shows active filters as removable chips and clears one on ✕", async () => {
    const user = userEvent.setup();
    const { calls } = mockApi({ rows: [entry()] });
    renderPage(<ActivityPage />, "/activity?model=gpt-4o&status=error");
    await screen.findByText("gpt-4o");

    // A chip per active entity filter (model + status); the time range is not a chip.
    expect(screen.getByRole("button", { name: "Remove Model filter" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Remove Status filter" })).toBeInTheDocument();

    // Removing the model chip drops just that filter from the query.
    await user.click(screen.getByRole("button", { name: "Remove Model filter" }));
    await waitFor(() => expect(listCalls(calls).some((url) => !url.includes("model=gpt-4o"))).toBe(true));
  });

  it("queries an unbounded window for the truthful 'All' preset", async () => {
    const user = userEvent.setup();
    const { calls } = mockApi({ rows: [entry()] });
    renderPage(<ActivityPage />);
    await screen.findByText("gpt-4o");

    await user.click(screen.getByRole("button", { name: "All" }));

    // Activity's list endpoint applies no default lookback, so "All" really omits
    // the start bound rather than silently scoping to a recent window.
    await waitFor(() => expect(listCalls(calls).some((url) => !url.includes("start_date"))).toBe(true));
  });

  it("frames a drill-down window that reaches outside the preset extent", async () => {
    const { calls } = mockApi({ rows: [entry()] });
    // A Usage-page drill-down: explicit multi-week bounds while `range` stays the
    // 24h default. The timeline must frame the drilled window (daily buckets over
    // its bounds), not the unrelated 24h extent.
    renderPage(<ActivityPage />, "/activity?start_date=2020-07-01T00:00:00.000Z&end_date=2020-07-15T00:00:00.000Z");
    await screen.findByText("gpt-4o");

    await waitFor(() =>
      expect(
        calls.some(
          (c) =>
            c.url.includes("/v1/usage/summary") &&
            c.url.includes("bucket=day") &&
            c.url.includes("start_date=2020-07-01") &&
            c.url.includes("end_date=2020-07-15"),
        ),
      ).toBe(true),
    );
    // The caption reflects the drilled window (end shown inclusively).
    expect(screen.getByText(/Showing/)).toHaveTextContent("Jul 1");
    expect(screen.getByText(/Showing/)).toHaveTextContent("Jul 14");
  });

  it("buckets the timeline histogram by the active preset's extent", async () => {
    const { calls } = mockApi({ rows: [entry()] });
    renderPage(<ActivityPage />); // default 24h
    await screen.findByText("gpt-4o");

    // The 24h extent buckets hourly, so the timeline's context summary is fetched
    // with bucket=hour (distinct from the day-bucketed model-suggestion summary).
    await waitFor(() =>
      expect(calls.some((c) => c.url.includes("/v1/usage/summary") && c.url.includes("bucket=hour"))).toBe(true),
    );
  });
});
