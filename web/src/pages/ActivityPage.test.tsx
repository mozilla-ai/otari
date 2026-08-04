import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactElement } from "react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { UsageEntry } from "@/api/types";
import { ActivityPage } from "@/pages/ActivityPage";

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
    status_code: null,
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
        by_source: Array.from(new Set(rows.map((r) => r.source))).map((s) => ({
          key: s,
          cost: 0,
          tokens: 0,
          requests: 0,
          is_other: false,
        })),
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

  it("asks the summary endpoint only for the breakdowns it reads", async () => {
    // Two summary reads back this page: the model typeahead (by_model) and the
    // timeline histogram (series only). Each breakdown is a separate GROUP BY over
    // the window server-side, so neither may request the full set.
    const { calls } = mockApi({ rows: [entry()] });
    renderPage(<ActivityPage />);

    await screen.findByText("gpt-4o");
    const summaryCalls = calls.filter((c) => c.url.includes("/v1/usage/summary")).map((c) => c.url);
    expect(summaryCalls.length).toBeGreaterThan(0);
    expect(summaryCalls.some((url) => url.includes("dimensions=model"))).toBe(true);
    expect(summaryCalls.some((url) => url.includes("dimensions=none"))).toBe(true);
    // No caller here reads a session/provider/user breakdown.
    expect(summaryCalls.some((url) => url.includes("dimensions=source_label"))).toBe(false);
    expect(summaryCalls.every((url) => url.includes("dimensions="))).toBe(true);
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

  it("honors a session drill-down and shows it as a clearable chip", async () => {
    // The Usage page's session breakdown links here scoped to one source_label.
    // Without the filter the log would silently show every session's requests.
    const { calls } = mockApi({ rows: [entry({ source: "claude_code", source_label: "sess-1" })] });
    renderPage(<ActivityPage />, "/activity?source_label=sess-1");

    await screen.findByText("gpt-4o");
    expect(listCalls(calls).some((url) => url.includes("source_label=sess-1"))).toBe(true);

    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: /Session/ }));
    await waitFor(() => expect(listCalls(calls).at(-1)).not.toContain("source_label="));
  });

  it("honors endpoint and provider drill-downs", async () => {
    const { calls } = mockApi({ rows: [entry()] });
    renderPage(<ActivityPage />, "/activity?endpoint=%2Fv1%2Fmessages&provider=anthropic");

    await screen.findByText("gpt-4o");
    const urls = listCalls(calls);
    expect(urls.some((url) => url.includes("endpoint=%2Fv1%2Fmessages"))).toBe(true);
    expect(urls.some((url) => url.includes("provider=anthropic"))).toBe(true);
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

  it("opens an error row's detail and shows the diagnostic with its status code", async () => {
    const user = userEvent.setup();
    mockApi({
      rows: [entry({ status: "error", error_message: "provider exploded: quota exceeded", status_code: 502 })],
    });
    renderPage(<ActivityPage />);

    const row = (await screen.findByText("gpt-4o")).closest("tr")!;
    expect(within(row).getByText("error")).toBeInTheDocument();

    await user.click(row);
    // The dashboard is admin-only, so the stored error text is shown verbatim,
    // with the classifying HTTP status alongside the "Error" heading.
    expect(screen.getByText("provider exploded: quota exceeded")).toBeInTheDocument();
    expect(screen.getByText("Error (502)")).toBeInTheDocument();
  });

  it("omits the status code from the error heading when none was recorded", async () => {
    const user = userEvent.setup();
    mockApi({ rows: [entry({ status: "error", error_message: "stream completed without usage data", status_code: null })] });
    renderPage(<ActivityPage />);

    const row = (await screen.findByText("gpt-4o")).closest("tr")!;
    await user.click(row);
    expect(screen.getByText("stream completed without usage data")).toBeInTheDocument();
    // Bare heading, no "(code)" suffix; scope to a span so the status filter's
    // <option>Error</option> does not match.
    expect(screen.getByText("Error", { selector: "span" })).toBeInTheDocument();
    expect(screen.queryByText(/Error \(/)).not.toBeInTheDocument();
  });

  it("copies a request id out of the detail panel", async () => {
    // The id an operator pastes into a log search or a support thread, where a
    // mistyped character makes it useless.
    const user = userEvent.setup();
    mockApi({ rows: [entry({ id: "3ba12b77-8841-42a5-b776-a0a1aacb347f" })] });
    renderPage(<ActivityPage />);

    await user.click((await screen.findByText("gpt-4o")).closest("tr")!);
    await user.click(screen.getByRole("button", { name: "Copy request id" }));

    expect(await navigator.clipboard.readText()).toBe("3ba12b77-8841-42a5-b776-a0a1aacb347f");
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

  it("offers the sources seen in the window and sends the picked one to the API", async () => {
    const { calls } = mockApi({
      rows: [entry(), entry({ id: "imp", model: "claude-sonnet-4", source: "claude_code", counts_toward_budget: false })],
    });
    const user = userEvent.setup();
    renderPage(<ActivityPage />);
    await screen.findByText("gpt-4o");

    // Options come from the log itself (the summary's provenance breakdown), with
    // friendly labels for the sources the page knows about.
    const select = screen.getByLabelText("Source");
    await waitFor(() => expect(within(select).getByRole("option", { name: "Claude Code" })).toBeInTheDocument());
    expect(within(select).getByRole("option", { name: "Gateway" })).toBeInTheDocument();

    await user.selectOptions(select, "claude_code");
    await waitFor(() => expect(listCalls(calls).at(-1)).toContain("source=claude_code"));
  });

  it("keeps a drill-down source listed even when the window holds none of its rows", async () => {
    // The select must show the filter that is actually applied, or the operator
    // sees a chip they cannot find in the picker.
    mockApi({ rows: [entry({ source: "gateway" })] });
    renderPage(<ActivityPage />, "/activity?source=codex");
    await screen.findByText("gpt-4o");

    const select = screen.getByLabelText("Source");
    expect(within(select).getByRole("option", { name: "Codex" })).toBeInTheDocument();
    expect(select).toHaveValue("codex");
  });

  it("surfaces a timeline summary failure instead of an empty strip", async () => {
    // If the series query fails, "No activity in this range" would misread as a
    // quiet gateway; the error banner must carry the failure.
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.includes("/v1/usage/summary")) {
        return jsonResponse({ detail: "summary exploded" }, 500);
      }
      if (url.includes("/v1/usage/count")) return jsonResponse({ total: 1 });
      if (url.includes("/v1/usage")) return jsonResponse([entry()]);
      return jsonResponse([]);
    });
    renderPage(<ActivityPage />);
    await screen.findByText("gpt-4o");

    expect(await screen.findByText(/summary exploded/)).toBeInTheDocument();
  });

  it("hides the source picker while only one source exists", async () => {
    // Most gateways only ever see their own traffic; a provenance select with a
    // single option is noise, so it only appears once a second source shows up.
    mockApi({ rows: [entry(), entry({ id: "b" })] });
    renderPage(<ActivityPage />);
    await screen.findAllByText("gpt-4o");

    await waitFor(() => expect(screen.queryByLabelText("Source")).not.toBeInTheDocument());
  });

  it("shows a row's token composition rather than one uninformative total", async () => {
    // A cached agent request: the total is ~98% cache read, so the total alone
    // makes every row look alike. The bar carries the split.
    mockApi({
      rows: [
        entry({
          prompt_tokens: 100_000,
          completion_tokens: 500,
          total_tokens: 100_500,
          cache_read_tokens: 98_000,
          cache_write_tokens: 1_500,
          billing_meters: {
            total_input_tokens: 100_000,
            fresh_input_tokens: 500,
            cache_read_tokens: 98_000,
            cache_write_tokens: 1_500,
            cache_write_1h_tokens: 0,
            completion_tokens: 500,
          },
        }),
      ],
    });
    renderPage(<ActivityPage />);

    const row = (await screen.findByText("gpt-4o")).closest("tr")!;
    expect(within(row).getByText("100,500")).toBeInTheDocument();
    const bar = within(row).getByRole("img", { name: /Token composition/ });
    expect(bar).toHaveAccessibleName(
      "Token composition: Fresh input 500, Cache read 98,000, Cache write 1,500, Output 500",
    );

    // Four segments, widest being the cache read, so the shape is what the eye
    // compares between rows.
    const widths = [...bar.querySelectorAll("rect")].map((r) => Number(r.getAttribute("width")));
    expect(widths).toHaveLength(4);
    expect(Math.max(...widths)).toBeCloseTo((98_000 / 100_500) * 100, 5);
    expect(widths.reduce((a, b) => a + b, 0)).toBeCloseTo(100, 5);
  });

  it("explains the column's total in the detail panel when it exceeds the raw one", async () => {
    // An additive-convention row reports its cache buckets outside the prompt, so
    // the billed total the column shows is far above the stored `total_tokens`.
    // Both are spelled out, or the two numbers look like a bug.
    const user = userEvent.setup();
    mockApi({
      rows: [
        entry({
          prompt_tokens: 1_000,
          completion_tokens: 200,
          total_tokens: 1_200,
          cache_read_tokens: 98_000,
          cache_write_tokens: 1_500,
          billing_meters: {
            total_input_tokens: 100_500,
            fresh_input_tokens: 1_000,
            cache_read_tokens: 98_000,
            cache_write_tokens: 1_500,
            cache_write_1h_tokens: 0,
            completion_tokens: 200,
          },
        }),
      ],
    });
    renderPage(<ActivityPage />);

    const row = (await screen.findByText("gpt-4o")).closest("tr")!;
    await user.click(row);

    const field = (label: string): string => screen.getByText(label).parentElement!.textContent ?? "";
    expect(field("Total tokens")).toContain("1,200");
    expect(field("Billed tokens")).toContain("100,700");
    // Which is the number the row itself shows.
    expect(within(row).getByText("100,700")).toBeInTheDocument();
  });

  it("splits an unmetered row from its raw columns, and shows no bar without usage", async () => {
    // An unpriced row carries no billing meters, so the composition falls back to
    // the raw columns (cache read counted inside the prompt).
    mockApi({
      rows: [
        entry({
          id: "unpriced",
          model: "unpriced-model",
          prompt_tokens: 1_000,
          completion_tokens: 200,
          total_tokens: 1_200,
          cache_read_tokens: 400,
          billing_meters: null,
          cost: null,
        }),
        entry({ id: "failed", model: "failed-model", status: "error", prompt_tokens: null, completion_tokens: null, total_tokens: null }),
      ],
    });
    renderPage(<ActivityPage />);

    const unpriced = (await screen.findByText("unpriced-model")).closest("tr")!;
    expect(within(unpriced).getByRole("img", { name: /Token composition/ })).toHaveAccessibleName(
      "Token composition: Fresh input 600, Cache read 400, Output 200",
    );

    // A request that failed before the provider reported usage has nothing to
    // compose, so the cell stays an em-dash instead of drawing an empty bar.
    const failed = screen.getByText("failed-model").closest("tr")!;
    expect(within(failed).queryByRole("img", { name: /Token composition/ })).not.toBeInTheDocument();
    expect(within(failed).getByText("—")).toBeInTheDocument();
  });

  it("opens a bookmarked deep page on that page", async () => {
    // The URL is the source of truth for `page`, but the mount effect used to
    // re-anchor the rolling window a few milliseconds later, which changed the
    // filter set and reset the page: every shared `?page=3` link opened on page 1.
    const { calls } = mockApi({ rows: Array.from({ length: 50 }, (_, i) => entry({ id: `r${i}` })), total: 500 });
    renderPage(<ActivityPage />, "/activity?page=2");

    await screen.findAllByText("gpt-4o");
    expect(await screen.findByText("101–150 of 500")).toBeInTheDocument();
    expect(listCalls(calls).every((url) => url.includes("skip=100"))).toBe(true);
  });

  it("keeps the current page when refreshing", async () => {
    // Refresh used to re-anchor the rolling window, which changed the filter set,
    // which reset the page: pressing it on page 3 dropped you back to page 1.
    const { calls } = mockApi({ rows: Array.from({ length: 50 }, (_, i) => entry({ id: `r${i}` })), total: 500 });
    const user = userEvent.setup();
    renderPage(<ActivityPage />, "/activity?page=2");
    await screen.findAllByText("gpt-4o");
    expect(await screen.findByText("101–150 of 500")).toBeInTheDocument();

    const before = listCalls(calls).length;
    const button = screen.getByRole("button", { name: "Refresh" });
    await waitFor(() => expect(button).toBeEnabled());
    await user.click(button);

    // The list is refetched, and every fetch stays on the third page's offset.
    await waitFor(() => expect(listCalls(calls).length).toBeGreaterThan(before));
    expect(listCalls(calls).every((url) => url.includes("skip=100"))).toBe(true);
    expect(screen.getByText("101–150 of 500")).toBeInTheDocument();
  });

  it("re-anchors a rolling window when its preset is re-picked", async () => {
    // Refresh no longer moves the window, so re-selecting the active preset is the
    // gesture that advances a rolling range to "now".
    const { calls } = mockApi({ rows: [entry()] });
    const user = userEvent.setup();
    renderPage(<ActivityPage />);
    await screen.findByText("gpt-4o");

    const startOf = (url: string): string | null => new URL(url, "http://x").searchParams.get("start_date");
    const before = startOf(listCalls(calls).at(-1)!);
    expect(before).not.toBeNull();

    await user.click(screen.getByRole("button", { name: "24h" }));

    await waitFor(() => expect(startOf(listCalls(calls).at(-1)!)).not.toBe(before));
  });

  it("distinguishes filtered-empty from never-used", async () => {
    const user = userEvent.setup();
    mockApi({ rows: [], total: 0 });
    renderPage(<ActivityPage />);

    // The default 24h preset is not itself a filter (mirroring UsagePage), so an
    // empty result on a brand-new gateway reads as "never used", not "filtered".
    expect(await screen.findByText("No requests recorded yet.")).toBeInTheDocument();

    // The unbounded "All" applies no window either, so it stays "never used"
    // rather than flipping to filtered-empty.
    await user.click(screen.getByRole("button", { name: "All" }));
    expect(await screen.findByText("No requests recorded yet.")).toBeInTheDocument();

    // Narrowing to a bounded non-default preset is a real time filter, so an empty
    // result then reads as filtered-to-empty.
    await user.click(screen.getByRole("button", { name: "7d" }));
    expect(await screen.findByText("No requests match these filters.")).toBeInTheDocument();
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

  it("carries the drill-down filters into an 'all matching' delete", async () => {
    // The count that sizes "select all N" is taken under the source/session/provider
    // scope, so the delete body has to repeat it. If it does not, the server
    // re-derives a wider set: omitting `source` alone widened the target from one
    // imported source to every imported row in the window.
    const user = userEvent.setup();
    const { calls } = mockApi({
      rows: [entry({ id: "imp-1", model: "imported-model", counts_toward_budget: false })],
      total: 5,
    });
    renderPage(
      <ActivityPage />,
      "/activity?source=claude_code&source_label=task-42&provider=anthropic&endpoint=external",
    );

    const row = (await screen.findByText("imported-model")).closest("tr")!;
    await user.click(within(row).getByRole("checkbox"));
    await user.click(await screen.findByRole("button", { name: /Select all 5 matching/ }));

    await user.click(screen.getByRole("button", { name: "Delete" }));
    const dialog = await screen.findByRole("alertdialog");
    await user.click(within(dialog).getByRole("button", { name: "Delete" }));

    await waitFor(() => {
      const del = calls.find((c) => c.url.endsWith("/v1/usage") && c.method === "DELETE");
      expect(del).toBeTruthy();
      const body = JSON.parse(del!.body ?? "{}");
      expect(body.by_filter).toBe(true);
      expect(body.source).toBe("claude_code");
      expect(body.source_label).toBe("task-42");
      expect(body.provider).toBe("anthropic");
      expect(body.endpoint).toBe("external");
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

  it("hides the selection column when nothing on the page can be selected", async () => {
    // A gateway-only deployment has no imported rows, so every checkbox would
    // render disabled: a column of dead controls rather than an explanation.
    mockApi({ rows: [entry({ id: "gw", model: "gateway-model", counts_toward_budget: true })] });
    renderPage(<ActivityPage />);

    await screen.findByText("gateway-model");
    expect(screen.queryAllByRole("checkbox")).toHaveLength(0);
  });

  it("prices the model from a request that carried no cost", async () => {
    const user = userEvent.setup();
    // A row stores the instance and the bare model separately, so the pricing
    // key has to be rebuilt from both: the model alone is prefix-less and the
    // dialog would (rightly) refuse it.
    const { calls } = mockApi({
      rows: [entry({ id: "free", model: "mistral-small", provider: "vllm", cost: null })],
    });
    renderPage(<ActivityPage />);

    const row = (await screen.findByText("mistral-small")).closest("tr")!;
    await user.click(row);
    await user.click(screen.getByRole("button", { name: "Price this model" }));

    const dialog = await screen.findByRole("alertdialog");
    expect(within(dialog).getByLabelText("Model key")).toHaveValue("vllm:mistral-small");
    await user.type(within(dialog).getByLabelText("Input $ / 1M"), "0.2");
    await user.type(within(dialog).getByLabelText("Output $ / 1M"), "0.6");
    await user.click(within(dialog).getByRole("button", { name: "Set price" }));

    await waitFor(() => {
      const call = calls.find((c) => c.url.includes("/v1/pricing") && c.method === "POST");
      expect(call).toBeTruthy();
      expect(JSON.parse(call!.body!)).toMatchObject({
        model_key: "vllm:mistral-small",
        input_price_per_million: 0.2,
        output_price_per_million: 0.6,
      });
    });
    // Setting the model's price must not rewrite what logged rows were billed.
    expect(calls.some((c) => c.url.includes("/v1/usage/set-price"))).toBe(false);
  });

  it("does not offer model pricing on a request that was costed", async () => {
    const user = userEvent.setup();
    mockApi({ rows: [entry({ id: "paid", model: "gpt-4o", provider: "openai", cost: 0.5 })] });
    renderPage(<ActivityPage />);

    const row = (await screen.findByText("gpt-4o")).closest("tr")!;
    await user.click(row);

    expect(screen.getByText("Request detail")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Price this model" })).not.toBeInTheDocument();
  });

  it("treats a $0 cost as priced, not as a model needing a price", async () => {
    // cost=0 is a real price (a model priced at zero), which is why the backend
    // marks a row unpriced on cost IS NULL rather than on falsiness.
    const user = userEvent.setup();
    mockApi({ rows: [entry({ id: "free-model", model: "mistral-small", provider: "vllm", cost: 0 })] });
    renderPage(<ActivityPage />);

    const row = (await screen.findByText("mistral-small")).closest("tr")!;
    await user.click(row);

    expect(screen.getByText("Request detail")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Price this model" })).not.toBeInTheDocument();
  });

  it("prices a selector that never resolved from the row's model alone", async () => {
    // A selector the gateway could not resolve is logged with no provider and
    // the raw selector as the model, so it is already the key to price.
    const user = userEvent.setup();
    mockApi({
      rows: [entry({ id: "unresolved", model: "vllm:mistral-small", provider: null, cost: null })],
    });
    renderPage(<ActivityPage />);

    const row = (await screen.findByText("vllm:mistral-small")).closest("tr")!;
    await user.click(row);
    await user.click(screen.getByRole("button", { name: "Price this model" }));

    const dialog = await screen.findByRole("alertdialog");
    expect(within(dialog).getByLabelText("Model key")).toHaveValue("vllm:mistral-small");
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

    // The histogram, however, sends an explicit start bound: without one the
    // summary endpoint would apply a hidden 30-day default, so the bars would show
    // a rolling month while the caption reads "All time". The list stays all-time.
    expect(
      calls.some((c) => c.url.includes("/v1/usage/summary") && c.url.includes("start_date=")),
    ).toBe(true);
  });

  it("gives the histogram an explicit start for the custom-range sentinel", async () => {
    const { calls } = mockApi({ rows: [entry()] });
    // `?range=custom` has no rolling window of its own, so without an explicit
    // extent the summary would fall back to the server's hidden 30-day default.
    renderPage(<ActivityPage />, "/activity?range=custom");
    await screen.findByText("gpt-4o");

    await waitFor(() =>
      expect(
        calls.some((c) => c.url.includes("/v1/usage/summary") && c.url.includes("start_date=")),
      ).toBe(true),
    );
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
    // The caption reflects the drilled window (end shown inclusively). Assert on
    // day numbers and the UTC marker, not a month abbreviation, since the caption
    // formats with the runtime locale ("Jul" would fail outside en-US).
    const caption = (screen.getByText(/Showing/).textContent ?? "").replace(/\s+/g, " ");
    expect(caption).toMatch(/\b1\b/);
    expect(caption).toMatch(/\b14\b/);
    expect(caption).toContain("UTC");
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
