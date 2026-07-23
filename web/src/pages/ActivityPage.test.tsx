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

// Mock fetch for /v1/usage (list), /v1/usage/count, and /v1/users. Records every
// list URL so tests can assert the query params the filters produced.
function mockApi(opts: { rows?: UsageEntry[]; total?: number } = {}) {
  const rows = opts.rows ?? [];
  const total = opts.total ?? rows.length;

  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input);
    if (url.includes("/v1/usage/count")) {
      return jsonResponse({ total });
    }
    if (url.includes("/v1/usage/summary")) {
      // Model-suggestion query: distinct models drawn from the mocked rows.
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
    if (url.includes("/v1/users")) {
      return jsonResponse([]);
    }
    return jsonResponse([]);
  });
}

function renderPage(ui: ReactElement, route = "/activity") {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[route]}>{ui}</MemoryRouter>
    </QueryClientProvider>,
  );
}

// Only the list requests (not /count, not the /summary model-suggestion query)
// carry the pagination + filter params.
function listCalls(fetchMock: ReturnType<typeof mockApi>): string[] {
  return fetchMock.mock.calls
    .map(([u]) => String(u))
    .filter((u) => u.includes("/v1/usage") && !u.includes("/count") && !u.includes("/summary"));
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
    // Null api_key_id (master-key / historical) renders as an em-dash.
    expect(within(importedRow).getByText("—")).toBeInTheDocument();
  });

  it("sends the api key filter to the API", async () => {
    const fetchMock = mockApi({ rows: [entry({ api_key_id: "key-1" })] });
    renderPage(<ActivityPage />, "/activity?api_key_id=key-1");

    await screen.findByText("gpt-4o");
    expect(listCalls(fetchMock).some((url) => url.includes("api_key_id=key-1"))).toBe(true);
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

  it("expands an error row without exposing provider diagnostics", async () => {
    const user = userEvent.setup();
    mockApi({ rows: [entry({ status: "error", error_message: "provider exploded: quota exceeded" })] });
    renderPage(<ActivityPage />);

    const row = (await screen.findByText("gpt-4o")).closest("tr")!;
    expect(within(row).getByText("error")).toBeInTheDocument();

    await user.click(row);
    expect(screen.getByText("The provider returned an error. Inspect gateway logs for details.")).toBeInTheDocument();
    expect(screen.queryByText("provider exploded: quota exceeded")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Copy" })).not.toBeInTheDocument();
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
    const fetchMock = mockApi({ rows: [entry()] });
    const user = userEvent.setup();
    renderPage(<ActivityPage />);

    await screen.findByText("gpt-4o");
    await user.selectOptions(screen.getByLabelText("Status"), "error");

    // The most recent list request must carry status=error.
    const calls = listCalls(fetchMock);
    expect(calls.at(-1)).toContain("status=error");
  });

  it("distinguishes filtered-empty from never-used", async () => {
    const user = userEvent.setup();
    mockApi({ rows: [], total: 0 });
    renderPage(<ActivityPage />);

    // Default 24h window counts as a filter, so an empty result reads as filtered.
    expect(await screen.findByText("No requests match these filters.")).toBeInTheDocument();

    // Clearing all filters (including the time window) shows the true-empty copy.
    await user.click(screen.getByRole("button", { name: "Clear filters" }));
    expect(await screen.findByText("No requests recorded yet.")).toBeInTheDocument();
  });

  it("filters by API key, not by endpoint or source", async () => {
    // The Activity log scopes by API key; the endpoint and source selects were
    // removed. (Source is still shown per-row in the expanded detail.)
    mockApi({ rows: [entry()] });
    renderPage(<ActivityPage />);

    await screen.findByText("gpt-4o");
    expect(screen.getByRole("combobox", { name: "API key" })).toBeInTheDocument();
    expect(screen.queryByLabelText("Endpoint")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Source")).not.toBeInTheDocument();
  });

  it("keeps Next reachable when the count request fails", async () => {
    // A failed /count must not strand the operator on page 1 with a full page of
    // rows they cannot page past.
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.includes("/v1/usage/count")) {
        return jsonResponse({ detail: "boom" }, 500);
      }
      if (url.includes("/v1/usage")) {
        return jsonResponse(Array.from({ length: 50 }, (_, i) => entry({ id: `r${i}` })));
      }
      return jsonResponse([]);
    });
    renderPage(<ActivityPage />);

    await screen.findAllByText("gpt-4o");
    expect(screen.getByRole("button", { name: "Next" })).toBeEnabled();
    // The range must reflect the rows on screen rather than reading "0 of 0",
    // which would contradict the 50 visible rows.
    expect(screen.getByText("1–50")).toBeInTheDocument();
    expect(screen.queryByText("0 of 0")).not.toBeInTheDocument();
  });

  it("suggests in-window models in the picker and commits the picked one", async () => {
    const fetchMock = mockApi({
      rows: [entry({ model: "gpt-4o" }), entry({ id: "b", model: "claude-sonnet-5" })],
    });
    const user = userEvent.setup();
    renderPage(<ActivityPage />);
    await screen.findAllByText("gpt-4o");

    const modelInput = screen.getByRole("combobox", { name: "Model" });
    await user.click(modelInput);
    await user.type(modelInput, "claude");
    // The suggestion comes from the window's usage, not a free-text guess.
    await user.click(await screen.findByRole("option", { name: "claude-sonnet-5" }));

    await waitFor(() => expect(listCalls(fetchMock).at(-1)).toContain("model=claude-sonnet-5"));
  });

  it("seeds filters from the drill-down query string", async () => {
    // A row click on the Usage page navigates here with the dimension + window.
    const fetchMock = mockApi({ rows: [entry()] });
    renderPage(<ActivityPage />, "/activity?model=gpt-4o&user_id=alice&status=error");

    await screen.findByText("gpt-4o");
    const latest = listCalls(fetchMock).at(-1)!;
    expect(latest).toContain("model=gpt-4o");
    expect(latest).toContain("user_id=alice");
    expect(latest).toContain("status=error");
  });

  it("shows the paginator range and total", async () => {
    mockApi({ rows: Array.from({ length: 50 }, (_, i) => entry({ id: `r${i}` })), total: 120 });
    renderPage(<ActivityPage />);

    expect(await screen.findByText("1–50 of 120")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Previous" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Next" })).toBeEnabled();
  });
});
