import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactElement } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { setMasterKey } from "@/api/client";
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
    cost: 0.0123,
    status: "success",
    error_message: null,
    latency_ms: 842,
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
    if (url.includes("/v1/usage")) {
      return jsonResponse(rows);
    }
    if (url.includes("/v1/users")) {
      return jsonResponse([]);
    }
    return jsonResponse([]);
  });
}

function renderPage(ui: ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

// Only the list requests (not /count) carry the pagination + filter params.
function listCalls(fetchMock: ReturnType<typeof mockApi>): string[] {
  return fetchMock.mock.calls
    .map(([u]) => String(u))
    .filter((u) => u.includes("/v1/usage") && !u.includes("/count"));
}

describe("ActivityPage", () => {
  beforeEach(() => setMasterKey("test-master-key"));
  afterEach(() => {
    vi.restoreAllMocks();
    setMasterKey(null);
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

  it("expands an error row to show the full error text and a copy affordance", async () => {
    const user = userEvent.setup();
    mockApi({ rows: [entry({ status: "error", error_message: "provider exploded: quota exceeded" })] });
    renderPage(<ActivityPage />);

    const row = (await screen.findByText("gpt-4o")).closest("tr")!;
    expect(within(row).getByText("error")).toBeInTheDocument();

    await user.click(row);
    expect(screen.getByText("provider exploded: quota exceeded")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Copy" })).toBeInTheDocument();
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

  it("offers every endpoint that writes a usage log, including batches", async () => {
    // Batch rows are written by log_batch_usage under /v1/batches and
    // /v1/batches/results; omitting them made those rows unreachable by filter.
    mockApi({ rows: [entry()] });
    renderPage(<ActivityPage />);

    await screen.findByText("gpt-4o");
    const select = screen.getByLabelText("Endpoint");
    const offered = Array.from(select.querySelectorAll("option")).map((o) => o.getAttribute("value"));
    for (const ep of ["/v1/chat/completions", "/v1/embeddings", "/v1/batches", "/v1/batches/results"]) {
      expect(offered).toContain(ep);
    }
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

  it("shows the paginator range and total", async () => {
    mockApi({ rows: Array.from({ length: 50 }, (_, i) => entry({ id: `r${i}` })), total: 120 });
    renderPage(<ActivityPage />);

    expect(await screen.findByText("1–50 of 120")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Previous" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Next" })).toBeEnabled();
  });
});
