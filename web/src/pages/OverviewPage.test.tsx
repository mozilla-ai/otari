import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import type { ReactElement } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { setMasterKey } from "@/api/client";
import type { UsageSummary } from "@/api/types";
import { OverviewPage } from "@/pages/OverviewPage";

// Totals no client could have produced from one /v1/usage page: the request
// count is well past that endpoint's row cap.
const SUMMARY: UsageSummary = {
  totals: {
    requests: 12_500,
    prompt_tokens: 900_000,
    completion_tokens: 100_000,
    total_tokens: 1_000_000,
    cost: 42.5,
    errors: 125,
  },
  by_model: [],
};

function renderWithClient(ui: ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

function mockSummary(summary: UsageSummary) {
  return vi
    .spyOn(globalThis, "fetch")
    .mockImplementation(
      async () => new Response(JSON.stringify(summary), { status: 200, headers: { "Content-Type": "application/json" } }),
    );
}

describe("OverviewPage", () => {
  beforeEach(() => setMasterKey("test-master-key"));
  afterEach(() => {
    vi.restoreAllMocks();
    setMasterKey(null);
  });

  it("reports totals over the whole usage log", async () => {
    const fetchMock = mockSummary(SUMMARY);

    renderWithClient(<OverviewPage />);

    expect(await screen.findByText("12,500")).toBeInTheDocument();
    expect(screen.getByText("1,000,000")).toBeInTheDocument();
    expect(screen.getByText("$42.50")).toBeInTheDocument();
    expect(screen.getByText("125")).toBeInTheDocument();
    expect(screen.getByText("1.0% of requests")).toBeInTheDocument();

    // The aggregate endpoint, not a page of rows the client would have to sum.
    expect(String(fetchMock.mock.calls[0]?.[0])).toContain("/v1/usage/summary");
  });

  it("does not divide by zero on an empty log", async () => {
    mockSummary({
      totals: { requests: 0, prompt_tokens: 0, completion_tokens: 0, total_tokens: 0, cost: 0, errors: 0 },
      by_model: [],
    });

    renderWithClient(<OverviewPage />);

    expect(await screen.findByText("0.0% of requests")).toBeInTheDocument();
    expect(screen.getByText("$0.00")).toBeInTheDocument();
  });
});
