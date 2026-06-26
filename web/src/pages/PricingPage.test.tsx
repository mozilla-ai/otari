import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactElement } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { setMasterKey } from "@/api/client";
import type { GatewaySettings, PricingResponse } from "@/api/types";
import { PricingPage } from "@/pages/PricingPage";

const ROW: PricingResponse = {
  model_key: "openai:gpt-4o",
  effective_at: "2026-01-01T00:00:00Z",
  input_price_per_million: 2.5,
  output_price_per_million: 10,
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
};

const SETTINGS: GatewaySettings = {
  mode: "standalone",
  version: "1.0.0",
  default_pricing: true,
  require_pricing: true,
};

function renderWithClient(ui: ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
}

// Route GETs to the right fixture; callers override the POST behavior.
function mockApi(post?: () => Response) {
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input);
    const method = (init?.method ?? "GET").toUpperCase();
    if (method === "POST") {
      return (post ?? (() => jsonResponse(ROW)))();
    }
    if (url.includes("/v1/settings")) {
      return jsonResponse(SETTINGS);
    }
    return jsonResponse([ROW]);
  });
}

describe("PricingPage", () => {
  beforeEach(() => setMasterKey("test-master-key"));
  afterEach(() => {
    vi.restoreAllMocks();
    setMasterKey(null);
  });

  it("lists the current price per model", async () => {
    mockApi();

    renderWithClient(<PricingPage />);

    expect(await screen.findByText("openai:gpt-4o")).toBeInTheDocument();
    expect(screen.getByText("openai")).toBeInTheDocument();
  });

  it("surfaces that the genai-prices default fallback is active", async () => {
    mockApi();

    renderWithClient(<PricingPage />);

    expect(await screen.findByText(/genai-prices/i)).toBeInTheDocument();
  });

  it("edits a price and posts the new value effective now", async () => {
    const fetchMock = mockApi(() => jsonResponse({ ...ROW, input_price_per_million: 4 }));

    const user = userEvent.setup();
    renderWithClient(<PricingPage />);

    await screen.findByText("openai:gpt-4o");
    await user.click(screen.getByRole("button", { name: "Edit" }));

    const inputPrice = screen.getByLabelText("Input price for openai:gpt-4o");
    await user.clear(inputPrice);
    await user.type(inputPrice, "4");
    await user.click(screen.getByRole("button", { name: "Save" }));

    const postCall = fetchMock.mock.calls.find(([, init]) => (init?.method ?? "GET").toUpperCase() === "POST");
    expect(postCall).toBeDefined();
    expect(postCall?.[0]).toBe("/v1/pricing");
    expect(JSON.parse(String(postCall?.[1]?.body))).toMatchObject({
      model_key: "openai:gpt-4o",
      input_price_per_million: 4,
      output_price_per_million: 10,
    });
  });
});
