import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactElement } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { setMasterKey } from "@/api/client";
import type { GatewaySettings, PricingResponse, UsageEntry } from "@/api/types";
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
    if (url.includes("/v1/models")) {
      return jsonResponse({ object: "list", data: [] });
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

  it("shows a default-priced used model and lets you override it", async () => {
    const usageEntry: UsageEntry = {
      id: "u1",
      user_id: "alice",
      api_key_id: "k1",
      timestamp: "2026-01-01T00:00:00Z",
      model: "gpt-4o",
      provider: "openai",
      endpoint: "/v1/chat/completions",
      prompt_tokens: 10,
      completion_tokens: 5,
      total_tokens: 15,
      cache_read_tokens: null,
      cache_write_tokens: null,
      cost: 0.001,
      status: "success",
      error_message: null,
    };

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.includes("/v1/settings")) {
        return jsonResponse(SETTINGS);
      }
      if (url.includes("/v1/models")) {
        return jsonResponse({
          object: "list",
          data: [
            {
              id: "openai:gpt-4o",
              object: "model",
              created: 0,
              owned_by: "openai",
              pricing: { input_price_per_million: 2.5, output_price_per_million: 10 },
              pricing_source: "default",
            },
          ],
        });
      }
      if (url.includes("/v1/usage")) {
        return jsonResponse([usageEntry]);
      }
      return jsonResponse([]); // no DB pricing
    });

    const user = userEvent.setup();
    renderWithClient(<PricingPage />);

    // The used-but-unpriced model appears as a read-only "default" row.
    expect(await screen.findByText("openai:gpt-4o")).toBeInTheDocument();
    expect(screen.getByText("default")).toBeInTheDocument();

    // Override pre-fills the form with the default rate.
    await user.click(screen.getByRole("button", { name: "Override" }));
    expect((screen.getByLabelText("Input price per million") as HTMLInputElement).value).toBe("2.5");
    expect((screen.getByLabelText("Output price per million") as HTMLInputElement).value).toBe("10");
  });

  it("surfaces that the genai-prices default fallback is active", async () => {
    mockApi();

    renderWithClient(<PricingPage />);

    expect(await screen.findByText(/genai-prices/i)).toBeInTheDocument();
  });

  it("sets a price for a used model and backfills its past usage", async () => {
    const usageEntry: UsageEntry = {
      id: "u1",
      user_id: "alice",
      api_key_id: "k1",
      timestamp: "2026-01-01T00:00:00Z",
      model: "gpt-4o",
      provider: "openai",
      endpoint: "/v1/chat/completions",
      prompt_tokens: 10,
      completion_tokens: 5,
      total_tokens: 15,
      cache_read_tokens: null,
      cache_write_tokens: null,
      cost: null,
      status: "success",
      error_message: null,
    };

    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      const method = (init?.method ?? "GET").toUpperCase();
      if (method === "POST" && url.includes("/v1/usage/backfill")) {
        return jsonResponse({ model_key: "openai:gpt-4o", rows_updated: 3, cost_added: 0.12, users_updated: 2 });
      }
      if (method === "POST") {
        return jsonResponse({ ...ROW, model_key: "openai:gpt-4o" });
      }
      if (url.includes("/v1/settings")) {
        return jsonResponse(SETTINGS);
      }
      if (url.includes("/v1/models")) {
        return jsonResponse({ object: "list", data: [] });
      }
      if (url.includes("/v1/usage")) {
        return jsonResponse([usageEntry]);
      }
      return jsonResponse([]); // no pricing yet
    });

    const user = userEvent.setup();
    renderWithClient(<PricingPage />);

    await user.click(await screen.findByRole("button", { name: "Set pricing" }));
    await user.selectOptions(screen.getByLabelText("Used model without a price"), "openai:gpt-4o");
    await user.type(screen.getByLabelText("Input price per million"), "1");
    await user.type(screen.getByLabelText("Output price per million"), "2");
    await user.click(screen.getByRole("button", { name: "Save pricing" }));

    await user.click(await screen.findByRole("button", { name: "Backfill past usage" }));

    expect(await screen.findByText(/Backfilled 3 requests/)).toBeInTheDocument();
    const backfillCall = fetchMock.mock.calls.find(
      ([u, i]) => String(u).includes("/v1/usage/backfill") && (i?.method ?? "").toUpperCase() === "POST",
    );
    expect(backfillCall).toBeDefined();
    expect(JSON.parse(String(backfillCall?.[1]?.body))).toMatchObject({ model_key: "openai:gpt-4o" });
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
