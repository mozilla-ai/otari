import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactElement } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { setMasterKey } from "@/api/client";
import type { GatewaySettings, ModelListResponse, ModelUsage, PricingResponse, UsageSummary } from "@/api/types";
import { ModelsPage } from "@/pages/ModelsPage";

const PRICED: PricingResponse = {
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
  require_pricing: false,
};

// A model the gateway is metering off the genai-prices fallback.
const CATALOG: ModelListResponse = {
  object: "list",
  data: [
    {
      id: "openai:gpt-4o",
      object: "model",
      created: 0,
      owned_by: "openai",
      pricing: { input_price_per_million: 2.5, output_price_per_million: 10 },
      pricing_source: "configured",
    },
    {
      id: "anthropic:claude-sonnet-4",
      object: "model",
      created: 0,
      owned_by: "anthropic",
      pricing: { input_price_per_million: 3, output_price_per_million: 15 },
      pricing_source: "default",
    },
    // A config.yml alias as the gateway reports one: owned_by "otari", and its
    // target's price, with pricing_source describing where that price came from.
    {
      id: "fast-model",
      object: "model",
      created: 0,
      owned_by: "otari",
      pricing: { input_price_per_million: 0.15, output_price_per_million: 0.6 },
      pricing_source: "configured",
    },
  ],
};

const EMPTY_SUMMARY: UsageSummary = {
  totals: { requests: 0, prompt_tokens: 0, completion_tokens: 0, total_tokens: 0, cost: 0, errors: 0 },
  by_model: [],
};

// The gateway groups usage by (provider, model) and keys it as "provider:model",
// the same identity the catalog and pricing keys use.
function modelUsage(provider: string, model: string, requests = 1): ModelUsage {
  return {
    key: `${provider}:${model}`,
    model,
    provider,
    requests,
    prompt_tokens: 10 * requests,
    completion_tokens: 5 * requests,
    total_tokens: 15 * requests,
    cost: 0,
  };
}

function renderWithClient(ui: ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
}

function mockApi(opts: { byModel?: ModelUsage[]; post?: (url: string) => Response } = {}) {
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input);
    const method = (init?.method ?? "GET").toUpperCase();
    if (method === "POST") {
      return opts.post ? opts.post(url) : jsonResponse(PRICED);
    }
    if (url.includes("/v1/settings")) {
      return jsonResponse(SETTINGS);
    }
    if (url.includes("/v1/models")) {
      return jsonResponse(CATALOG);
    }
    if (url.includes("/v1/usage/summary")) {
      return jsonResponse({ ...EMPTY_SUMMARY, by_model: opts.byModel ?? [] });
    }
    return jsonResponse([PRICED]);
  });
}

describe("ModelsPage", () => {
  beforeEach(() => setMasterKey("test-master-key"));
  afterEach(() => {
    vi.restoreAllMocks();
    setMasterKey(null);
  });

  it("lists catalog models with their price source", async () => {
    mockApi();

    renderWithClient(<ModelsPage />);

    expect(await screen.findByText("openai:gpt-4o")).toBeInTheDocument();
    expect(screen.getByText("anthropic:claude-sonnet-4")).toBeInTheDocument();
    // The configured price wins; the fallback-priced model is marked default.
    expect(screen.getByText("configured")).toBeInTheDocument();
    expect(screen.getByText("default")).toBeInTheDocument();
  });

  it("reports traffic beyond a single usage page", async () => {
    // Counts come from the server-side aggregate, so they are not bounded by
    // the row limit /v1/usage would impose on a client-side tally.
    mockApi({ byModel: [modelUsage("openai", "gpt-4o", 4200)] });

    renderWithClient(<ModelsPage />);
    await screen.findByText("openai:gpt-4o");

    expect(screen.getByText("4,200")).toBeInTheDocument();
    expect(screen.getByText("63,000")).toBeInTheDocument();
  });

  it("edits a configured price inline", async () => {
    const fetchMock = mockApi();
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await screen.findByText("openai:gpt-4o");

    await user.click(screen.getAllByRole("button", { name: "Edit" })[0]);
    const input = screen.getByLabelText("Input price for openai:gpt-4o");
    await user.clear(input);
    await user.type(input, "4");
    await user.click(screen.getByRole("button", { name: "Save" }));

    const call = fetchMock.mock.calls.find(([, init]) => (init?.method ?? "") === "POST");
    expect(call).toBeDefined();
    expect(String(call?.[0])).toContain("/v1/pricing");
    expect(JSON.parse(String(call?.[1]?.body))).toMatchObject({
      model_key: "openai:gpt-4o",
      input_price_per_million: 4,
    });
  });

  it("offers a backfill after pricing a model that already has traffic", async () => {
    mockApi({
      byModel: [modelUsage("anthropic", "claude-sonnet-4")],
      post: (url) =>
        url.includes("/v1/usage/backfill")
          ? jsonResponse({ model_key: "anthropic:claude-sonnet-4", rows_updated: 3, cost_added: 0.12, users_updated: 1 })
          : jsonResponse(PRICED),
    });
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await screen.findByText("anthropic:claude-sonnet-4");

    // The default-priced model has traffic, so it offers "Set price".
    await user.click(screen.getByRole("button", { name: "Set price" }));
    await user.click(screen.getByRole("button", { name: "Save" }));

    await user.click(await screen.findByRole("button", { name: "Backfill past usage" }));
    expect(await screen.findByText(/Backfilled 3 requests/)).toBeInTheDocument();
  });

  it("shows an alias with its target's price and no way to price it", async () => {
    mockApi();

    renderWithClient(<ModelsPage />);
    await screen.findByText("fast-model");

    // The alias inherits the target's rate, and is labelled an alias from
    // owned_by rather than pricing_source, which describes where the target's
    // price came from and here reads "configured" like any DB-priced model.
    expect(screen.getByText("alias")).toBeInTheDocument();
    expect(screen.getByText("$0.15")).toBeInTheDocument();
    // The API rejects a price posted against an alias name, so none is offered.
    expect(screen.getByText("priced by its target")).toBeInTheDocument();
    expect(screen.queryByLabelText("Input price for fast-model")).not.toBeInTheDocument();
  });

  it("ignores a dead pricing row stored under an alias's own name", async () => {
    // Nothing reads a price keyed on the alias, so showing it would advertise a
    // rate the gateway never charges.
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if ((init?.method ?? "GET").toUpperCase() === "POST") {
        return jsonResponse(PRICED);
      }
      if (url.includes("/v1/settings")) {
        return jsonResponse(SETTINGS);
      }
      if (url.includes("/v1/models")) {
        return jsonResponse(CATALOG);
      }
      if (url.includes("/v1/usage/summary")) {
        return jsonResponse(EMPTY_SUMMARY);
      }
      return jsonResponse([
        PRICED,
        { ...PRICED, model_key: "fast-model", input_price_per_million: 99, output_price_per_million: 99 },
      ]);
    });

    renderWithClient(<ModelsPage />);
    await screen.findByText("fast-model");

    expect(screen.queryByText("$99.00")).not.toBeInTheDocument();
    expect(screen.getByText("$0.15")).toBeInTheDocument();
    expect(screen.getByText("priced by its target")).toBeInTheDocument();
  });

  it("does not offer a backfill for a model with no traffic", async () => {
    mockApi({ byModel: [] });
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await screen.findByText("openai:gpt-4o");

    await user.click(screen.getAllByRole("button", { name: "Edit" })[0]);
    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(screen.queryByRole("button", { name: "Backfill past usage" })).not.toBeInTheDocument();
  });
});
