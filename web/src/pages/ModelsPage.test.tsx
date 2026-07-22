import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import userEvent from "@testing-library/user-event";
import type { ReactElement } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { setMasterKey } from "@/api/client";
import type {
  AliasResponse,
  DiscoverableModelsResponse,
  GatewaySettings,
  ModelListResponse,
  ModelMetadata,
  ModelMetadataResponse,
  ModelObject,
  PricingResponse,
} from "@/api/types";
import { ModelsPage } from "@/pages/ModelsPage";

const PRICED: PricingResponse = {
  model_key: "openai:gpt-4o",
  effective_at: "2026-01-01T00:00:00Z",
  input_price_per_million: 2.5,
  output_price_per_million: 10,
  cache_read_price_per_million: null,
  cache_write_price_per_million: null,
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
};

const SETTINGS: GatewaySettings = {
  mode: "standalone",
  version: "1.0.0",
  model_discovery: true,
  default_pricing: true,
  require_pricing: false,
  master_key_source: "configured",
  config: [],
};

function catalogModel(
  id: string,
  owned_by: string,
  source: "configured" | "default" | "none",
  price: [number, number] | null,
  context_window: number | null = null,
  cache: [number | null, number | null] = [null, null],
): ModelObject {
  return {
    id,
    object: "model",
    created: 0,
    owned_by,
    pricing: price
      ? {
          input_price_per_million: price[0],
          output_price_per_million: price[1],
          cache_read_price_per_million: cache[0],
          cache_write_price_per_million: cache[1],
        }
      : null,
    pricing_source: source,
    context_window,
  };
}

const CATALOG: ModelListResponse = {
  object: "list",
  data: [
    catalogModel("openai:gpt-4o", "openai", "configured", [2.5, 10], 128000),
    catalogModel("openai:gpt-4o-mini", "openai", "default", [0.15, 0.6], 128000),
    catalogModel("anthropic:claude-sonnet-4", "anthropic", "default", [3, 15], 200000),
    catalogModel("fast-model", "otari", "default", [0.15, 0.6], 128000),
  ],
};

const DISCOVERABLE: DiscoverableModelsResponse = {
  providers: [
    {
      provider: "openai",
      ok: true,
      error: null,
      models: [
        { id: "gpt-4o", key: "openai:gpt-4o" },
        { id: "gpt-4o-mini", key: "openai:gpt-4o-mini" },
      ],
    },
    {
      provider: "anthropic",
      ok: true,
      error: null,
      models: [{ id: "claude-sonnet-4", key: "anthropic:claude-sonnet-4" }],
    },
  ],
};

function meta(overrides: Partial<ModelMetadata>): ModelMetadata {
  return {
    name: null,
    description: null,
    family: null,
    input_modalities: ["text"],
    output_modalities: ["text"],
    reasoning: false,
    tool_call: false,
    structured_output: false,
    attachment: false,
    temperature: false,
    context_window: null,
    max_output_tokens: null,
    knowledge_cutoff: null,
    release_date: null,
    last_updated: null,
    open_weights: false,
    deprecated: false,
    cost_input: null,
    cost_output: null,
    ...overrides,
  };
}

const METADATA: ModelMetadataResponse = {
  source: "models.dev",
  available: true,
  models: {
    "openai:gpt-4o": meta({
      input_modalities: ["text", "image"],
      tool_call: true,
      context_window: 128000,
      knowledge_cutoff: "2023-09",
      release_date: "2024-05-13",
      description: "GPT-4o multimodal model.",
    }),
    "openai:gpt-4o-mini": meta({
      input_modalities: ["text", "image"],
      tool_call: true,
      context_window: 128000,
      release_date: "2024-07-18",
    }),
    "anthropic:claude-sonnet-4": meta({
      input_modalities: ["text", "image"],
      tool_call: true,
      reasoning: true,
      context_window: 200000,
      release_date: "2025-05-14",
    }),
  },
};

const ALIASES: AliasResponse[] = [
  { name: "fast-model", target: "openai:gpt-4o-mini", source: "config", created_at: null, updated_at: null },
];

function table(): HTMLElement {
  return screen.getByRole("table");
}

function tableRow(text: string): HTMLElement {
  return within(table()).getByText(text).closest("tr") as HTMLElement;
}

// The persistent detail panel is the page's <aside> (role complementary).
function panel(): HTMLElement {
  return screen.getByRole("complementary");
}

function renderWithClient(ui: ReactElement, initialEntries: string[] = ["/"]) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <MemoryRouter initialEntries={initialEntries}>
      <QueryClientProvider client={client}>{ui}</QueryClientProvider>
    </MemoryRouter>,
  );
}

function notFound(): Response {
  return new Response(JSON.stringify({ detail: "Model not found" }), {
    status: 404,
    headers: { "Content-Type": "application/json" },
  });
}

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
}

function mockApi(
  opts: {
    post?: (url: string) => Response;
    catalog?: ModelListResponse;
    pricing?: PricingResponse[];
    aliases?: AliasResponse[];
    discoverable?: DiscoverableModelsResponse;
    metadata?: ModelMetadataResponse;
  } = {},
) {
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input);
    const method = (init?.method ?? "GET").toUpperCase();
    if (method === "POST") {
      return opts.post ? opts.post(url) : jsonResponse(PRICED);
    }
    if (method === "DELETE") {
      return jsonResponse(null);
    }
    if (url.includes("/v1/settings")) {
      return jsonResponse(SETTINGS);
    }
    // Specific /v1/models/* routes before the /v1/models/ catch-all (which 404s,
    // matching the server's route order).
    if (url.includes("/v1/models/discoverable")) {
      return jsonResponse(opts.discoverable ?? DISCOVERABLE);
    }
    if (url.includes("/v1/models/metadata")) {
      return jsonResponse(opts.metadata ?? METADATA);
    }
    if (url.includes("/v1/aliases")) {
      return jsonResponse(opts.aliases ?? ALIASES);
    }
    if (url.includes("/v1/models/")) {
      return notFound();
    }
    if (url.includes("/v1/models")) {
      return jsonResponse(opts.catalog ?? CATALOG);
    }
    if (url.includes("/v1/pricing")) {
      return jsonResponse(opts.pricing ?? [PRICED]);
    }
    return jsonResponse([]);
  });
}

// Selects a model row and returns the detail panel scoped for assertions.
async function selectModel(user: ReturnType<typeof userEvent.setup>, key: string): Promise<HTMLElement> {
  await user.click(tableRow(key));
  return panel();
}

function modelOrder(): string[] {
  return within(table())
    .getAllByRole("row")
    .map((row) => row.textContent ?? "")
    .filter((text) => text.includes(":"));
}

describe("ModelsPage", () => {
  beforeEach(() => {
    setMasterKey("test-master-key");
    // The page persists sort choice in localStorage; start each test clean so
    // one test's sort does not leak into the next.
    window.localStorage.clear();
  });
  afterEach(() => {
    vi.restoreAllMocks();
    setMasterKey(null);
    window.localStorage.clear();
  });

  it("shows no usage columns anywhere", async () => {
    mockApi();

    renderWithClient(<ModelsPage />);
    await screen.findByText("openai:gpt-4o");

    for (const heading of ["Requests", "Tokens", "Cost"]) {
      expect(screen.queryByRole("columnheader", { name: heading })).not.toBeInTheDocument();
    }
  });

  // -- model list ----------------------------------------------------------

  it("does not show a context column in the model table", async () => {
    mockApi();

    renderWithClient(<ModelsPage />);
    await screen.findByText("openai:gpt-4o");

    expect(within(table()).queryByRole("columnheader", { name: "Context" })).not.toBeInTheDocument();
    expect(within(tableRow("openai:gpt-4o")).queryByText("128K")).not.toBeInTheDocument();
  });

  it("surfaces the default-pricing note as a tooltip on the pricing column, not a banner", async () => {
    mockApi();

    renderWithClient(<ModelsPage />);
    await screen.findByText("openai:gpt-4o");

    // The explanation lives in a tooltip anchored to the pricing header, reached
    // via a labelled info trigger, rather than a persistent banner.
    expect(within(table()).getByRole("button", { name: /How unpriced models are metered/ })).toBeInTheDocument();
    const tip = within(table()).getByRole("tooltip");
    expect(tip).toHaveTextContent(/Default pricing is on/);
    expect(tip).toHaveTextContent(/genai-prices/);
  });

  it("hides the detail panel until a model is selected, then closes it again", async () => {
    mockApi();
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await screen.findByText("openai:gpt-4o");

    // Hidden initially so the table has the page to itself.
    expect(screen.queryByRole("complementary")).not.toBeInTheDocument();

    // Selecting a row opens it.
    await user.click(tableRow("openai:gpt-4o"));
    expect(screen.getByRole("complementary")).toBeInTheDocument();

    // The close button dismisses it back to the table-only layout.
    await user.click(screen.getByRole("button", { name: "Close model details" }));
    expect(screen.queryByRole("complementary")).not.toBeInTheDocument();
  });

  it("fills the detail panel when a model row is selected", async () => {
    mockApi();
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await screen.findByText("openai:gpt-4o");

    const detail = await selectModel(user, "openai:gpt-4o");

    expect(within(detail).getByText("configured")).toBeInTheDocument();
    expect(within(detail).getByText("128K")).toBeInTheDocument();
    expect(within(detail).getByText("2023-09")).toBeInTheDocument();
    expect(within(detail).getByText("Tool calling")).toBeInTheDocument();
    expect(within(detail).queryByText("Provider")).not.toBeInTheDocument();
  });

  it("filters the model list by search", async () => {
    mockApi();
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await screen.findByText("anthropic:claude-sonnet-4");

    await user.type(screen.getByRole("searchbox"), "gpt-4o-mini");

    expect(within(table()).getByText("openai:gpt-4o-mini")).toBeInTheDocument();
    expect(within(table()).queryByText("anthropic:claude-sonnet-4")).not.toBeInTheDocument();
  });

  it("filters by provider", async () => {
    mockApi();
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await screen.findByText("openai:gpt-4o");

    await user.selectOptions(screen.getByLabelText("Filter by provider"), "anthropic");

    expect(within(table()).getByText("anthropic:claude-sonnet-4")).toBeInTheDocument();
    expect(within(table()).queryByText("openai:gpt-4o")).not.toBeInTheDocument();
  });

  it("reads the provider filter from the URL query parameter", async () => {
    mockApi();

    renderWithClient(<ModelsPage />, ["/models?provider=anthropic"]);
    await screen.findByText("anthropic:claude-sonnet-4");

    // The provider select is pre-set to the URL's provider, and only that
    // provider's models are shown.
    expect(screen.getByLabelText("Filter by provider")).toHaveValue("anthropic");
    expect(within(table()).getByText("anthropic:claude-sonnet-4")).toBeInTheDocument();
    expect(within(table()).queryByText("openai:gpt-4o")).not.toBeInTheDocument();
  });

  it("falls back to all providers when the URL names an unknown provider", async () => {
    mockApi();

    renderWithClient(<ModelsPage />, ["/models?provider=doesnotexist"]);
    await screen.findByText("anthropic:claude-sonnet-4");

    // A stale/misspelled ?provider= resets to "all" once the catalogue loads,
    // so the select is usable and every provider's models remain visible.
    expect(screen.getByLabelText("Filter by provider")).toHaveValue("all");
    expect(within(table()).getByText("anthropic:claude-sonnet-4")).toBeInTheDocument();
    expect(within(table()).getByText("openai:gpt-4o")).toBeInTheDocument();
  });

  it("filters to custom prices only", async () => {
    mockApi();
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await screen.findByText("openai:gpt-4o");

    await user.selectOptions(screen.getByLabelText("Filter by pricing"), "Custom price");

    expect(within(table()).getByText("openai:gpt-4o")).toBeInTheDocument();
    expect(within(table()).queryByText("openai:gpt-4o-mini")).not.toBeInTheDocument();
    expect(within(table()).queryByText("anthropic:claude-sonnet-4")).not.toBeInTheDocument();
  });

  it("filters to custom (not discovered) models", async () => {
    mockApi({
      pricing: [PRICED, { ...PRICED, model_key: "mistral:large", input_price_per_million: 4, output_price_per_million: 12 }],
    });
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await screen.findByText("mistral:large");

    await user.selectOptions(screen.getByLabelText("Filter by source"), "Custom (not discovered)");

    expect(within(table()).getByText("mistral:large")).toBeInTheDocument();
    expect(within(table()).queryByText("openai:gpt-4o")).not.toBeInTheDocument();
  });

  it("filters by a per-model capability, matching the Features chips", async () => {
    mockApi();
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await screen.findByText("anthropic:claude-sonnet-4");

    // Reasoning is a model-level flag (models.dev), not a provider capability:
    // only claude-sonnet-4 reports it, so the two gpt-4o models drop out.
    await user.selectOptions(screen.getByLabelText("Filter by capability"), "Reasoning");

    expect(within(table()).getByText("anthropic:claude-sonnet-4")).toBeInTheDocument();
    expect(within(table()).queryByText("openai:gpt-4o")).not.toBeInTheDocument();
    expect(within(table()).queryByText("openai:gpt-4o-mini")).not.toBeInTheDocument();
  });

  it("treats missing input modalities as not matching a modality filter", async () => {
    const withoutModalities = {
      ...METADATA,
      models: {
        ...METADATA.models,
        "openai:gpt-4o": { ...METADATA.models["openai:gpt-4o"], input_modalities: undefined },
      },
    } as unknown as ModelMetadataResponse;
    mockApi({ metadata: withoutModalities });
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await screen.findByText("openai:gpt-4o");
    await user.selectOptions(screen.getByLabelText("Filter by capability"), "Vision");

    expect(within(table()).queryByText("openai:gpt-4o")).not.toBeInTheDocument();
  });

  it("explains an empty list is due to filters, not missing credentials", async () => {
    mockApi();
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await screen.findByText("openai:gpt-4o");

    // No mock model reports PDF input, so the list empties for a filter reason.
    await user.selectOptions(screen.getByLabelText("Filter by capability"), "PDF");

    expect(within(table()).getByText("No models match your filters.")).toBeInTheDocument();
  });

  it("filters by minimum context window", async () => {
    mockApi();
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await screen.findByText("openai:gpt-4o");

    await user.selectOptions(screen.getByLabelText("Minimum context window"), "≥ 200K");

    expect(within(table()).getByText("anthropic:claude-sonnet-4")).toBeInTheDocument();
    expect(within(table()).queryByText("openai:gpt-4o")).not.toBeInTheDocument();
  });

  it("sorts by input price when the column header is clicked", async () => {
    mockApi();
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await screen.findByText("openai:gpt-4o");

    await user.click(screen.getByRole("button", { name: /Base in \/ out \$ \/ 1M/ }));

    const order = modelOrder();
    const miniIndex = order.findIndex((text) => text.includes("gpt-4o-mini"));
    const claudeIndex = order.findIndex((text) => text.includes("claude-sonnet-4"));
    expect(miniIndex).toBeLessThan(claudeIndex);
  });

  it("defaults to sorting by model name, ascending", async () => {
    mockApi();

    renderWithClient(<ModelsPage />);
    await screen.findByText("openai:gpt-4o");

    // No header click needed: default is model A→Z, so anthropic sorts before openai.
    const order = modelOrder();
    const claude = order.findIndex((text) => text.includes("claude-sonnet-4"));
    const base = order.findIndex((text) => text.includes("openai:gpt-4o") && !text.includes("mini"));
    const mini = order.findIndex((text) => text.includes("gpt-4o-mini"));
    expect(claude).toBeLessThan(base);
    expect(base).toBeLessThan(mini);
  });

  it("remembers the chosen sort across a remount", async () => {
    mockApi();
    const user = userEvent.setup();

    const { unmount } = renderWithClient(<ModelsPage />);
    await screen.findByText("openai:gpt-4o");

    // Flip the Model column to Z→A and confirm it is persisted.
    await user.click(screen.getByRole("button", { name: /^Model/ }));
    expect(JSON.parse(window.localStorage.getItem("otari.dashboard.modelsSort") ?? "{}")).toEqual({
      col: "model",
      dir: "desc",
    });

    unmount();

    // A fresh mount restores Z→A rather than the A→Z default.
    renderWithClient(<ModelsPage />);
    await screen.findByText("openai:gpt-4o");
    const order = modelOrder();
    const base = order.findIndex((text) => text.includes("openai:gpt-4o") && !text.includes("mini"));
    const claude = order.findIndex((text) => text.includes("claude-sonnet-4"));
    expect(base).toBeLessThan(claude);
  });

  it("defaults to 15 models per page and lets operators change the page size", async () => {
    const models = Array.from({ length: 30 }, (_, i) => {
      const id = `m${String(i).padStart(2, "0")}`;
      return { id, key: `openai:${id}` };
    });
    mockApi({
      catalog: { object: "list", data: [] },
      pricing: [],
      discoverable: { providers: [{ provider: "openai", ok: true, error: null, models }] },
      metadata: { source: "models.dev", available: true, models: {} },
    });
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await screen.findByText("openai:m00");

    expect(within(table()).getByText("openai:m14")).toBeInTheDocument();
    expect(within(table()).queryByText("openai:m15")).not.toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText("Rows per page"), "25");

    expect(within(table()).getByText("openai:m24")).toBeInTheDocument();
    expect(within(table()).queryByText("openai:m25")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Next" }));
    expect(within(table()).getByText("openai:m25")).toBeInTheDocument();
    expect(within(table()).queryByText("openai:m00")).not.toBeInTheDocument();
  });

  it("names a provider discovery could not list", async () => {
    mockApi({
      discoverable: {
        providers: [
          { provider: "openai", ok: true, error: null, models: [{ id: "gpt-4o", key: "openai:gpt-4o" }] },
          { provider: "anthropic", ok: false, error: "401 authentication_error", models: [] },
        ],
      },
    });

    renderWithClient(<ModelsPage />);

    expect(await screen.findByText(/Could not list anthropic/)).toBeInTheDocument();
  });

  // -- inline pricing (in the detail panel) --------------------------------

  it("edits a configured price from the detail panel", async () => {
    const fetchMock = mockApi();
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await screen.findByText("openai:gpt-4o");

    const detail = await selectModel(user, "openai:gpt-4o");
    await user.click(within(detail).getByRole("button", { name: "Edit price" }));
    const input = within(detail).getByLabelText("Input price for openai:gpt-4o");
    await user.clear(input);
    await user.type(input, "4");
    await user.click(within(detail).getByRole("button", { name: "Save" }));

    const call = fetchMock.mock.calls.find(([, init]) => (init?.method ?? "") === "POST");
    expect(call).toBeDefined();
    expect(String(call?.[0])).toContain("/v1/pricing");
    expect(JSON.parse(String(call?.[1]?.body))).toMatchObject({
      model_key: "openai:gpt-4o",
      input_price_per_million: 4,
    });
  });

  it("sets cache, TTL, and long-context prices from the detail panel", async () => {
    const fetchMock = mockApi();
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await screen.findByText("openai:gpt-4o");

    const detail = await selectModel(user, "openai:gpt-4o");
    await user.click(within(detail).getByRole("button", { name: "Edit price" }));
    await user.type(within(detail).getByLabelText("Cache read price for openai:gpt-4o"), "0.3");
    await user.type(within(detail).getByLabelText("Cache write price for openai:gpt-4o"), "3.75");
    await user.type(within(detail).getByLabelText("1 hour cache write price for openai:gpt-4o"), "6");
    await user.click(within(detail).getByRole("button", { name: "Add tier" }));
    await user.clear(within(detail).getByLabelText("Tier context threshold"));
    await user.type(within(detail).getByLabelText("Tier context threshold"), "200000");
    await user.type(within(detail).getByLabelText("Tier input price"), "6");
    await user.click(within(detail).getByRole("button", { name: "Save" }));

    const call = fetchMock.mock.calls.find(([, init]) => (init?.method ?? "") === "POST");
    expect(String(call?.[0])).toContain("/v1/pricing");
    expect(JSON.parse(String(call?.[1]?.body))).toMatchObject({
      model_key: "openai:gpt-4o",
      cache_read_price_per_million: 0.3,
      cache_write_price_per_million: 3.75,
      cache_write_1h_price_per_million: 6,
      pricing_tiers: [{ min_input_tokens: 200000, input_price_per_million: 6 }],
    });
  });

  it("shows configured cache prices in the table and detail panel", async () => {
    mockApi({
      catalog: {
        object: "list",
        data: [catalogModel("anthropic:claude-sonnet-4", "anthropic", "configured", [3, 15], 200000, [0.3, 3.75])],
      },
      pricing: [],
    });
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await screen.findByText("anthropic:claude-sonnet-4");

    expect(screen.getByRole("columnheader", { name: "Caching policy" })).toBeInTheDocument();

    // The compact cache-policy cell keeps the table readable while opening the
    // full structured editor on click.
    const row = tableRow("anthropic:claude-sonnet-4");
    expect(within(row).getByRole("button", { name: "Edit caching price for anthropic:claude-sonnet-4" })).toHaveTextContent(
      "R $0.30 · W $3.75",
    );

    const detail = await selectModel(user, "anthropic:claude-sonnet-4");
    expect(within(detail).getByText("Cache read")).toBeInTheDocument();
    expect(within(detail).getByText("$0.30 / 1M")).toBeInTheDocument();
    expect(within(detail).getByText("$3.75 / 1M")).toBeInTheDocument();
  });

  it("compares effective rates at a selected context threshold", async () => {
    mockApi({
      pricing: [
        {
          ...PRICED,
          pricing_tiers: [
            {
              min_input_tokens: 200000,
              input_price_per_million: 5,
              output_price_per_million: 20,
              cache_read_price_per_million: 0.5,
            },
          ],
        },
      ],
    });
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await screen.findByText("openai:gpt-4o");

    await user.selectOptions(screen.getByLabelText("Compare prices at context"), "200000");

    expect(screen.getByRole("columnheader", { name: /at 200K in \/ out \$ \/ 1M/ })).toBeInTheDocument();
    const row = tableRow("openai:gpt-4o");
    expect(within(row).getByRole("button", { name: "Edit input price for openai:gpt-4o" })).toHaveTextContent("$5.00");
    expect(within(row).getByRole("button", { name: "Edit pricing policy for openai:gpt-4o" })).toHaveTextContent(
      "1 tier · ≥ 200K",
    );
  });

  it("removes the repeated provider instance from a model's visible name", async () => {
    mockApi({
      catalog: {
        object: "list",
        data: [catalogModel("homelab:work-anthropic:claude-sonnet-5", "homelab", "none", null, 200000)],
      },
      pricing: [],
      discoverable: { providers: [] },
      providers: { providers: [] },
      metadata: { source: "models.dev", available: true, models: {} },
    });
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await screen.findByText("homelab:work-anthropic:claude-sonnet-5");

    const row = tableRow("homelab:work-anthropic:claude-sonnet-5");
    expect(within(row).getByText("work-anthropic:claude-sonnet-5")).toBeInTheDocument();
    expect(within(row).getByText("homelab")).toBeInTheDocument();

    const detail = await selectModel(user, "homelab:work-anthropic:claude-sonnet-5");
    expect(within(detail).getByText(/Selector:/)).toBeInTheDocument();
    expect(within(detail).getByText("homelab:work-anthropic:claude-sonnet-5")).toBeInTheDocument();
  });

  it("opens the inline editor by clicking a price number and saves cache rates", async () => {
    const fetchMock = mockApi();
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await screen.findByText("openai:gpt-4o");

    const row = tableRow("openai:gpt-4o");
    await user.click(within(row).getByRole("button", { name: "Edit input price for openai:gpt-4o" }));

    // The inline editor row appears with all four fields.
    const cacheRead = await screen.findByLabelText("Cache read price for openai:gpt-4o");
    await user.type(cacheRead, "0.3");
    await user.click(screen.getByRole("button", { name: "Save" }));

    const call = fetchMock.mock.calls.find(([, init]) => (init?.method ?? "") === "POST");
    expect(String(call?.[0])).toContain("/v1/pricing");
    expect(JSON.parse(String(call?.[1]?.body))).toMatchObject({
      model_key: "openai:gpt-4o",
      cache_read_price_per_million: 0.3,
    });
  });

  it("does not prompt to backfill usage after setting a price", async () => {
    mockApi();
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await screen.findByText("openai:gpt-4o");

    const detail = await selectModel(user, "openai:gpt-4o");
    await user.click(within(detail).getByRole("button", { name: "Edit price" }));
    await user.click(within(detail).getByRole("button", { name: "Save" }));

    expect(await within(panel()).findByRole("button", { name: "Edit price" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /backfill/i })).not.toBeInTheDocument();
  });

});
