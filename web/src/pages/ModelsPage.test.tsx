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
  ProviderCapabilities,
  ProvidersResponse,
} from "@/api/types";
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
  model_discovery: true,
  default_pricing: true,
  require_pricing: false,
};

function catalogModel(
  id: string,
  owned_by: string,
  source: "configured" | "default" | "none",
  price: [number, number] | null,
  context_window: number | null = null,
): ModelObject {
  return {
    id,
    object: "model",
    created: 0,
    owned_by,
    pricing: price ? { input_price_per_million: price[0], output_price_per_million: price[1] } : null,
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

function caps(overrides: Partial<ProviderCapabilities>): ProviderCapabilities {
  return {
    streaming: false,
    reasoning: false,
    vision: false,
    pdf: false,
    embeddings: false,
    image_generation: false,
    audio: false,
    rerank: false,
    responses_api: false,
    moderation: false,
    list_models: false,
    ...overrides,
  };
}

const PROVIDERS: ProvidersResponse = {
  providers: [
    {
      instance: "openai",
      provider_type: "openai",
      name: "OpenAI",
      doc_url: "https://platform.openai.com/docs",
      description: "OpenAI models.",
      env_key: "OPENAI_API_KEY",
      pricing_urls: ["https://openai.com/pricing"],
      capabilities: caps({ vision: true, streaming: true, embeddings: true, list_models: true }),
    },
    {
      instance: "anthropic",
      provider_type: "anthropic",
      name: "Anthropic",
      doc_url: "https://docs.anthropic.com",
      description: null,
      env_key: "ANTHROPIC_API_KEY",
      pricing_urls: [],
      capabilities: caps({ reasoning: true, streaming: true }),
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

function renderWithClient(ui: ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <MemoryRouter>
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
    providers?: ProvidersResponse;
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
    if (url.includes("/v1/providers")) {
      return jsonResponse(opts.providers ?? PROVIDERS);
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

async function openPriceForm(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole("button", { name: /Price a model not listed here/i }));
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

  it("lists models with their context window", async () => {
    mockApi();

    renderWithClient(<ModelsPage />);
    await screen.findByText("openai:gpt-4o");

    expect(within(tableRow("openai:gpt-4o")).getByText("128K")).toBeInTheDocument();
    expect(within(tableRow("anthropic:claude-sonnet-4")).getByText("200K")).toBeInTheDocument();
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

  it("starts with an empty detail panel until a model is selected", async () => {
    mockApi();

    renderWithClient(<ModelsPage />);
    await screen.findByText("openai:gpt-4o");

    expect(within(panel()).getByText(/Select a model/)).toBeInTheDocument();
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
    // Provider block: name, a provider capability, and the discovered model count.
    expect(within(detail).getByText("OpenAI")).toBeInTheDocument();
    expect(within(detail).getByText("Vision")).toBeInTheDocument();
    expect(within(detail).getByText(/2 models reported/)).toBeInTheDocument();
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

    await user.click(screen.getByRole("button", { name: /Input \$ \/ 1M/ }));

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

  it("paginates a long model list", async () => {
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

  // -- Price an unlisted model: the picker -----------------------------------

  it("offers discovered models in the picker and puts the full selector in the field", async () => {
    mockApi();
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await screen.findByText("openai:gpt-4o");
    await openPriceForm(user);

    const picker = screen.getByRole("combobox", { name: /model/i });
    await user.type(picker, "4o-mini");
    const option = await screen.findByRole("option", { name: "openai:gpt-4o-mini" });
    await user.click(option);
    expect(picker).toHaveValue("openai:gpt-4o-mini");
  });

  it("keeps a model the picker never offered typeable", async () => {
    mockApi();
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await screen.findByText("openai:gpt-4o");
    await openPriceForm(user);

    const picker = screen.getByRole("combobox", { name: /model/i });
    await user.type(picker, "bedrock:claude-3-sonnet");
    expect(picker).toHaveValue("bedrock:claude-3-sonnet");
    await user.keyboard("{Escape}");

    await user.type(screen.getByLabelText(/input price per million/i), "1");
    await user.type(screen.getByLabelText(/output price per million/i), "2");
    await user.click(screen.getByRole("button", { name: /save price/i }));

    const post = fetchSpy.mock.calls.find(
      ([url, init]) => String(url).includes("/v1/pricing") && init?.method === "POST",
    );
    expect(JSON.parse(String(post?.[1]?.body))).toMatchObject({
      model_key: "bedrock:claude-3-sonnet",
      input_price_per_million: 1,
      output_price_per_million: 2,
    });
  });
});
