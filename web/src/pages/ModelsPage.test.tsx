import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactElement } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { setMasterKey } from "@/api/client";
import type {
  AliasResponse,
  DiscoverableModelsResponse,
  GatewaySettings,
  ModelListResponse,
  ModelObject,
  PricingResponse,
  ProviderCapabilities,
  ProvidersResponse,
} from "@/api/types";
import { ModelsPage } from "@/pages/ModelsPage";

// The one model with a configured DB price. Its catalog pricing_source is
// "configured" to match, so the two sources stay consistent.
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

// The catalog GET /v1/models serves. Kept consistent with DISCOVERABLE below:
// every real (non-alias) model here is also reported by discovery, since a model
// only lands in the catalog with a default price by having been discovered.
const CATALOG: ModelListResponse = {
  object: "list",
  data: [
    catalogModel("openai:gpt-4o", "openai", "configured", [2.5, 10], 128000),
    catalogModel("openai:gpt-4o-mini", "openai", "default", [0.15, 0.6], 128000),
    catalogModel("anthropic:claude-sonnet-4", "anthropic", "default", [3, 15], 200000),
    // A config.yml alias as the gateway reports one: owned_by "otari", carrying
    // its target's price (gpt-4o-mini, default-rated).
    catalogModel("fast-model", "otari", "default", [0.15, 0.6], 128000),
  ],
};

// What discovery reports, consistent with the catalog. Every provider succeeds
// here; a specific test overrides this to exercise a failed provider.
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

// "fast-model" is the config.yml alias the catalog reports above.
const ALIASES: AliasResponse[] = [
  { name: "fast-model", target: "openai:gpt-4o-mini", source: "config", created_at: null, updated_at: null },
];

function rowFor(text: string): HTMLElement {
  return screen.getByText(text).closest("tr") as HTMLElement;
}

// The row for the "fast-model" alias, so assertions about its actions cannot
// accidentally match a button belonging to some other model.
function aliasRow(): HTMLElement {
  return rowFor("fast-model");
}

function renderWithClient(ui: ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
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
    // Before the /v1/models/ catch-all, which would answer it with a 404 the
    // same way the real route order matters server-side.
    if (url.includes("/v1/models/discoverable")) {
      return jsonResponse(opts.discoverable ?? DISCOVERABLE);
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

// Tab accessible names carry a trailing count ("Models 3"), so match on a prefix.
async function goToTab(user: ReturnType<typeof userEvent.setup>, name: string) {
  await user.click(screen.getByRole("tab", { name: new RegExp(`^${name}`) }));
}

async function openAdd(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole("button", { name: "Add" }));
}

// The row text order in the table body, for asserting sort/filter results.
function modelOrder(): string[] {
  return screen
    .getAllByRole("row")
    .map((row) => row.textContent ?? "")
    .filter((text) => text.includes(":"));
}

describe("ModelsPage", () => {
  beforeEach(() => setMasterKey("test-master-key"));
  afterEach(() => {
    vi.restoreAllMocks();
    setMasterKey(null);
  });

  // -- tab structure -------------------------------------------------------

  it("opens on the Models tab and keeps aliases separate, each with a count", async () => {
    mockApi();

    renderWithClient(<ModelsPage />);
    await screen.findByText("openai:gpt-4o");

    // Three real models; the alias is not counted among them.
    expect(screen.getByRole("tab", { name: /Models 3/ })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /Aliases 1/ })).toBeInTheDocument();
    expect(screen.queryByText("fast-model")).not.toBeInTheDocument();
  });

  it("shows no usage columns anywhere", async () => {
    mockApi();

    renderWithClient(<ModelsPage />);
    await screen.findByText("openai:gpt-4o");

    // The page is about configuration, not usage.
    for (const heading of ["Requests", "Tokens", "Cost"]) {
      expect(screen.queryByRole("columnheader", { name: heading })).not.toBeInTheDocument();
    }
  });

  // -- model list ----------------------------------------------------------

  it("lists models with their price source and context window", async () => {
    mockApi();

    renderWithClient(<ModelsPage />);

    expect(await screen.findByText("openai:gpt-4o")).toBeInTheDocument();
    expect(screen.getByText("anthropic:claude-sonnet-4")).toBeInTheDocument();
    // The configured price shows as configured; the fallback-priced one as default.
    expect(screen.getByText("configured")).toBeInTheDocument();
    expect(screen.getAllByText("default").length).toBeGreaterThan(0);
    // Context windows are pulled from the catalog and shown compactly.
    expect(within(rowFor("openai:gpt-4o")).getByText("128K")).toBeInTheDocument();
    expect(within(rowFor("anthropic:claude-sonnet-4")).getByText("200K")).toBeInTheDocument();
  });

  it("filters the model list by search", async () => {
    mockApi();
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await screen.findByText("anthropic:claude-sonnet-4");

    await user.type(screen.getByRole("searchbox"), "gpt-4o-mini");

    expect(screen.getByText("openai:gpt-4o-mini")).toBeInTheDocument();
    expect(screen.queryByText("anthropic:claude-sonnet-4")).not.toBeInTheDocument();
  });

  it("filters by provider", async () => {
    mockApi();
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await screen.findByText("openai:gpt-4o");

    await user.selectOptions(screen.getByLabelText("Filter by provider"), "anthropic");

    expect(screen.getByText("anthropic:claude-sonnet-4")).toBeInTheDocument();
    expect(screen.queryByText("openai:gpt-4o")).not.toBeInTheDocument();
  });

  it("filters to custom prices only", async () => {
    mockApi();
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await screen.findByText("openai:gpt-4o");

    await user.selectOptions(screen.getByLabelText("Filter by pricing"), "Custom price");

    // Only gpt-4o carries a configured (custom) price; the default-priced ones drop.
    expect(screen.getByText("openai:gpt-4o")).toBeInTheDocument();
    expect(screen.queryByText("openai:gpt-4o-mini")).not.toBeInTheDocument();
    expect(screen.queryByText("anthropic:claude-sonnet-4")).not.toBeInTheDocument();
  });

  it("filters to custom (not discovered) models", async () => {
    // A priced model no provider reports: present only because it was priced.
    mockApi({
      pricing: [PRICED, { ...PRICED, model_key: "mistral:large", input_price_per_million: 4, output_price_per_million: 12 }],
    });
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await screen.findByText("mistral:large");

    await user.selectOptions(screen.getByLabelText("Filter by source"), "Custom (not discovered)");

    expect(screen.getByText("mistral:large")).toBeInTheDocument();
    expect(screen.queryByText("openai:gpt-4o")).not.toBeInTheDocument();
  });

  it("filters by provider capability", async () => {
    mockApi();
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await screen.findByText("anthropic:claude-sonnet-4");

    // Only openai reports vision, so anthropic's model drops.
    await user.selectOptions(screen.getByLabelText("Filter by capability"), "Vision");

    expect(screen.getByText("openai:gpt-4o")).toBeInTheDocument();
    expect(screen.queryByText("anthropic:claude-sonnet-4")).not.toBeInTheDocument();
  });

  it("filters by minimum context window", async () => {
    mockApi();
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await screen.findByText("openai:gpt-4o");

    await user.selectOptions(screen.getByLabelText("Minimum context window"), "≥ 200K");

    // Only the 200K model clears the bar.
    expect(screen.getByText("anthropic:claude-sonnet-4")).toBeInTheDocument();
    expect(screen.queryByText("openai:gpt-4o")).not.toBeInTheDocument();
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
    // Cheapest input price first: gpt-4o-mini (0.15) ahead of claude (3.00).
    expect(miniIndex).toBeLessThan(claudeIndex);
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
    });
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await screen.findByText("openai:m00");

    // 25 per page: m00-m24 on page one, m25 on page two.
    expect(screen.getByText("openai:m24")).toBeInTheDocument();
    expect(screen.queryByText("openai:m25")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Next" }));
    expect(screen.getByText("openai:m25")).toBeInTheDocument();
    expect(screen.queryByText("openai:m00")).not.toBeInTheDocument();
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

  // -- provider drawer -----------------------------------------------------

  it("opens a provider detail drawer with capabilities and model count", async () => {
    mockApi();
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await screen.findByText("openai:gpt-4o");

    // The provider cell is a button; click OpenAI's to open its drawer.
    await user.click(screen.getAllByRole("button", { name: "openai" })[0]);

    const drawer = screen.getByRole("dialog", { name: /Provider openai/ });
    expect(within(drawer).getByText("OpenAI")).toBeInTheDocument();
    expect(within(drawer).getByText("Vision")).toBeInTheDocument();
    // openai reports two models in DISCOVERABLE.
    expect(within(drawer).getByText(/2 models reported/)).toBeInTheDocument();
    expect(within(drawer).getByRole("link", { name: /API documentation/ })).toBeInTheDocument();
  });

  // -- inline pricing ------------------------------------------------------

  it("edits a configured price inline", async () => {
    const fetchMock = mockApi();
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await screen.findByText("openai:gpt-4o");

    await user.click(within(rowFor("openai:gpt-4o")).getByRole("button", { name: "Edit" }));
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

  it("does not prompt to backfill usage after setting a price", async () => {
    // Backfill is a usage concern; the config page no longer offers it.
    mockApi();
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await screen.findByText("openai:gpt-4o");

    await user.click(within(rowFor("openai:gpt-4o")).getByRole("button", { name: "Edit" }));
    await user.click(screen.getByRole("button", { name: "Save" }));

    // The editor closes back to an Edit action, and nothing usage-related appears.
    expect(await within(rowFor("openai:gpt-4o")).findByRole("button", { name: "Edit" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /backfill/i })).not.toBeInTheDocument();
  });

  // -- Aliases tab ---------------------------------------------------------

  it("shows an alias with its target and resolved price, and no way to price it", async () => {
    mockApi();
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await goToTab(user, "Aliases");
    await screen.findByText("fast-model");

    // Target and its default-rated price are shown; a config.yml alias cannot be
    // deleted or priced from here.
    expect(within(aliasRow()).getByText("openai:gpt-4o-mini")).toBeInTheDocument();
    expect(within(aliasRow()).getByText("$0.15")).toBeInTheDocument();
    expect(within(aliasRow()).getByText("set in config.yml")).toBeInTheDocument();
    expect(within(aliasRow()).queryByRole("button", { name: /price|delete/i })).not.toBeInTheDocument();
  });

  it("ignores a dead pricing row stored under an alias's own name", async () => {
    // Nothing reads a price keyed on the alias, so showing it would advertise a
    // rate the gateway never charges.
    mockApi({
      pricing: [
        PRICED,
        { ...PRICED, model_key: "fast-model", input_price_per_million: 99, output_price_per_million: 99 },
      ],
    });
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await goToTab(user, "Aliases");
    await screen.findByText("fast-model");

    expect(within(aliasRow()).queryByText("$99.00")).not.toBeInTheDocument();
    expect(within(aliasRow()).getByText("$0.15")).toBeInTheDocument();
  });

  it("deletes a stored alias but not one from config.yml", async () => {
    mockApi({
      aliases: [
        { name: "fast-model", target: "openai:gpt-4o-mini", source: "stored", created_at: null, updated_at: null },
      ],
    });
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await goToTab(user, "Aliases");
    await screen.findByText("fast-model");

    // Stored, so it is removable from here.
    expect(screen.queryByText("set in config.yml")).not.toBeInTheDocument();
    await user.click(within(aliasRow()).getByRole("button", { name: "Delete" }));
    // ConfirmButton asks once before acting.
    await user.click(within(aliasRow()).getByRole("button", { name: "Delete" }));

    const call = fetchSpy.mock.calls.find(([, init]) => (init?.method ?? "") === "DELETE");
    expect(String(call?.[0])).toContain("/v1/aliases/fast-model");
  });

  // -- Add form: picker + alias creation -----------------------------------

  it("offers discovered models in the picker and puts the full selector in the field", async () => {
    mockApi();
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await screen.findByText("openai:gpt-4o");
    await openAdd(user);

    // Scope to the picker by its label: the filter <select>s also report the
    // combobox role, so an unscoped query would be ambiguous.
    const picker = screen.getByRole("combobox", { name: /model/i });
    await user.type(picker, "4o-mini");
    // Offered as the full selector, which is what /v1/pricing keys on: picking
    // "gpt-4o-mini" alone would store a price nothing ever reads.
    const option = await screen.findByRole("option", { name: "openai:gpt-4o-mini" });
    await user.click(option);
    expect(picker).toHaveValue("openai:gpt-4o-mini");
  });

  it("keeps a model the picker never offered typeable", async () => {
    // Discovery only sees configured providers. A model behind an unconfigured
    // one still has to be priceable, so the field is not a whitelist.
    mockApi();
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await screen.findByText("openai:gpt-4o");
    await openAdd(user);

    const picker = screen.getByRole("combobox", { name: /model/i });
    await user.type(picker, "bedrock:claude-3-sonnet");
    expect(picker).toHaveValue("bedrock:claude-3-sonnet");
    // The open popover aria-hides the rest of the form, so dismiss it the way a
    // user would before reaching the price fields.
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

  it("opens the Add form in alias mode from the Aliases tab and creates against the picked target", async () => {
    mockApi({
      post: (url) => (url.includes("/v1/aliases") ? jsonResponse(ALIASES[0]) : jsonResponse(PRICED)),
    });
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await goToTab(user, "Aliases");
    await openAdd(user);

    // Aliases tab opens the form straight to alias creation.
    await user.type(screen.getByRole("textbox", { name: /alias name/i }), "smart");
    await user.type(screen.getByRole("combobox"), "openai:gpt-4o-mini");
    await user.keyboard("{Escape}");
    await user.click(screen.getByRole("button", { name: /create alias/i }));

    const aliasPost = fetchSpy.mock.calls.find(
      ([url, init]) => String(url).includes("/v1/aliases") && (init?.method ?? "") === "POST",
    );
    expect(aliasPost).toBeDefined();
    expect(JSON.parse(String(aliasPost?.[1]?.body))).toEqual({ name: "smart", target: "openai:gpt-4o-mini" });
  });

  it("refuses an alias name that could be mistaken for a model key", async () => {
    // The gateway rejects a name carrying ":" or "/", since it could never be
    // told apart from a real provider:model selector.
    mockApi();
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await goToTab(user, "Aliases");
    await openAdd(user);

    await user.type(screen.getByRole("textbox", { name: /alias name/i }), "openai:fast");

    expect(screen.getByText(/cannot contain/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /create alias/i })).toBeDisabled();
  });
});
