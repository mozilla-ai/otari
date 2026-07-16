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
  ModelUsage,
  PricingResponse,
  UsageSummary,
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
): ModelObject {
  return {
    id,
    object: "model",
    created: 0,
    owned_by,
    pricing: price ? { input_price_per_million: price[0], output_price_per_million: price[1] } : null,
    pricing_source: source,
  };
}

// The catalog GET /v1/models serves. Kept consistent with DISCOVERABLE below:
// every real (non-alias) model here is also reported by discovery, since a model
// only lands in the catalog with a default price by having been discovered.
const CATALOG: ModelListResponse = {
  object: "list",
  data: [
    catalogModel("openai:gpt-4o", "openai", "configured", [2.5, 10]),
    catalogModel("openai:gpt-4o-mini", "openai", "default", [0.15, 0.6]),
    catalogModel("anthropic:claude-sonnet-4", "anthropic", "default", [3, 15]),
    // A config.yml alias as the gateway reports one: owned_by "otari", carrying
    // its target's price (gpt-4o-mini, default-rated).
    catalogModel("fast-model", "otari", "default", [0.15, 0.6]),
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

// "fast-model" is the config.yml alias the catalog reports above.
const ALIASES: AliasResponse[] = [
  { name: "fast-model", target: "openai:gpt-4o-mini", source: "config", created_at: null, updated_at: null },
];

const EMPTY_SUMMARY: UsageSummary = {
  totals: { requests: 0, prompt_tokens: 0, completion_tokens: 0, total_tokens: 0, cost: 0, errors: 0 },
  by_model: [],
};

// The gateway groups usage by (provider, model) and keys it as "provider:model",
// the same identity the catalog and pricing keys use.
function modelUsage(provider: string, model: string, requests = 1, cost = 0): ModelUsage {
  return {
    key: `${provider}:${model}`,
    model,
    provider,
    requests,
    prompt_tokens: 10 * requests,
    completion_tokens: 5 * requests,
    total_tokens: 15 * requests,
    cost,
  };
}

// The row for the "fast-model" alias, so assertions about its actions cannot
// accidentally match a button belonging to some other model.
function aliasRow(): HTMLElement {
  return screen.getByText("fast-model").closest("tr") as HTMLElement;
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
    byModel?: ModelUsage[];
    post?: (url: string) => Response;
    unlisted?: (url: string) => Response;
    catalog?: ModelListResponse;
    pricing?: PricingResponse[];
    aliases?: AliasResponse[];
    discoverable?: DiscoverableModelsResponse;
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
    // Before the /v1/models/ catch-all, which would answer it with a 404 the
    // same way the real route order matters server-side.
    if (url.includes("/v1/models/discoverable")) {
      return jsonResponse(opts.discoverable ?? DISCOVERABLE);
    }
    if (url.includes("/v1/aliases")) {
      return jsonResponse(opts.aliases ?? ALIASES);
    }
    if (url.includes("/v1/models/")) {
      return opts.unlisted ? opts.unlisted(url) : notFound();
    }
    if (url.includes("/v1/models")) {
      return jsonResponse(opts.catalog ?? CATALOG);
    }
    if (url.includes("/v1/usage/summary")) {
      return jsonResponse({ ...EMPTY_SUMMARY, by_model: opts.byModel ?? [] });
    }
    if (url.includes("/v1/pricing")) {
      return jsonResponse(opts.pricing ?? [PRICED]);
    }
    return jsonResponse([]);
  });
}

// Tab accessible names carry a trailing count ("Priced 1"), so match on a prefix.
async function goToTab(user: ReturnType<typeof userEvent.setup>, name: string) {
  await user.click(screen.getByRole("tab", { name: new RegExp(`^${name}`) }));
}

async function openAdd(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole("button", { name: "Add" }));
}

describe("ModelsPage", () => {
  beforeEach(() => setMasterKey("test-master-key"));
  afterEach(() => {
    vi.restoreAllMocks();
    setMasterKey(null);
  });

  // -- tab structure -------------------------------------------------------

  it("opens on the In use tab and shows only models with traffic, costliest first", async () => {
    mockApi({
      byModel: [modelUsage("openai", "gpt-4o", 10, 2), modelUsage("anthropic", "claude-sonnet-4", 40, 9)],
    });

    renderWithClient(<ModelsPage />);
    // Wait for data before reading the rows, so the loading row is gone.
    await screen.findByText("anthropic:claude-sonnet-4");

    const rows = screen.getAllByRole("row");
    const modelCells = rows.slice(1).map((row) => row.querySelector("td")?.textContent);
    // Costliest first: claude ($9) before gpt-4o ($2). Untrafficked models absent.
    expect(modelCells).toEqual(["anthropic:claude-sonnet-4", "openai:gpt-4o"]);
    expect(screen.queryByText("openai:gpt-4o-mini")).not.toBeInTheDocument();
  });

  it("splits models across tabs, each counting its own category", async () => {
    mockApi({ byModel: [modelUsage("openai", "gpt-4o", 5, 1)] });
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    // Counts fill in as each query resolves; wait for the priced count to land.
    await screen.findByRole("tab", { name: /Priced 1/ });
    // Discovered: the three real models discovery reports.
    expect(screen.getByRole("tab", { name: /Discovered 3/ })).toBeInTheDocument();
    // Aliases: the single config alias.
    expect(screen.getByRole("tab", { name: /Aliases 1/ })).toBeInTheDocument();

    await goToTab(user, "Priced");
    expect(screen.getByText("openai:gpt-4o")).toBeInTheDocument();
    expect(screen.queryByText("anthropic:claude-sonnet-4")).not.toBeInTheDocument();
  });

  // -- Discovered tab ------------------------------------------------------

  it("lists discovered models with their price source", async () => {
    mockApi();
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await goToTab(user, "Discovered");

    expect(await screen.findByText("openai:gpt-4o")).toBeInTheDocument();
    expect(screen.getByText("anthropic:claude-sonnet-4")).toBeInTheDocument();
    // The configured price shows as configured; the fallback-priced one as default.
    expect(screen.getByText("configured")).toBeInTheDocument();
    expect(screen.getAllByText("default").length).toBeGreaterThan(0);
  });

  it("filters the discovered list by search", async () => {
    mockApi();
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await goToTab(user, "Discovered");
    await screen.findByText("anthropic:claude-sonnet-4");

    await user.type(screen.getByRole("searchbox"), "gpt-4o-mini");

    expect(screen.getByText("openai:gpt-4o-mini")).toBeInTheDocument();
    expect(screen.queryByText("anthropic:claude-sonnet-4")).not.toBeInTheDocument();
  });

  it("paginates a long discovered list", async () => {
    const models = Array.from({ length: 30 }, (_, i) => {
      const id = `m${String(i).padStart(2, "0")}`;
      return { id, key: `openai:${id}` };
    });
    mockApi({
      catalog: { object: "list", data: [] },
      discoverable: { providers: [{ provider: "openai", ok: true, error: null, models }] },
    });
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await goToTab(user, "Discovered");
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
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await goToTab(user, "Discovered");

    expect(await screen.findByText(/Could not list anthropic/)).toBeInTheDocument();
  });

  // -- In use tab: spend + pricing -----------------------------------------

  it("reports traffic beyond a single usage page", async () => {
    // Counts come from the server-side aggregate, so they are not bounded by
    // the row limit /v1/usage would impose on a client-side tally.
    mockApi({ byModel: [modelUsage("openai", "gpt-4o", 4200, 5)] });

    renderWithClient(<ModelsPage />);
    await screen.findByText("openai:gpt-4o");

    expect(screen.getByText("4,200")).toBeInTheDocument();
    expect(screen.getByText("63,000")).toBeInTheDocument();
  });

  it("prices a used model the catalog does not list", async () => {
    // A model with traffic that discovery and the catalog never surfaced is still
    // billed by name, so its rate is fetched per-key rather than shown as unpriced.
    mockApi({
      byModel: [modelUsage("openai", "o1-preview", 1250, 3)],
      unlisted: () =>
        jsonResponse({
          id: "openai:o1-preview",
          object: "model",
          created: 0,
          owned_by: "openai",
          pricing: { input_price_per_million: 0.99, output_price_per_million: 1.99 },
          pricing_source: "default",
        }),
    });

    renderWithClient(<ModelsPage />);

    expect(await screen.findByText("$0.99")).toBeInTheDocument();
    expect(screen.getByText("$1.99")).toBeInTheDocument();
    expect(screen.getByText("1,250")).toBeInTheDocument();
  });

  it("leaves a used model unpriced when the gateway cannot price it either", async () => {
    // 404 means nothing can name a rate for it, which is what "not priced" says.
    mockApi({ byModel: [modelUsage("mystery", "unknown-model", 3, 0)] });

    renderWithClient(<ModelsPage />);
    await screen.findByText("unknown-model");

    expect(screen.getByText("not priced")).toBeInTheDocument();
  });

  it("offers a backfill after pricing a used model", async () => {
    mockApi({
      byModel: [modelUsage("anthropic", "claude-sonnet-4", 4, 1)],
      post: (url) =>
        url.includes("/v1/usage/backfill")
          ? jsonResponse({ model_key: "anthropic:claude-sonnet-4", rows_updated: 3, cost_added: 0.12, users_updated: 1 })
          : jsonResponse(PRICED),
    });
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await screen.findByText("anthropic:claude-sonnet-4");

    // Default-priced with traffic, so it offers "Set price".
    await user.click(screen.getByRole("button", { name: "Set price" }));
    await user.click(screen.getByRole("button", { name: "Save" }));

    await user.click(await screen.findByRole("button", { name: "Backfill past usage" }));
    expect(await screen.findByText(/Backfilled 3 requests/)).toBeInTheDocument();
  });

  // -- Priced tab ----------------------------------------------------------

  it("edits a configured price inline", async () => {
    const fetchMock = mockApi();
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await goToTab(user, "Priced");
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

  it("does not offer a backfill for a model with no traffic", async () => {
    mockApi({ byModel: [] });
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await goToTab(user, "Priced");
    await screen.findByText("openai:gpt-4o");

    await user.click(screen.getAllByRole("button", { name: "Edit" })[0]);
    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(screen.queryByRole("button", { name: "Backfill past usage" })).not.toBeInTheDocument();
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
    await screen.findByRole("tab", { name: /In use/ });
    await openAdd(user);

    const picker = screen.getByRole("combobox");
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
    await screen.findByRole("tab", { name: /In use/ });
    await openAdd(user);

    const picker = screen.getByRole("combobox");
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
