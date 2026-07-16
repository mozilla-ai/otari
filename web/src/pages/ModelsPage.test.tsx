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
  ModelUsage,
  PricingResponse,
  UsageSummary,
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

// What the picker offers, and one provider that could not be listed.
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
    { provider: "anthropic", ok: false, error: "401 authentication_error", models: [] },
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

  it("prices a model with traffic that the catalog does not list", async () => {
    // An alias's target is withheld from the catalog on purpose, yet usage bills
    // it by name. Claiming "not priced" next to its spend would be a lie, so the
    // rate is fetched for it.
    mockApi({
      byModel: [modelUsage("openai", "gpt-4o-mini", 1250)],
      unlisted: () =>
        jsonResponse({
          id: "openai:gpt-4o-mini",
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

  it("leaves a model unpriced when the gateway cannot price it either", async () => {
    // 404 means nothing can name a rate for it, which is what "not priced" says.
    mockApi({ byModel: [modelUsage("mystery", "unknown-model", 3)] });

    renderWithClient(<ModelsPage />);
    await screen.findByText("unknown-model");

    expect(screen.getByText("not priced")).toBeInTheDocument();
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
    // This one comes from config.yml, which the dashboard cannot edit.
    expect(screen.getByText("set in config.yml")).toBeInTheDocument();
    expect(within(aliasRow()).queryByRole("button", { name: /set price/i })).not.toBeInTheDocument();
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
    expect(screen.getByText("set in config.yml")).toBeInTheDocument();
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

  it("offers discovered models in the picker and posts the one chosen", async () => {
    mockApi();
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await screen.findByText("openai:gpt-4o");
    await user.click(screen.getByRole("button", { name: "Add a model" }));

    const picker = screen.getByRole("combobox");
    await user.type(picker, "4o-mini");
    // Offered as the full selector, which is what /v1/pricing keys on: picking
    // "gpt-4o-mini" alone would store a price nothing ever reads.
    const option = await screen.findByRole("option", { name: "openai:gpt-4o-mini" });
    expect(option).toBeInTheDocument();

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
    await user.click(screen.getByRole("button", { name: "Add a model" }));

    const picker = screen.getByRole("combobox");
    await user.type(picker, "bedrock:claude-3-sonnet");
    expect(picker).toHaveValue("bedrock:claude-3-sonnet");
    // The open popover aria-hides the rest of the form, so dismiss it the way a
    // user would before reaching the price fields.
    await user.keyboard("{Escape}");

    await user.type(screen.getByLabelText(/input price per million/i), "1");
    await user.type(screen.getByLabelText(/output price per million/i), "2");
    await user.click(screen.getByRole("button", { name: /save price/i }));

    const post = fetchSpy.mock.calls.find(([url, init]) => String(url).includes("/v1/pricing") && init?.method === "POST");
    expect(JSON.parse(String(post?.[1]?.body))).toMatchObject({
      model_key: "bedrock:claude-3-sonnet",
      input_price_per_million: 1,
      output_price_per_million: 2,
    });
  });

  it("says which provider could not be listed rather than showing an empty picker", async () => {
    // An unreachable provider contributes no models. Without this the list just
    // looks short, and a bad key is indistinguishable from a provider with
    // nothing to offer.
    mockApi();
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await screen.findByText("openai:gpt-4o");
    await user.click(screen.getByRole("button", { name: "Add a model" }));

    expect(await screen.findByText(/Could not list models for anthropic/)).toBeInTheDocument();
  });

  it("creates an alias against the target chosen in the picker", async () => {
    mockApi({
      post: (url) => {
        if (url.includes("/v1/aliases")) {
          return jsonResponse(ALIASES[0]);
        }
        return jsonResponse(PRICED);
      },
    });
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await screen.findByText("openai:gpt-4o");
    await user.click(screen.getByRole("button", { name: "Add a model" }));
    await user.click(screen.getByRole("button", { name: "Add an alias" }));

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
    await screen.findByText("openai:gpt-4o");
    await user.click(screen.getByRole("button", { name: "Add a model" }));
    await user.click(screen.getByRole("button", { name: "Add an alias" }));

    await user.type(screen.getByRole("textbox", { name: /alias name/i }), "openai:fast");

    expect(screen.getByText(/cannot contain/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /create alias/i })).toBeDisabled();
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
    await screen.findByText("fast-model");

    // Stored, so it is removable from here.
    expect(screen.queryByText("set in config.yml")).not.toBeInTheDocument();
    const del = within(aliasRow()).getByRole("button", { name: "Delete" });
    await user.click(del);
    // ConfirmButton asks once before acting.
    await user.click(within(aliasRow()).getByRole("button", { name: "Delete" }));

    const call = fetchSpy.mock.calls.find(([, init]) => (init?.method ?? "") === "DELETE");
    expect(String(call?.[0])).toContain("/v1/aliases/fast-model");
  });
});
