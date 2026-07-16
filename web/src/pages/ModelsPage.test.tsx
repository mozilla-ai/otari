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
    post?: (url: string) => Response;
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

  it("opens on Discovered and splits models across config tabs, each with its count", async () => {
    mockApi();
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    // Discovered is the default; wait for its models before checking counts.
    await screen.findByText("openai:gpt-4o");

    expect(screen.getByRole("tab", { name: /Discovered 3/ })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /Priced 1/ })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /Aliases 1/ })).toBeInTheDocument();

    await goToTab(user, "Priced");
    expect(screen.getByText("openai:gpt-4o")).toBeInTheDocument();
    // claude is default-priced, not configured, so it is not in the Priced tab.
    expect(screen.queryByText("anthropic:claude-sonnet-4")).not.toBeInTheDocument();
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

  // -- Discovered tab ------------------------------------------------------

  it("lists discovered models with their price source", async () => {
    mockApi();

    renderWithClient(<ModelsPage />);

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

  it("does not prompt to backfill usage after setting a price", async () => {
    // Backfill is a usage concern; the config page no longer offers it.
    mockApi();
    const user = userEvent.setup();

    renderWithClient(<ModelsPage />);
    await goToTab(user, "Priced");
    await screen.findByText("openai:gpt-4o");

    await user.click(screen.getAllByRole("button", { name: "Edit" })[0]);
    await user.click(screen.getByRole("button", { name: "Save" }));

    // The editor closes back to an Edit action, and nothing usage-related appears.
    expect(await screen.findByRole("button", { name: "Edit" })).toBeInTheDocument();
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
    await screen.findByText("openai:gpt-4o");
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
