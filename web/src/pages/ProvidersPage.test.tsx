import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactElement } from "react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { setMasterKey } from "@/api/client";
import { PROVIDER_HEALTH_REFRESH_MS } from "@/api/hooks";
import type {
  GatewaySettings,
  KnownProvider,
  ProviderHealth,
  ProviderHealthResponse,
  ProviderInfo,
  StoredProvider,
  TestProviderResult,
} from "@/api/types";
import { ProvidersPage } from "@/pages/ProvidersPage";

const CAPS = {
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
};

function providerInfo(instance: string, envKey: string | null = null): ProviderInfo {
  return {
    instance,
    provider_type: instance,
    name: instance,
    doc_url: null,
    description: null,
    env_key: envKey,
    pricing_urls: [],
    capabilities: CAPS,
  };
}

function storedProvider(instance: string, last4: string | null, decryptable = true): StoredProvider {
  return {
    instance,
    provider_type: null,
    api_base: null,
    last4,
    client_args: {},
    created_at: null,
    updated_at: "2026-01-01T00:00:00+00:00",
    decryptable,
  };
}

const SETTINGS: GatewaySettings = {
  mode: "standalone",
  version: "1.0.0",
  model_discovery: true,
  default_pricing: true,
  require_pricing: false,
  master_key_source: "configured",
  config: [],
};

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

// Build a health response, defaulting every provider in `meta` to reachable so
// tests that don't care about health still get a well-formed payload.
function healthResponse(providers: ProviderHealth[]): ProviderHealthResponse {
  // Mirror the backend: the summary checked_at is the most recent per-provider
  // checked_at, or null when no provider has ever been checked.
  const checkedAts = providers.map((p) => p.checked_at).filter((t): t is string => t !== null);
  return {
    providers,
    healthy: providers.filter((p) => p.ok).length,
    total: providers.length,
    checked_at: checkedAts.length > 0 ? checkedAts.sort().at(-1)! : null,
  };
}

interface MockOpts {
  meta?: ProviderInfo[];
  stored?: StoredProvider[];
  settings?: GatewaySettings;
  testResult?: TestProviderResult;
  catalog?: KnownProvider[];
  // Per-provider health; defaults to every `meta` provider reachable. `healthRefresh`
  // is served for the forced-refresh (refresh=true) request, if given.
  health?: ProviderHealth[];
  healthRefresh?: ProviderHealth[];
}

function mockApi(opts: MockOpts = {}) {
  let storedList = [...(opts.stored ?? [])];
  let settings = { ...(opts.settings ?? SETTINGS) };
  const meta = opts.meta ?? [];
  const testResult = opts.testResult ?? { ok: true, model_count: 3, error: null };
  const catalog = opts.catalog ?? [];
  const health =
    opts.health ??
    meta.map((info) => ({ instance: info.instance, ok: true, model_count: 3, error: null, checked_at: null }));
  const healthRefresh = opts.healthRefresh ?? health;

  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input);
    const method = (init?.method ?? "GET").toUpperCase();

    if (url.includes("/v1/provider-credentials")) {
      if (url.endsWith("/test") && method === "POST") {
        return jsonResponse(testResult);
      }
      if (method === "POST") {
        const body = JSON.parse(String(init?.body)) as { instance: string; api_key?: string | null };
        const row = storedProvider(body.instance, body.api_key ? body.api_key.slice(-4) : null);
        storedList = [...storedList, row];
        return jsonResponse(row, 201);
      }
      if (method === "DELETE") {
        const instance = decodeURIComponent(url.split("/").pop() ?? "");
        storedList = storedList.filter((p) => p.instance !== instance);
        return new Response(null, { status: 204 });
      }
      return jsonResponse(storedList);
    }
    if (url.includes("/v1/providers/catalog")) {
      return jsonResponse(catalog);
    }
    if (url.includes("/v1/providers/health")) {
      return jsonResponse(healthResponse(url.includes("refresh=true") ? healthRefresh : health));
    }
    if (url.includes("/v1/providers")) {
      return jsonResponse({ providers: meta });
    }
    if (url.includes("/v1/settings")) {
      if (method === "PATCH") {
        settings = { ...settings, ...JSON.parse(String(init?.body)) };
      }
      return jsonResponse(settings);
    }
    return jsonResponse([]);
  });
}

function renderPage(ui: ReactElement, client = new QueryClient({ defaultOptions: { queries: { retry: false } } })) {
  return render(
    <MemoryRouter>
      <QueryClientProvider client={client}>{ui}</QueryClientProvider>
    </MemoryRouter>,
  );
}

function healthRequestCount(fetchMock: ReturnType<typeof mockApi>): number {
  return fetchMock.mock.calls.filter(([url]) => String(url).includes("/v1/providers/health")).length;
}

describe("ProvidersPage", () => {
  beforeEach(() => setMasterKey("test-master-key"));
  afterEach(() => {
    vi.restoreAllMocks();
    setMasterKey(null);
  });

  it("lists config and stored providers with provenance and redacted keys", async () => {
    mockApi({
      meta: [providerInfo("openai", "OPENAI_API_KEY"), providerInfo("anthropic")],
      stored: [storedProvider("anthropic", "4242")],
    });

    renderPage(<ProvidersPage />);

    // Key off cells unique to each row (the instance name appears in two columns).
    const storedRow = (await screen.findByText("••••4242")).closest("tr")!;
    expect(within(storedRow).getByText("stored")).toBeInTheDocument();

    const configRow = screen.getByText("OPENAI_API_KEY").closest("tr")!;
    expect(within(configRow).getByText("config")).toBeInTheDocument();
    // The plaintext key is never shown, only the last 4.
    expect(document.body.textContent).not.toContain("sk-");
  });

  it("adds a custom provider and posts a write-only key, never rendering it", async () => {
    const fetchMock = mockApi({ meta: [], stored: [] });
    const user = userEvent.setup();
    renderPage(<ProvidersPage />);

    await screen.findByText(/No providers yet/);
    await user.click(screen.getByRole("button", { name: "Add your first provider" }));
    await user.click(screen.getByRole("button", { name: "Custom endpoint" }));
    await user.type(screen.getByLabelText("Name"), "my-llm");
    await user.type(screen.getByLabelText("API base"), "http://box:8000/v1");
    await user.type(screen.getByLabelText("API key (optional)"), "sk-live-9999");
    await user.click(screen.getByRole("button", { name: "Add provider" }));

    const post = fetchMock.mock.calls.find(
      ([u, init]) => String(u).endsWith("/v1/provider-credentials") && (init?.method ?? "") === "POST",
    );
    expect(post).toBeDefined();
    expect(JSON.parse(String(post?.[1]?.body))).toMatchObject({
      instance: "my-llm",
      provider_type: "openai-compatible",
      api_base: "http://box:8000/v1",
      api_key: "sk-live-9999",
    });

    // After the round trip the row shows the redacted key, never the plaintext.
    expect(await screen.findByText("••••9999")).toBeInTheDocument();
    expect(document.body.textContent).not.toContain("sk-live-9999");
  });

  it("offers the known-provider picker with an Advanced disclosure", async () => {
    mockApi({
      stored: [storedProvider("anthropic", "0000")],
      catalog: [
        {
          id: "openai",
          name: "OpenAI",
          env_key: "OPENAI_API_KEY",
          default_api_base: "https://api.openai.com/v1",
          requires_api_key: true,
          env_key_present: false,
        },
      ],
    });
    const user = userEvent.setup();
    renderPage(<ProvidersPage />);

    await screen.findByText("••••0000");
    await user.click(screen.getByRole("button", { name: "Add provider" }));

    // Known provider is the default tab: a provider picker plus a collapsed Advanced section.
    expect(screen.getByPlaceholderText("Search providers…")).toBeInTheDocument();
    expect(screen.getByText("Advanced (API base, rename)")).toBeInTheDocument();
    expect(screen.queryByLabelText("API base")).not.toBeInTheDocument();
  });

  it("keeps Add disabled for a key-requiring provider until a key is entered", async () => {
    mockApi({
      stored: [storedProvider("anthropic", "0000")],
      catalog: [
        {
          id: "openai",
          name: "OpenAI",
          env_key: "OPENAI_API_KEY",
          default_api_base: "https://api.openai.com/v1",
          requires_api_key: true,
          env_key_present: false,
        },
      ],
    });
    const user = userEvent.setup();
    renderPage(<ProvidersPage />);

    await screen.findByText("••••0000");
    await user.click(screen.getByRole("button", { name: "Add provider" }));

    await user.type(screen.getByPlaceholderText("Search providers…"), "OpenAI");
    await user.click(await screen.findByRole("option", { name: /OpenAI/ }));
    // Close the combobox popover, which otherwise aria-hides the submit button.
    await user.keyboard("{Escape}");

    const submit = screen.getByRole("button", { name: "Add provider" });
    expect(submit).toBeDisabled();

    await user.type(screen.getByLabelText("API key"), "sk-live-xxxx");
    expect(submit).toBeEnabled();
  });

  it("lets a key-requiring provider submit without a key when its env var is already set", async () => {
    const fetchMock = mockApi({
      stored: [storedProvider("anthropic", "0000")],
      catalog: [
        {
          id: "openai",
          name: "OpenAI",
          env_key: "OPENAI_API_KEY",
          default_api_base: "https://api.openai.com/v1",
          requires_api_key: true,
          env_key_present: true,
        },
      ],
    });
    const user = userEvent.setup();
    renderPage(<ProvidersPage />);

    await screen.findByText("••••0000");
    await user.click(screen.getByRole("button", { name: "Add provider" }));

    await user.type(screen.getByPlaceholderText("Search providers…"), "OpenAI");
    await user.click(await screen.findByRole("option", { name: /OpenAI/ }));
    // Close the combobox popover, which otherwise aria-hides the submit button.
    await user.keyboard("{Escape}");

    // The field is optional and the copy explains the env fallback.
    expect(screen.getByText(/OPENAI_API_KEY is set on the server/)).toBeInTheDocument();
    expect(screen.getByLabelText("API key (optional)")).toBeInTheDocument();

    // Submit with no key: the server stores none and any-llm reads OPENAI_API_KEY.
    const submit = screen.getByRole("button", { name: "Add provider" });
    expect(submit).toBeEnabled();
    await user.click(submit);

    const post = fetchMock.mock.calls.find(
      ([u, init]) => String(u).endsWith("/v1/provider-credentials") && (init?.method ?? "") === "POST",
    );
    expect(post).toBeDefined();
    expect(JSON.parse(String(post?.[1]?.body))).toMatchObject({ instance: "openai", api_key: null });
  });

  it("replaces the welcome onboarding with the add form", async () => {
    mockApi({ meta: [], stored: [] });
    const user = userEvent.setup();
    renderPage(<ProvidersPage />);

    expect(await screen.findByText("Welcome to Otari")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Add provider" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Add your first provider" }));

    expect(screen.queryByText("Welcome to Otari")).not.toBeInTheDocument();
    expect(screen.getByPlaceholderText("Search providers…")).toBeInTheDocument();
  });

  it("hides the onboarding once a provider exists", async () => {
    mockApi({ stored: [storedProvider("openai", "1234")] });
    renderPage(<ProvidersPage />);

    await screen.findByText("••••1234");
    expect(screen.queryByText("Welcome to Otari")).not.toBeInTheDocument();
  });

  it("flags a stored provider whose key can't be decrypted", async () => {
    mockApi({ stored: [storedProvider("home-lab", "0000", false)] });
    renderPage(<ProvidersPage />);

    expect(await screen.findByText(/key unreadable/)).toBeInTheDocument();
    // Test is disabled for an unreadable key; Edit/Delete remain to recover it.
    expect(screen.getByRole("button", { name: "Test" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Edit" })).toBeEnabled();
  });

  it("reports a successful connection test", async () => {
    mockApi({ stored: [storedProvider("openai", "1234")], testResult: { ok: true, model_count: 5, error: null } });
    const user = userEvent.setup();
    renderPage(<ProvidersPage />);

    await screen.findByText("••••1234");
    await user.click(screen.getByRole("button", { name: "Test" }));

    expect(await screen.findByText(/Connected\. 5 models available\./)).toBeInTheDocument();
  });

  // The gateway-wide require_pricing alarm moved to the app shell; its behavior
  // (show, enable default pricing, dismiss) is covered in PricingWarning.test.tsx.

  it("shows each provider's reachability, including config-only providers", async () => {
    mockApi({
      meta: [providerInfo("openai"), providerInfo("anthropic")],
      health: [
        { instance: "openai", ok: true, model_count: 12, error: null, checked_at: "2026-07-21T00:00:00+00:00" },
        {
          instance: "anthropic",
          ok: false,
          model_count: 0,
          error: "authentication failed: invalid key",
          checked_at: "2026-07-21T00:00:00+00:00",
        },
      ],
    });
    renderPage(<ProvidersPage />);

    // Scope by the status pill's row (the provider name repeats in the Type cell).
    const reachableRow = (await screen.findByText("Reachable")).closest("tr")!;
    expect(within(reachableRow).getAllByText("openai").length).toBeGreaterThan(0);

    const unreachablePill = screen.getByText("Unreachable");
    const unreachableRow = unreachablePill.closest("tr")!;
    expect(within(unreachableRow).getAllByText("anthropic").length).toBeGreaterThan(0);
    // The provider error rides along as the pill's tooltip.
    expect(unreachablePill).toHaveAttribute("title", expect.stringContaining("authentication failed"));
  });

  it("summarizes how many providers are reachable", async () => {
    mockApi({
      meta: [providerInfo("openai"), providerInfo("anthropic")],
      health: [
        { instance: "openai", ok: true, model_count: 3, error: null, checked_at: null },
        { instance: "anthropic", ok: false, model_count: 0, error: "down", checked_at: null },
      ],
    });
    renderPage(<ProvidersPage />);

    expect(await screen.findByText("1 of 2 providers reachable")).toBeInTheDocument();
  });

  it("does not automatically re-check all providers within an hour", async () => {
    const fetchMock = mockApi({ meta: [providerInfo("openai")] });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const first = renderPage(<ProvidersPage />, client);

    await screen.findByText("1 of 1 provider reachable");
    expect(healthRequestCount(fetchMock)).toBe(1);

    first.unmount();
    client.setQueryData(["provider-health"], healthResponse([{ instance: "openai", ok: true, model_count: 3, error: null, checked_at: null }]), {
      updatedAt: Date.now() - (PROVIDER_HEALTH_REFRESH_MS - 5_000),
    });
    renderPage(<ProvidersPage />, client);

    await screen.findByText("1 of 1 provider reachable");
    await waitFor(() => expect(healthRequestCount(fetchMock)).toBe(1));
  });

  it("forces a live re-check of every provider on Re-check all", async () => {
    const user = userEvent.setup();
    mockApi({
      meta: [providerInfo("openai")],
      health: [{ instance: "openai", ok: true, model_count: 3, error: null, checked_at: null }],
      healthRefresh: [{ instance: "openai", ok: false, model_count: 0, error: "provider down", checked_at: null }],
    });
    renderPage(<ProvidersPage />);

    const row = (await screen.findByText("Reachable")).closest("tr")!;
    expect(within(row).getAllByText("openai").length).toBeGreaterThan(0);

    await user.click(screen.getByRole("button", { name: "Re-check all" }));

    expect(await within(row).findByText("Unreachable")).toBeInTheDocument();
    expect(await screen.findByText("0 of 1 provider reachable")).toBeInTheDocument();
  });

  it("links a provider name to the filtered models page", async () => {
    mockApi({
      meta: [providerInfo("openai"), providerInfo("anthropic")],
    });

    renderPage(<ProvidersPage />);

    // Clicking a provider navigates to the Models page filtered to that provider.
    const openaiLink = await screen.findByRole("link", { name: "openai" });
    expect(openaiLink).toHaveAttribute("href", "/models?provider=openai");

    const anthropicLink = screen.getByRole("link", { name: "anthropic" });
    expect(anthropicLink).toHaveAttribute("href", "/models?provider=anthropic");
  });
});
