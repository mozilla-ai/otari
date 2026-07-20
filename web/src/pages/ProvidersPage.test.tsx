import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactElement } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { setMasterKey } from "@/api/client";
import type { GatewaySettings, KnownProvider, ProviderInfo, StoredProvider, TestProviderResult } from "@/api/types";
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
};

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

interface MockOpts {
  meta?: ProviderInfo[];
  stored?: StoredProvider[];
  settings?: GatewaySettings;
  testResult?: TestProviderResult;
  catalog?: KnownProvider[];
}

function mockApi(opts: MockOpts = {}) {
  let storedList = [...(opts.stored ?? [])];
  let settings = { ...(opts.settings ?? SETTINGS) };
  const meta = opts.meta ?? [];
  const testResult = opts.testResult ?? { ok: true, model_count: 3, error: null };
  const catalog = opts.catalog ?? [];

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

function renderPage(ui: ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
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
    await user.click(screen.getByRole("button", { name: "Add provider" }));
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

  it("shows the welcome onboarding when no provider is configured", async () => {
    mockApi({ meta: [], stored: [] });
    renderPage(<ProvidersPage />);

    expect(await screen.findByText("Welcome to Otari")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Add your first provider" })).toBeInTheDocument();
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
});
