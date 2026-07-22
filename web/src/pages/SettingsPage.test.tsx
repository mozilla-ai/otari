import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactElement } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { setMasterKey } from "@/api/client";
import type { ConfigField, GatewaySettings, ReencryptProviderCredentialsResult, StoredProvider } from "@/api/types";
import { AuthProvider } from "@/auth/AuthContext";
import { SettingsPage, fieldMatches } from "@/pages/SettingsPage";

describe("fieldMatches", () => {
  const field: ConfigField = {
    key: "model_cache_ttl_seconds",
    value: 300,
    type: "int",
    settable: true,
    group: "Models & discovery",
    description: "TTL for the discovery cache.",
  };

  it("matches an empty query", () => {
    expect(fieldMatches(field, "")).toBe(true);
  });

  it("matches a substring of the key, description, or group", () => {
    expect(fieldMatches(field, "cache")).toBe(true);
    expect(fieldMatches(field, "discovery")).toBe(true);
    expect(fieldMatches(field, "ttl for")).toBe(true);
  });

  it("matches a fuzzy subsequence of the key", () => {
    expect(fieldMatches(field, "mctts")).toBe(true);
  });

  it("does not match unrelated text", () => {
    expect(fieldMatches(field, "database")).toBe(false);
  });
});

const CONFIG: ConfigField[] = [
  { key: "host", value: "0.0.0.0", type: "str", settable: false, group: "Server & database", description: "Host to bind." },
  { key: "cors_allow_origins", value: [], type: "list", settable: false, group: "Server & database", description: "Allowed origins." },
  { key: "budget_strategy", value: "for_update", type: "str", settable: false, group: "Metering & budgets", description: "Budget strategy." },
  { key: "model_discovery", value: true, type: "bool", settable: true, group: "Models & discovery", description: "Auto-discover models." },
  { key: "default_pricing", value: false, type: "bool", settable: true, group: "Metering & budgets", description: "Community default pricing." },
  { key: "require_pricing", value: false, type: "bool", settable: true, group: "Metering & budgets", description: "Reject unpriced models." },
  { key: "model_cache_ttl_seconds", value: 300, type: "int", settable: true, group: "Models & discovery", description: "Discovery cache TTL." },
  {
    key: "stream_missing_usage_policy",
    value: "estimate",
    type: "str",
    settable: true,
    group: "Metering & budgets",
    description: "How to bill a streamed response with no usage.",
    options: ["estimate", "fail", "allow_free"],
  },
  { key: "model_discovery_negative_ttl_seconds", value: 30, type: "float", settable: true, group: "Models & discovery", description: "Negative discovery cache TTL.", minimum: 0 },
  { key: "model_discovery_timeout_seconds", value: 10, type: "float", settable: true, group: "Models & discovery", description: "Per-provider discovery timeout.", exclusive_minimum: 0 },
  { key: "vision_describe_model", value: null, type: "str", settable: true, group: "Vision & file understanding", description: "Model used to caption images." },
];

const SETTINGS: GatewaySettings = {
  mode: "standalone",
  version: "1.2.3",
  model_discovery: true,
  default_pricing: false,
  require_pricing: false,
  master_key_source: "generated",
  config: CONFIG,
};

const PRICE_REFRESH = {
  fetched_at: "2026-07-22T00:00:00Z",
  added_count: 1,
  changed_count: 2,
  removed_count: 0,
  protected_model_count: 1,
  changes: [
    { model_key: "openai:gpt-4o-mini", change: "changed" },
    { model_key: "openai:gpt-5", change: "added" },
  ],
  changes_truncated: false,
};

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

function renderWithClient(ui: ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <AuthProvider>{ui}</AuthProvider>
    </QueryClientProvider>,
  );
}

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
}

// Serves the settings, and echoes PATCH bodies back merged onto the base (both
// the top-level flags and the matching config entries) so the UI reflects the
// change after a round trip, like the real backend.
function mockApi(
  initial: GatewaySettings = SETTINGS,
  stored: StoredProvider[] = [],
  reencryptResult: ReencryptProviderCredentialsResult = { reencrypted: 1, unreadable: 0 },
) {
  let current = initial;
  let storedList = [...stored];
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input);
    const method = (init?.method ?? "GET").toUpperCase();
    if (url.includes("/v1/pricing/refresh/confirm") && method === "POST") {
      return jsonResponse({ applied: true });
    }
    if (url.includes("/v1/pricing/refresh/reject") && method === "POST") {
      return new Response(null, { status: 204 });
    }
    if (url.includes("/v1/pricing/refresh") && method === "POST") {
      return jsonResponse(PRICE_REFRESH);
    }
    if (url.includes("/v1/provider-credentials/reencrypt") && method === "POST") {
      storedList = storedList.map((provider) => ({ ...provider, decryptable: reencryptResult.unreadable === 0 }));
      return jsonResponse(reencryptResult);
    }
    if (url.includes("/v1/provider-credentials")) {
      return jsonResponse(storedList);
    }
    if (url.includes("/v1/settings/master-key/rotate")) {
      return jsonResponse({ master_key: "otari-mk-new" });
    }
    if (url.includes("/v1/settings")) {
      if (method === "PATCH") {
        const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
        current = {
          ...current,
          ...body,
          config: current.config.map((field) => (field.key in body ? { ...field, value: body[field.key] as ConfigField["value"] } : field)),
        };
      }
      return jsonResponse(current);
    }
    return jsonResponse([]);
  });
}

describe("SettingsPage", () => {
  beforeEach(() => {
    window.sessionStorage.setItem("otari.dashboard.masterKey", "test-master-key");
    setMasterKey("test-master-key");
  });
  afterEach(() => {
    vi.restoreAllMocks();
    window.sessionStorage.clear();
    setMasterKey(null);
  });

  it("reflects the current settings on its switches", async () => {
    mockApi();

    renderWithClient(<SettingsPage />);

    // Wait for the settings to load (the switch renders unchecked while loading).
    await screen.findByText(/Version 1.2.3/);
    expect(screen.getByRole("switch", { name: "model_discovery" })).toHaveAttribute("aria-checked", "true");
    expect(screen.getByRole("switch", { name: "default_pricing" })).toHaveAttribute("aria-checked", "false");
    expect(screen.getByText("Credential security")).toBeInTheDocument();
    expect(screen.getByText("OTARI_SECRET_KEY=<new-key>,<old-key>")).toBeInTheDocument();
  });

  it("patches a setting when its switch is toggled", async () => {
    const fetchMock = mockApi();
    const user = userEvent.setup();

    renderWithClient(<SettingsPage />);
    // Wait for settings to load: the switch renders disabled until then, so
    // clicking too early is ignored.
    await screen.findByText(/Version 1.2.3/);

    await user.click(screen.getByRole("switch", { name: "model_discovery" }));

    const call = fetchMock.mock.calls.find(([, init]) => (init?.method ?? "") === "PATCH");
    expect(call).toBeDefined();
    expect(String(call?.[0])).toContain("/v1/settings");
    expect(JSON.parse(String(call?.[1]?.body))).toEqual({ model_discovery: false });

    // The switch reflects the new value after the round trip.
    expect(await screen.findByRole("switch", { name: "model_discovery" })).toHaveAttribute("aria-checked", "false");
  });

  it("regenerates the master key in a one-time dialog without dropping the settings view", async () => {
    const fetchMock = mockApi();
    const user = userEvent.setup();

    renderWithClient(<SettingsPage />);
    await screen.findByText(/Version 1.2.3/);

    await user.click(screen.getByRole("button", { name: "Regenerate" }));
    expect(screen.getByRole("alertdialog", { name: "Regenerate master key?" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Regenerate key" }));

    const revealedKey = await screen.findByDisplayValue("otari-mk-new");
    expect(revealedKey).toHaveAttribute("autocomplete", "off");
    expect(revealedKey).toHaveAttribute("data-1p-ignore");
    expect(revealedKey).toHaveAttribute("data-lpignore", "true");
    expect(screen.getByRole("alertdialog", { name: "Master key regenerated" })).toBeInTheDocument();
    expect(screen.getByText("Credential security")).toBeInTheDocument();
    expect(window.sessionStorage.getItem("otari.dashboard.masterKey")).toBe("otari-mk-new");

    await user.keyboard("{Escape}");
    expect(screen.getByDisplayValue("otari-mk-new")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "I’ve saved this key" }));
    expect(screen.queryByDisplayValue("otari-mk-new")).not.toBeInTheDocument();

    await user.click(screen.getByRole("switch", { name: "default_pricing" }));
    const patch = fetchMock.mock.calls.find(([, init]) => (init?.method ?? "") === "PATCH");
    expect(new Headers(patch?.[1]?.headers).get("Authorization")).toBe("Bearer otari-mk-new");
  });

  it("explains when a master key is managed in configuration", async () => {
    mockApi({ ...SETTINGS, master_key_source: "configured" });

    renderWithClient(<SettingsPage />);

    expect(await screen.findByText(/uses a key managed through OTARI_MASTER_KEY or config.yml/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Managed in configuration" })).toBeDisabled();
  });

  it("re-encrypts stored provider keys from the Credential security category", async () => {
    const fetchMock = mockApi(SETTINGS, [storedProvider("openai", "1234")]);
    const user = userEvent.setup();

    renderWithClient(<SettingsPage />);
    await screen.findByText("Credential security");

    await user.click(screen.getByRole("button", { name: "Re-encrypt provider keys" }));

    const call = fetchMock.mock.calls.find(
      ([url, init]) => String(url).endsWith("/v1/provider-credentials/reencrypt") && (init?.method ?? "") === "POST",
    );
    expect(call).toBeDefined();
    expect(await screen.findByText(/Re-encrypted 1 provider key/)).toBeInTheDocument();
  });

  it("reviews and accepts persisted default price updates", async () => {
    const fetchMock = mockApi();
    const user = userEvent.setup();

    renderWithClient(<SettingsPage />);
    await screen.findByText("Default pricing catalog");

    await user.click(screen.getByRole("button", { name: "Check for price updates" }));

    expect(await screen.findByRole("alertdialog", { name: "Review default price updates" })).toBeInTheDocument();
    expect(screen.getByText(/1 added, 2 changed, and 0 removed/)).toBeInTheDocument();
    expect(screen.getAllByText("genai-prices")).toHaveLength(2);
    expect(screen.getByText("openai:gpt-4o-mini: changed")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Accept price updates" }));

    expect(screen.queryByRole("alertdialog", { name: "Review default price updates" })).not.toBeInTheDocument();
    expect(
      fetchMock.mock.calls.some(
        ([url, init]) => String(url).endsWith("/v1/pricing/refresh/confirm") && init?.method === "POST",
      ),
    ).toBe(true);
  });

  it("rejects a reviewed default price update", async () => {
    const fetchMock = mockApi();
    const user = userEvent.setup();

    renderWithClient(<SettingsPage />);
    await screen.findByText("Default pricing catalog");
    await user.click(screen.getByRole("button", { name: "Check for price updates" }));
    await screen.findByRole("alertdialog", { name: "Review default price updates" });

    await user.click(screen.getByRole("button", { name: "Reject changes" }));

    expect(screen.queryByRole("alertdialog", { name: "Review default price updates" })).not.toBeInTheDocument();
    expect(
      fetchMock.mock.calls.some(
        ([url, init]) => String(url).endsWith("/v1/pricing/refresh/reject") && init?.method === "POST",
      ),
    ).toBe(true);
  });

  it("shows unreadable provider key recovery guidance in the Credential security category", async () => {
    mockApi(SETTINGS, [storedProvider("openai", "1234", false)]);

    renderWithClient(<SettingsPage />);

    expect(await screen.findByText(/1 stored provider key cannot be decrypted/)).toBeInTheDocument();
  });

  it("enables default pricing from off", async () => {
    const fetchMock = mockApi();
    const user = userEvent.setup();

    renderWithClient(<SettingsPage />);
    await screen.findByText(/Version 1.2.3/);

    await user.click(screen.getByRole("switch", { name: "default_pricing" }));

    const call = fetchMock.mock.calls.find(([, init]) => (init?.method ?? "") === "PATCH");
    expect(JSON.parse(String(call?.[1]?.body))).toEqual({ default_pricing: true });
  });

  it("changes the stream policy through its select", async () => {
    const fetchMock = mockApi();
    const user = userEvent.setup();

    renderWithClient(<SettingsPage />);
    await screen.findByText(/Version 1.2.3/);

    await user.selectOptions(screen.getByRole("combobox", { name: "stream_missing_usage_policy" }), "fail");

    const call = fetchMock.mock.calls.find(([, init]) => (init?.method ?? "") === "PATCH");
    expect(JSON.parse(String(call?.[1]?.body))).toEqual({ stream_missing_usage_policy: "fail" });
  });

  it("saves an integer setting only after Save is pressed", async () => {
    const fetchMock = mockApi();
    const user = userEvent.setup();

    renderWithClient(<SettingsPage />);
    await screen.findByText(/Version 1.2.3/);

    const input = screen.getByRole("spinbutton", { name: "model_cache_ttl_seconds" });
    await user.clear(input);
    await user.type(input, "60");

    // Nothing is sent while typing.
    expect(fetchMock.mock.calls.some(([, init]) => (init?.method ?? "") === "PATCH")).toBe(false);

    await user.click(screen.getByRole("button", { name: "Save model_cache_ttl_seconds" }));

    const call = fetchMock.mock.calls.find(([, init]) => (init?.method ?? "") === "PATCH");
    expect(JSON.parse(String(call?.[1]?.body))).toEqual({ model_cache_ttl_seconds: 60 });
  });

  it("renders startup-only fields read-only, with no control", async () => {
    mockApi();

    renderWithClient(<SettingsPage />);
    await screen.findByText(/Version 1.2.3/);

    // The host row shows its value and a startup-only marker, and offers no
    // interactive control keyed by its name.
    expect(screen.getByText("host")).toBeInTheDocument();
    expect(screen.getAllByText("startup-only").length).toBeGreaterThan(0);
    expect(screen.queryByRole("switch", { name: "host" })).toBeNull();
    expect(screen.queryByRole("spinbutton", { name: "host" })).toBeNull();
  });

  it("saves a free-text (nullable) setting", async () => {
    const fetchMock = mockApi();
    const user = userEvent.setup();

    renderWithClient(<SettingsPage />);
    await screen.findByText(/Version 1.2.3/);

    const input = screen.getByRole("textbox", { name: "vision_describe_model" });
    await user.type(input, "ollama/qwen2-vl");
    await user.click(screen.getByRole("button", { name: "Save vision_describe_model" }));

    const call = fetchMock.mock.calls.find(([, init]) => (init?.method ?? "") === "PATCH");
    expect(JSON.parse(String(call?.[1]?.body))).toEqual({ vision_describe_model: "ollama/qwen2-vl" });
  });

  it("clears a nullable setting to null when the field is emptied", async () => {
    // Start with the describe model set, then clear it: the PATCH must send an
    // explicit null (not "") so the backend distinguishes "unset" from "omitted".
    const withValue = {
      ...SETTINGS,
      config: SETTINGS.config.map((field) =>
        field.key === "vision_describe_model" ? { ...field, value: "ollama/qwen2-vl" } : field,
      ),
    };
    const fetchMock = mockApi(withValue);
    const user = userEvent.setup();

    renderWithClient(<SettingsPage />);
    await screen.findByText(/Version 1.2.3/);

    await user.clear(screen.getByRole("textbox", { name: "vision_describe_model" }));
    await user.click(screen.getByRole("button", { name: "Save vision_describe_model" }));

    const call = fetchMock.mock.calls.find(([, init]) => (init?.method ?? "") === "PATCH");
    expect(JSON.parse(String(call?.[1]?.body))).toEqual({ vision_describe_model: null });
  });

  it("saves a decimal to a float setting", async () => {
    const fetchMock = mockApi();
    const user = userEvent.setup();

    renderWithClient(<SettingsPage />);
    await screen.findByText(/Version 1.2.3/);

    const input = screen.getByRole("spinbutton", { name: "model_discovery_negative_ttl_seconds" });
    await user.clear(input);
    await user.type(input, "7.5");
    await user.click(screen.getByRole("button", { name: "Save model_discovery_negative_ttl_seconds" }));

    const call = fetchMock.mock.calls.find(([, init]) => (init?.method ?? "") === "PATCH");
    expect(JSON.parse(String(call?.[1]?.body))).toEqual({ model_discovery_negative_ttl_seconds: 7.5 });
  });

  it("disables Save at 0 for a gt=0 field, and enables it for a positive value", async () => {
    mockApi();
    const user = userEvent.setup();

    renderWithClient(<SettingsPage />);
    await screen.findByText(/Version 1.2.3/);

    const input = screen.getByRole("spinbutton", { name: "model_discovery_timeout_seconds" });
    const save = screen.getByRole("button", { name: "Save model_discovery_timeout_seconds" });

    // 0 violates gt=0, so Save stays disabled (no round-trip to a 422).
    await user.clear(input);
    await user.type(input, "0");
    expect(save).toBeDisabled();

    // A positive value is accepted.
    await user.clear(input);
    await user.type(input, "5");
    expect(save).not.toBeDisabled();
  });

  it("shows an empty state when nothing matches the search", async () => {
    mockApi();
    const user = userEvent.setup();

    renderWithClient(<SettingsPage />);
    await screen.findByText(/Version 1.2.3/);

    await user.type(screen.getByRole("searchbox", { name: "Search settings" }), "zzzznomatch");
    expect(screen.getByText("No settings match your search.")).toBeInTheDocument();
  });

  it("filters fields by fuzzy search, including a subsequence of the key", async () => {
    mockApi();
    const user = userEvent.setup();

    renderWithClient(<SettingsPage />);
    await screen.findByText(/Version 1.2.3/);

    // Both a startup-only and a settable field are visible up front.
    expect(screen.getByText("host")).toBeInTheDocument();
    expect(screen.getByText("model_cache_ttl_seconds")).toBeInTheDocument();

    // A fuzzy subsequence of the key still matches; unrelated fields drop out.
    await user.type(screen.getByRole("searchbox", { name: "Search settings" }), "mctts");
    expect(screen.getByText("model_cache_ttl_seconds")).toBeInTheDocument();
    expect(screen.queryByText("host")).toBeNull();
  });

  it("limits the list to settable fields when 'Settable only' is checked", async () => {
    mockApi();
    const user = userEvent.setup();

    renderWithClient(<SettingsPage />);
    await screen.findByText(/Version 1.2.3/);

    await user.click(screen.getByRole("checkbox", { name: "Settable only" }));

    // Startup-only fields drop out; settable ones remain.
    expect(screen.queryByText("host")).toBeNull();
    expect(screen.queryByText("budget_strategy")).toBeNull();
    expect(screen.getByText("model_cache_ttl_seconds")).toBeInTheDocument();
  });
});
