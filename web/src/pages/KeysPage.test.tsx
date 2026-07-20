import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactElement } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { setMasterKey } from "@/api/client";
import type { ApiKey } from "@/api/types";
import { KeysPage } from "@/pages/KeysPage";

function apiKey(overrides: Partial<ApiKey> = {}): ApiKey {
  return {
    id: "key-1",
    key_prefix: "gw-AbC3dE",
    key_name: "ci-bot",
    user_id: "alice",
    created_at: "2026-01-01T00:00:00+00:00",
    last_used_at: null,
    expires_at: null,
    is_active: true,
    allowed_models: null,
    metadata: {},
    ...overrides,
  };
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

const NEW_SECRET = "gw-NEWSECRET0000000000000000000000000000000000000000000000";
const REGEN_SECRET = "gw-REGEN00000000000000000000000000000000000000000000000000";

function mockApi(opts: { keys?: ApiKey[] } = {}) {
  let list = [...(opts.keys ?? [])];

  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input);
    const method = (init?.method ?? "GET").toUpperCase();

    if (url.includes("/v1/keys")) {
      if (url.endsWith("/rotate") && method === "POST") {
        const id = url.split("/").slice(-2)[0];
        const prefix = REGEN_SECRET.slice(0, 10);
        list = list.map((k) => (k.id === id ? { ...k, key_prefix: prefix } : k));
        const row = list.find((k) => k.id === id) ?? apiKey({ id });
        return jsonResponse({ ...row, key: REGEN_SECRET, key_prefix: prefix });
      }
      if (method === "POST" && url.endsWith("/v1/keys")) {
        const body = JSON.parse(String(init?.body)) as {
          key_name?: string | null;
          user_id?: string | null;
          allowed_models?: string[] | null;
        };
        const row = apiKey({
          id: "key-new",
          key_prefix: NEW_SECRET.slice(0, 10),
          key_name: body.key_name ?? null,
          user_id: body.user_id ?? "apikey-key-new",
          allowed_models: body.allowed_models ?? null,
        });
        list = [...list, row];
        return jsonResponse({ ...row, key: NEW_SECRET });
      }
      if (method === "PATCH") {
        const id = decodeURIComponent(url.split("/").pop() ?? "");
        const body = JSON.parse(String(init?.body)) as Partial<ApiKey>;
        list = list.map((k) => (k.id === id ? { ...k, ...body } : k));
        return jsonResponse(list.find((k) => k.id === id));
      }
      if (method === "DELETE") {
        const id = decodeURIComponent(url.split("/").pop() ?? "");
        list = list.filter((k) => k.id !== id);
        return new Response(null, { status: 204 });
      }
      return jsonResponse(list);
    }
    if (url.includes("/v1/models/discoverable")) {
      return jsonResponse({
        providers: [{ provider: "openai", ok: true, error: null, models: [{ id: "gpt-4o", key: "openai:gpt-4o" }] }],
      });
    }
    if (url.includes("/v1/providers")) {
      return jsonResponse({ providers: [{ instance: "openai" }] });
    }
    if (url.includes("/v1/aliases")) {
      return jsonResponse([]);
    }
    return jsonResponse([]);
  });
}

function renderPage(ui: ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

describe("KeysPage", () => {
  beforeEach(() => setMasterKey("test-master-key"));
  afterEach(() => {
    vi.restoreAllMocks();
    setMasterKey(null);
  });

  it("lists keys with status and prefix, never the full secret", async () => {
    mockApi({
      keys: [
        apiKey({ id: "key-1", key_name: "ci-bot", key_prefix: "gw-AbC3dE", is_active: true }),
        apiKey({ id: "key-2", key_name: "legacy", key_prefix: null, is_active: false }),
      ],
    });
    renderPage(<KeysPage />);

    const activeRow = (await screen.findByText("ci-bot")).closest("tr")!;
    expect(within(activeRow).getByText("Active")).toBeInTheDocument();
    expect(within(activeRow).getByText("gw-AbC3dE…")).toBeInTheDocument();

    // A key minted before the prefix existed renders "—", not a crash.
    const legacyRow = screen.getByText("legacy").closest("tr")!;
    expect(within(legacyRow).getByText("Disabled")).toBeInTheDocument();
    expect(within(legacyRow).getByText("—")).toBeInTheDocument();

    expect(document.body.textContent).not.toContain(NEW_SECRET);
  });

  it("shows the plaintext + first-call snippet once, then only the prefix", async () => {
    mockApi({ keys: [] });
    const user = userEvent.setup();
    renderPage(<KeysPage />);

    await screen.findByText("No API keys yet");
    await user.click(screen.getByRole("button", { name: "Create your first key" }));
    await user.type(screen.getByLabelText("Name"), "deploy-key");
    await user.click(screen.getByRole("button", { name: "Create key" }));

    // The reveal shows the secret and a runnable curl snippet with the key injected.
    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByDisplayValue(NEW_SECRET)).toBeInTheDocument();
    const curl = within(dialog).getByDisplayValue(new RegExp(`Otari-Key: ${NEW_SECRET}`));
    expect(curl).toBeInTheDocument();
    expect((curl as HTMLTextAreaElement).value).toContain(`${window.location.origin}/v1/chat/completions`);

    await user.click(within(dialog).getByRole("button", { name: /I.?ve saved this key/ }));

    // After closing, the list shows only the prefix and the secret is gone from the DOM.
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(await screen.findByText(`${NEW_SECRET.slice(0, 10)}…`)).toBeInTheDocument();
    expect(document.body.textContent).not.toContain(NEW_SECRET);
  });

  it("does not close the reveal on Escape; requires the explicit save button", async () => {
    mockApi({ keys: [] });
    const user = userEvent.setup();
    renderPage(<KeysPage />);

    await screen.findByText("No API keys yet");
    await user.click(screen.getByRole("button", { name: "Create your first key" }));
    await user.click(screen.getByRole("button", { name: "Create key" }));

    const dialog = await screen.findByRole("dialog");
    await user.keyboard("{Escape}");
    expect(screen.getByRole("dialog")).toBeInTheDocument();

    await user.click(within(dialog).getByRole("button", { name: /I.?ve saved this key/ }));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("confirms Copied when the clipboard API is available", async () => {
    mockApi({ keys: [] });
    const user = userEvent.setup();
    // Install after userEvent.setup(), which otherwise replaces navigator.clipboard
    // with its own stub.
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", { configurable: true, value: { writeText } });
    renderPage(<KeysPage />);

    await screen.findByText("No API keys yet");
    await user.click(screen.getByRole("button", { name: "Create your first key" }));
    await user.click(screen.getByRole("button", { name: "Create key" }));

    const dialog = await screen.findByRole("dialog");
    const copyButtons = within(dialog).getAllByRole("button", { name: "Copy" });
    await user.click(copyButtons[0]);

    expect(writeText).toHaveBeenCalledWith(NEW_SECRET);
    expect(await within(dialog).findByText("Copied to clipboard.")).toBeInTheDocument();
  });

  it("disables an active key via PATCH, then offers permanent delete", async () => {
    const fetchMock = mockApi({ keys: [apiKey({ id: "key-1", key_name: "ci-bot", is_active: true })] });
    const user = userEvent.setup();
    renderPage(<KeysPage />);

    const row = (await screen.findByText("ci-bot")).closest("tr")!;
    // An active key offers no Delete (require-disable-first).
    expect(within(row).queryByRole("button", { name: "Delete" })).not.toBeInTheDocument();

    await user.click(within(row).getByRole("button", { name: "Disable" }));

    const patch = fetchMock.mock.calls.find(
      ([u, init]) => String(u).includes("/v1/keys/key-1") && (init?.method ?? "") === "PATCH",
    );
    expect(JSON.parse(String(patch?.[1]?.body))).toEqual({ is_active: false });

    // Once disabled, a Delete action appears.
    const disabledRow = (await screen.findByText("Disabled")).closest("tr")!;
    expect(within(disabledRow).getByRole("button", { name: "Delete" })).toBeInTheDocument();
  });

  it("regenerates a secret after an explicit confirm", async () => {
    mockApi({ keys: [apiKey({ id: "key-1", key_name: "ci-bot", is_active: true })] });
    const user = userEvent.setup();
    renderPage(<KeysPage />);

    const row = (await screen.findByText("ci-bot")).closest("tr")!;
    await user.click(within(row).getByRole("button", { name: "Regenerate" }));
    // Arming shows a named warning; a second click confirms.
    expect(within(row).getByText(/stops working immediately/)).toBeInTheDocument();
    await user.click(within(row).getByRole("button", { name: "Regenerate" }));

    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByDisplayValue(REGEN_SECRET)).toBeInTheDocument();
  });

  it("permanently deletes a disabled key after confirm", async () => {
    const fetchMock = mockApi({ keys: [apiKey({ id: "key-1", key_name: "legacy", is_active: false })] });
    const user = userEvent.setup();
    renderPage(<KeysPage />);

    const row = (await screen.findByText("legacy")).closest("tr")!;
    await user.click(within(row).getByRole("button", { name: "Delete" }));
    expect(within(row).getByText(/unlinks its usage history/)).toBeInTheDocument();
    await user.click(within(row).getByRole("button", { name: "Delete permanently" }));

    const del = fetchMock.mock.calls.find(
      ([u, init]) => String(u).includes("/v1/keys/key-1") && (init?.method ?? "") === "DELETE",
    );
    expect(del).toBeDefined();
    expect(screen.queryByText("legacy")).not.toBeInTheDocument();
  });

  it("creates a key restricted to selected models", async () => {
    const fetchMock = mockApi({ keys: [] });
    const user = userEvent.setup();
    renderPage(<KeysPage />);

    await screen.findByText("No API keys yet");
    await user.click(screen.getByRole("button", { name: "Create your first key" }));
    await user.click(screen.getByRole("button", { name: "Advanced (user, model access)" }));
    await user.click(screen.getByRole("button", { name: "Only selected" }));
    // The scope picker is a catalog combobox, not free text: type to filter, then
    // pick the discovered model.
    await user.type(screen.getByLabelText("Add a model"), "gpt-4o");
    await user.click(await screen.findByRole("option", { name: "openai:gpt-4o" }));
    // Close the combobox popover, which otherwise aria-hides the submit button.
    await user.keyboard("{Escape}");
    await user.click(screen.getByRole("button", { name: "Create key" }));

    const post = fetchMock.mock.calls.find(
      ([u, init]) => String(u).endsWith("/v1/keys") && (init?.method ?? "") === "POST",
    );
    expect(JSON.parse(String(post?.[1]?.body)).allowed_models).toEqual(["openai:gpt-4o"]);
  });

  it("blocks all models by posting an empty list", async () => {
    const fetchMock = mockApi({ keys: [] });
    const user = userEvent.setup();
    renderPage(<KeysPage />);

    await screen.findByText("No API keys yet");
    await user.click(screen.getByRole("button", { name: "Create your first key" }));
    await user.click(screen.getByRole("button", { name: "Advanced (user, model access)" }));
    await user.click(screen.getByRole("button", { name: "Block all" }));
    await user.click(screen.getByRole("button", { name: "Create key" }));

    const post = fetchMock.mock.calls.find(
      ([u, init]) => String(u).endsWith("/v1/keys") && (init?.method ?? "") === "POST",
    );
    expect(JSON.parse(String(post?.[1]?.body)).allowed_models).toEqual([]);
  });

  it("disables Create when 'Only selected' has no models (never a silent deny-all)", async () => {
    mockApi({ keys: [] });
    const user = userEvent.setup();
    renderPage(<KeysPage />);

    await screen.findByText("No API keys yet");
    await user.click(screen.getByRole("button", { name: "Create your first key" }));
    await user.click(screen.getByRole("button", { name: "Advanced (user, model access)" }));
    await user.click(screen.getByRole("button", { name: "Only selected" }));

    expect(screen.getByRole("button", { name: "Create key" })).toBeDisabled();
  });

  it("opens the edit form when a key row is clicked", async () => {
    mockApi({ keys: [apiKey({ id: "key-1", key_name: "ci-bot" })] });
    const user = userEvent.setup();
    renderPage(<KeysPage />);

    await user.click(await screen.findByText("ci-bot"));

    // The inline edit card appears (its Save button is unique to edit mode).
    expect(await screen.findByRole("button", { name: "Save changes" })).toBeInTheDocument();
  });

  it("resets the edit form when switching to a different key row", async () => {
    mockApi({
      keys: [apiKey({ id: "k1", key_name: "alpha" }), apiKey({ id: "k2", key_name: "bravo" })],
    });
    const user = userEvent.setup();
    renderPage(<KeysPage />);

    await user.click(await screen.findByText("alpha"));
    expect(await screen.findByLabelText("Name")).toHaveValue("alpha");

    // Switching to another key must remount the form; without a keyed remount it
    // would keep "alpha" and PATCH the wrong key.
    await user.click(screen.getByText("bravo"));
    expect(await screen.findByLabelText("Name")).toHaveValue("bravo");
  });

  it("clicking a row action does not also open the edit form", async () => {
    mockApi({ keys: [apiKey({ id: "key-1", key_name: "ci-bot", is_active: true })] });
    const user = userEvent.setup();
    renderPage(<KeysPage />);

    const row = (await screen.findByText("ci-bot")).closest("tr")!;
    await user.click(within(row).getByRole("button", { name: "Disable" }));

    expect(screen.queryByRole("button", { name: "Save changes" })).not.toBeInTheDocument();
  });

  it("shows a key's access scope in its row without a misleading count", async () => {
    mockApi({
      keys: [
        apiKey({ id: "k1", key_name: "scoped", allowed_models: ["openai:*", "openai:gpt-4o"] }),
        apiKey({ id: "k2", key_name: "open", allowed_models: null }),
        apiKey({ id: "k3", key_name: "locked", allowed_models: [] }),
      ],
    });
    renderPage(<KeysPage />);

    const scoped = (await screen.findByText("scoped")).closest("tr")!;
    // A wildcard is many models, so the chip says "Selected models", not "2 models".
    expect(within(scoped).getByText("Selected models")).toBeInTheDocument();
    expect(within(screen.getByText("open").closest("tr")!).getByText("All models")).toBeInTheDocument();
    expect(within(screen.getByText("locked").closest("tr")!).getByText("No models")).toBeInTheDocument();
  });

  it("flags an expired key and marks a virtual owner", async () => {
    mockApi({
      keys: [
        apiKey({
          id: "key-1",
          key_name: "old",
          is_active: true,
          expires_at: "2020-01-01T00:00:00+00:00",
          user_id: "apikey-abcdef",
        }),
      ],
    });
    renderPage(<KeysPage />);

    const row = (await screen.findByText("old")).closest("tr")!;
    expect(within(row).getByText("Expired")).toBeInTheDocument();
    expect(within(row).getByText("virtual")).toBeInTheDocument();
  });
});
