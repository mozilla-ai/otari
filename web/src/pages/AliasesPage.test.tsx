import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import type { ReactElement } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { AliasResponse } from "@/api/types";
import { AliasesPage } from "@/pages/AliasesPage";

const stored = (name: string, target: string, user_id: string | null = null): AliasResponse => ({
  name,
  target,
  source: "stored",
  user_id,
  created_at: null,
  updated_at: null,
});

const ALIASES: AliasResponse[] = [
  { name: "fast-model", target: "openai:gpt-4o-mini", source: "config", user_id: null, created_at: null, updated_at: null },
  stored("smart", "anthropic:claude-opus-4"),
];

const USERS = [
  { user_id: "alice", alias: "alice", spend: 0, is_blocked: false },
  { user_id: "bob", alias: "bob", spend: 0, is_blocked: false },
];

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
}

function mockApi(aliases: AliasResponse[] = ALIASES) {
  let list = [...aliases];
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input);
    const method = (init?.method ?? "GET").toUpperCase();
    if (url.includes("/v1/users")) {
      return jsonResponse(USERS);
    }
    if (url.includes("/v1/aliases")) {
      if (method === "POST") {
        const body = JSON.parse(String(init?.body)) as { name: string; target: string; user_id?: string | null };
        const userId = body.user_id ?? null;
        // Scope is part of the identity, matching the backend upsert.
        const existing = list.findIndex((alias) => alias.name === body.name && alias.user_id === userId);
        const row = stored(body.name, body.target, userId);
        list = existing >= 0 ? list.map((alias, i) => (i === existing ? row : alias)) : [...list, row];
        return jsonResponse(row);
      }
      if (method === "DELETE") {
        const [path, query] = url.split("?");
        const name = decodeURIComponent(path.split("/").pop() ?? "");
        const userId = new URLSearchParams(query ?? "").get("user_id");
        list = list.filter((alias) => !(alias.name === name && alias.user_id === userId));
        return new Response(null, { status: 204 });
      }
      return jsonResponse(list);
    }
    // ModelComboBox loads discoverable models; none needed here.
    return jsonResponse({ providers: [] });
  });
}

function renderPage(ui: ReactElement, route = "/aliases") {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <MemoryRouter initialEntries={[route]}>
      <QueryClientProvider client={client}>{ui}</QueryClientProvider>
    </MemoryRouter>,
  );
}

describe("AliasesPage", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("lists aliases with provenance; config is read-only, stored is deletable", async () => {
    mockApi();
    renderPage(<AliasesPage />);

    const configRow = (await screen.findByText("fast-model")).closest("tr")!;
    expect(within(configRow).getByText("config")).toBeInTheDocument();
    expect(within(configRow).getByText("set in config.yml")).toBeInTheDocument();

    const storedRow = screen.getByText("smart").closest("tr")!;
    expect(within(storedRow).getByText("stored")).toBeInTheDocument();
    expect(within(storedRow).getByRole("button", { name: "Delete" })).toBeInTheDocument();
  });

  it("creates a stored alias", async () => {
    const fetchMock = mockApi([]);
    const user = userEvent.setup();
    renderPage(<AliasesPage />);

    await user.click(screen.getByRole("button", { name: "New alias" }));
    await user.type(screen.getByRole("textbox", { name: /alias name/i }), "smart");
    await user.type(screen.getByRole("combobox", { name: /target/i }), "anthropic:claude-opus-4");
    // Close the combobox popover, which otherwise aria-hides the submit button.
    await user.keyboard("{Escape}");
    await user.click(screen.getByRole("button", { name: /create alias/i }));

    const post = fetchMock.mock.calls.find(
      ([u, init]) => String(u).includes("/v1/aliases") && (init?.method ?? "") === "POST",
    );
    // user_id null: "Every caller" is the default scope, same as before scoping existed.
    expect(JSON.parse(String(post?.[1]?.body))).toEqual({
      name: "smart",
      target: "anthropic:claude-opus-4",
      user_id: null,
    });
  });

  it("deletes a stored alias", async () => {
    const fetchMock = mockApi([stored("smart", "anthropic:claude-opus-4")]);
    const user = userEvent.setup();
    renderPage(<AliasesPage />);

    const row = (await screen.findByText("smart")).closest("tr")!;
    await user.click(within(row).getByRole("button", { name: "Delete" }));
    await user.click(within(row).getByRole("button", { name: "Delete" }));

    const del = fetchMock.mock.calls.find(([, init]) => (init?.method ?? "") === "DELETE");
    expect(String(del?.[0])).toContain("/v1/aliases/smart");
  });

  it("edits a stored alias target", async () => {
    const fetchMock = mockApi([stored("smart", "anthropic:claude-opus-4")]);
    const user = userEvent.setup();
    renderPage(<AliasesPage />);

    const row = (await screen.findByText("smart")).closest("tr")!;
    await user.click(within(row).getByRole("button", { name: "Edit" }));

    const targetInput = screen.getByRole("combobox", { name: /target/i });
    await user.clear(targetInput);
    await user.type(targetInput, "openai:gpt-4o");
    await user.keyboard("{Escape}");
    await user.click(screen.getByRole("button", { name: /save changes/i }));

    const post = fetchMock.mock.calls.find(
      ([u, init]) => String(u).includes("/v1/aliases") && (init?.method ?? "") === "POST",
    );
    expect(JSON.parse(String(post?.[1]?.body))).toEqual({ name: "smart", target: "openai:gpt-4o", user_id: null });
  });

  it("opening the edit form closes the create form", async () => {
    mockApi([
      stored("smart", "anthropic:claude-opus-4"),
    ]);
    const user = userEvent.setup();
    renderPage(<AliasesPage />);

    await user.click(await screen.findByRole("button", { name: "New alias" }));
    expect(screen.getByText("New alias")).toBeInTheDocument();

    const row = screen.getByText("smart").closest("tr")!;
    await user.click(within(row).getByRole("button", { name: "Edit" }));

    expect(screen.queryByText("New alias")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /save changes/i })).toBeInTheDocument();
  });

  it("refuses an alias name that could be mistaken for a model key", async () => {
    mockApi([]);
    const user = userEvent.setup();
    renderPage(<AliasesPage />);

    await user.click(screen.getByRole("button", { name: "New alias" }));
    await user.type(screen.getByRole("textbox", { name: /alias name/i }), "openai:fast");

    expect(screen.getByText(/cannot contain/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /create alias/i })).toBeDisabled();
  });

  it("shows who each alias applies to", async () => {
    mockApi([stored("smart", "anthropic:claude-opus-4"), stored("smart", "openai:gpt-4o", "alice")]);
    renderPage(<AliasesPage />);

    // Both rows survive: the name alone is not the identity, so a per-user
    // override must not replace the global row in the table.
    const rows = (await screen.findAllByText("smart")).map((cell) => cell.closest("tr")!);
    expect(rows).toHaveLength(2);
    const scopes = rows.map((row) => (within(row).queryByText("alice") ? "alice" : "Every caller"));
    expect(scopes.sort()).toEqual(["Every caller", "alice"]);
  });

  it("creates a user-scoped alias", async () => {
    const fetchMock = mockApi([]);
    const user = userEvent.setup();
    renderPage(<AliasesPage />);

    await user.click(screen.getByRole("button", { name: "New alias" }));
    await user.type(screen.getByRole("textbox", { name: /alias name/i }), "smart");
    await user.type(screen.getByRole("combobox", { name: /target/i }), "anthropic:claude-opus-4");
    await user.keyboard("{Escape}");
    await user.click(screen.getByRole("button", { name: "One user" }));
    await user.type(screen.getByRole("combobox", { name: /^user$/i }), "alice");
    await user.keyboard("{Escape}");
    await user.click(screen.getByRole("button", { name: /create alias/i }));

    const post = fetchMock.mock.calls.find(
      ([u, init]) => String(u).includes("/v1/aliases") && (init?.method ?? "") === "POST",
    );
    expect(JSON.parse(String(post?.[1]?.body))).toEqual({
      name: "smart",
      target: "anthropic:claude-opus-4",
      user_id: "alice",
    });
  });

  it("will not submit a user-scoped alias with no user picked", async () => {
    mockApi([]);
    const user = userEvent.setup();
    renderPage(<AliasesPage />);

    await user.click(screen.getByRole("button", { name: "New alias" }));
    await user.type(screen.getByRole("textbox", { name: /alias name/i }), "smart");
    await user.type(screen.getByRole("combobox", { name: /target/i }), "anthropic:claude-opus-4");
    await user.keyboard("{Escape}");
    await user.click(screen.getByRole("button", { name: "One user" }));

    // An empty user must be an incomplete form, never a silent global alias.
    expect(screen.getByRole("button", { name: /create alias/i })).toBeDisabled();
  });

  it("deletes only the scope that was asked for", async () => {
    const fetchMock = mockApi([stored("smart", "anthropic:claude-opus-4"), stored("smart", "openai:gpt-4o", "alice")]);
    const user = userEvent.setup();
    renderPage(<AliasesPage />);

    const scopedRow = (await screen.findByText("alice")).closest("tr")!;
    await user.click(within(scopedRow).getByRole("button", { name: "Delete" }));
    await user.click(within(scopedRow).getByRole("button", { name: "Delete" }));

    const del = fetchMock.mock.calls.find(([, init]) => (init?.method ?? "") === "DELETE");
    expect(String(del?.[0])).toContain("user_id=alice");
    // The global row is untouched.
    await vi.waitFor(() => {
      expect(screen.queryByText("alice")).not.toBeInTheDocument();
    });
    expect(screen.getByText("smart")).toBeInTheDocument();
  });

  it("bulk-deletes the right row when a user id is not delimiter-free", async () => {
    // Row keys pack scope and name together and a user_id has no format
    // restriction, so this pins the delete down to the row that was selected
    // however that key is built.
    const fetchMock = mockApi([stored("smart", "openai:gpt-4o", "team alpha")]);
    const user = userEvent.setup();
    renderPage(<AliasesPage />);

    const row = (await screen.findByText("smart")).closest("tr")!;
    await user.click(within(row).getByRole("checkbox"));
    const bar = (await screen.findByText("1 selected")).closest("div")!;
    await user.click(within(bar).getByRole("button", { name: "Delete" }));
    const dialog = await screen.findByRole("alertdialog");
    await user.click(within(dialog).getByRole("button", { name: "Delete" }));

    await vi.waitFor(() => {
      const del = fetchMock.mock.calls.find(([, init]) => (init?.method ?? "").toUpperCase() === "DELETE");
      expect(del).toBeTruthy();
      expect(new URL(String(del![0]), "http://x").searchParams.get("user_id")).toBe("team alpha");
    });
  });

  it("treats an empty-string user id as a scope, not as global", async () => {
    // "" is a legal user id, so a truthiness check on the scope would drop the
    // query param and delete the global alias instead of this one.
    const fetchMock = mockApi([stored("shared", "openai:gpt-4o"), stored("shared", "openai:gpt-4o-mini", "")]);
    const user = userEvent.setup();
    renderPage(<AliasesPage />);

    const scopedRow = (await screen.findByText("openai:gpt-4o-mini")).closest("tr")!;
    await user.click(within(scopedRow).getByRole("button", { name: "Delete" }));
    await user.click(within(scopedRow).getByRole("button", { name: "Delete" }));

    const del = fetchMock.mock.calls.find(([, init]) => (init?.method ?? "").toUpperCase() === "DELETE");
    expect(String(del?.[0])).toContain("user_id=");
  });

  it("only lets stored aliases be selected, and bulk-deletes them", async () => {
    const fetchMock = mockApi();
    const user = userEvent.setup();
    renderPage(<AliasesPage />);

    const configRow = (await screen.findByText("fast-model")).closest("tr")!;
    const storedRow = screen.getByText("smart").closest("tr")!;
    // config alias is read-only, so its checkbox is disabled.
    expect(within(configRow).getByRole("checkbox")).toBeDisabled();
    await user.click(within(storedRow).getByRole("checkbox"));

    const bar = (await screen.findByText("1 selected")).closest("div")!;
    await user.click(within(bar).getByRole("button", { name: "Delete" }));

    const dialog = await screen.findByRole("alertdialog");
    await user.click(within(dialog).getByRole("button", { name: "Delete" }));

    await vi.waitFor(() => {
      const del = fetchMock.mock.calls.find(
        ([u, init]) => String(u).includes("/v1/aliases/") && (init?.method ?? "").toUpperCase() === "DELETE",
      );
      expect(del).toBeTruthy();
      expect(String(del![0])).toContain("smart");
    });
  });
});
