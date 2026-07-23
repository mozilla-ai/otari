import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactElement } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { Budget, User } from "@/api/types";
import { UsersPage } from "@/pages/UsersPage";

function user(overrides: Partial<User> = {}): User {
  return {
    user_id: "alice",
    alias: "Alice",
    spend: 0,
    reserved: 0,
    budget_id: null,
    allowed_models: null,
    budget_started_at: null,
    next_budget_reset_at: null,
    blocked: false,
    created_at: "2026-01-01T00:00:00+00:00",
    updated_at: "2026-01-01T00:00:00+00:00",
    metadata: {},
    ...overrides,
  };
}

function budget(overrides: Partial<Budget> = {}): Budget {
  return {
    budget_id: "bud-1111-2222",
    name: "team-free",
    max_budget: 100,
    budget_duration_sec: null,
    created_at: "2026-01-01T00:00:00+00:00",
    updated_at: "2026-01-01T00:00:00+00:00",
    user_count: 0,
    total_spend: 0,
    total_reserved: 0,
    ...overrides,
  };
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

function mockApi(opts: { users?: User[]; budgets?: Budget[] } = {}) {
  let list = [...(opts.users ?? [])];
  const budgets = opts.budgets ?? [];

  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input);
    const method = (init?.method ?? "GET").toUpperCase();

    if (url.includes("/v1/users")) {
      if (method === "POST") {
        const body = JSON.parse(String(init?.body)) as Partial<User>;
        const row = user({
          user_id: body.user_id ?? "new",
          alias: body.alias ?? null,
          budget_id: body.budget_id ?? null,
          allowed_models: body.allowed_models ?? null,
        });
        list = [...list, row];
        return jsonResponse(row);
      }
      if (method === "PATCH") {
        const id = decodeURIComponent(url.split("/").pop() ?? "");
        const body = JSON.parse(String(init?.body)) as Partial<User>;
        list = list.map((u) => (u.user_id === id ? { ...u, ...body } : u));
        return jsonResponse(list.find((u) => u.user_id === id));
      }
      if (method === "DELETE") {
        const id = decodeURIComponent(url.split("/").pop() ?? "");
        list = list.filter((u) => u.user_id !== id);
        return new Response(null, { status: 204 });
      }
      return jsonResponse(list);
    }
    if (url.includes("/v1/budgets")) {
      return jsonResponse(budgets);
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

describe("UsersPage", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("shows onboarding when there are no users", async () => {
    mockApi({ users: [] });
    renderPage(<UsersPage />);

    expect(await screen.findByText("No users yet")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Create your first user" })).toBeInTheDocument();
  });

  it("lists a user with status, assigned budget name, spend, and access", async () => {
    mockApi({
      users: [user({ user_id: "alice", budget_id: "bud-1111-2222", spend: 12.5, allowed_models: ["openai:*"] })],
      budgets: [budget({ budget_id: "bud-1111-2222", name: "team-free" })],
    });
    renderPage(<UsersPage />);

    const row = (await screen.findByText("alice")).closest("tr")!;
    expect(within(row).getByText("Active")).toBeInTheDocument();
    expect(within(row).getByText("team-free")).toBeInTheDocument();
    expect(within(row).getByText("$12.50")).toBeInTheDocument();
    // A wildcard is many models, so the chip says "Selected models".
    expect(within(row).getByText("Selected models")).toBeInTheDocument();
  });

  it("hides auto-created virtual users until the toggle is enabled", async () => {
    mockApi({ users: [user({ user_id: "apikey-abcdef", alias: null })] });
    const user_ = userEvent.setup();
    renderPage(<UsersPage />);

    // Hidden by default: a lone virtual user reads as an empty managed list.
    expect(await screen.findByText("No users yet")).toBeInTheDocument();
    expect(screen.queryByText("apikey-abcdef")).not.toBeInTheDocument();

    await user_.click(screen.getByRole("checkbox", { name: /Show auto-created/ }));

    const row = (await screen.findByText("apikey-abcdef")).closest("tr")!;
    expect(within(row).getByText("virtual")).toBeInTheDocument();
  });

  it("creates a user with an assigned budget", async () => {
    const fetchMock = mockApi({ users: [], budgets: [budget({ budget_id: "bud-1111-2222", name: "team-free" })] });
    const user_ = userEvent.setup();
    renderPage(<UsersPage />);

    await screen.findByText("No users yet");
    await user_.click(screen.getByRole("button", { name: "Create your first user" }));
    await user_.type(screen.getByLabelText("User ID"), "bob");
    await user_.type(screen.getByLabelText("Alias (optional)"), "Bob");
    await user_.selectOptions(screen.getByLabelText("Budget"), "bud-1111-2222");
    await user_.click(screen.getByRole("button", { name: "Create user" }));

    const post = fetchMock.mock.calls.find(
      ([u, init]) => String(u).endsWith("/v1/users") && (init?.method ?? "") === "POST",
    );
    expect(JSON.parse(String(post?.[1]?.body))).toEqual({
      user_id: "bob",
      alias: "Bob",
      budget_id: "bud-1111-2222",
      allowed_models: null,
    });
  });

  it("sets a per-user default model access via the scope control", async () => {
    const fetchMock = mockApi({ users: [], budgets: [] });
    const user_ = userEvent.setup();
    renderPage(<UsersPage />);

    await screen.findByText("No users yet");
    await user_.click(screen.getByRole("button", { name: "Create your first user" }));
    await user_.type(screen.getByLabelText("User ID"), "carol");
    await user_.click(screen.getByRole("button", { name: "Only selected" }));
    await user_.type(screen.getByLabelText("Add a model"), "gpt-4o");
    await user_.click(await screen.findByRole("option", { name: "openai:gpt-4o" }));
    await user_.keyboard("{Escape}");
    await user_.click(screen.getByRole("button", { name: "Create user" }));

    const post = fetchMock.mock.calls.find(
      ([u, init]) => String(u).endsWith("/v1/users") && (init?.method ?? "") === "POST",
    );
    expect(JSON.parse(String(post?.[1]?.body)).allowed_models).toEqual(["openai:gpt-4o"]);
  });

  it("blocks a user via PATCH, then offers to unblock", async () => {
    const fetchMock = mockApi({ users: [user({ user_id: "alice", blocked: false })] });
    const user_ = userEvent.setup();
    renderPage(<UsersPage />);

    const row = (await screen.findByText("alice")).closest("tr")!;
    await user_.click(within(row).getByRole("button", { name: "Block" }));

    const patch = fetchMock.mock.calls.find(
      ([u, init]) => String(u).includes("/v1/users/alice") && (init?.method ?? "") === "PATCH",
    );
    expect(JSON.parse(String(patch?.[1]?.body))).toEqual({ blocked: true });

    const blockedRow = (await screen.findByText("Blocked")).closest("tr")!;
    expect(within(blockedRow).getByRole("button", { name: "Unblock" })).toBeInTheDocument();
  });

  it("opens the edit form when a user row is clicked, seeded from the row", async () => {
    mockApi({ users: [user({ user_id: "alice", alias: "Alice" })] });
    const user_ = userEvent.setup();
    renderPage(<UsersPage />);

    await user_.click(await screen.findByText("alice"));
    expect(await screen.findByRole("button", { name: "Save changes" })).toBeInTheDocument();
    expect(screen.getByLabelText("Alias")).toHaveValue("Alice");
  });

  it("deletes a user after an explicit confirm", async () => {
    const fetchMock = mockApi({ users: [user({ user_id: "alice" })] });
    const user_ = userEvent.setup();
    renderPage(<UsersPage />);

    const row = (await screen.findByText("alice")).closest("tr")!;
    await user_.click(within(row).getByRole("button", { name: "Delete" }));
    expect(within(row).getByText(/deactivates its API keys/)).toBeInTheDocument();
    await user_.click(within(row).getByRole("button", { name: "Delete user" }));

    const del = fetchMock.mock.calls.find(
      ([u, init]) => String(u).includes("/v1/users/alice") && (init?.method ?? "") === "DELETE",
    );
    expect(del).toBeDefined();
    expect(screen.queryByText("alice")).not.toBeInTheDocument();
  });
});
