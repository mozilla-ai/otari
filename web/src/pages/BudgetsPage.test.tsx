import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactElement } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { setMasterKey } from "@/api/client";
import type { Budget, BudgetResetLog, User } from "@/api/types";
import { BudgetsPage } from "@/pages/BudgetsPage";

function testUser(user_id: string): User {
  return {
    user_id,
    alias: null,
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
  };
}

function budget(overrides: Partial<Budget> = {}): Budget {
  return {
    budget_id: "11111111-2222-3333-4444-555555555555",
    name: null,
    max_budget: 100,
    budget_duration_sec: 86_400,
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

function mockApi(
  opts: {
    budgets?: Budget[];
    resetLogs?: BudgetResetLog[];
    users?: User[];
    failedUserUpdates?: string[];
    updateUser?: (userId: string) => Response | Promise<Response>;
  } = {},
) {
  let list = [...(opts.budgets ?? [])];
  const resetLogs = opts.resetLogs ?? [];
  const users = opts.users ?? [];

  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input);
    const method = (init?.method ?? "GET").toUpperCase();

    if (url.includes("/v1/users")) {
      if (method === "PATCH") {
        const userId = decodeURIComponent(url.split("/").pop() ?? "");
        if (opts.failedUserUpdates?.includes(userId)) {
          return jsonResponse({ detail: "User update failed" }, 500);
        }
        if (opts.updateUser) {
          return opts.updateUser(userId);
        }
        return jsonResponse(testUser(userId));
      }
      return jsonResponse(users);
    }

    if (url.includes("/v1/budgets")) {
      if (url.includes("/reset-logs")) {
        return jsonResponse(resetLogs);
      }
      if (method === "POST") {
        const body = JSON.parse(String(init?.body)) as Partial<Budget>;
        const row = budget({
          budget_id: "new-budget-id-0000-0000-000000000000",
          name: body.name ?? null,
          max_budget: body.max_budget ?? null,
          budget_duration_sec: body.budget_duration_sec ?? null,
        });
        list = [...list, row];
        return jsonResponse(row);
      }
      if (method === "PATCH") {
        const id = decodeURIComponent(url.split("/").pop() ?? "");
        const body = JSON.parse(String(init?.body)) as Partial<Budget>;
        list = list.map((b) => (b.budget_id === id ? { ...b, ...body } : b));
        return jsonResponse(list.find((b) => b.budget_id === id));
      }
      if (method === "DELETE") {
        const id = decodeURIComponent(url.split("/").pop() ?? "");
        list = list.filter((b) => b.budget_id !== id);
        return new Response(null, { status: 204 });
      }
      return jsonResponse(list);
    }
    return jsonResponse([]);
  });
}

function renderPage(ui: ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

describe("BudgetsPage", () => {
  beforeEach(() => setMasterKey("test-master-key"));
  afterEach(() => {
    vi.restoreAllMocks();
    setMasterKey(null);
  });

  it("shows onboarding when there are no budgets", async () => {
    mockApi({ budgets: [] });
    renderPage(<BudgetsPage />);

    expect(await screen.findByText("No budgets yet")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Create your first budget" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Create budget" })).not.toBeInTheDocument();
  });

  it("lists a budget with its limit and humanized reset period", async () => {
    mockApi({ budgets: [budget({ max_budget: 100, budget_duration_sec: 604_800 })] });
    renderPage(<BudgetsPage />);

    const row = (await screen.findByText("11111111")).closest("tr")!;
    expect(within(row).getByText("$100.00")).toBeInTheDocument();
    expect(within(row).getByText("Weekly")).toBeInTheDocument();
  });

  it("renders an unlimited budget without a spend bar", async () => {
    mockApi({ budgets: [budget({ max_budget: null, budget_duration_sec: null, user_count: 0 })] });
    renderPage(<BudgetsPage />);

    const row = (await screen.findByText("11111111")).closest("tr")!;
    expect(within(row).getByText("Unlimited")).toBeInTheDocument();
    expect(within(row).getByText("No reset")).toBeInTheDocument();
    expect(within(row).getByText("No users assigned")).toBeInTheDocument();
  });

  it("shows an aggregate spend bar across assigned users", async () => {
    mockApi({ budgets: [budget({ max_budget: 100, user_count: 3, total_spend: 150 })] });
    renderPage(<BudgetsPage />);

    const row = (await screen.findByText("11111111")).closest("tr")!;
    // 3 users × $100 = $300 allocated; $150 spent.
    expect(within(row).getByText("$150.00")).toBeInTheDocument();
    expect(within(row).getByText("of $300.00")).toBeInTheDocument();
    const bar = within(row).getByRole("progressbar");
    expect(bar).toHaveAttribute("aria-valuenow", "50");
  });

  it("creates a budget, posting the limit and chosen period", async () => {
    const fetchMock = mockApi({ budgets: [] });
    const user = userEvent.setup();
    renderPage(<BudgetsPage />);

    await screen.findByText("No budgets yet");
    await user.click(screen.getByRole("button", { name: "Create your first budget" }));
    await user.type(screen.getByLabelText("Name (optional)"), "team-free-tier");
    await user.type(screen.getByLabelText("Spending limit (USD)"), "250");
    await user.click(screen.getByRole("button", { name: "Weekly" }));
    await user.click(screen.getByRole("button", { name: "Create budget" }));

    const post = fetchMock.mock.calls.find(
      ([u, init]) => String(u).includes("/v1/budgets") && (init?.method ?? "") === "POST",
    );
    expect(JSON.parse(String(post?.[1]?.body))).toEqual({
      name: "team-free-tier",
      max_budget: 250,
      budget_duration_sec: 604_800,
    });

    // The created budget shows its name in the table.
    expect(await screen.findByText("team-free-tier")).toBeInTheDocument();
  });

  it("assigns the new budget to chosen users on create", async () => {
    const fetchMock = mockApi({ budgets: [], users: [testUser("alice"), testUser("bob")] });
    const user = userEvent.setup();
    renderPage(<BudgetsPage />);

    await screen.findByText("No budgets yet");
    await user.click(screen.getByRole("button", { name: "Create your first budget" }));
    await user.type(screen.getByLabelText("Spending limit (USD)"), "100");
    // Pick a user from the assignment combobox, then submit.
    await user.type(screen.getByLabelText("Add a user"), "alice");
    await user.click(await screen.findByRole("option", { name: /alice/ }));
    await user.keyboard("{Escape}");
    await user.click(screen.getByRole("button", { name: "Create budget" }));

    // The chosen user is PATCHed onto the newly created budget's id.
    const patch = await vi.waitFor(() => {
      const call = fetchMock.mock.calls.find(
        ([u, init]) => String(u).includes("/v1/users/alice") && (init?.method ?? "") === "PATCH",
      );
      if (!call) throw new Error("no PATCH yet");
      return call;
    });
    expect(JSON.parse(String(patch[1]?.body))).toEqual({ budget_id: "new-budget-id-0000-0000-000000000000" });
  });

  it("keeps failed initial assignments retryable without creating another budget", async () => {
    const fetchMock = mockApi({ budgets: [], users: [testUser("alice")], failedUserUpdates: ["alice"] });
    const user = userEvent.setup();
    renderPage(<BudgetsPage />);

    await user.click(await screen.findByRole("button", { name: "Create your first budget" }));
    await user.type(screen.getByLabelText("Add a user"), "alice");
    await user.click(await screen.findByRole("option", { name: /alice/ }));
    await user.keyboard("{Escape}");
    await user.click(screen.getByRole("button", { name: "Create budget" }));

    expect(await screen.findByText(/could not assign it to: alice/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Retry assignments" }));
    await vi.waitFor(() => {
      const patches = fetchMock.mock.calls.filter(
        ([url, init]) => String(url).includes("/v1/users/alice") && (init?.method ?? "") === "PATCH",
      );
      expect(patches).toHaveLength(2);
    });

    const budgetPosts = fetchMock.mock.calls.filter(
      ([url, init]) => String(url).includes("/v1/budgets") && (init?.method ?? "") === "POST",
    );
    expect(budgetPosts).toHaveLength(1);
  });

  it("prevents closing the form while initial user assignments are pending", async () => {
    let resolveUserUpdate: ((response: Response) => void) | undefined;
    const userUpdate = new Promise<Response>((resolve) => {
      resolveUserUpdate = resolve;
    });
    mockApi({ budgets: [], users: [testUser("alice")], updateUser: () => userUpdate });
    const user = userEvent.setup();
    renderPage(<BudgetsPage />);

    await user.click(await screen.findByRole("button", { name: "Create your first budget" }));
    await user.type(screen.getByLabelText("Add a user"), "alice");
    await user.click(await screen.findByRole("option", { name: /alice/ }));
    await user.keyboard("{Escape}");
    await user.click(screen.getByRole("button", { name: "Create budget" }));

    await vi.waitFor(() => expect(screen.getByRole("button", { name: "Cancel" })).toBeDisabled());
    await act(async () => {
      resolveUserUpdate?.(jsonResponse(testUser("alice")));
      await userUpdate;
    });
    await vi.waitFor(() => expect(screen.queryByRole("button", { name: "Cancel" })).not.toBeInTheDocument());
  });

  it("does not submit a non-finite budget limit", async () => {
    const fetchMock = mockApi({ budgets: [] });
    const user = userEvent.setup();
    renderPage(<BudgetsPage />);

    await user.click(await screen.findByRole("button", { name: "Create your first budget" }));
    await user.type(screen.getByLabelText("Spending limit (USD)"), "1e309");

    expect(screen.getByRole("button", { name: "Create budget" })).toBeDisabled();
    expect(
      fetchMock.mock.calls.some(
        ([url, init]) => String(url).includes("/v1/budgets") && (init?.method ?? "") === "POST",
      ),
    ).toBe(false);
  });

  it("does not submit a custom period that rounds down to zero days", async () => {
    const fetchMock = mockApi({ budgets: [] });
    const user = userEvent.setup();
    renderPage(<BudgetsPage />);

    await user.click(await screen.findByRole("button", { name: "Create your first budget" }));
    await user.click(screen.getByRole("button", { name: "Custom" }));
    await user.click(screen.getByLabelText("Every N days"));
    await user.paste("0.1");
    await user.click(screen.getByRole("button", { name: "Create budget" }));

    const post = await vi.waitFor(() => {
      const call = fetchMock.mock.calls.find(
        ([url, init]) => String(url).includes("/v1/budgets") && (init?.method ?? "") === "POST",
      );
      if (!call) throw new Error("no budget POST");
      return call;
    });
    expect(JSON.parse(String(post[1]?.body))).toMatchObject({ budget_duration_sec: null });
  });

  it("creates an unlimited budget when the limit is left blank", async () => {
    const fetchMock = mockApi({ budgets: [] });
    const user = userEvent.setup();
    renderPage(<BudgetsPage />);

    await screen.findByText("No budgets yet");
    await user.click(screen.getByRole("button", { name: "Create your first budget" }));
    // Leave the limit blank; keep "No reset" (the default selection).
    await user.click(screen.getByRole("button", { name: "Create budget" }));

    const post = fetchMock.mock.calls.find(
      ([u, init]) => String(u).includes("/v1/budgets") && (init?.method ?? "") === "POST",
    );
    expect(JSON.parse(String(post?.[1]?.body))).toEqual({ name: null, max_budget: null, budget_duration_sec: null });
  });

  it("opens the edit form seeded from the row when a budget is clicked", async () => {
    mockApi({ budgets: [budget({ max_budget: 42, budget_duration_sec: 86_400 })] });
    const user = userEvent.setup();
    renderPage(<BudgetsPage />);

    await user.click(await screen.findByText("11111111"));

    expect(await screen.findByRole("button", { name: "Save changes" })).toBeInTheDocument();
    expect(screen.getByLabelText("Spending limit (USD)")).toHaveValue("42");
  });

  it("reveals per-user reset history on demand", async () => {
    mockApi({
      budgets: [budget()],
      resetLogs: [
        {
          id: 1,
          user_id: "alice",
          budget_id: "11111111-2222-3333-4444-555555555555",
          previous_spend: 12.5,
          reset_at: "2026-02-01T00:00:00+00:00",
          next_reset_at: "2026-02-02T00:00:00+00:00",
        },
      ],
    });
    const user = userEvent.setup();
    renderPage(<BudgetsPage />);

    const row = (await screen.findByText("11111111")).closest("tr")!;
    await user.click(within(row).getByRole("button", { name: "History" }));

    expect(await screen.findByText("alice")).toBeInTheDocument();
    expect(screen.getByText("$12.50")).toBeInTheDocument();
  });

  it("deletes a budget after an explicit confirm", async () => {
    const fetchMock = mockApi({ budgets: [budget()] });
    const user = userEvent.setup();
    renderPage(<BudgetsPage />);

    const row = (await screen.findByText("11111111")).closest("tr")!;
    await user.click(within(row).getByRole("button", { name: "Delete" }));
    expect(within(row).getByText(/lose this limit/)).toBeInTheDocument();
    await user.click(within(row).getByRole("button", { name: "Delete permanently" }));

    const del = fetchMock.mock.calls.find(
      ([u, init]) => String(u).includes("/v1/budgets/") && (init?.method ?? "") === "DELETE",
    );
    expect(del).toBeDefined();
    expect(screen.queryByText("11111111")).not.toBeInTheDocument();
  });
});
