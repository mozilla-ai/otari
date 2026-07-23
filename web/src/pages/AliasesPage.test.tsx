import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import type { ReactElement } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { AliasResponse } from "@/api/types";
import { AliasesPage } from "@/pages/AliasesPage";

const ALIASES: AliasResponse[] = [
  { name: "fast-model", target: "openai:gpt-4o-mini", source: "config", created_at: null, updated_at: null },
  { name: "smart", target: "anthropic:claude-opus-4", source: "stored", created_at: null, updated_at: null },
];

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
}

function mockApi(aliases: AliasResponse[] = ALIASES) {
  let list = [...aliases];
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input);
    const method = (init?.method ?? "GET").toUpperCase();
    if (url.includes("/v1/aliases")) {
      if (method === "POST") {
        const body = JSON.parse(String(init?.body)) as { name: string; target: string };
        const existing = list.findIndex((alias) => alias.name === body.name);
        const row: AliasResponse = { ...body, source: "stored", created_at: null, updated_at: null };
        list = existing >= 0 ? list.map((alias, i) => (i === existing ? row : alias)) : [...list, row];
        return jsonResponse(row);
      }
      if (method === "DELETE") {
        const name = decodeURIComponent(url.split("/").pop() ?? "");
        list = list.filter((alias) => alias.name !== name);
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
    expect(JSON.parse(String(post?.[1]?.body))).toEqual({ name: "smart", target: "anthropic:claude-opus-4" });
  });

  it("deletes a stored alias", async () => {
    const fetchMock = mockApi([
      { name: "smart", target: "anthropic:claude-opus-4", source: "stored", created_at: null, updated_at: null },
    ]);
    const user = userEvent.setup();
    renderPage(<AliasesPage />);

    const row = (await screen.findByText("smart")).closest("tr")!;
    await user.click(within(row).getByRole("button", { name: "Delete" }));
    await user.click(within(row).getByRole("button", { name: "Delete" }));

    const del = fetchMock.mock.calls.find(([, init]) => (init?.method ?? "") === "DELETE");
    expect(String(del?.[0])).toContain("/v1/aliases/smart");
  });

  it("edits a stored alias target", async () => {
    const fetchMock = mockApi([
      { name: "smart", target: "anthropic:claude-opus-4", source: "stored", created_at: null, updated_at: null },
    ]);
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
    expect(JSON.parse(String(post?.[1]?.body))).toEqual({ name: "smart", target: "openai:gpt-4o" });
  });

  it("opening the edit form closes the create form", async () => {
    mockApi([
      { name: "smart", target: "anthropic:claude-opus-4", source: "stored", created_at: null, updated_at: null },
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
});
