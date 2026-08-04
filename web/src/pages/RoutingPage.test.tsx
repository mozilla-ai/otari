import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import type { ReactElement } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { PolicySpec, RoutingPolicyResponse } from "@/api/types";
import { RoutingPage } from "@/pages/RoutingPage";

const policy = (
  name: string,
  spec: PolicySpec,
  overrides: Partial<RoutingPolicyResponse> = {},
): RoutingPolicyResponse => ({
  name,
  spec,
  source: "stored",
  user_id: null,
  is_dynamic: false,
  created_at: null,
  updated_at: null,
  ...overrides,
});

const CHAIN: PolicySpec = {
  select: [{ default: "openai:gpt-5-mini" }],
  on_failure: ["anthropic:claude-haiku-4-5"],
};

const POLICIES: RoutingPolicyResponse[] = [
  policy("fast", CHAIN),
  policy(
    "auto",
    {
      select: [
        { when: { budget_used_pct: { gte: 80 } }, target: "openai:gpt-5-nano" },
        { default: "openai:gpt-5-mini" },
      ],
    },
    { source: "config", is_dynamic: true },
  ),
];

const USERS = [{ user_id: "alice", alias: "alice", spend: 0, is_blocked: false }];

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
}

function mockApi(policies: RoutingPolicyResponse[] = POLICIES, guardrailsUrl: string | null = "http://guardrails:8000") {
  let list = [...policies];
  const calls: { url: string; method: string; body: unknown }[] = [];
  const spy = vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input);
    const method = (init?.method ?? "GET").toUpperCase();
    const body = init?.body === undefined ? undefined : JSON.parse(String(init.body));
    calls.push({ url, method, body });

    if (url.includes("/v1/routing/policies/explain")) {
      return jsonResponse({
        name: "fast",
        selection_reason: "default",
        is_dynamic: false,
        candidates: [
          {
            position: 1,
            instance: "openai",
            model: "gpt-5-mini",
            selection_reason: "default",
            dispatch_model: "openai:gpt-5-mini",
          },
        ],
        dropped: [
          {
            selector: "anthropic:claude-haiku-4-5",
            reason: "not_allowed",
            detail: "is not in allowed_models for this caller",
          },
        ],
        guardrails: [],
      });
    }
    if (url.includes("/v1/routing/policies")) {
      if (method === "POST") {
        const row = policy(body.name, body.spec, { user_id: body.user_id ?? null });
        list = [...list, row];
        return jsonResponse(row);
      }
      if (method === "DELETE") {
        const name = decodeURIComponent((url.split("?")[0].split("/").pop() ?? ""));
        list = list.filter((item) => item.name !== name);
        return new Response(null, { status: 204 });
      }
      return jsonResponse(list);
    }
    if (url.includes("/v1/tool-settings")) {
      return jsonResponse({
        fields: [{ key: "guardrails_url", service: "guardrails", type: "url", value: guardrailsUrl }],
      });
    }
    if (url.includes("/v1/users")) return jsonResponse(USERS);
    if (url.includes("/v1/models")) return jsonResponse({ object: "list", data: [] });
    return jsonResponse([]);
  });
  return { spy, calls };
}

function renderPage(ui: ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>{ui}</MemoryRouter>
    </QueryClientProvider>,
  );
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("RoutingPage", () => {
  it("lists policies with what they serve and where they come from", async () => {
    mockApi();
    renderPage(<RoutingPage />);

    const fastRow = (await screen.findByText("fast")).closest("tr")!;
    // The chain is summarised rather than hidden: an operator scanning the table
    // needs to see that a fallback exists without opening the policy.
    expect(within(fastRow).getByText(/openai:gpt-5-mini/)).toBeInTheDocument();
    expect(within(fastRow).getByText(/\+1 on failure/)).toBeInTheDocument();
    expect(within(fastRow).getByText("stored")).toBeInTheDocument();
  });

  it("marks a policy that decides per request, since it has no single target", async () => {
    mockApi();
    renderPage(<RoutingPage />);

    const autoRow = (await screen.findByText("auto")).closest("tr")!;
    expect(within(autoRow).getByText("Dynamic")).toBeInTheDocument();
    expect(within(autoRow).getByText(/Chosen per request/)).toBeInTheDocument();
  });

  it("does not offer to edit or delete a policy that lives in config.yml", async () => {
    mockApi();
    renderPage(<RoutingPage />);

    const autoRow = (await screen.findByText("auto")).closest("tr")!;
    expect(within(autoRow).getByText("set in config.yml")).toBeInTheDocument();
    expect(within(autoRow).queryByRole("button", { name: "Delete" })).not.toBeInTheDocument();
  });

  it("creates a one-target policy from three fields", async () => {
    const { calls } = mockApi([]);
    const user = userEvent.setup();
    renderPage(<RoutingPage />);

    await user.click(await screen.findByRole("button", { name: "New policy" }));
    await user.type(screen.getByRole("textbox", { name: /policy name/i }), "cheap");
    await user.type(screen.getByRole("combobox", { name: /^serves$/i }), "openai:gpt-5-nano");
    // Close the combobox popover, which otherwise aria-hides the submit button.
    await user.keyboard("{Escape}");
    await user.click(screen.getByRole("button", { name: "Create policy" }));

    const post = calls.find((call) => call.method === "POST");
    expect(post).toBeDefined();
    const body = post!.body as { name: string; spec: PolicySpec };
    expect(body.name).toBe("cheap");
    // The fallthrough is explicit and last, which is what the schema requires.
    expect(body.spec.select).toEqual([{ default: "openai:gpt-5-nano" }]);
    expect(body.spec.on_failure).toBeUndefined();
  });

  it("keeps the failure chain and guardrails out of the way until asked for", async () => {
    mockApi([]);
    const user = userEvent.setup();
    renderPage(<RoutingPage />);

    await user.click(await screen.findByRole("button", { name: "New policy" }));
    // Naming one model must stay a short task, so neither section is present yet.
    expect(screen.queryByText("If that fails, try")).not.toBeInTheDocument();
    expect(screen.queryByText("Always check")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Add a fallback chain/ }));
    expect(screen.getByText("If that fails, try")).toBeInTheDocument();
    // Adding another one belongs inside the section it extends, not in the row of
    // links that start a section.
    const section = screen.getByText("If that fails, try").closest("div")!.parentElement!;
    expect(within(section).getByRole("button", { name: /Another fallback/ })).toBeInTheDocument();
  });

  it("disables the guardrails affordance when no guardrails service is configured", async () => {
    mockApi([], null);
    const user = userEvent.setup();
    renderPage(<RoutingPage />);

    await user.click(await screen.findByRole("button", { name: "New policy" }));
    const add = await screen.findByRole("button", { name: /Add guardrails/ });

    // Disabled, and never silently: the reason and the route to fixing it sit next
    // to the control as text, so it works on touch and for a screen reader.
    expect(add).toBeDisabled();
    expect(screen.getByText(/No guardrails service is configured/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Tools & Guardrails/ })).toHaveAttribute("href", "/tools");
  });

  it("refuses a policy name that would shadow a real model selector", async () => {
    mockApi([]);
    const user = userEvent.setup();
    renderPage(<RoutingPage />);

    await user.click(await screen.findByRole("button", { name: "New policy" }));
    await user.type(screen.getByRole("textbox", { name: /policy name/i }), "openai:gpt-4o");
    await user.type(screen.getByRole("combobox", { name: /^serves$/i }), "openai:gpt-5-nano");
    await user.keyboard("{Escape}");

    expect(screen.getByText(/cannot contain/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Create policy" })).toBeDisabled();
  });

  it("warns when a guardrail makes the guardrails service a hard dependency", async () => {
    mockApi([]);
    const user = userEvent.setup();
    renderPage(<RoutingPage />);

    await user.click(await screen.findByRole("button", { name: "New policy" }));
    await user.click(screen.getByRole("button", { name: /Add guardrails/ }));

    // block + block is the honest default, and its cost has to be visible where
    // the choice is made: an outage then refuses every request through the policy.
    expect(screen.getByText(/rejects every request through this policy/)).toBeInTheDocument();
  });

  it("refuses a tier-down threshold that could never fire", async () => {
    mockApi([]);
    const user = userEvent.setup();
    renderPage(<RoutingPage />);

    await user.click(await screen.findByRole("button", { name: "New policy" }));
    await user.type(screen.getByRole("textbox", { name: /policy name/i }), "thrifty");
    await user.type(screen.getByRole("combobox", { name: /^serves$/i }), "openai:gpt-5-mini");
    await user.keyboard("{Escape}");
    await user.click(screen.getByRole("button", { name: /Tier down/ }));

    const threshold = screen.getByRole("textbox", { name: /budget used at least/i });
    await user.clear(threshold);
    await user.type(threshold, "100");

    // The budget gate refuses the request before selection at 100%, so such a rule
    // is dead config. Saying so here beats a 400 from the server.
    expect(screen.getByText("Must be under 100.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Create policy" })).toBeDisabled();
  });
});
