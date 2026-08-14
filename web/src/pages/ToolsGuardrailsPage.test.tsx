import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactElement } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ToolSettingField, ToolSettingsResponse, ToolsResponse } from "@/api/types";
import { ToolsGuardrailsPage } from "@/pages/ToolsGuardrailsPage";

const FIELDS: ToolSettingField[] = [
  { key: "web_search_url", service: "web_search", type: "url", value: "http://searxng:8080", description: "Web search backend URL." },
  { key: "web_search_engines", service: "web_search", type: "str", value: null, description: "Engine list." },
  { key: "web_search_max_results", service: "web_search", type: "int", value: 5, description: "Result cap." },
  { key: "web_search_extract", service: "web_search", type: "bool", value: null, description: "Extract page content." },
  { key: "web_search_intercept", service: "web_search", type: "bool", value: null, description: "Claim provider-named web_search keywords." },
  { key: "web_search_purpose_hint", service: "web_search", type: "str", value: null, description: "Purpose hint." },
  { key: "sandbox_url", service: "sandbox", type: "url", value: null, description: "Sandbox backend URL." },
  { key: "sandbox_purpose_hint", service: "sandbox", type: "str", value: null, description: "Purpose hint." },
  { key: "guardrails_url", service: "guardrails", type: "url", value: "http://guardrails:8000", description: "Guardrails URL." },
];

const RESPONSE: ToolSettingsResponse = { fields: FIELDS };

// GET /v1/tools drives the "how to call this" card. `accepted_types` is what the
// deployment currently honors, so it is the interesting axis here.
const TOOLS: ToolsResponse = {
  object: "list",
  data: [
    {
      id: "otari_web_search",
      object: "tool",
      description: "Search the web for current information.",
      available: true,
      accepted_types: ["otari_web_search"],
      input_schema: { type: "object", properties: { query: { type: "string" } }, required: ["query"] },
      example: { type: "otari_web_search" },
    },
    {
      id: "otari_code_execution",
      object: "tool",
      description: "Execute Python code in a sandboxed REPL.",
      available: false,
      accepted_types: ["otari_code_execution"],
      input_schema: { type: "object", properties: { code: { type: "string" } }, required: ["code"] },
      example: { type: "otari_code_execution" },
    },
  ],
};

function renderWithClient(ui: ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

interface MockOpts {
  patchStatus?: number;
  patchDetail?: string;
  testBody?: { ok: boolean; reason: string };
  tools?: ToolsResponse;
  toolsStatus?: number;
}

function mockApi(opts: MockOpts = {}) {
  let current = RESPONSE;
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input);
    const method = (init?.method ?? "GET").toUpperCase();
    if (url.includes("/v1/tools")) {
      if (opts.toolsStatus && opts.toolsStatus >= 400) {
        return jsonResponse({ detail: "nope" }, opts.toolsStatus);
      }
      return jsonResponse(opts.tools ?? TOOLS);
    }
    if (url.includes("/tool-settings/") && url.endsWith("/test")) {
      return jsonResponse(opts.testBody ?? { ok: true, reason: "reachable (HTTP 200)" });
    }
    if (url.includes("/v1/tool-settings")) {
      if (method === "PATCH") {
        if (opts.patchStatus && opts.patchStatus >= 400) {
          return jsonResponse({ detail: opts.patchDetail ?? "bad" }, opts.patchStatus);
        }
        const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
        current = {
          fields: current.fields.map((f) => (f.key in body ? { ...f, value: body[f.key] as ToolSettingField["value"] } : f)),
        };
      }
      return jsonResponse(current);
    }
    return jsonResponse([]);
  });
}

describe("ToolsGuardrailsPage", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders the three service sections and effective values", async () => {
    mockApi();
    renderWithClient(<ToolsGuardrailsPage />);

    expect(await screen.findByText("Web search")).toBeInTheDocument();
    expect(screen.getByText("Code execution")).toBeInTheDocument();
    expect(screen.getByText("Guardrails")).toBeInTheDocument();
    expect(screen.getByLabelText("web_search_url")).toHaveValue("http://searxng:8080");
    expect(screen.getByLabelText("guardrails_url")).toHaveValue("http://guardrails:8000");
  });

  it("saves a URL change with a PATCH", async () => {
    const fetchMock = mockApi();
    const user = userEvent.setup();
    renderWithClient(<ToolsGuardrailsPage />);
    await screen.findByText("Web search");

    const input = screen.getByLabelText("web_search_url");
    await user.clear(input);
    await user.type(input, "http://new-searxng:9000");
    await user.click(screen.getByRole("button", { name: "Save web_search_url" }));

    const call = fetchMock.mock.calls.find(([, init]) => (init?.method ?? "") === "PATCH");
    expect(call).toBeDefined();
    expect(JSON.parse(String(call?.[1]?.body))).toEqual({ web_search_url: "http://new-searxng:9000" });
  });

  it("clears a URL to null when emptied and saved", async () => {
    const fetchMock = mockApi();
    const user = userEvent.setup();
    renderWithClient(<ToolsGuardrailsPage />);
    await screen.findByText("Web search");

    const input = screen.getByLabelText("web_search_url");
    await user.clear(input);
    await user.click(screen.getByRole("button", { name: "Save web_search_url" }));

    const call = fetchMock.mock.calls.find(([, init]) => (init?.method ?? "") === "PATCH");
    expect(JSON.parse(String(call?.[1]?.body))).toEqual({ web_search_url: null });
  });

  it("tests a URL for reachability and announces the result", async () => {
    mockApi({ testBody: { ok: true, reason: "reachable (HTTP 200)" } });
    const user = userEvent.setup();
    renderWithClient(<ToolsGuardrailsPage />);
    await screen.findByText("Web search");

    await user.click(screen.getByRole("button", { name: "Test web_search" }));
    expect(await screen.findByText("reachable (HTTP 200)")).toBeInTheDocument();
  });

  it("does not keep a test result against a URL that changed underneath it", async () => {
    // Editing the field already drops a stale result (the onChange reset). This
    // covers the path that reset does not: the committed URL changing from a
    // refetch (e.g. a background refresh, or saving a sibling field) re-seeds the
    // input with no keystroke, so a result must be gated on the URL it tested.
    const user = userEvent.setup();
    let searxngUrl = "http://searxng:8080";
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      const method = (init?.method ?? "GET").toUpperCase();
      if (url.includes("/tool-settings/") && url.endsWith("/test")) {
        return jsonResponse({ ok: true, reason: "reachable (HTTP 200)" });
      }
      if (url.includes("/v1/tool-settings")) {
        // Saving the engines field surfaces a server-changed web_search_url, so
        // the next GET re-seeds the URL field without an operator keystroke.
        if (method === "PATCH") searxngUrl = "http://searxng:9999";
        return jsonResponse({
          fields: FIELDS.map((f) => (f.key === "web_search_url" ? { ...f, value: searxngUrl } : f)),
        });
      }
      return jsonResponse([]);
    });
    renderWithClient(<ToolsGuardrailsPage />);
    await screen.findByText("Web search");

    // Test the URL; its result shows while the field still holds the tested URL.
    await user.click(screen.getByRole("button", { name: "Test web_search" }));
    expect(await screen.findByText("reachable (HTTP 200)")).toBeInTheDocument();

    // Save a sibling field; the refetch re-seeds web_search_url to a new value.
    await user.type(screen.getByLabelText("web_search_engines"), "google");
    await user.click(screen.getByRole("button", { name: "Save web_search_engines" }));

    await waitFor(() => expect(screen.getByLabelText("web_search_url")).toHaveValue("http://searxng:9999"));
    // The result belonged to the old URL, so it must no longer be shown.
    expect(screen.queryByText("reachable (HTTP 200)")).not.toBeInTheDocument();
  });

  it("shows an inline error under the field when a save is rejected", async () => {
    mockApi({ patchStatus: 422, patchDetail: "URL must use http or https, got no scheme." });
    const user = userEvent.setup();
    renderWithClient(<ToolsGuardrailsPage />);
    await screen.findByText("Web search");

    const input = screen.getByLabelText("sandbox_url");
    await user.type(input, "ftp://bad");
    await user.click(screen.getByRole("button", { name: "Save sandbox_url" }));

    expect(await screen.findByText(/must use http or https/)).toBeInTheDocument();
  });

  it("sends web_search_extract=false when the tri-state select is set to Off", async () => {
    const fetchMock = mockApi();
    const user = userEvent.setup();
    renderWithClient(<ToolsGuardrailsPage />);
    await screen.findByText("Web search");

    await user.selectOptions(screen.getByLabelText("web_search_extract"), "off");

    const call = fetchMock.mock.calls.find(([, init]) => (init?.method ?? "") === "PATCH");
    expect(JSON.parse(String(call?.[1]?.body))).toEqual({ web_search_extract: false });
  });

  it("renders a backend field not in the frontend's ordered list (fallback)", async () => {
    // A field the backend reports for a service but the frontend hasn't listed in
    // SERVICES[*].order must still render, so a backend addition is not hidden.
    const withExtra: ToolSettingsResponse = {
      fields: [
        ...FIELDS,
        { key: "web_search_timeout_s", service: "web_search", type: "int", value: 30, description: "New knob." },
      ],
    };
    vi.spyOn(globalThis, "fetch").mockImplementation(async () => jsonResponse(withExtra));
    renderWithClient(<ToolsGuardrailsPage />);
    await screen.findByText("Web search");
    expect(await screen.findByLabelText("web_search_timeout_s")).toBeInTheDocument();
  });

  it("surfaces a failed boolean save inline (not silently)", async () => {
    mockApi({ patchStatus: 422, patchDetail: "web_search_extract must be a boolean." });
    const user = userEvent.setup();
    renderWithClient(<ToolsGuardrailsPage />);
    await screen.findByText("Web search");

    await user.selectOptions(screen.getByLabelText("web_search_extract"), "off");

    expect(await screen.findByText(/must be a boolean/)).toBeInTheDocument();
  });

  it("aligns fields on one grid: every row shares the same fixed input/action columns", async () => {
    // The alignment fix (issue #355) lays each field out as a grid row with
    // fixed-width input and action tracks, so the boxes and Save buttons line up
    // in columns regardless of a row's type or whether it also has a Test button.
    // Before the fix, rows were flex with per-type input widths (w-64 vs w-28)
    // and a Test button that shoved Save off the shared right edge.
    mockApi();
    renderWithClient(<ToolsGuardrailsPage />);
    await screen.findByText("Web search");

    const rowOf = (labeledBy: string) =>
      screen.getByLabelText(labeledBy).closest("div.grid") as HTMLElement | null;

    const urlRow = rowOf("web_search_url"); // has Save + Test
    const textRow = rowOf("web_search_engines"); // has Save only
    const numberRow = rowOf("web_search_max_results"); // narrower numeric input

    for (const row of [urlRow, textRow, numberRow]) {
      expect(row).not.toBeNull();
      // Same three-track template on every row keeps the columns aligned.
      expect(row?.className).toContain("sm:grid-cols-[minmax(0,1fr)_16rem_10rem]");
    }

    // The URL and text inputs fill the shared input column (no more w-64 vs w-28
    // mismatch); the numeric input is pinned to that column's right edge.
    expect(screen.getByLabelText("web_search_url").className).toContain("w-full");
    expect(screen.getByLabelText("web_search_engines").className).toContain("w-full");
    expect(screen.getByLabelText("web_search_max_results").className).toContain("sm:justify-self-end");
  });

  it("trims surrounding whitespace when saving a text field", async () => {
    const fetchMock = mockApi();
    const user = userEvent.setup();
    renderWithClient(<ToolsGuardrailsPage />);
    await screen.findByText("Web search");

    const input = screen.getByLabelText("web_search_engines");
    await user.type(input, "  google,bing  ");
    await user.click(screen.getByRole("button", { name: "Save web_search_engines" }));

    const call = fetchMock.mock.calls.find(([, init]) => (init?.method ?? "") === "PATCH");
    expect(JSON.parse(String(call?.[1]?.body))).toEqual({ web_search_engines: "google,bing" });
  });
});

describe("ToolsGuardrailsPage how-to-call card", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("shows the accepted declaration types and a runnable example per tool", async () => {
    mockApi();
    renderWithClient(<ToolsGuardrailsPage />);

    // One card per gateway-run tool; guardrails has none (it is not declared in
    // `tools[]` at all).
    await screen.findByText("Web search");
    await waitFor(() => expect(screen.getAllByText("Accepted tools[].type")).toHaveLength(2));
    // The type to declare is the thing an operator cannot guess from the settings.
    expect(screen.getAllByText("otari_web_search").length).toBeGreaterThan(0);
    expect(screen.getAllByText(/POST \/v1\/chat\/completions/)).toHaveLength(2);
  });

  it("advertises the provider-named keywords when interception is on", async () => {
    mockApi({
      tools: {
        object: "list",
        data: [
          {
            ...TOOLS.data[0],
            accepted_types: ["otari_web_search", "web_search", "web_search_<date>"],
          },
        ],
      },
    });
    renderWithClient(<ToolsGuardrailsPage />);

    expect(await screen.findByText("web_search_<date>")).toBeInTheDocument();
    expect(screen.getByText("web_search")).toBeInTheDocument();
  });

  it("flags a tool whose backend is not configured", async () => {
    mockApi();
    renderWithClient(<ToolsGuardrailsPage />);

    // The code-execution fixture has available: false.
    expect(await screen.findByText("No backend configured")).toBeInTheDocument();
  });

  it("keeps the editable settings usable when /v1/tools fails", async () => {
    // The card is reference material; a failed discovery fetch must not take the
    // settings form down with it.
    mockApi({ toolsStatus: 500 });
    renderWithClient(<ToolsGuardrailsPage />);

    expect(await screen.findByLabelText("web_search_url")).toHaveValue("http://searxng:8080");
    await waitFor(() => expect(screen.queryByText("Accepted tools[].type", { exact: false })).not.toBeInTheDocument());
  });

  it("saves web_search_intercept from the tri-state select", async () => {
    const fetchMock = mockApi();
    const user = userEvent.setup();
    renderWithClient(<ToolsGuardrailsPage />);
    await screen.findByText("Web search");

    await user.selectOptions(screen.getByLabelText("web_search_intercept"), "on");

    const call = fetchMock.mock.calls.find(([, init]) => (init?.method ?? "") === "PATCH");
    expect(JSON.parse(String(call?.[1]?.body))).toEqual({ web_search_intercept: true });
  });

  it("re-reads the search providers after the web-search URL is saved", async () => {
    // A searxng search tool with no api_base inherits web_search_url, so the
    // catalogue that tells the Search tools card what a blank box resolves to
    // (and whether one is required at all) changes with this very PATCH. Without
    // the invalidation the card keeps offering the old inherited URL, or keeps
    // demanding one the gateway would now supply.
    const fetchMock = mockApi();
    const user = userEvent.setup();
    renderWithClient(<ToolsGuardrailsPage />);
    await screen.findByText("Web search");

    const providerFetches = () =>
      fetchMock.mock.calls.filter(([url]) => String(url).includes("/v1/search-tools/providers")).length;
    await waitFor(() => expect(providerFetches()).toBeGreaterThan(0));
    const before = providerFetches();

    const input = screen.getByLabelText("web_search_url");
    await user.clear(input);
    await user.type(input, "http://new-searxng:9000");
    await user.click(screen.getByRole("button", { name: "Save web_search_url" }));

    await waitFor(() => expect(providerFetches()).toBeGreaterThan(before));
  });
});
