import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactElement } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import type {
  OrganizationContext,
  ToolSettingField,
  ToolSettingsResponse,
  ToolsResponse,
} from "@/client"
import { ToolsGuardrailsPage } from "@/features/tools/ToolsGuardrailsPage"
import { API_ROOT } from "@/shared/api/client"
import { organizationContext } from "@/tests/fixtures"
import { pickOption } from "@/tests/select"

const FIELDS: ToolSettingField[] = [
  {
    key: "web_search_url",
    service: "web_search",
    type: "url",
    value: "http://searxng:8080",
    description: "Web search backend URL.",
  },
  {
    key: "web_search_engines",
    service: "web_search",
    type: "str",
    value: null,
    description: "Engine list.",
  },
  {
    key: "web_search_max_results",
    service: "web_search",
    type: "int",
    value: 5,
    description: "Result cap.",
  },
  {
    key: "web_search_extract",
    service: "web_search",
    type: "bool",
    value: null,
    description: "Extract page content.",
  },
  {
    key: "web_search_intercept",
    service: "web_search",
    type: "bool",
    value: null,
    description: "Claim provider-named web_search keywords.",
  },
  {
    key: "web_search_purpose_hint",
    service: "web_search",
    type: "str",
    value: null,
    description: "Purpose hint.",
  },
  {
    key: "sandbox_url",
    service: "sandbox",
    type: "url",
    value: null,
    description: "Sandbox backend URL.",
  },
  {
    key: "sandbox_purpose_hint",
    service: "sandbox",
    type: "str",
    value: null,
    description: "Purpose hint.",
  },
  {
    key: "guardrails_url",
    service: "guardrails",
    type: "url",
    value: "http://guardrails:8000",
    description: "Guardrails URL.",
  },
]

const RESPONSE: ToolSettingsResponse = { fields: FIELDS }

// What the server hands a non-operator: the same fields with the three
// service endpoints withheld rather than masked (otari-ai#1969).
const TENANT_RESPONSE: ToolSettingsResponse = {
  fields: FIELDS.filter((field) => field.type !== "url"),
}

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
      input_schema: {
        type: "object",
        properties: { query: { type: "string" } },
        required: ["query"],
      },
      example: { type: "otari_web_search" },
    },
    {
      id: "otari_code_execution",
      object: "tool",
      description: "Execute Python code in a sandboxed REPL.",
      available: false,
      accepted_types: ["otari_code_execution"],
      input_schema: {
        type: "object",
        properties: { code: { type: "string" } },
        required: ["code"],
      },
      example: { type: "otari_code_execution" },
    },
  ],
}

function renderWithClient(ui: ReactElement) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>)
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

interface MockOpts {
  /** The tool-settings body, for the tenant read that comes back without URLs. */
  settings?: ToolSettingsResponse
  patchStatus?: number
  patchDetail?: string
  testBody?: { ok: boolean; reason: string }
  tools?: ToolsResponse
  toolsStatus?: number
  // The caller's membership context, an operator's by default: the page gates
  // its deployment-wide reads on `deployment_operator`, so most tests here are
  // about the forms only an operator sees.
  context?: OrganizationContext
}

function mockApi(opts: MockOpts = {}) {
  let current = opts.settings ?? RESPONSE
  return vi
    .spyOn(globalThis, "fetch")
    .mockImplementation(async (input, init) => {
      const url = String(input)
      const method = (init?.method ?? "GET").toUpperCase()
      if (url.endsWith(`${API_ROOT}/organizations/me`)) {
        return jsonResponse(opts.context ?? organizationContext())
      }
      if (url.includes(`${API_ROOT}/tools`)) {
        if (opts.toolsStatus && opts.toolsStatus >= 400) {
          return jsonResponse({ detail: "nope" }, opts.toolsStatus)
        }
        return jsonResponse(opts.tools ?? TOOLS)
      }
      if (url.includes("/tool-settings/") && url.endsWith("/test")) {
        return jsonResponse(
          opts.testBody ?? { ok: true, reason: "reachable (HTTP 200)" },
        )
      }
      if (url.includes(`${API_ROOT}/tool-settings`)) {
        if (method === "PATCH") {
          if (opts.patchStatus && opts.patchStatus >= 400) {
            return jsonResponse(
              { detail: opts.patchDetail ?? "bad" },
              opts.patchStatus,
            )
          }
          const body = JSON.parse(String(init?.body)) as Record<string, unknown>
          current = {
            fields: current.fields.map((f) =>
              f.key in body
                ? { ...f, value: body[f.key] as ToolSettingField["value"] }
                : f,
            ),
          }
        }
        return jsonResponse(current)
      }
      return jsonResponse([])
    })
}

// The label a control now carries: the visible label and the config key beside
// it, which is what makes "Backend URL" unique on a page configuring three
// services.
const named = (label: string, key: string) => `${label} ${key}`
const WEB_SEARCH_URL = named("Backend URL", "web_search_url")
const ENGINES = named("Engines", "web_search_engines")
const MAX_RESULTS = named("Max results", "web_search_max_results")
const EXTRACT = named("Extract page content", "web_search_extract")
const INTERCEPT = named("Intercept provider web search", "web_search_intercept")
const SANDBOX_URL = named("Backend URL", "sandbox_url")
const GUARDRAILS_URL = named("Backend URL", "guardrails_url")

/** The last PATCH body the page sent, parsed. */
function lastPatch(fetchMock: ReturnType<typeof mockApi>) {
  const call = fetchMock.mock.calls
    .filter(([, init]) => (init?.method ?? "") === "PATCH")
    .at(-1)
  return call ? (JSON.parse(String(call[1]?.body)) as unknown) : undefined
}

describe("ToolsGuardrailsPage", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("renders every service's groups and effective values", async () => {
    mockApi()
    renderWithClient(<ToolsGuardrailsPage />)

    // The combined page names the service in each group heading, since three
    // groups called "Backend" would not say which one they configure.
    expect(
      await screen.findByRole("heading", { name: "Web search · Backend" }),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("heading", { name: "Code execution · Backend" }),
    ).toBeInTheDocument()
    expect(
      await screen.findByRole("heading", { name: "Guardrails · Backend" }),
    ).toBeInTheDocument()
    expect(screen.getByLabelText(WEB_SEARCH_URL)).toHaveValue(
      "http://searxng:8080",
    )
    expect(screen.getByLabelText(GUARDRAILS_URL)).toHaveValue(
      "http://guardrails:8000",
    )
  })

  it("saves a URL change on blur, with no Save button on the page", async () => {
    const fetchMock = mockApi()
    const user = userEvent.setup()
    renderWithClient(<ToolsGuardrailsPage />)
    await screen.findByLabelText(WEB_SEARCH_URL)

    const input = screen.getByLabelText(WEB_SEARCH_URL)
    await user.clear(input)
    await user.type(input, "http://new-searxng:9000")
    await user.tab()

    await waitFor(() =>
      expect(lastPatch(fetchMock)).toEqual({
        web_search_url: "http://new-searxng:9000",
      }),
    )
    expect(screen.queryByRole("button", { name: /^Save/ })).toBeNull()
  })

  it("commits a field on Enter without leaving it by hand", async () => {
    const fetchMock = mockApi()
    const user = userEvent.setup()
    renderWithClient(<ToolsGuardrailsPage />)
    await screen.findByLabelText(ENGINES)

    await user.type(screen.getByLabelText(ENGINES), "google,bing{Enter}")

    await waitFor(() =>
      expect(lastPatch(fetchMock)).toEqual({
        web_search_engines: "google,bing",
      }),
    )
  })

  it("does not save a field that was focused and left unchanged", async () => {
    const fetchMock = mockApi()
    const user = userEvent.setup()
    renderWithClient(<ToolsGuardrailsPage />)
    await screen.findByLabelText(WEB_SEARCH_URL)

    await user.click(screen.getByLabelText(WEB_SEARCH_URL))
    await user.tab()

    expect(lastPatch(fetchMock)).toBeUndefined()
  })

  it("clears a URL to null when emptied", async () => {
    const fetchMock = mockApi()
    const user = userEvent.setup()
    renderWithClient(<ToolsGuardrailsPage />)
    await screen.findByLabelText(WEB_SEARCH_URL)

    await user.clear(screen.getByLabelText(WEB_SEARCH_URL))
    await user.tab()

    await waitFor(() =>
      expect(lastPatch(fetchMock)).toEqual({ web_search_url: null }),
    )
  })

  it("trims surrounding whitespace when saving a text field", async () => {
    const fetchMock = mockApi()
    const user = userEvent.setup()
    renderWithClient(<ToolsGuardrailsPage />)
    await screen.findByLabelText(ENGINES)

    await user.type(screen.getByLabelText(ENGINES), "  google,bing  ")
    await user.tab()

    await waitFor(() =>
      expect(lastPatch(fetchMock)).toEqual({
        web_search_engines: "google,bing",
      }),
    )
  })

  it("refuses a ceiling that is not a whole number without asking the server", async () => {
    const fetchMock = mockApi()
    const user = userEvent.setup()
    renderWithClient(<ToolsGuardrailsPage />)
    await screen.findByLabelText(MAX_RESULTS)

    const input = screen.getByLabelText(MAX_RESULTS)
    await user.clear(input)
    await user.type(input, "1e1")
    await user.tab()

    expect(
      await screen.findByText("A whole number, 1 or more."),
    ).toBeInTheDocument()
    expect(lastPatch(fetchMock)).toBeUndefined()
  })

  it.each(["1e3", "0x10", "abc"])(
    "refuses %s as a price rather than letting Number read it",
    async (raw) => {
      // The other numeric rows guard against this; the money row is the one
      // that admits a decimal, so its guard is a different regex, not none.
      const fetchMock = mockApi()
      const user = userEvent.setup()
      renderWithClient(<ToolsGuardrailsPage only="web_search" />)
      const price = await screen.findByLabelText(
        "Price per call for otari:web_search",
      )

      // The row is disabled until /v1/pricing answers, so a rate is never
      // typed over one nobody can see.
      await waitFor(() => expect(price).toBeEnabled())
      await user.type(price, raw)
      await user.tab()

      expect(
        await screen.findByText("An amount in dollars, such as 0.01."),
      ).toBeInTheDocument()
      expect(
        fetchMock.mock.calls.some(([url]) =>
          String(url).includes(`${API_ROOT}/pricing`),
        ) &&
          fetchMock.mock.calls.some(
            ([, init]) => (init?.method ?? "") === "POST",
          ),
      ).toBe(false)
    },
  )

  it("rounds a price onto the wire rather than shipping float noise", async () => {
    // The stored column is per million, so 0.07 * 1e6 is 70000.00000000001.
    const fetchMock = mockApi()
    const user = userEvent.setup()
    renderWithClient(<ToolsGuardrailsPage only="web_search" />)
    const price = await screen.findByLabelText(
      "Price per call for otari:web_search",
    )

    await waitFor(() => expect(price).toBeEnabled())
    await user.type(price, "0.07")
    await user.tab()

    await waitFor(() => {
      const call = fetchMock.mock.calls.find(
        ([url, init]) =>
          String(url).includes(`${API_ROOT}/pricing`) &&
          (init?.method ?? "") === "POST",
      )
      expect(call).toBeDefined()
      expect(JSON.parse(String(call?.[1]?.body)).input_price_per_million).toBe(
        70000,
      )
    })
  })

  it("tests a URL for reachability and announces the result", async () => {
    mockApi({ testBody: { ok: true, reason: "reachable (HTTP 200)" } })
    const user = userEvent.setup()
    renderWithClient(<ToolsGuardrailsPage />)
    await screen.findByLabelText(WEB_SEARCH_URL)

    await user.click(screen.getByRole("button", { name: "Test web_search" }))
    expect(await screen.findByText("reachable (HTTP 200)")).toBeInTheDocument()
  })

  it("does not keep a test result against a URL that changed underneath it", async () => {
    // Editing the field already drops a stale result (the onChange reset). This
    // covers the path that reset does not: the committed URL changing from a
    // refetch re-seeds the input with no keystroke, so a result must be gated on
    // the URL it tested.
    const user = userEvent.setup()
    let searxngUrl = "http://searxng:8080"
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input)
      const method = (init?.method ?? "GET").toUpperCase()
      if (url.endsWith(`${API_ROOT}/organizations/me`)) {
        return jsonResponse(organizationContext())
      }
      if (url.includes("/tool-settings/") && url.endsWith("/test")) {
        return jsonResponse({ ok: true, reason: "reachable (HTTP 200)" })
      }
      if (url.includes(`${API_ROOT}/tool-settings`)) {
        // Saving the engines field surfaces a server-changed web_search_url, so
        // the next GET re-seeds the URL field without an operator keystroke.
        if (method === "PATCH") searxngUrl = "http://searxng:9999"
        return jsonResponse({
          fields: FIELDS.map((f) =>
            f.key === "web_search_url" ? { ...f, value: searxngUrl } : f,
          ),
        })
      }
      return jsonResponse([])
    })
    renderWithClient(<ToolsGuardrailsPage />)
    await screen.findByLabelText(WEB_SEARCH_URL)

    await user.click(screen.getByRole("button", { name: "Test web_search" }))
    expect(await screen.findByText("reachable (HTTP 200)")).toBeInTheDocument()

    await user.type(screen.getByLabelText(ENGINES), "google")
    await user.tab()

    await waitFor(() =>
      expect(screen.getByLabelText(WEB_SEARCH_URL)).toHaveValue(
        "http://searxng:9999",
      ),
    )
    expect(screen.queryByText("reachable (HTTP 200)")).not.toBeInTheDocument()
  })

  it("shows a rejected save in the row it came from, keeping the typed value", async () => {
    mockApi({
      patchStatus: 422,
      patchDetail: "URL must use http or https, got no scheme.",
    })
    const user = userEvent.setup()
    renderWithClient(<ToolsGuardrailsPage />)
    await screen.findByLabelText(SANDBOX_URL)

    const input = screen.getByLabelText(SANDBOX_URL)
    await user.type(input, "ftp://bad")
    await user.tab()

    expect(
      await screen.findByText(/must use http or https/),
    ).toBeInTheDocument()
    expect(input).toHaveValue("ftp://bad")
    expect(input).toHaveAttribute("aria-invalid", "true")
  })

  it("sends web_search_extract=false when the tri-state select is set to Off", async () => {
    const fetchMock = mockApi()
    const user = userEvent.setup()
    renderWithClient(<ToolsGuardrailsPage />)
    await screen.findByLabelText(WEB_SEARCH_URL)

    await pickOption(user, EXTRACT, "Off")

    await waitFor(() =>
      expect(lastPatch(fetchMock)).toEqual({ web_search_extract: false }),
    )
  })

  it("saves web_search_intercept from the tri-state select", async () => {
    const fetchMock = mockApi()
    const user = userEvent.setup()
    renderWithClient(<ToolsGuardrailsPage />)
    await screen.findByLabelText(WEB_SEARCH_URL)

    await pickOption(user, INTERCEPT, "On")

    await waitFor(() =>
      expect(lastPatch(fetchMock)).toEqual({ web_search_intercept: true }),
    )
  })

  it("surfaces a failed boolean save inline (not silently)", async () => {
    mockApi({
      patchStatus: 422,
      patchDetail: "web_search_extract must be a boolean.",
    })
    const user = userEvent.setup()
    renderWithClient(<ToolsGuardrailsPage />)
    await screen.findByLabelText(WEB_SEARCH_URL)

    await pickOption(user, EXTRACT, "Off")

    expect(await screen.findByText(/must be a boolean/)).toBeInTheDocument()
  })

  it("renders a backend field no group lists (fallback)", async () => {
    // A field the backend reports for a service that no group in SERVICES names
    // must still render, so a backend addition is not hidden.
    const withExtra: ToolSettingsResponse = {
      fields: [
        ...FIELDS,
        {
          key: "web_search_timeout_s",
          service: "web_search",
          type: "int",
          value: 30,
          description: "New knob.",
        },
      ],
    }
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) =>
      String(input).endsWith(`${API_ROOT}/organizations/me`)
        ? jsonResponse(organizationContext())
        : jsonResponse(withExtra),
    )
    renderWithClient(<ToolsGuardrailsPage />)

    expect(
      await screen.findByLabelText("web_search_timeout_s"),
    ).toBeInTheDocument()
  })

  it("lays every row out with the label left and the control in a shared lane", async () => {
    // One row shape for every field type, which is what keeps the controls in a
    // column down a group whether or not a row also carries a Test button.
    mockApi()
    renderWithClient(<ToolsGuardrailsPage />)
    await screen.findByLabelText(WEB_SEARCH_URL)

    for (const label of [WEB_SEARCH_URL, ENGINES, MAX_RESULTS]) {
      expect(screen.getByLabelText(label).className).toContain("w-full")
    }
    // Only the numeric field narrows, and it does so in the same lane.
    expect(screen.getByLabelText(MAX_RESULTS).className).toContain(
      "md:w-[5.5rem]",
    )
    expect(screen.getByLabelText(ENGINES).className).toContain(
      "md:w-[13.75rem]",
    )
  })
})

describe("ToolsGuardrailsPage tool status", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("heads each tool's settings with whether the deployment can run it", async () => {
    mockApi()
    renderWithClient(<ToolsGuardrailsPage />)

    // One row per gateway-run tool; guardrails declares none, so it has no row.
    expect(
      await screen.findByRole("button", { name: /otari_web_search/ }),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: /otari_code_execution/ }),
    ).toBeInTheDocument()
    // The code-execution fixture has available: false.
    expect(
      screen.getByRole("button", { name: /Unavailable · no backend/ }),
    ).toBeInTheDocument()
  })

  it("opens to the type a client declares and the reason a tool is unavailable", async () => {
    mockApi()
    const user = userEvent.setup()
    renderWithClient(<ToolsGuardrailsPage only="sandbox" />)

    const row = await screen.findByRole("button", {
      name: /otari_code_execution/,
    })
    expect(row).toHaveAttribute("aria-expanded", "false")
    await user.click(row)

    expect(row).toHaveAttribute("aria-expanded", "true")
    // Two rows in the page's own grammar, not a pair of eyebrowed paragraphs.
    expect(screen.getByText("Why unavailable")).toBeInTheDocument()
    expect(screen.getByText("Declare in a request")).toBeInTheDocument()
    // The value is selectable text beside a copy button, sized to its content:
    // the Clipboard API is absent on plain-HTTP origins, so the manual path is
    // the one that always works.
    expect(
      screen.getByText('"type": "otari_code_execution"'),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", {
        name: "Copy tools[].type for otari_code_execution",
      }),
    ).toBeInTheDocument()
  })

  it("names the provider keywords interception adds", async () => {
    mockApi({
      tools: {
        object: "list",
        data: [
          {
            ...TOOLS.data[0],
            accepted_types: [
              "otari_web_search",
              "web_search",
              "web_search_<date>",
            ],
          },
        ],
      },
    })
    const user = userEvent.setup()
    renderWithClient(<ToolsGuardrailsPage only="web_search" />)

    await user.click(
      await screen.findByRole("button", { name: /otari_web_search/ }),
    )
    expect(
      screen.getByText("web_search, web_search_<date>"),
    ).toBeInTheDocument()
  })

  it("keeps the editable settings usable when /v1/tools fails", async () => {
    // The status row is reference material; a failed discovery fetch must not
    // take the settings form down with it.
    mockApi({ toolsStatus: 500 })
    renderWithClient(<ToolsGuardrailsPage only="web_search" />)

    expect(await screen.findByLabelText(WEB_SEARCH_URL)).toHaveValue(
      "http://searxng:8080",
    )
    await waitFor(() =>
      expect(
        screen.queryByRole("button", { name: /otari_web_search/ }),
      ).not.toBeInTheDocument(),
    )
  })

  it("re-reads the search providers after the web-search URL is saved", async () => {
    // A searxng search tool with no api_base inherits web_search_url, so the
    // catalogue that tells the Search tools card what a blank box resolves to
    // (and whether one is required at all) changes with this very PATCH.
    const fetchMock = mockApi()
    const user = userEvent.setup()
    renderWithClient(<ToolsGuardrailsPage />)
    await screen.findByLabelText(WEB_SEARCH_URL)

    const providerFetches = () =>
      fetchMock.mock.calls.filter(([url]) =>
        String(url).includes(`${API_ROOT}/search-tools/providers`),
      ).length
    await waitFor(() => expect(providerFetches()).toBeGreaterThan(0))
    const before = providerFetches()

    const input = screen.getByLabelText(WEB_SEARCH_URL)
    await user.clear(input)
    await user.type(input, "http://new-searxng:9000")
    await user.tab()

    await waitFor(() => expect(providerFetches()).toBeGreaterThan(before))
  })

  it("carries the MCP servers section on the combined page and no narrowed one", async () => {
    // The card has no service above it to be filtered with, so the `only` guard
    // is the only thing keeping it off the per-service views.
    mockApi()
    const { unmount } = renderWithClient(<ToolsGuardrailsPage />)
    expect(
      await screen.findByRole("heading", { name: "MCP servers" }),
    ).toBeInTheDocument()
    unmount()

    renderWithClient(<ToolsGuardrailsPage only="sandbox" />)
    await screen.findByLabelText(SANDBOX_URL)
    expect(
      screen.queryByRole("heading", { name: "MCP servers" }),
    ).not.toBeInTheDocument()
  })
})

// otari-ai#1930: the Tools group is member-visible, but the service settings,
// the pricing row, and the /api/v1/search tools are operator-only reads, so for
// everyone else the page was a 403 banner, and the empty field list also
// dropped the member-appropriate workspace cards nested under it.
describe("ToolsGuardrailsPage by caller role", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("shows a non-operator the workspace cards and the still-operator-only reads are not fired", async () => {
    const fetchMock = mockApi({
      settings: TENANT_RESPONSE,
      context: organizationContext({
        deployment_operator: false,
        role: "member",
      }),
    })
    renderWithClient(<ToolsGuardrailsPage />)

    // The two workspace cards, in the state a harness with no selected
    // workspace lands in; their real forms are covered by their own suites.
    expect(
      await screen.findByText(/Per-workspace web search is set on a workspace/),
    ).toBeInTheDocument()
    expect(
      screen.getByText(/Per-workspace code execution is set on a workspace/),
    ).toBeInTheDocument()

    // No deployment search-tools group, whose read is still operator-only.
    expect(
      screen.queryByRole("heading", { name: "Search tools" }),
    ).not.toBeInTheDocument()
    const urls = fetchMock.mock.calls.map(([input]) => String(input))
    expect(urls.some((url) => url.includes(`${API_ROOT}/search-tools`))).toBe(
      false,
    )
  })

  it("renders a non-operator's tool settings as values rather than controls", async () => {
    mockApi({
      settings: TENANT_RESPONSE,
      context: organizationContext({
        deployment_operator: false,
        role: "member",
      }),
    })
    renderWithClient(<ToolsGuardrailsPage />)

    // The field is shown, so a member is told what the tools do to their
    // requests, and it is text: there is no control to press.
    expect(
      await screen.findByText("web_search_max_results"),
    ).toBeInTheDocument()
    expect(screen.queryByLabelText(MAX_RESULTS)).not.toBeInTheDocument()
    // The service endpoints never arrive, so nothing renders them either.
    expect(screen.queryByText("web_search_url")).not.toBeInTheDocument()
    expect(screen.queryByText("guardrails_url")).not.toBeInTheDocument()
    // Nor the per-call rate: /api/v1/pricing is still operator-only, so this row
    // would show a member an editable "unpriced" field that only fails on save.
    expect(screen.queryByText("Price per call")).not.toBeInTheDocument()
  })

  it("keeps a narrowed service page working for a non-operator", async () => {
    // The sidebar's per-service children render this page with `only`, so the
    // member-appropriate card has to survive the narrowing too.
    mockApi({
      context: organizationContext({
        deployment_operator: false,
        role: "member",
      }),
    })
    renderWithClient(<ToolsGuardrailsPage only="sandbox" />)

    expect(
      await screen.findByText(
        /Per-workspace code execution is set on a workspace/,
      ),
    ).toBeInTheDocument()
    expect(screen.queryByLabelText(SANDBOX_URL)).not.toBeInTheDocument()
  })

  it("still renders the operator forms alongside the workspace cards for an operator", async () => {
    // The operator is a member too: gating the deployment-wide reads must not
    // have cost them either half of the page.
    mockApi()
    renderWithClient(<ToolsGuardrailsPage only="web_search" />)

    expect(await screen.findByLabelText(WEB_SEARCH_URL)).toBeInTheDocument()
    expect(
      screen.getByRole("heading", { name: "Search tools" }),
    ).toBeInTheDocument()
    expect(
      screen.getByText(/Per-workspace web search is set on a workspace/),
    ).toBeInTheDocument()
  })
})
