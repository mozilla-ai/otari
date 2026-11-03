import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type {
  PricingResponse,
  SearchProviderInfo,
  SearchToolsResponse,
  ToolSettingField,
} from "@/client"
import { API_ROOT } from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  bootstrap,
  organizationContext,
  pricingResponse,
} from "@/tests/fixtures"
import { renderWithRouter } from "@/tests/router"
import { pickOption, selectTrigger } from "@/tests/select"

import { WebSearchPage } from "./WebSearchPage"

interface Request {
  url: string
  method: string
  body: unknown
}

const SEARCH_TOOLS: SearchToolsResponse = {
  stored: [
    {
      name: "exa",
      kind: "search",
      provider: "exa",
      last4: "a1b2",
      options: {},
      decryptable: true,
      shadows_config: false,
      updated_at: "2026-10-20T10:00:00",
    },
  ],
  config: [
    {
      name: "local",
      kind: "search",
      provider: "searxng",
      api_base: "http://searxng:8080",
      has_api_key: false,
      shadowed: false,
    },
  ],
}

const FETCH_TOOLS: SearchToolsResponse = {
  stored: [
    {
      name: "tabstack",
      kind: "fetch",
      provider: "tabstack",
      last4: "9f3c",
      options: {},
      decryptable: true,
      shadows_config: false,
    },
  ],
  config: [
    {
      name: "builtin_fetch",
      kind: "fetch",
      provider: "builtin",
      has_api_key: false,
      shadowed: false,
    },
  ],
}

const SEARCH_CATALOG: SearchProviderInfo[] = [
  {
    id: "exa",
    kind: "search",
    requires_api_key: true,
    requires_api_base: false,
    default_api_base: "https://api.exa.ai",
    tier: "production",
    options: [
      {
        name: "type",
        type: "string",
        enum: ["auto", "fast"],
        default: "auto",
        description: "How Exa searches.",
        operator_only: false,
      },
    ],
  },
  {
    id: "searxng",
    kind: "search",
    requires_api_key: false,
    requires_api_base: true,
    tier: null,
    options: null,
  },
]

const FETCH_CATALOG: SearchProviderInfo[] = [
  {
    id: "tabstack",
    kind: "fetch",
    requires_api_key: true,
    requires_api_base: false,
    tier: "production",
    options: [],
    reports_final_url: false,
  },
]

const field = (
  key: string,
  type: ToolSettingField["type"],
  value: ToolSettingField["value"] = null,
): ToolSettingField => ({ key, type, value, service: "web_search" })

function settingsFields(
  values: Record<string, ToolSettingField["value"]> = {},
): ToolSettingField[] {
  return [
    field("web_search_default_tool", "str", values.web_search_default_tool),
    field("web_fetch_default_tool", "str", values.web_fetch_default_tool),
    field("web_search_max_calls", "int"),
    field("web_search_max_results", "int"),
    field("web_search_extract", "bool"),
    field("web_search_intercept", "bool"),
    field("web_search_url", "url", values.web_search_url),
    field("web_search_engines", "str"),
    field("web_search_purpose_hint", "str"),
  ]
}

const rate = (modelKey: string, dollars: number, kind = "model") =>
  pricingResponse({
    model_key: modelKey,
    kind,
    input_price_per_million: dollars * 1_000_000,
    unit: "requests",
  })

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(body === undefined ? null : JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

function mockApi(
  opts: {
    operator?: boolean
    requirePricing?: boolean
    /** The rates in force, keyed `kind:model_key`; anything else answers 404. */
    rates?: Record<string, PricingResponse>
    settings?: Record<string, ToolSettingField["value"]>
    /** Answers a price write with this refusal. */
    priceRefusal?: string
    /** Answers a settings write with this refusal. */
    settingsRefusal?: string
    /** Refuse only the first price write, or price delete, then accept. */
    priceFailsOnce?: boolean
    /** Rate reads, as `kind:model_key`, that answer 500. */
    failingReads?: string[]
  } = {},
) {
  const requests: Request[] = []
  let priceFailures = opts.priceFailsOnce ? 1 : 0
  const rates = opts.rates ?? {
    "model:otari:web_search": rate("otari:web_search", 0.01),
    "search:exa": rate("exa", 0.005, "search"),
  }
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
    const body = init?.body ? JSON.parse(String(init.body)) : undefined
    requests.push({ url, method, body })
    const path = new URL(url, "http://localhost")

    if (url.endsWith(`${API_ROOT}/organizations/me`)) {
      return jsonResponse(
        organizationContext({ deployment_operator: opts.operator ?? true }),
      )
    }
    if (path.pathname === `${API_ROOT}/settings`) {
      return jsonResponse({ require_pricing: opts.requirePricing ?? false })
    }
    if (path.pathname === `${API_ROOT}/tool-settings`) {
      if (method === "PATCH" && opts.settingsRefusal) {
        return jsonResponse({ detail: opts.settingsRefusal }, 422)
      }
      return jsonResponse({ fields: settingsFields(opts.settings) })
    }
    if (path.pathname === `${API_ROOT}/search-tools/providers`) {
      return jsonResponse(
        path.searchParams.get("kind") === "fetch"
          ? FETCH_CATALOG
          : SEARCH_CATALOG,
      )
    }
    if (path.pathname.endsWith("/test")) {
      return jsonResponse({ ok: true, hits: 3 })
    }
    if (path.pathname === `${API_ROOT}/search-tools`) {
      if (method === "POST") {
        return jsonResponse(
          {
            ...body,
            last4: "zz99",
            decryptable: true,
            shadows_config: false,
            notice:
              "web_search_default_tool is now 'exa', so adding this one did not turn in-loop search off.",
          },
          201,
        )
      }
      return jsonResponse(
        path.searchParams.get("kind") === "fetch" ? FETCH_TOOLS : SEARCH_TOOLS,
      )
    }
    if (path.pathname.startsWith(`${API_ROOT}/search-tools/`)) {
      return method === "DELETE"
        ? new Response(null, { status: 204 })
        : jsonResponse({ ...SEARCH_TOOLS.stored[0], ...body })
    }
    if (path.pathname.startsWith(`${API_ROOT}/pricing`)) {
      if ((method === "POST" || method === "DELETE") && priceFailures > 0) {
        priceFailures -= 1
        return jsonResponse({ detail: "Database unavailable" }, 500)
      }
      if (method === "POST") {
        return opts.priceRefusal
          ? jsonResponse({ detail: opts.priceRefusal }, 422)
          : jsonResponse(body, 201)
      }
      if (method === "DELETE") return new Response(null, { status: 204 })
      const key = decodeURIComponent(
        path.pathname.slice(`${API_ROOT}/pricing/`.length),
      )
      const id = `${path.searchParams.get("kind")}:${key}`
      if (opts.failingReads?.includes(id)) {
        return jsonResponse({ detail: "Database unavailable" }, 500)
      }
      const found = rates[id]
      return found
        ? jsonResponse(found)
        : jsonResponse({ detail: "Not found" }, 404)
    }
    return jsonResponse({})
  })
  return requests
}

async function renderPage() {
  return renderWithRouter(
    <DeploymentProvider value={bootstrap()}>
      <QueryClientProvider
        client={
          new QueryClient({ defaultOptions: { queries: { retry: false } } })
        }
      >
        <WebSearchPage />
      </QueryClientProvider>
    </DeploymentProvider>,
    { url: "/admin/web-search" },
  )
}

const findRow = async (table: string, name: string) => {
  const grid = await screen.findByRole("grid", { name: table })
  // A tool named after its provider is named twice in its row, so the first
  // match is enough to find the row.
  const [cell] = await within(grid).findAllByText(name, { exact: true })
  const row = cell.closest('[role="row"]')
  if (!(row instanceof HTMLElement)) throw new Error(`no row for ${name}`)
  return row
}

const writes = (requests: Request[], method: string, path: string) =>
  requests.filter(
    (request) =>
      request.method === method &&
      new URL(request.url, "http://localhost").pathname ===
        `${API_ROOT}${path}`,
  )

afterEach(() => {
  vi.restoreAllMocks()
})

describe("WebSearchPage", () => {
  it("refuses a caller who does not operate the deployment without asking for its tools", async () => {
    const requests = mockApi({ operator: false })
    await renderPage()

    expect(
      await screen.findByText("Web search is not available to you"),
    ).toBeInTheDocument()
    expect(
      requests.some((request) => request.url.includes("/search-tools")),
    ).toBe(false)
  })

  it("lists each kind's tools beside their prices, and the rate each falls back to", async () => {
    mockApi({
      requirePricing: true,
      settings: { web_search_default_tool: "exa" },
    })
    await renderPage()

    const exa = await findRow("Search tools", "exa")
    expect(await within(exa).findByText("$0.005")).toBeInTheDocument()
    expect(within(exa).getByText("Default")).toBeInTheDocument()
    expect(within(exa).getByText("••••a1b2")).toBeInTheDocument()

    // No rate of its own: the model's searches take Web search's, while the
    // search API, which only the tool's own rate prices, goes unpriced.
    const local = await findRow("Search tools", "local")
    expect(await within(local).findByText("$0.01")).toBeInTheDocument()
    expect(
      within(local).getByText("Unpriced on the search API"),
    ).toBeInTheDocument()
    expect(within(local).getByText("config.yml")).toBeInTheDocument()
    expect(within(local).getByText(/Search API only/)).toBeInTheDocument()

    // The built-in fetcher is the fetch default when nothing else is named.
    const builtin = await findRow("Fetch tools", "builtin_fetch")
    expect(within(builtin).getByText("Default")).toBeInTheDocument()
    expect(within(builtin).getByText("Not needed")).toBeInTheDocument()
    expect(await within(builtin).findByText("Unpriced")).toBeInTheDocument()

    expect(await screen.findByText(/require_pricing is on/)).toBeInTheDocument()
  })

  it("stars a tool as the model's default, and stops using it", async () => {
    const user = userEvent.setup()
    const requests = mockApi({ settings: { web_search_default_tool: "exa" } })
    await renderPage()

    await user.click(
      await screen.findByRole("button", {
        name: /^Stop using exa as the default/,
      }),
    )
    await waitFor(() =>
      expect(writes(requests, "PATCH", "/tool-settings")).toHaveLength(1),
    )
    expect(writes(requests, "PATCH", "/tool-settings")[0].body).toEqual({
      web_search_default_tool: null,
    })

    // The model cannot search with searxng, so it cannot be the default.
    expect(
      screen.getByRole("button", {
        name: /^Make local the default\. Unavailable/,
      }),
    ).toBeDisabled()

    await user.click(
      screen.getByRole("button", { name: "Make tabstack the default" }),
    )
    await waitFor(() =>
      expect(writes(requests, "PATCH", "/tool-settings")).toHaveLength(2),
    )
    expect(writes(requests, "PATCH", "/tool-settings")[1].body).toEqual({
      web_fetch_default_tool: "tabstack",
    })
  })

  it("names builtin_fetch outright, since a cleared fetch default falls back to config.yml's", async () => {
    const user = userEvent.setup()
    const requests = mockApi({
      settings: { web_fetch_default_tool: "tabstack" },
    })
    await renderPage()

    await user.click(
      await screen.findByRole("button", {
        name: /^Stop using tabstack as the default/,
      }),
    )
    await waitFor(() =>
      expect(writes(requests, "PATCH", "/tool-settings")[0]?.body).toEqual({
        web_fetch_default_tool: "builtin_fetch",
      }),
    )
    await user.click(
      screen.getByRole("button", { name: "Make builtin_fetch the default" }),
    )
    await waitFor(() =>
      expect(writes(requests, "PATCH", "/tool-settings")[1]?.body).toEqual({
        web_fetch_default_tool: "builtin_fetch",
      }),
    )
  })

  it("reads none in any case as the model's search turned off", async () => {
    mockApi({ settings: { web_search_default_tool: "None" } })
    await renderPage()

    expect(
      await screen.findByRole("switch", { name: "The model can search" }),
    ).not.toBeChecked()
  })

  it("turns the model's search off with none", async () => {
    const user = userEvent.setup()
    const requests = mockApi()
    await renderPage()

    await user.click(
      await screen.findByRole("switch", { name: "The model can search" }),
    )
    await waitFor(() =>
      expect(writes(requests, "PATCH", "/tool-settings")[0]?.body).toEqual({
        web_search_default_tool: "none",
      }),
    )
  })

  it("adds a search tool with its price, and says what the create changed", async () => {
    const user = userEvent.setup()
    const requests = mockApi()
    await renderPage()

    await user.click(
      await screen.findByRole("button", { name: "Add search tool" }),
    )
    const dialog = await screen.findByRole("dialog")
    await pickOption(user, "Provider", "exa", dialog)
    // exa is taken, so the blank name becomes exa-2.
    expect(
      within(dialog).getByText(/leave blank to name it exa-2/),
    ).toBeInTheDocument()
    await user.type(within(dialog).getByLabelText(/API key/), "sk-new")
    await user.type(
      within(dialog).getByLabelText("Price per call, USD"),
      "0.004",
    )
    await user.click(
      within(dialog).getByRole("button", { name: "Add search tool" }),
    )

    expect(await screen.findByText("Search tool added")).toBeInTheDocument()
    expect(writes(requests, "POST", "/search-tools")[0].body).toMatchObject({
      name: "exa-2",
      kind: "search",
      provider: "exa",
      api_key: "sk-new",
    })
    expect(writes(requests, "POST", "/pricing")[0].body).toEqual({
      kind: "search",
      model_key: "exa-2",
      input_price_per_million: 4000,
      output_price_per_million: 0,
      unit: "requests",
    })
    expect(
      screen.getByText(/web_search_default_tool is now 'exa'/),
    ).toBeInTheDocument()
  })

  it("keeps a tool whose price was refused, and says so", async () => {
    const user = userEvent.setup()
    mockApi({ priceRefusal: "Pricing is read-only here." })
    await renderPage()

    await user.click(
      await screen.findByRole("button", { name: "Add fetch tool" }),
    )
    const dialog = await screen.findByRole("dialog")
    await pickOption(user, "Provider", "tabstack", dialog)
    // Tabstack does not report where a redirect led.
    expect(
      within(dialog).getByText(/does not report where a redirect led/),
    ).toBeInTheDocument()
    await user.type(within(dialog).getByLabelText(/API key/), "ts-key")
    await user.type(
      within(dialog).getByLabelText("Price per call, USD"),
      "0.001",
    )
    await user.click(
      within(dialog).getByRole("button", { name: "Add fetch tool" }),
    )

    expect(await screen.findByText("Fetch tool added")).toBeInTheDocument()
    expect(
      screen.getByText(/Its price was not saved: Pricing is read-only here\./),
    ).toBeInTheDocument()
  })

  it("tests a tool from its row and says what came back", async () => {
    const user = userEvent.setup()
    const requests = mockApi()
    await renderPage()

    await user.click(await screen.findByRole("button", { name: "Test exa" }))

    expect(await screen.findByText("Worked: 3 hits.")).toBeInTheDocument()
    expect(writes(requests, "POST", "/search-tools/exa/test")[0].body).toEqual({
      query: "otari web search",
      url: null,
    })
  })

  it("removes a tool and, unless told not to, its price", async () => {
    const user = userEvent.setup()
    const requests = mockApi()
    await renderPage()

    await user.click(await screen.findByRole("button", { name: "Remove exa" }))
    const confirm = await screen.findByRole("alertdialog")
    await user.click(
      within(confirm).getByRole("button", { name: "Remove permanently" }),
    )

    await waitFor(() =>
      expect(writes(requests, "DELETE", "/pricing/exa")).toHaveLength(1),
    )
    expect(writes(requests, "DELETE", "/search-tools/exa")).toHaveLength(1)
    expect(writes(requests, "DELETE", "/pricing/exa")[0].url).toContain(
      "kind=search",
    )
  })

  it("configures a stored tool by sending only what changed", async () => {
    const user = userEvent.setup()
    const requests = mockApi()
    await renderPage()

    await user.click(await screen.findByRole("button", { name: "Edit exa" }))
    const dialog = await screen.findByRole("dialog", { name: "Configure exa" })
    expect(within(dialog).getByLabelText(/API key/)).toHaveAttribute(
      "placeholder",
      "Keep ••••a1b2",
    )
    await user.type(within(dialog).getByLabelText(/API key/), "sk-rotated")
    await user.click(within(dialog).getByRole("button", { name: "Save" }))

    await waitFor(() =>
      expect(writes(requests, "PATCH", "/search-tools/exa")).toHaveLength(1),
    )
    expect(writes(requests, "PATCH", "/search-tools/exa")[0].body).toEqual({
      expected_updated_at: "2026-10-20T10:00:00",
      api_key: "sk-rotated",
    })
    // The price did not change, so nothing was written for it.
    expect(writes(requests, "POST", "/pricing")).toHaveLength(0)
  })

  it("shows the legacy SearXNG backend only while it is set", async () => {
    mockApi()
    const { unmount } = await renderPage()
    expect(
      await screen.findByRole("heading", { name: "The model's web tools" }),
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("heading", { name: "Legacy SearXNG backend" }),
    ).not.toBeInTheDocument()
    unmount()
    vi.restoreAllMocks()

    mockApi({ settings: { web_search_url: "http://searxng:8080" } })
    await renderPage()
    expect(
      await screen.findByRole("heading", { name: "Legacy SearXNG backend" }),
    ).toBeInTheDocument()
  })

  it("saves the model's settings from their own rows, and says when one is refused", async () => {
    const user = userEvent.setup()
    const requests = mockApi()
    const { unmount } = await renderPage()

    await pickOption(
      user,
      "Read pages for hits without text web_search_extract",
      "Off",
    )
    await waitFor(() =>
      expect(writes(requests, "PATCH", "/tool-settings")[0]?.body).toEqual({
        web_search_extract: false,
      }),
    )
    await pickOption(
      user,
      "Run provider web search here web_search_intercept",
      "On",
    )
    await waitFor(() =>
      expect(writes(requests, "PATCH", "/tool-settings")[1]?.body).toEqual({
        web_search_intercept: true,
      }),
    )
    // The purpose hint stays with the API.
    expect(
      screen.queryByText("web_search_purpose_hint"),
    ).not.toBeInTheDocument()
    unmount()
    vi.restoreAllMocks()

    mockApi({ settingsRefusal: "web_search_extract must be a boolean." })
    await renderPage()
    await pickOption(
      user,
      "Read pages for hits without text web_search_extract",
      "Off",
    )
    expect(await screen.findByText(/must be a boolean/)).toBeInTheDocument()
    expect(
      selectTrigger("Read pages for hits without text web_search_extract"),
    ).toBeInTheDocument()
  })

  it("refuses a call cap that is not a whole number without asking the server", async () => {
    const user = userEvent.setup()
    const requests = mockApi()
    await renderPage()

    const input = await screen.findByLabelText(
      "Calls per request web_search_max_calls",
    )
    // A number fills the lane like the other controls, digits at its end.
    expect(input.className).toContain("text-right")
    await user.type(input, "1e1")
    await user.tab()

    expect(
      await screen.findByText("A whole number, 1 or more."),
    ).toBeInTheDocument()
    expect(writes(requests, "PATCH", "/tool-settings")).toHaveLength(0)
  })

  it("re-reads the search providers after the legacy backend URL is saved", async () => {
    // A searxng search tool with no endpoint of its own inherits
    // web_search_url, so the catalog that says what a blank endpoint resolves
    // to changes with this very write.
    const user = userEvent.setup()
    const requests = mockApi({
      settings: { web_search_url: "http://searxng:8080" },
    })
    await renderPage()

    const input = await screen.findByLabelText("Backend URL web_search_url")
    const providerReads = () =>
      requests.filter((request) =>
        request.url.includes(`${API_ROOT}/search-tools/providers`),
      ).length
    await waitFor(() => expect(providerReads()).toBeGreaterThan(0))
    const before = providerReads()

    await user.clear(input)
    await user.type(input, "http://searxng:9000")
    await user.tab()

    await waitFor(() => expect(providerReads()).toBeGreaterThan(before))
  })

  it("retries only the price when an edit's update went through and its price did not", async () => {
    const user = userEvent.setup()
    const requests = mockApi({ priceFailsOnce: true })
    await renderPage()

    await user.click(await screen.findByRole("button", { name: "Edit exa" }))
    const dialog = await screen.findByRole("dialog", { name: "Configure exa" })
    await user.type(within(dialog).getByLabelText(/API key/), "sk-rotated")
    const price = within(dialog).getByLabelText("Price per call, USD")
    await user.clear(price)
    await user.type(price, "0.007")
    await user.click(within(dialog).getByRole("button", { name: "Save" }))
    await waitFor(() =>
      expect(writes(requests, "POST", "/pricing")).toHaveLength(1),
    )

    await user.click(within(dialog).getByRole("button", { name: "Save" }))
    await waitFor(() =>
      expect(writes(requests, "POST", "/pricing")).toHaveLength(2),
    )
    // The update is not sent again with a stale updated_at, which the API
    // would refuse with 412.
    expect(writes(requests, "PATCH", "/search-tools/exa")).toHaveLength(1)
  })

  it("retries only the price delete when removing a tool went through and its price did not", async () => {
    const user = userEvent.setup()
    const requests = mockApi({ priceFailsOnce: true })
    await renderPage()

    await user.click(await screen.findByRole("button", { name: "Remove exa" }))
    const confirm = await screen.findByRole("alertdialog")
    await user.click(
      within(confirm).getByRole("button", { name: "Remove permanently" }),
    )
    await waitFor(() =>
      expect(writes(requests, "DELETE", "/pricing/exa")).toHaveLength(1),
    )
    await user.click(
      within(confirm).getByRole("button", { name: "Remove permanently" }),
    )

    await waitFor(() =>
      expect(writes(requests, "DELETE", "/pricing/exa")).toHaveLength(2),
    )
    expect(writes(requests, "DELETE", "/search-tools/exa")).toHaveLength(1)
  })

  it("drops an unsaved test's result once the settings it ran on change", async () => {
    const user = userEvent.setup()
    mockApi()
    await renderPage()

    await user.click(
      await screen.findByRole("button", { name: "Add search tool" }),
    )
    const dialog = await screen.findByRole("dialog")
    await pickOption(user, "Provider", "exa", dialog)
    await user.type(within(dialog).getByLabelText(/API key/), "sk-new")
    await user.click(within(dialog).getByRole("button", { name: "Test" }))
    expect(
      await within(dialog).findByText("Worked: 3 hits."),
    ).toBeInTheDocument()

    await user.type(within(dialog).getByLabelText(/API key/), "x")
    expect(
      within(dialog).queryByText("Worked: 3 hits."),
    ).not.toBeInTheDocument()
  })

  it("prices a config.yml tool as a row of its own kind", async () => {
    const user = userEvent.setup()
    const requests = mockApi()
    await renderPage()

    await user.click(
      await screen.findByRole("button", { name: "Edit the price of local" }),
    )
    await user.type(await screen.findByLabelText(/Price per call, USD/), "0.07")
    await user.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() =>
      expect(writes(requests, "POST", "/pricing")[0]?.body).toEqual({
        kind: "search",
        model_key: "local",
        // Rounded: 0.07 * 1e6 is 70000.00000000001 in binary floating point.
        input_price_per_million: 70000,
        output_price_per_million: 0,
        unit: "requests",
      }),
    )
  })

  it("leaves a rate config.yml sets to config.yml", async () => {
    mockApi({
      rates: {
        "search:local": pricingResponse({
          model_key: "local",
          kind: "search",
          input_price_per_million: 3000,
          unit: "requests",
          origin: "config",
        }),
      },
    })
    await renderPage()

    const local = await findRow("Search tools", "local")
    expect(
      await within(local).findByText("Price in config.yml"),
    ).toBeInTheDocument()
    expect(
      within(local).queryByRole("button", { name: "Edit the price of local" }),
    ).not.toBeInTheDocument()
  })

  it("calls a tool unreadable, not unpriced, when the rate it falls back to could not be read", async () => {
    mockApi({
      failingReads: ["model:otari:web_search"],
      requirePricing: true,
    })
    await renderPage()

    const local = await findRow("Search tools", "local")
    expect(await within(local).findByText("Could not read")).toBeInTheDocument()
    expect(within(local).queryByText("Unpriced")).not.toBeInTheDocument()
  })
})
