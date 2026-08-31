import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactElement } from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type {
  AliasResponse,
  DiscoverableModelsResponse,
  GatewaySettings,
  ModelListResponse,
  ModelMetadata,
  ModelMetadataResponse,
  ModelObject,
  OrganizationContext,
  PricingResponse,
} from "@/client"
import { ModelsPage } from "@/features/models/ModelsPage"
import * as apiClient from "@/shared/api/client"
import { organizationContext, pricingResponse } from "@/tests/fixtures"
import { withRouter } from "@/tests/router"
import { pickOption, selectTrigger } from "@/tests/select"

const PRICED: PricingResponse = pricingResponse({
  model_key: "openai:gpt-4o",
  effective_at: "2026-01-01T00:00:00Z",
  input_price_per_million: 2.5,
  output_price_per_million: 10,
  cache_read_price_per_million: null,
  cache_write_price_per_million: null,
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
})

const SETTINGS: GatewaySettings = {
  mode: "standalone",
  version: "1.0.0",
  model_discovery: true,
  default_pricing: true,
  require_pricing: false,
  master_key_source: "configured",
  secret_key_configured: true,
  config: [],
}

function catalogModel(
  id: string,
  owned_by: string,
  source: "configured" | "default" | "none",
  price: [number, number] | null,
  context_window: number | null = null,
  cache: [number | null, number | null] = [null, null],
): ModelObject {
  return {
    id,
    object: "model",
    created: 0,
    owned_by,
    pricing: price
      ? {
          input_price_per_million: price[0],
          output_price_per_million: price[1],
          cache_read_price_per_million: cache[0],
          cache_write_price_per_million: cache[1],
        }
      : null,
    pricing_source: source,
    context_window,
  }
}

const CATALOG: ModelListResponse = {
  object: "list",
  data: [
    catalogModel("openai:gpt-4o", "openai", "configured", [2.5, 10], 128000),
    catalogModel(
      "openai:gpt-4o-mini",
      "openai",
      "default",
      [0.15, 0.6],
      128000,
    ),
    catalogModel(
      "anthropic:claude-sonnet-4",
      "anthropic",
      "default",
      [3, 15],
      200000,
    ),
    catalogModel("fast-model", "otari", "default", [0.15, 0.6], 128000),
  ],
}

const DISCOVERABLE: DiscoverableModelsResponse = {
  providers: [
    {
      provider: "openai",
      ok: true,
      error: null,
      discovery_unsupported: false,
      models: [
        { id: "gpt-4o", key: "openai:gpt-4o" },
        { id: "gpt-4o-mini", key: "openai:gpt-4o-mini" },
      ],
    },
    {
      provider: "anthropic",
      ok: true,
      error: null,
      discovery_unsupported: false,
      models: [{ id: "claude-sonnet-4", key: "anthropic:claude-sonnet-4" }],
    },
  ],
}

function meta(overrides: Partial<ModelMetadata>): ModelMetadata {
  return {
    name: null,
    description: null,
    family: null,
    input_modalities: ["text"],
    output_modalities: ["text"],
    reasoning: false,
    tool_call: false,
    structured_output: false,
    attachment: false,
    temperature: false,
    context_window: null,
    max_output_tokens: null,
    knowledge_cutoff: null,
    release_date: null,
    last_updated: null,
    open_weights: false,
    deprecated: false,
    cost_input: null,
    cost_output: null,
    ...overrides,
  }
}

const METADATA: ModelMetadataResponse = {
  source: "models.dev",
  available: true,
  models: {
    "openai:gpt-4o": meta({
      input_modalities: ["text", "image"],
      tool_call: true,
      context_window: 128000,
      knowledge_cutoff: "2023-09",
      release_date: "2024-05-13",
      description: "GPT-4o multimodal model.",
    }),
    "openai:gpt-4o-mini": meta({
      input_modalities: ["text", "image"],
      tool_call: true,
      context_window: 128000,
      release_date: "2024-07-18",
    }),
    "anthropic:claude-sonnet-4": meta({
      input_modalities: ["text", "image"],
      tool_call: true,
      reasoning: true,
      context_window: 200000,
      release_date: "2025-05-14",
    }),
  },
}

const ALIASES: AliasResponse[] = [
  {
    name: "fast-model",
    target: "openai:gpt-4o-mini",
    source: "config",
    user_id: null,
    created_at: null,
    updated_at: null,
  },
]

function table(): HTMLElement {
  // react-aria's Table exposes role="grid" once row selection is enabled.
  return screen.getByRole("grid")
}

function tableRow(text: string): HTMLElement {
  return within(table()).getByText(text).closest("tr") as HTMLElement
}

// The persistent detail panel is the page's <aside> (role complementary).
function panel(): HTMLElement {
  return screen.getByRole("complementary")
}

function renderWithClient(ui: ReactElement, url = "/") {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>{ui}</QueryClientProvider>,
    { wrapper: withRouter({ url }) },
  )
}

function mockApi(
  opts: {
    post?: (url: string) => unknown
    catalog?: ModelListResponse
    pricing?: PricingResponse[]
    aliases?: AliasResponse[]
    discoverable?: DiscoverableModelsResponse
    metadata?: ModelMetadataResponse
    settings?: GatewaySettings
    // The caller's membership context, an operator's by default: the page
    // gates its deployment-wide reads and every write affordance on
    // `deployment_operator`, so most tests here are about the operator view.
    context?: OrganizationContext
  } = {},
) {
  return vi
    .spyOn(apiClient, "apiFetch")
    .mockImplementation(async (input, init) => {
      const url = String(input)
      const method = (init?.method ?? "GET").toUpperCase()
      if (method === "POST") {
        return (opts.post ? opts.post(url) : PRICED) as never
      }
      if (method === "DELETE") {
        return null as never
      }
      if (url.endsWith("/v1/organizations/me")) {
        return (opts.context ?? organizationContext()) as never
      }
      if (url.includes("/v1/settings")) {
        return (opts.settings ?? SETTINGS) as never
      }
      // Specific /v1/models/* routes before the /v1/models/ catch-all (which 404s,
      // matching the server's route order).
      if (url.includes("/v1/models/discoverable")) {
        return (opts.discoverable ?? DISCOVERABLE) as never
      }
      if (url.includes("/v1/models/metadata")) {
        return (opts.metadata ?? METADATA) as never
      }
      if (url.includes("/v1/aliases")) {
        return (opts.aliases ?? ALIASES) as never
      }
      if (url.includes("/v1/models/")) {
        throw new Error("Model not found")
      }
      if (url.includes("/v1/models")) {
        return (opts.catalog ?? CATALOG) as never
      }
      if (url.includes("/v1/pricing")) {
        return (opts.pricing ?? [PRICED]) as never
      }
      return [] as never
    })
}

// Selects a model row and returns the detail panel scoped for assertions.
async function selectModel(
  user: ReturnType<typeof userEvent.setup>,
  key: string,
): Promise<HTMLElement> {
  await user.click(tableRow(key))
  return panel()
}

function modelOrder(): string[] {
  return within(table())
    .getAllByRole("row")
    .map((row) => row.textContent ?? "")
    .filter((text) => text.includes(":"))
}

describe("ModelsPage", () => {
  beforeEach(() => {
    // The page persists sort choice in localStorage; start each test clean so
    // one test's sort does not leak into the next.
    window.localStorage.clear()
  })
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("shows no usage columns anywhere", async () => {
    mockApi()

    renderWithClient(<ModelsPage />)
    await screen.findByText("openai:gpt-4o")

    for (const heading of ["Requests", "Tokens", "Cost"]) {
      expect(
        screen.queryByRole("columnheader", { name: heading }),
      ).not.toBeInTheDocument()
    }
  })

  // -- model list ----------------------------------------------------------

  it("copies the provider-qualified model id, which the visible name omits", async () => {
    // The Model column drops the provider prefix, and a selectable row cannot be
    // highlighted, so the id a caller sends as `model` is otherwise unreachable
    // from this table (#478).
    mockApi()
    const user = userEvent.setup()

    renderWithClient(<ModelsPage />)
    await screen.findByText("openai:gpt-4o")

    await user.click(
      within(tableRow("openai:gpt-4o")).getByRole("button", {
        name: "Copy model id",
      }),
    )

    expect(await navigator.clipboard.readText()).toBe("openai:gpt-4o")
  })

  it("keeps the screen-reader-only key out of a hand-made selection", async () => {
    // `sr-only` is clipped, not hidden, so its text joins a range selection and a
    // drag across this cell used to yield "gpt-4oopenai:gpt-4o". Its own
    // `select-none` outranks the inherited `user-select: text`.
    mockApi()
    renderWithClient(<ModelsPage />)
    await screen.findByText("openai:gpt-4o")

    const srOnly = within(tableRow("openai:gpt-4o")).getByText(
      "openai:gpt-4o",
      {
        selector: "span.sr-only",
      },
    )
    expect(srOnly.className).toContain("select-none")
  })

  it("does not show a context column in the model table", async () => {
    mockApi()

    renderWithClient(<ModelsPage />)
    await screen.findByText("openai:gpt-4o")

    expect(
      within(table()).queryByRole("columnheader", { name: "Context" }),
    ).not.toBeInTheDocument()
    expect(
      within(tableRow("openai:gpt-4o")).queryByText("128K"),
    ).not.toBeInTheDocument()
  })

  it("surfaces the default-pricing note as a tooltip on the pricing column, not a banner", async () => {
    mockApi()

    renderWithClient(<ModelsPage />)
    await screen.findByText("openai:gpt-4o")

    // The explanation lives in a tooltip anchored to the pricing header, reached
    // via a labeled info trigger, rather than a persistent banner. Awaited: the
    // trigger waits on the settings read, which itself waits on the membership
    // context that says an operator is asking.
    expect(
      await within(table()).findByRole("button", {
        name: /How unpriced models are metered/,
      }),
    ).toBeInTheDocument()
    const tip = within(table()).getByRole("tooltip")
    expect(tip).toHaveTextContent(/Default pricing is on/)
    expect(tip).toHaveTextContent(/genai-prices/)
  })

  it("hides the detail panel until a model is selected, then closes it again", async () => {
    mockApi()
    const user = userEvent.setup()

    renderWithClient(<ModelsPage />)
    await screen.findByText("openai:gpt-4o")

    // Hidden initially so the table has the page to itself.
    expect(screen.queryByRole("complementary")).not.toBeInTheDocument()

    // Selecting a row opens it.
    await user.click(tableRow("openai:gpt-4o"))
    expect(screen.getByRole("complementary")).toBeInTheDocument()

    // The close button dismisses it back to the table-only layout.
    await user.click(
      screen.getByRole("button", { name: "Close model details" }),
    )
    expect(screen.queryByRole("complementary")).not.toBeInTheDocument()
  })

  it("fills the detail panel when a model row is selected", async () => {
    mockApi()
    const user = userEvent.setup()

    renderWithClient(<ModelsPage />)
    await screen.findByText("openai:gpt-4o")

    const detail = await selectModel(user, "openai:gpt-4o")

    expect(within(detail).getByText("configured")).toBeInTheDocument()
    expect(within(detail).getByText("128K")).toBeInTheDocument()
    expect(within(detail).getByText("2023-09")).toBeInTheDocument()
    expect(within(detail).getByText("Tool calling")).toBeInTheDocument()
    expect(within(detail).queryByText("Provider")).not.toBeInTheDocument()
  })

  it("filters the model list by search", async () => {
    mockApi()
    const user = userEvent.setup()

    renderWithClient(<ModelsPage />)
    await screen.findByText("anthropic:claude-sonnet-4")

    await user.type(screen.getByRole("searchbox"), "gpt-4o-mini")

    expect(within(table()).getByText("openai:gpt-4o-mini")).toBeInTheDocument()
    expect(
      within(table()).queryByText("anthropic:claude-sonnet-4"),
    ).not.toBeInTheDocument()
  })

  it("filters by provider", async () => {
    mockApi()
    const user = userEvent.setup()

    renderWithClient(<ModelsPage />)
    await screen.findByText("openai:gpt-4o")

    await pickOption(user, "Filter by provider", "anthropic")

    expect(
      within(table()).getByText("anthropic:claude-sonnet-4"),
    ).toBeInTheDocument()
    expect(within(table()).queryByText("openai:gpt-4o")).not.toBeInTheDocument()
  })

  it("reads the provider filter from the URL query parameter", async () => {
    mockApi()

    renderWithClient(<ModelsPage />, "/models?provider=anthropic")
    await screen.findByText("anthropic:claude-sonnet-4")

    // The provider select is pre-set to the URL's provider, and only that
    // provider's models are shown.
    expect(selectTrigger("Filter by provider")).toHaveTextContent("anthropic")
    expect(
      within(table()).getByText("anthropic:claude-sonnet-4"),
    ).toBeInTheDocument()
    expect(within(table()).queryByText("openai:gpt-4o")).not.toBeInTheDocument()
  })

  it("falls back to all providers when the URL names an unknown provider", async () => {
    mockApi()

    renderWithClient(<ModelsPage />, "/models?provider=doesnotexist")
    await screen.findByText("anthropic:claude-sonnet-4")

    // A stale/misspelled ?provider= resets to "all" once the catalog loads,
    // so the select is usable and every provider's models remain visible.
    expect(selectTrigger("Filter by provider")).toHaveTextContent(
      "All providers",
    )
    expect(
      within(table()).getByText("anthropic:claude-sonnet-4"),
    ).toBeInTheDocument()
    expect(within(table()).getByText("openai:gpt-4o")).toBeInTheDocument()
  })

  it("filters to custom prices only", async () => {
    mockApi()
    const user = userEvent.setup()

    renderWithClient(<ModelsPage />)
    await screen.findByText("openai:gpt-4o")

    await pickOption(user, "Filter by pricing", "Custom price")

    expect(within(table()).getByText("openai:gpt-4o")).toBeInTheDocument()
    expect(
      within(table()).queryByText("openai:gpt-4o-mini"),
    ).not.toBeInTheDocument()
    expect(
      within(table()).queryByText("anthropic:claude-sonnet-4"),
    ).not.toBeInTheDocument()
  })

  it("filters to custom (not discovered) models", async () => {
    mockApi({
      pricing: [
        PRICED,
        {
          ...PRICED,
          model_key: "mistral:large",
          input_price_per_million: 4,
          output_price_per_million: 12,
        },
      ],
    })
    const user = userEvent.setup()

    renderWithClient(<ModelsPage />)
    await screen.findByText("mistral:large")

    await pickOption(user, "Filter by source", "Custom (not discovered)")

    expect(within(table()).getByText("mistral:large")).toBeInTheDocument()
    expect(within(table()).queryByText("openai:gpt-4o")).not.toBeInTheDocument()
  })

  it("filters by a per-model capability, matching the Features chips", async () => {
    mockApi()
    const user = userEvent.setup()

    renderWithClient(<ModelsPage />)
    await screen.findByText("anthropic:claude-sonnet-4")

    // Reasoning is a model-level flag (models.dev), not a provider capability:
    // only claude-sonnet-4 reports it, so the two gpt-4o models drop out.
    await pickOption(user, "Filter by capability", "Reasoning")

    expect(
      within(table()).getByText("anthropic:claude-sonnet-4"),
    ).toBeInTheDocument()
    expect(within(table()).queryByText("openai:gpt-4o")).not.toBeInTheDocument()
    expect(
      within(table()).queryByText("openai:gpt-4o-mini"),
    ).not.toBeInTheDocument()
  })

  it("treats missing input modalities as not matching a modality filter", async () => {
    const withoutModalities = {
      ...METADATA,
      models: {
        ...(METADATA.models ?? {}),
        "openai:gpt-4o": {
          ...METADATA.models?.["openai:gpt-4o"],
          input_modalities: undefined,
        },
      },
    } as unknown as ModelMetadataResponse
    mockApi({ metadata: withoutModalities })
    const user = userEvent.setup()

    renderWithClient(<ModelsPage />)
    await screen.findByText("openai:gpt-4o")
    await pickOption(user, "Filter by capability", "Vision")

    expect(within(table()).queryByText("openai:gpt-4o")).not.toBeInTheDocument()
  })

  it("explains an empty list is due to filters, not missing credentials", async () => {
    mockApi()
    const user = userEvent.setup()

    renderWithClient(<ModelsPage />)
    await screen.findByText("openai:gpt-4o")

    // No mock model reports PDF input, so the list empties for a filter reason.
    await pickOption(user, "Filter by capability", "PDF")

    expect(
      within(table()).getByText("No models match your filters."),
    ).toBeInTheDocument()
  })

  it("filters by minimum context window", async () => {
    mockApi()
    const user = userEvent.setup()

    renderWithClient(<ModelsPage />)
    await screen.findByText("openai:gpt-4o")

    await pickOption(user, "Minimum context window", "≥ 200K")

    expect(
      within(table()).getByText("anthropic:claude-sonnet-4"),
    ).toBeInTheDocument()
    expect(within(table()).queryByText("openai:gpt-4o")).not.toBeInTheDocument()
  })

  it("sorts by input price when the column header is clicked", async () => {
    mockApi()
    const user = userEvent.setup()

    renderWithClient(<ModelsPage />)
    await screen.findByText("openai:gpt-4o")

    await user.click(
      screen.getByRole("columnheader", { name: /Base in \/ out \/ 1M/ }),
    )

    const order = modelOrder()
    const miniIndex = order.findIndex((text) => text.includes("gpt-4o-mini"))
    const claudeIndex = order.findIndex((text) =>
      text.includes("claude-sonnet-4"),
    )
    expect(miniIndex).toBeLessThan(claudeIndex)
  })

  it("defaults to sorting by model name, ascending", async () => {
    mockApi()

    renderWithClient(<ModelsPage />)
    await screen.findByText("openai:gpt-4o")

    // No header click needed: default is model A→Z, so anthropic sorts before openai.
    const order = modelOrder()
    const claude = order.findIndex((text) => text.includes("claude-sonnet-4"))
    const base = order.findIndex(
      (text) => text.includes("openai:gpt-4o") && !text.includes("mini"),
    )
    const mini = order.findIndex((text) => text.includes("gpt-4o-mini"))
    expect(claude).toBeLessThan(base)
    expect(base).toBeLessThan(mini)
  })

  it("remembers the chosen sort across a remount", async () => {
    mockApi()
    const user = userEvent.setup()

    const { unmount } = renderWithClient(<ModelsPage />)
    await screen.findByText("openai:gpt-4o")

    // Flip the Model column to Z→A and confirm it is persisted.
    await user.click(screen.getByRole("columnheader", { name: /^Model/ }))
    expect(
      JSON.parse(
        window.localStorage.getItem("otari.dashboard.modelsSort") ?? "{}",
      ),
    ).toEqual({
      col: "model",
      dir: "desc",
    })

    unmount()

    // A fresh mount restores Z→A rather than the A→Z default.
    renderWithClient(<ModelsPage />)
    await screen.findByText("openai:gpt-4o")
    const order = modelOrder()
    const base = order.findIndex(
      (text) => text.includes("openai:gpt-4o") && !text.includes("mini"),
    )
    const claude = order.findIndex((text) => text.includes("claude-sonnet-4"))
    expect(base).toBeLessThan(claude)
  })

  it("defaults to 15 models per page and lets operators change the page size", async () => {
    const models = Array.from({ length: 30 }, (_, i) => {
      const id = `m${String(i).padStart(2, "0")}`
      return { id, key: `openai:${id}` }
    })
    mockApi({
      catalog: { object: "list", data: [] },
      pricing: [],
      discoverable: {
        providers: [
          {
            provider: "openai",
            ok: true,
            error: null,
            discovery_unsupported: false,
            models,
          },
        ],
      },
      metadata: { source: "models.dev", available: true, models: {} },
    })
    const user = userEvent.setup()

    renderWithClient(<ModelsPage />)
    await screen.findByText("openai:m00")

    expect(within(table()).getByText("openai:m14")).toBeInTheDocument()
    expect(within(table()).queryByText("openai:m15")).not.toBeInTheDocument()

    await pickOption(user, "Rows per page", "25")

    expect(within(table()).getByText("openai:m24")).toBeInTheDocument()
    expect(within(table()).queryByText("openai:m25")).not.toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Next page" }))
    expect(within(table()).getByText("openai:m25")).toBeInTheDocument()
    expect(within(table()).queryByText("openai:m00")).not.toBeInTheDocument()
  })

  it("names a provider discovery could not list", async () => {
    mockApi({
      discoverable: {
        providers: [
          {
            provider: "openai",
            ok: true,
            error: null,
            discovery_unsupported: false,
            models: [{ id: "gpt-4o", key: "openai:gpt-4o" }],
          },
          {
            provider: "anthropic",
            ok: false,
            error: "401 authentication_error",
            discovery_unsupported: false,
            models: [],
          },
        ],
      },
    })

    renderWithClient(<ModelsPage />)

    expect(
      await screen.findByText(/Could not list anthropic/),
    ).toBeInTheDocument()
  })

  it("says a provider offers no model discovery rather than blaming its credentials", async () => {
    // otari#447: pointing the operator at config.yml credentials would send them
    // after a problem that isn't there.
    mockApi({
      discoverable: {
        providers: [
          {
            provider: "otari",
            ok: false,
            error: "Error code: 404 - {'detail': 'Not Found'}",
            discovery_unsupported: true,
            models: [],
          },
        ],
      },
    })

    renderWithClient(<ModelsPage />)

    expect(
      await screen.findByText(/otari does not offer model discovery/),
    ).toBeInTheDocument()
    expect(screen.queryByText(/Could not list/)).not.toBeInTheDocument()
  })

  it("agrees in number when several providers offer no discovery", async () => {
    mockApi({
      discoverable: {
        providers: [
          {
            provider: "otari",
            ok: false,
            error: "404",
            discovery_unsupported: true,
            models: [],
          },
          {
            provider: "vllm",
            ok: false,
            error: "404",
            discovery_unsupported: true,
            models: [],
          },
        ],
      },
    })

    renderWithClient(<ModelsPage />)

    expect(
      await screen.findByText(/otari, vllm do not offer model discovery/),
    ).toBeInTheDocument()
    expect(screen.getByText(/their models are missing/)).toBeInTheDocument()
  })

  // -- inline pricing (in the detail panel) --------------------------------

  it("edits a configured price from the detail panel", async () => {
    const fetchMock = mockApi()
    const user = userEvent.setup()

    renderWithClient(<ModelsPage />)
    await screen.findByText("openai:gpt-4o")

    const detail = await selectModel(user, "openai:gpt-4o")
    await user.click(within(detail).getByRole("button", { name: "Edit price" }))
    const input = within(detail).getByLabelText("Input price for openai:gpt-4o")
    await user.clear(input)
    await user.type(input, "4")
    await user.click(within(detail).getByRole("button", { name: "Save" }))

    const call = fetchMock.mock.calls.find(
      ([, init]) => (init?.method ?? "") === "POST",
    )
    expect(call).toBeDefined()
    expect(String(call?.[0])).toContain("/v1/pricing")
    expect(JSON.parse(String(call?.[1]?.body))).toMatchObject({
      model_key: "openai:gpt-4o",
      input_price_per_million: 4,
    })
  })

  it("sets cache, TTL, and long-context prices from the detail panel", async () => {
    const fetchMock = mockApi()
    const user = userEvent.setup()

    renderWithClient(<ModelsPage />)
    await screen.findByText("openai:gpt-4o")

    const detail = await selectModel(user, "openai:gpt-4o")
    await user.click(within(detail).getByRole("button", { name: "Edit price" }))
    await user.type(
      within(detail).getByLabelText("Cache read price for openai:gpt-4o"),
      "0.3",
    )
    await user.type(
      within(detail).getByLabelText("Cache write price for openai:gpt-4o"),
      "3.75",
    )
    await user.type(
      within(detail).getByLabelText(
        "1 hour cache write price for openai:gpt-4o",
      ),
      "6",
    )
    await user.click(within(detail).getByRole("button", { name: "Add tier" }))
    await user.clear(within(detail).getByLabelText("Tier context threshold"))
    await user.type(
      within(detail).getByLabelText("Tier context threshold"),
      "200000",
    )
    await user.type(within(detail).getByLabelText("Tier input price"), "6")
    await user.click(within(detail).getByRole("button", { name: "Save" }))

    const call = fetchMock.mock.calls.find(
      ([, init]) => (init?.method ?? "") === "POST",
    )
    expect(String(call?.[0])).toContain("/v1/pricing")
    expect(JSON.parse(String(call?.[1]?.body))).toMatchObject({
      model_key: "openai:gpt-4o",
      cache_read_price_per_million: 0.3,
      cache_write_price_per_million: 3.75,
      cache_write_1h_price_per_million: 6,
      pricing_tiers: [{ min_input_tokens: 200000, input_price_per_million: 6 }],
    })
  })

  it("shows configured cache prices in the table and detail panel", async () => {
    mockApi({
      catalog: {
        object: "list",
        data: [
          catalogModel(
            "anthropic:claude-sonnet-4",
            "anthropic",
            "configured",
            [3, 15],
            200000,
            [0.3, 3.75],
          ),
        ],
      },
      pricing: [],
    })
    const user = userEvent.setup()

    renderWithClient(<ModelsPage />)
    await screen.findByText("anthropic:claude-sonnet-4")

    expect(
      screen.getByRole("columnheader", { name: "Caching" }),
    ).toBeInTheDocument()

    // The compact cache-policy cell keeps the table readable while opening the
    // full structured editor on click.
    const row = tableRow("anthropic:claude-sonnet-4")
    expect(
      within(row).getByRole("button", {
        name: "Edit caching price for anthropic:claude-sonnet-4",
      }),
    ).toHaveTextContent("R $0.30 · W $3.75")

    const detail = await selectModel(user, "anthropic:claude-sonnet-4")
    expect(within(detail).getByText("Cache read")).toBeInTheDocument()
    expect(within(detail).getByText("$0.30 / 1M")).toBeInTheDocument()
    expect(within(detail).getByText("$3.75 / 1M")).toBeInTheDocument()
  })

  it("compares effective rates at a selected context threshold", async () => {
    mockApi({
      pricing: [
        {
          ...PRICED,
          pricing_tiers: [
            {
              min_input_tokens: 200000,
              input_price_per_million: 5,
              output_price_per_million: 20,
              cache_read_price_per_million: 0.5,
            },
          ],
        },
      ],
    })
    const user = userEvent.setup()

    renderWithClient(<ModelsPage />)
    await screen.findByText("openai:gpt-4o")

    await pickOption(user, "Compare prices at context", "Compare at 200K")

    expect(
      screen.getByRole("columnheader", { name: /at 200K in \/ out \/ 1M/ }),
    ).toBeInTheDocument()
    const row = tableRow("openai:gpt-4o")
    expect(
      within(row).getByRole("button", {
        name: "Edit input price for openai:gpt-4o",
      }),
    ).toHaveTextContent("$5.00")
    expect(
      within(row).getByRole("button", {
        name: "Edit pricing policy for openai:gpt-4o",
      }),
    ).toHaveTextContent("1 tier · ≥ 200K")
  })

  it("orders pricing-policy thresholds by their underlying token count", async () => {
    mockApi({
      pricing: [
        {
          ...PRICED,
          pricing_tiers: [
            { min_input_tokens: 1_000_000, input_price_per_million: 6 },
            { min_input_tokens: 999_000, input_price_per_million: 5 },
          ],
        },
      ],
    })

    renderWithClient(<ModelsPage />)
    await screen.findByText("openai:gpt-4o")

    expect(
      within(tableRow("openai:gpt-4o")).getByRole("button", {
        name: "Edit pricing policy for openai:gpt-4o",
      }),
    ).toHaveTextContent("2 tiers · ≥ 999K, 1M")
  })

  it("removes the repeated provider instance from a model's visible name", async () => {
    mockApi({
      catalog: {
        object: "list",
        data: [
          catalogModel(
            "homelab:work-anthropic:claude-sonnet-5",
            "homelab",
            "none",
            null,
            200000,
          ),
        ],
      },
      pricing: [],
      discoverable: { providers: [] },
      metadata: { source: "models.dev", available: true, models: {} },
    })
    const user = userEvent.setup()

    renderWithClient(<ModelsPage />)
    await screen.findByText("homelab:work-anthropic:claude-sonnet-5")

    const row = tableRow("homelab:work-anthropic:claude-sonnet-5")
    expect(
      within(row).getByText("work-anthropic:claude-sonnet-5"),
    ).toBeInTheDocument()
    expect(within(row).getByText("homelab")).toBeInTheDocument()

    const detail = await selectModel(
      user,
      "homelab:work-anthropic:claude-sonnet-5",
    )
    expect(within(detail).getByText(/Selector:/)).toBeInTheDocument()
    expect(
      within(detail).getByText("homelab:work-anthropic:claude-sonnet-5"),
    ).toBeInTheDocument()
  })

  it("opens the inline editor by clicking a price number and saves cache rates", async () => {
    const fetchMock = mockApi()
    const user = userEvent.setup()

    renderWithClient(<ModelsPage />)
    await screen.findByText("openai:gpt-4o")

    const row = tableRow("openai:gpt-4o")
    await user.click(
      within(row).getByRole("button", {
        name: "Edit input price for openai:gpt-4o",
      }),
    )

    // The inline editor row appears with all four fields.
    const cacheRead = await screen.findByLabelText(
      "Cache read price for openai:gpt-4o",
    )
    await user.type(cacheRead, "0.3")
    await user.click(screen.getByRole("button", { name: "Save" }))

    const call = fetchMock.mock.calls.find(
      ([, init]) => (init?.method ?? "") === "POST",
    )
    expect(String(call?.[0])).toContain("/v1/pricing")
    expect(JSON.parse(String(call?.[1]?.body))).toMatchObject({
      model_key: "openai:gpt-4o",
      cache_read_price_per_million: 0.3,
    })
  })

  it("does not prompt to backfill usage after setting a price", async () => {
    mockApi()
    const user = userEvent.setup()

    renderWithClient(<ModelsPage />)
    await screen.findByText("openai:gpt-4o")

    const detail = await selectModel(user, "openai:gpt-4o")
    await user.click(within(detail).getByRole("button", { name: "Edit price" }))
    await user.click(within(detail).getByRole("button", { name: "Save" }))

    expect(
      await within(panel()).findByRole("button", { name: "Edit price" }),
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: /backfill/i }),
    ).not.toBeInTheDocument()
  })

  it("bulk-sets pricing on the selected models", async () => {
    const fetchMock = mockApi()
    const user = userEvent.setup()

    renderWithClient(<ModelsPage />)
    await screen.findByText("openai:gpt-4o")

    const row = tableRow("openai:gpt-4o")
    await user.click(within(row).getByRole("checkbox"))

    const bar = (await screen.findByText("1 selected")).closest("div")!
    await user.click(within(bar).getByRole("button", { name: "Set pricing" }))

    const dialog = await screen.findByRole("alertdialog")
    await user.type(within(dialog).getByLabelText("Input $ / 1M"), "5")
    await user.type(within(dialog).getByLabelText("Output $ / 1M"), "12")
    await user.click(within(dialog).getByRole("button", { name: "Set price" }))

    await vi.waitFor(() => {
      const call = fetchMock.mock.calls.find(
        ([u, init]) =>
          String(u).includes("/v1/pricing") &&
          (init?.method ?? "").toUpperCase() === "POST",
      )
      expect(call).toBeTruthy()
      expect(JSON.parse(String(call![1]!.body))).toMatchObject({
        model_key: "openai:gpt-4o",
        input_price_per_million: 5,
        output_price_per_million: 12,
      })
    })
  })

  // -- pricing a model the catalog does not list --------------------------

  it("prices the searched selector when nothing matches, seeded from the search box", async () => {
    const created: PricingResponse = {
      ...PRICED,
      model_key: "vllm:mistral-small",
      input_price_per_million: 0.2,
      output_price_per_million: 0.6,
    }
    // The pricing list is read per request, so appending on POST models the
    // refetch that puts the hand-priced model into the catalog.
    const pricingRows = [PRICED]
    const fetchMock = mockApi({
      pricing: pricingRows,
      post: () => {
        pricingRows.push(created)
        return created
      },
    })
    const user = userEvent.setup()

    renderWithClient(<ModelsPage />)
    await screen.findByText("openai:gpt-4o")

    // The operator searches for the model they just called successfully; the
    // table comes up empty and offers to price that exact selector.
    await user.type(screen.getByRole("searchbox"), "vllm:mistral-small")
    await user.click(
      await screen.findByRole("button", { name: "Price vllm:mistral-small" }),
    )

    const dialog = await screen.findByRole("alertdialog")
    expect(within(dialog).getByLabelText("Model key")).toHaveValue(
      "vllm:mistral-small",
    )
    await user.type(within(dialog).getByLabelText("Input $ / 1M"), "0.2")
    await user.type(within(dialog).getByLabelText("Output $ / 1M"), "0.6")
    await user.click(within(dialog).getByRole("button", { name: "Set price" }))

    await vi.waitFor(() => {
      const call = fetchMock.mock.calls.find(
        ([u, init]) =>
          String(u).includes("/v1/pricing") &&
          (init?.method ?? "").toUpperCase() === "POST",
      )
      expect(call).toBeTruthy()
      expect(JSON.parse(String(call![1]!.body))).toMatchObject({
        model_key: "vllm:mistral-small",
        input_price_per_million: 0.2,
        output_price_per_million: 0.6,
      })
    })

    // Selected on success, so the operator lands on the price they just set
    // even though the model is not discoverable.
    expect(
      await within(panel()).findByText("vllm:mistral-small"),
    ).toBeInTheDocument()
    expect(within(panel()).getByText("not discovered")).toBeInTheDocument()
    expect(tableRow("vllm:mistral-small")).toBeInTheDocument()
  })

  it("rejects a model key with no provider prefix", async () => {
    mockApi()
    const user = userEvent.setup()

    renderWithClient(<ModelsPage />)
    await screen.findByText("openai:gpt-4o")

    // A search that is a word rather than a selector gets the unseeded button.
    await user.type(screen.getByRole("searchbox"), "mistral")
    await user.click(
      await screen.findByRole("button", { name: "Price a model by hand" }),
    )

    const dialog = await screen.findByRole("alertdialog")
    await user.type(within(dialog).getByLabelText("Model key"), "mistral-small")
    await user.type(within(dialog).getByLabelText("Input $ / 1M"), "0.2")
    await user.type(within(dialog).getByLabelText("Output $ / 1M"), "0.6")

    expect(
      within(dialog).getByRole("button", { name: "Set price" }),
    ).toBeDisabled()
    expect(
      within(dialog).getByText(/Include the provider or instance prefix/),
    ).toBeInTheDocument()
  })

  it("prices a model from the no-discovery banner, seeded with the provider prefix", async () => {
    mockApi({
      discoverable: {
        providers: [
          {
            provider: "vllm",
            ok: false,
            error: "404",
            discovery_unsupported: true,
            models: [],
          },
        ],
      },
    })
    const user = userEvent.setup()

    renderWithClient(<ModelsPage />)
    await screen.findByText(/vllm does not offer model discovery/)

    await user.click(screen.getByRole("button", { name: "Price a model" }))

    const dialog = await screen.findByRole("alertdialog")
    expect(within(dialog).getByLabelText("Model key")).toHaveValue("vllm:")
    // A bare prefix is not yet a key, so the price cannot be submitted.
    expect(
      within(dialog).getByRole("button", { name: "Set price" }),
    ).toBeDisabled()
  })

  it("does not seed a prefix when several providers lack discovery", async () => {
    mockApi({
      discoverable: {
        providers: [
          {
            provider: "vllm",
            ok: false,
            error: "404",
            discovery_unsupported: true,
            models: [],
          },
          {
            provider: "ollama",
            ok: false,
            error: "404",
            discovery_unsupported: true,
            models: [],
          },
        ],
      },
    })
    const user = userEvent.setup()

    renderWithClient(<ModelsPage />)
    await screen.findByText(/vllm, ollama do not offer model discovery/)

    await user.click(screen.getByRole("button", { name: "Price a model" }))

    expect(
      within(await screen.findByRole("alertdialog")).getByLabelText(
        "Model key",
      ),
    ).toHaveValue("")
  })

  // A model the catalog withheld (an alias target) still reaches the table
  // through the discovery endpoint, and the price the gateway would charge for it
  // does not come with it. Calling that "not priced" claims free metering for a
  // model the fallback bills, which is the one thing the pricing table must not
  // get wrong.
  const WITHHELD = {
    catalog: {
      object: "list" as const,
      data: [catalogModel("myopus", "otari", "default", [3, 15], 200000)],
    },
    pricing: [],
    discoverable: {
      providers: [
        {
          provider: "anthropic",
          ok: true,
          error: null,
          discovery_unsupported: false,
          models: [{ id: "claude-opus-4", key: "anthropic:claude-opus-4" }],
        },
      ],
    },
  }

  it("reports an unknown rate for a withheld model when default pricing is on", async () => {
    mockApi(WITHHELD)
    const user = userEvent.setup()

    renderWithClient(<ModelsPage />)
    await screen.findByText("anthropic:claude-opus-4")

    expect(
      within(await selectModel(user, "anthropic:claude-opus-4")).getByText(
        "rate unknown",
      ),
    ).toBeInTheDocument()
  })

  it("reports it as unpriced when default pricing is off, where that is the truth", async () => {
    mockApi({ ...WITHHELD, settings: { ...SETTINGS, default_pricing: false } })
    const user = userEvent.setup()

    renderWithClient(<ModelsPage />)
    await screen.findByText("anthropic:claude-opus-4")

    expect(
      within(await selectModel(user, "anthropic:claude-opus-4")).getByText(
        "not priced",
      ),
    ).toBeInTheDocument()
  })

  it("shows a non-operator the catalog read-only and never fires the operator reads", async () => {
    // The member half of otari-ai#1942: the catalog and the price list serve
    // any session, and everything that writes or is operator-only is absent
    // rather than refused.
    const fetchMock = mockApi({
      context: organizationContext({
        deployment_operator: false,
        role: "member",
      }),
    })
    const user = userEvent.setup()
    renderWithClient(<ModelsPage />)
    await screen.findByText("openai:gpt-4o")

    // The price numbers render as text rather than as edit controls.
    expect(
      screen.queryByRole("button", { name: /Edit input price for/ }),
    ).not.toBeInTheDocument()

    // The detail panel still reports the rates, without the write affordances.
    await user.click(
      screen.getByText("openai:gpt-4o").closest("tr") as HTMLElement,
    )
    const aside = panel()
    expect(within(aside).getByText("Input")).toBeInTheDocument()
    expect(
      within(aside).queryByRole("button", { name: /Set price|Edit price/ }),
    ).not.toBeInTheDocument()
    expect(
      within(aside).queryByRole("button", { name: "Reset" }),
    ).not.toBeInTheDocument()

    const urls = fetchMock.mock.calls.map(([input]) => String(input))
    expect(urls.some((url) => url.includes("/v1/settings"))).toBe(false)
    expect(urls.some((url) => url.includes("/v1/models/discoverable"))).toBe(
      false,
    )
    expect(urls.some((url) => url.includes("/v1/models/metadata"))).toBe(false)
  })
})
