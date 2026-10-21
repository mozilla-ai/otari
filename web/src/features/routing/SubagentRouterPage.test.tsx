import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { AgentRecommender, DecisionProviderOption } from "@/client"
import { SubagentRouterPage } from "@/features/routing/SubagentRouterPage"
import { API_ROOT } from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap } from "@/tests/fixtures"
import { renderWithRouter } from "@/tests/router"

interface Request {
  url: string
  method: string
  body: unknown
}

const PROVIDERS: DecisionProviderOption[] = [
  { provider: "typesafe", self_hosted: false },
  { provider: "openrouter", self_hosted: false },
  { provider: "llamacpp", self_hosted: true },
]

function router(overrides: Partial<AgentRecommender> = {}): AgentRecommender {
  return {
    id: "11111111-1111-4111-8111-111111111111",
    provider: "openrouter",
    model: "openai/gpt-5-mini",
    api_base: null,
    api_key_last4: "9876",
    has_api_key: true,
    input_price_per_million: 2,
    output_price_per_million: 10,
    enabled: true,
    created_at: "2026-10-01T00:00:00Z",
    updated_at: null,
    ...overrides,
  }
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

function mockApi(
  opts: {
    granted?: boolean
    /** Answers the list 404, as the router refusing a non-operator. */
    listRefused?: boolean
    routers?: AgentRecommender[]
    /** Answers the PATCH that switches a router on with this refusal. */
    switchOnRefusal?: string
  } = {},
) {
  const granted = opts.granted ?? true
  const routers = opts.routers ?? [router()]
  const requests: Request[] = []

  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
    const body = init?.body ? JSON.parse(String(init.body)) : undefined
    requests.push({ url, method, body })

    if (url.includes(`${API_ROOT}/admin/access`)) {
      return jsonResponse({ granted })
    }
    if (url.includes(`${API_ROOT}/routing/recommenders`)) {
      if (method === "GET") {
        if (opts.listRefused) return jsonResponse({ detail: "Not found" }, 404)
        return jsonResponse({ data: routers, decision_providers: PROVIDERS })
      }
      if (method === "DELETE") return new Response(null, { status: 204 })
      if (method === "POST") {
        return jsonResponse(
          router({ id: "22222222-2222-4222-8222-222222222222", ...body }),
          201,
        )
      }
      const id = url.split("/").pop() ?? ""
      const patch = body as Partial<AgentRecommender>
      if (patch.enabled === true && opts.switchOnRefusal) {
        return jsonResponse({ detail: opts.switchOnRefusal }, 409)
      }
      return jsonResponse(router({ id, ...patch }))
    }
    return jsonResponse({})
  })

  return requests
}

function renderPage(surfaces?: string[]) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return renderWithRouter(
    <DeploymentProvider value={bootstrap(surfaces ? { surfaces } : {})}>
      <QueryClientProvider client={client}>
        <SubagentRouterPage />
      </QueryClientProvider>
    </DeploymentProvider>,
    { url: "/admin/subagent-router" },
  )
}

const writes = (requests: Request[]) =>
  requests
    .filter(
      (request) =>
        request.url.includes("/routing/recommenders") &&
        request.method !== "GET",
    )
    .map(({ method, body }) => ({ method, body }))

afterEach(() => {
  vi.restoreAllMocks()
})

describe("SubagentRouterPage", () => {
  it("lists each router by the tail of its key and its price", async () => {
    mockApi({
      routers: [
        router(),
        router({
          id: "33333333-3333-4333-8333-333333333333",
          provider: "llamacpp",
          model: "jev-local",
          api_base: "http://decisions:8080",
          api_key_last4: null,
          has_api_key: false,
          enabled: false,
        }),
      ],
    })
    await renderPage()

    const table = await screen.findByRole("grid", { name: "Subagent routers" })
    expect(await within(table).findByText("••••9876")).toBeInTheDocument()
    expect(within(table).getByText("No key stored")).toBeInTheDocument()
    expect(within(table).getByText("llama.cpp server")).toBeInTheDocument()
    expect(
      within(table).getAllByText("$2.00 in, $10.00 out per 1M tokens"),
    ).toHaveLength(2)
    expect(
      within(table).getByRole("switch", { name: "Route with jev-local" }),
    ).toHaveAttribute("aria-checked", "false")
  })

  it("refuses a caller who does not operate the deployment without asking for the list", async () => {
    const requests = mockApi({ granted: false })
    await renderPage()

    expect(
      await screen.findByText("Subagent router is not available to you"),
    ).toBeInTheDocument()
    expect(
      requests.filter((request) => request.url.includes("/routing/")),
    ).toEqual([])
    expect(screen.queryByRole("button", { name: "Add router" })).toBeNull()
  })

  it("refuses an operator the list itself 404s, without offering a write", async () => {
    mockApi({ listRefused: true })
    await renderPage()

    expect(
      await screen.findByText("Subagent router is not available to you"),
    ).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Add router" })).toBeNull()
  })

  it("says the page is not served where the surface is not published, asking nothing", async () => {
    const requests = mockApi()
    await renderPage([])

    expect(
      await screen.findByText("Subagent router is not available here"),
    ).toBeInTheDocument()
    expect(requests).toEqual([])
  })

  it("says what answers while no router is added", async () => {
    mockApi({ routers: [] })
    await renderPage()

    expect(await screen.findByText("No router added")).toBeInTheDocument()
    expect(screen.getByText(/agent_recommender_model/)).toBeInTheDocument()
  })

  it("says nothing answers while every router is switched off", async () => {
    mockApi({ routers: [router({ enabled: false })] })
    await renderPage()

    expect(
      await screen.findByText(/No router is switched on/),
    ).toBeInTheDocument()
  })

  it("adds the first router switched off, then switches it on", async () => {
    const requests = mockApi({ routers: [] })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      (await screen.findAllByRole("button", { name: "Add router" }))[0],
    )
    const dialog = await screen.findByRole("dialog")
    await user.type(within(dialog).getByLabelText("Model"), "jev-latest")
    await user.type(within(dialog).getByLabelText(/API key/), "ts-secret")
    await user.type(
      within(dialog).getByLabelText("Input, per 1M tokens (USD)"),
      "0.15",
    )
    // The base build adds nothing to the price, so there is no margin field.
    expect(within(dialog).queryByLabelText(/margin/i)).toBeNull()
    await user.click(within(dialog).getByRole("button", { name: "Add router" }))

    await waitFor(() => expect(writes(requests)).toHaveLength(2))
    expect(writes(requests)).toEqual([
      {
        method: "POST",
        body: {
          provider: "typesafe",
          model: "jev-latest",
          api_key: "ts-secret",
          input_price_per_million: 0.15,
          output_price_per_million: 0,
          enabled: false,
        },
      },
      { method: "PATCH", body: { enabled: true } },
    ])
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull())
  })

  it("adds a later router switched off and leaves it off", async () => {
    const requests = mockApi()
    const user = userEvent.setup()
    await renderPage()

    await user.click(await screen.findByRole("button", { name: "Add router" }))
    const dialog = await screen.findByRole("dialog")
    await user.type(within(dialog).getByLabelText("Model"), "jev-latest")
    await user.type(within(dialog).getByLabelText(/API key/), "ts-secret")
    await user.type(
      within(dialog).getByLabelText("Output, per 1M tokens (USD)"),
      "0.6",
    )
    await user.click(within(dialog).getByRole("button", { name: "Add router" }))

    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull())
    expect(writes(requests).map(({ method }) => method)).toEqual(["POST"])
  })

  it("checks the router before anything is sent", async () => {
    const requests = mockApi({ routers: [] })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      (await screen.findAllByRole("button", { name: "Add router" }))[0],
    )
    const dialog = await screen.findByRole("dialog")
    await user.type(within(dialog).getByLabelText("Model"), "jev-latest")
    await user.click(within(dialog).getByRole("button", { name: "Add router" }))

    expect(
      await within(dialog).findByText("Enter the API key for this provider."),
    ).toBeInTheDocument()
    expect(
      within(dialog).getByText(
        "Set an input or output token price above zero.",
      ),
    ).toBeInTheDocument()
    expect(writes(requests)).toEqual([])
  })

  it("edits a router that is on without resending its key, and switches it back on", async () => {
    const requests = mockApi()
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      await screen.findByRole("button", { name: "Edit openai/gpt-5-mini" }),
    )
    const dialog = await screen.findByRole("dialog")
    expect(
      within(dialog).getByPlaceholderText(
        "Stored key ends in 9876. Leave blank to keep it.",
      ),
    ).toBeInTheDocument()
    const output = within(dialog).getByLabelText("Output, per 1M tokens (USD)")
    await user.clear(output)
    await user.type(output, "12.5")
    await user.click(within(dialog).getByRole("button", { name: "Save" }))

    await waitFor(() => expect(writes(requests)).toHaveLength(2))
    expect(writes(requests)).toEqual([
      {
        method: "PATCH",
        body: {
          provider: "openrouter",
          model: "openai/gpt-5-mini",
          api_base: "",
          input_price_per_million: 2,
          output_price_per_million: 12.5,
          enabled: false,
        },
      },
      { method: "PATCH", body: { enabled: true } },
    ])
  })

  it("keeps the dialog open and the router off when switching it on is refused", async () => {
    const requests = mockApi({
      routers: [],
      switchOnRefusal: "Another subagent recommender was switched on",
    })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      (await screen.findAllByRole("button", { name: "Add router" }))[0],
    )
    const dialog = await screen.findByRole("dialog")
    await user.type(within(dialog).getByLabelText("Model"), "jev-latest")
    await user.type(within(dialog).getByLabelText(/API key/), "ts-secret")
    await user.type(
      within(dialog).getByLabelText("Input, per 1M tokens (USD)"),
      "0.15",
    )
    await user.click(within(dialog).getByRole("button", { name: "Add router" }))

    expect(
      await within(dialog).findByText(
        /Another subagent recommender was switched on/,
      ),
    ).toBeInTheDocument()

    // Saving again finishes the router already added rather than adding a second.
    await user.click(within(dialog).getByRole("button", { name: "Add router" }))
    await waitFor(() => expect(writes(requests)).toHaveLength(4))
    expect(writes(requests).map(({ method }) => method)).toEqual([
      "POST",
      "PATCH",
      "PATCH",
      "PATCH",
    ])
  })

  it("switches a router on from its row", async () => {
    const requests = mockApi({ routers: [router({ enabled: false })] })
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      await screen.findByRole("switch", {
        name: "Route with openai/gpt-5-mini",
      }),
    )

    await waitFor(() =>
      expect(writes(requests)).toEqual([
        { method: "PATCH", body: { enabled: true } },
      ]),
    )
  })

  it("removes a router once confirmed", async () => {
    const requests = mockApi()
    const user = userEvent.setup()
    await renderPage()

    await user.click(
      await screen.findByRole("button", { name: "Remove openai/gpt-5-mini" }),
    )
    const dialog = await screen.findByRole("alertdialog")
    expect(
      within(dialog).getByText(/agent_recommender_model names answers/),
    ).toBeInTheDocument()
    await user.click(
      within(dialog).getByRole("button", { name: "Remove router" }),
    )

    await waitFor(() =>
      expect(writes(requests)).toEqual([{ method: "DELETE", body: undefined }]),
    )
  })
})
