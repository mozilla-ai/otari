import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { AgentRecommender, ConfiguredRecommender } from "@/client"
import { AgentRecommendersPage } from "@/features/agentRecommenders/AgentRecommendersPage"
import { API_ROOT } from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap } from "@/tests/fixtures"
import { renderWithRouter } from "@/tests/router"

interface Request {
  url: string
  method: string
  body: unknown
}

const CONFIGURED: ConfiguredRecommender = {
  selector: "typesafe:jev-latest",
  provider: "typesafe",
  model: "jev-latest",
}

function recommender(
  overrides: Partial<AgentRecommender> = {},
): AgentRecommender {
  return {
    id: "11111111-1111-4111-8111-111111111111",
    provider: "openrouter",
    model: "stored-decider",
    api_base: null,
    api_key_last4: "1234",
    has_api_key: true,
    input_price_per_million: 0.15,
    output_price_per_million: 0,
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
    rows?: AgentRecommender[]
    configured?: ConfiguredRecommender | null
  } = {},
) {
  const granted = opts.granted ?? true
  const rows = opts.rows ?? [recommender()]
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
      if (opts.listRefused) return jsonResponse({ detail: "Not found" }, 404)
      if (method === "GET") {
        return jsonResponse({
          data: rows,
          configured:
            opts.configured === undefined ? CONFIGURED : opts.configured,
        })
      }
      if (method === "DELETE") return new Response(null, { status: 204 })
      return jsonResponse(
        recommender({ ...(body as object) }),
        method === "POST" ? 201 : 200,
      )
    }
    return jsonResponse({ detail: `unmocked ${method} ${url}` }, 500)
  })
  return requests
}

function renderPage() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return renderWithRouter(
    <DeploymentProvider value={bootstrap()}>
      <QueryClientProvider client={client}>
        <AgentRecommendersPage />
      </QueryClientProvider>
    </DeploymentProvider>,
    { url: "/admin/recommender" },
  )
}

const routerRequests = (requests: Request[], method?: string) =>
  requests.filter(
    (request) =>
      request.url.includes("/routing/recommenders") &&
      (method === undefined || request.method === method),
  )

afterEach(() => {
  vi.restoreAllMocks()
})

describe("AgentRecommendersPage", () => {
  it("lists each router by model, provider, endpoint, key tail and price", async () => {
    mockApi({
      rows: [
        recommender(),
        recommender({
          id: "22222222-2222-4222-8222-222222222222",
          provider: "llamacpp",
          model: "local-decider",
          api_base: "http://decisions.internal:8080",
          has_api_key: false,
          api_key_last4: null,
          output_price_per_million: 2,
          enabled: false,
        }),
      ],
    })
    await renderPage()

    const table = await screen.findByRole("grid", { name: "Subagent routers" })
    const first = await within(table).findByRole("row", {
      name: /stored-decider/,
    })
    expect(within(first).getByText("OpenRouter")).toBeInTheDocument()
    expect(within(first).getByText("Provider default")).toBeInTheDocument()
    expect(within(first).getByText("••••1234")).toBeInTheDocument()
    expect(
      within(first).getByText("$0.15 in, $0 out per 1M tokens"),
    ).toBeInTheDocument()
    expect(
      within(first).getByRole("switch", { name: "Route with stored-decider" }),
    ).toBeChecked()

    const second = within(table).getByRole("row", {
      name: /local-decider/,
    })
    expect(within(second).getByText("llama.cpp server")).toBeInTheDocument()
    expect(within(second).getByText("No key stored")).toBeInTheDocument()
    expect(
      within(second).getByText("$0.15 in, $2 out per 1M tokens"),
    ).toBeInTheDocument()
    expect(
      within(second).getByRole("switch", { name: "Route with local-decider" }),
    ).not.toBeChecked()
  })

  it("refuses a caller who does not operate the deployment without asking for the list", async () => {
    const requests = mockApi({ granted: false })
    await renderPage()

    expect(
      await screen.findByText("Subagent router is not available to you"),
    ).toBeInTheDocument()
    expect(routerRequests(requests)).toEqual([])
    expect(
      screen.queryByRole("button", { name: "Add router" }),
    ).not.toBeInTheDocument()
  })

  it("refuses an operator the list itself 404s, without offering a write", async () => {
    mockApi({ listRefused: true })
    await renderPage()

    expect(
      await screen.findByText("Subagent router is not available to you"),
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Add router" }),
    ).not.toBeInTheDocument()
  })

  it("says which configured decision model answers while no router is added", async () => {
    mockApi({ rows: [] })
    await renderPage()

    expect(await screen.findByText("No router added")).toBeInTheDocument()
    expect(
      screen.getByText(
        /typesafe:jev-latest from config\.yml answers, unpriced/,
      ),
    ).toBeInTheDocument()
  })

  it("warns when every router is switched off", async () => {
    mockApi({ rows: [recommender({ enabled: false })] })
    await renderPage()

    expect(
      await screen.findByText(/No router is switched on/),
    ).toBeInTheDocument()
  })

  it("switches a router on through a patch of its own row", async () => {
    const user = userEvent.setup()
    const requests = mockApi({ rows: [recommender({ enabled: false })] })
    await renderPage()

    await user.click(
      await screen.findByRole("switch", { name: "Route with stored-decider" }),
    )

    await waitFor(() => {
      expect(routerRequests(requests, "PATCH")).toHaveLength(1)
    })
    const [patch] = routerRequests(requests, "PATCH")
    expect(patch.url).toContain(
      "/routing/recommenders/11111111-1111-4111-8111-111111111111",
    )
    expect(patch.body).toEqual({ enabled: true })
  })

  it("removes a router only once the removal is confirmed", async () => {
    const user = userEvent.setup()
    const requests = mockApi()
    await renderPage()

    await user.click(
      await screen.findByRole("button", { name: "Remove stored-decider" }),
    )
    const dialog = await screen.findByRole("alertdialog")
    expect(dialog).toHaveTextContent(
      "stored-decider and its stored key are deleted",
    )
    expect(routerRequests(requests, "DELETE")).toEqual([])

    await user.click(
      within(dialog).getByRole("button", { name: "Remove router" }),
    )

    await waitFor(() => {
      expect(routerRequests(requests, "DELETE")).toHaveLength(1)
    })
  })

  it("adds a router with the trimmed fields and no margin, seeded from the configured model", async () => {
    const user = userEvent.setup()
    const requests = mockApi({ rows: [] })
    await renderPage()

    await user.click(
      (await screen.findAllByRole("button", { name: "Add router" }))[0],
    )
    const dialog = await screen.findByRole("dialog", { name: "New router" })
    expect(within(dialog).getByRole("textbox", { name: "Model" })).toHaveValue(
      "jev-latest",
    )
    expect(within(dialog).queryByLabelText(/Margin/)).not.toBeInTheDocument()
    await user.type(
      within(dialog).getByLabelText("API key"),
      " ts-secret-wxyz ",
    )
    await user.type(
      within(dialog).getByRole("textbox", {
        name: "Input, per 1M tokens (USD)",
      }),
      "0.15",
    )
    await user.click(within(dialog).getByRole("button", { name: "Add router" }))

    await waitFor(() => {
      expect(routerRequests(requests, "POST")).toHaveLength(1)
    })
    expect(routerRequests(requests, "POST")[0].body).toEqual({
      provider: "typesafe",
      model: "jev-latest",
      api_key: "ts-secret-wxyz",
      input_price_per_million: 0.15,
      output_price_per_million: 0,
    })
  })

  it("holds a new router back until it names a price", async () => {
    const user = userEvent.setup()
    const requests = mockApi({ rows: [], configured: null })
    await renderPage()

    await user.click(
      (await screen.findAllByRole("button", { name: "Add router" }))[0],
    )
    const dialog = await screen.findByRole("dialog", { name: "New router" })
    await user.type(within(dialog).getByRole("textbox", { name: "Model" }), "m")
    await user.type(within(dialog).getByLabelText("API key"), "k")
    await user.click(within(dialog).getByRole("button", { name: "Add router" }))

    expect(
      await within(dialog).findByText(
        "Set an input or output token price above zero.",
      ),
    ).toBeInTheDocument()
    expect(routerRequests(requests, "POST")).toEqual([])
  })

  it("drops a typed key when the address it would be sent to changes", async () => {
    const user = userEvent.setup()
    mockApi({ rows: [] })
    await renderPage()

    await user.click(
      (await screen.findAllByRole("button", { name: "Add router" }))[0],
    )
    const dialog = await screen.findByRole("dialog", { name: "New router" })
    const key = within(dialog).getByLabelText("API key")
    await user.type(key, "ts-secret-wxyz")
    await user.type(
      within(dialog).getByRole("textbox", { name: "API base" }),
      "https://elsewhere.example",
    )

    expect(key).toHaveValue("")
  })

  it("edits a router, naming the kept key by its tail and leaving it out of the patch", async () => {
    const user = userEvent.setup()
    const requests = mockApi()
    await renderPage()

    await user.click(
      await screen.findByRole("button", { name: "Edit stored-decider" }),
    )
    const dialog = await screen.findByRole("dialog", { name: "Edit router" })
    expect(within(dialog).getByLabelText("API key")).toHaveAttribute(
      "placeholder",
      "Stored key ends in 1234. Leave blank to keep it.",
    )
    const output = within(dialog).getByRole("textbox", {
      name: "Output, per 1M tokens (USD)",
    })
    await user.clear(output)
    await user.type(output, "0.6")
    await user.click(within(dialog).getByRole("button", { name: "Save" }))

    await waitFor(() => {
      expect(routerRequests(requests, "PATCH")).toHaveLength(1)
    })
    expect(routerRequests(requests, "PATCH")[0].body).toEqual({
      provider: "openrouter",
      model: "stored-decider",
      api_base: "",
      input_price_per_million: 0.15,
      output_price_per_million: 0.6,
    })
  })
})
