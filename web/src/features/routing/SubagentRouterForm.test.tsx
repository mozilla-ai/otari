/**
 * The router form with `overlayRecommenderMargin` replaced, as an edition's
 * build replaces it: a field after the output price, a check beside the
 * form's own, a save that lands between the router's save and its switch-on,
 * and a summary the table appends to the price.
 */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type { AgentRecommender, DecisionProviderOption } from "@/client"
import { SubagentRouterPage } from "@/features/routing/SubagentRouterPage"
import { API_ROOT } from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap } from "@/tests/fixtures"
import { renderWithRouter } from "@/tests/router"

const edition = vi.hoisted(() => ({
  events: [] as string[],
  save: vi.fn<(id: string, value: string) => Promise<void>>(),
}))

vi.mock("@/features/routing/overlayRecommenderMargin", () => ({
  RecommenderMarginField: ({
    value,
    onChange,
    error,
  }: {
    value: string
    onChange: (value: string) => void
    error: string
  }) => (
    <label>
      Margin (%)
      <input value={value} onChange={(event) => onChange(event.target.value)} />
      {error ? <span>{error}</span> : null}
    </label>
  ),
  useRecommenderMargin: () => ({
    initialValue: () => "20",
    validate: (value: string) =>
      /^\d+$/.test(value) ? {} : { margin: "Enter a whole percent." },
    save: edition.save,
    summary: () => ", plus 20%",
  }),
}))

const PROVIDERS: DecisionProviderOption[] = [
  { provider: "typesafe", self_hosted: false },
]

function router(overrides: Partial<AgentRecommender> = {}): AgentRecommender {
  return {
    id: "11111111-1111-4111-8111-111111111111",
    provider: "typesafe",
    model: "jev-latest",
    api_base: null,
    api_key_last4: "9876",
    has_api_key: true,
    input_price_per_million: 0.15,
    output_price_per_million: 0,
    enabled: false,
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

function mockApi(routers: AgentRecommender[]) {
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
    const body = init?.body ? JSON.parse(String(init.body)) : undefined
    if (url.includes(`${API_ROOT}/admin/access`)) {
      return jsonResponse({ granted: true })
    }
    if (url.includes(`${API_ROOT}/routing/recommenders`)) {
      if (method === "GET") {
        return jsonResponse({ data: routers, decision_providers: PROVIDERS })
      }
      edition.events.push(`${method} ${JSON.stringify(body)}`)
      const id = method === "POST" ? "new-router" : (url.split("/").pop() ?? "")
      return jsonResponse(
        router({ id, ...body }),
        method === "POST" ? 201 : 200,
      )
    }
    return jsonResponse({})
  })
}

async function renderPage() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return renderWithRouter(
    <DeploymentProvider value={bootstrap()}>
      <QueryClientProvider client={client}>
        <SubagentRouterPage />
      </QueryClientProvider>
    </DeploymentProvider>,
    { url: "/admin/subagent-router" },
  )
}

async function addFirstRouter(margin: string) {
  const user = userEvent.setup()
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
  await user.type(within(dialog).getByLabelText("Margin (%)"), margin)
  await user.click(within(dialog).getByRole("button", { name: "Add router" }))
  return dialog
}

beforeEach(() => {
  edition.events.length = 0
  edition.save.mockReset()
  edition.save.mockImplementation(async (id, value) => {
    edition.events.push(`save ${id} ${value}`)
  })
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe("SubagentRouterForm with an edition's margin", () => {
  it("saves the margin after the router lands off, and only then switches it on", async () => {
    mockApi([])
    await renderPage()

    await addFirstRouter("25")

    await waitFor(() => expect(edition.events).toHaveLength(3))
    expect(edition.events[0]).toMatch(/^POST .*"enabled":false/)
    expect(edition.events[1]).toBe("save new-router 25")
    expect(edition.events[2]).toBe('PATCH {"enabled":true}')
  })

  it("leaves the router off and shows the error when the margin cannot be saved", async () => {
    mockApi([])
    edition.save.mockRejectedValue(new Error("The margin could not be saved"))
    await renderPage()

    const dialog = await addFirstRouter("25")

    expect(
      await within(dialog).findByText(/The margin could not be saved/),
    ).toBeInTheDocument()
    expect(edition.events).toHaveLength(1)
    expect(edition.events[0]).toMatch(/^POST /)
  })

  it("checks the margin beside the router's own fields, before anything is sent", async () => {
    mockApi([])
    await renderPage()

    const dialog = await addFirstRouter("lots")

    expect(
      await within(dialog).findByText("Enter a whole percent."),
    ).toBeInTheDocument()
    expect(edition.events).toEqual([])
  })

  it("starts an edit from the stored margin and shows it beside the price", async () => {
    mockApi([router()])
    const user = userEvent.setup()
    await renderPage()

    expect(
      await screen.findByText("$0.15 in, $0.00 out per 1M tokens, plus 20%"),
    ).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Edit jev-latest" }))
    const dialog = await screen.findByRole("dialog")
    expect(within(dialog).getByLabelText("Margin (%)")).toHaveValue("20")
  })
})
