import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { OrganizationContext, OrganizationPricingOverride } from "@/client"
import { RateOverridesCard } from "@/features/organization/RateOverridesCard"
import { organizationContext } from "@/tests/fixtures"
import { renderWithRouter } from "@/tests/router"

interface RecordedRequest {
  url: string
  method: string
  body: unknown
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(status === 204 ? null : JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

function pricingOverride(
  fields: Partial<OrganizationPricingOverride> = {},
): OrganizationPricingOverride {
  return {
    id: "11111111-1111-1111-1111-111111111111",
    organization_id: "org-1",
    model_key: "openai:gpt-4o",
    input_price_per_million: 2.5,
    output_price_per_million: 5,
    cache_read_price_per_million: null,
    cache_write_price_per_million: null,
    cache_write_1h_price_per_million: null,
    pricing_tiers: [],
    effective_from: "2026-08-01T00:00:00Z",
    effective_to: null,
    created_at: "2026-08-01T00:00:00Z",
    updated_at: "2026-08-01T00:00:00Z",
    ...fields,
  }
}

// Mocked at the `@/client` boundary (a real `fetch`), which is what the standards
// call for: the hooks and their invalidation are part of what is under test, so
// stubbing them would leave the interesting half uncovered.
function mockApi({
  context = organizationContext(),
  overrides = [] as OrganizationPricingOverride[],
  writeStatus = 201,
  writeBody = pricingOverride() as unknown,
}: {
  context?: OrganizationContext
  overrides?: OrganizationPricingOverride[]
  writeStatus?: number
  writeBody?: unknown
} = {}) {
  const requests: RecordedRequest[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
    requests.push({
      url,
      method,
      body: init?.body ? JSON.parse(String(init.body)) : undefined,
    })
    if (url.includes("/v1/organizations/me/pricing")) {
      if (method === "GET") {
        return jsonResponse({ data: overrides, count: overrides.length })
      }
      return jsonResponse(writeBody, writeStatus)
    }
    return jsonResponse(context)
  })
  return requests
}

// The real router, per the frontend standards: the harness mounts what the app
// mounts, so a page that later grows a <Link> or URL state is already covered.
// Awaited, because the router resolves its first location asynchronously and a
// synchronous DOM read would race it.
function renderPage() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return renderWithRouter(
    <QueryClientProvider client={client}>
      <RateOverridesCard />
    </QueryClientProvider>,
  )
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe("RateOverridesCard", () => {
  it("lists the organization's overrides with their rates and period", async () => {
    mockApi({ overrides: [pricingOverride()] })

    await renderPage()

    expect(
      await screen.findByRole("heading", { name: /rate overrides/i }),
    ).toBeInTheDocument()
    expect(await screen.findByText("openai:gpt-4o")).toBeInTheDocument()
    expect(await screen.findByText("$2.50")).toBeInTheDocument()
    expect(await screen.findByText(/^From /)).toBeInTheDocument()
    expect(await screen.findByText("Active")).toBeInTheDocument()
  })

  it("shows an unset cache rate as absent rather than as zero", async () => {
    mockApi({ overrides: [pricingOverride()] })

    await renderPage()

    await screen.findByText("openai:gpt-4o")
    // `formatCost` renders null as "$0.00", so the absent check has to sit in
    // front of it; an unset cache rate must read as "no rate stored", not as a
    // negotiated zero.
    expect(screen.queryByText("$0.00")).not.toBeInTheDocument()
    expect(await screen.findAllByText("—")).not.toHaveLength(0)
  })

  it("explains itself when the organization has no overrides", async () => {
    mockApi({ overrides: [] })

    await renderPage()

    expect(await screen.findByText(/no override yet/i)).toBeInTheDocument()
  })

  // A failed list also leaves `rows` empty, and the empty state asserts that
  // every model is priced by the deployment list, which the page cannot know
  // when the list never arrived.
  it("shows the error rather than the empty state when the list fails", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input)
      if (url.includes("/v1/organizations/me/pricing")) {
        return jsonResponse({ detail: "pricing is unavailable" }, 503)
      }
      return jsonResponse(organizationContext())
    })

    await renderPage()

    expect(
      await screen.findByText(/pricing is unavailable/i),
    ).toBeInTheDocument()
    // The empty row says nothing about the catalog, so it cannot assert a fact
    // the request never returned.
    expect(
      screen.queryByText(/priced by the deployment price list/i),
    ).not.toBeInTheDocument()
  })

  it("sends a create when an override is added", async () => {
    const requests = mockApi({ overrides: [] })
    const user = userEvent.setup()

    await renderPage()

    await user.click(
      await screen.findByRole("button", { name: /add override/i }),
    )
    await user.type(
      await screen.findByLabelText(/model key/i),
      "anthropic:claude-sonnet-5",
    )
    await user.type(screen.getByLabelText(/input, per 1m tokens/i), "3")
    await user.type(screen.getByLabelText(/output, per 1m tokens/i), "15")
    await user.click(screen.getByRole("button", { name: /^add override$/i }))

    await waitFor(() => {
      const write = requests.find(
        (request) =>
          request.method === "POST" &&
          request.url.includes("/v1/organizations/me/pricing"),
      )
      expect(write?.body).toMatchObject({
        model_key: "anthropic:claude-sonnet-5",
        input_price_per_million: 3,
        output_price_per_million: 15,
      })
    })
  })

  it("refuses a model key with no provider prefix before sending it", async () => {
    const requests = mockApi({ overrides: [] })
    const user = userEvent.setup()

    await renderPage()

    await user.click(
      await screen.findByRole("button", { name: /add override/i }),
    )
    await user.type(await screen.findByLabelText(/model key/i), "gpt-4o")
    await user.type(screen.getByLabelText(/input, per 1m tokens/i), "3")
    await user.type(screen.getByLabelText(/output, per 1m tokens/i), "15")

    expect(screen.getByText(/needs the provider prefix/i)).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: /^add override$/i }),
    ).toBeDisabled()
    expect(requests.some((request) => request.method === "POST")).toBe(false)
  })

  // The server refuses this with a 409. Saying so before the request is sent
  // means the operator sees which period they collided with, not just a failure.
  it("refuses a period that overlaps a stored override before sending it", async () => {
    mockApi({
      overrides: [
        pricingOverride({
          effective_from: "2020-01-01T00:00:00Z",
          effective_to: null,
        }),
      ],
    })
    const user = userEvent.setup()

    await renderPage()

    await user.click(
      await screen.findByRole("button", { name: /add override/i }),
    )
    await user.type(await screen.findByLabelText(/model key/i), "openai:gpt-4o")
    await user.type(screen.getByLabelText(/input, per 1m tokens/i), "3")
    await user.type(screen.getByLabelText(/output, per 1m tokens/i), "15")

    expect(
      screen.getByText(/overlaps an override already stored/i),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: /^add override$/i }),
    ).toBeDisabled()
  })

  it("sends a replace without the model key when an override is edited", async () => {
    const requests = mockApi({
      overrides: [pricingOverride()],
      writeStatus: 200,
    })
    const user = userEvent.setup()

    await renderPage()

    await user.click(await screen.findByRole("button", { name: /edit/i }))
    const input = await screen.findByLabelText(/input, per 1m tokens/i)
    await user.clear(input)
    await user.type(input, "1.25")
    await user.click(screen.getByRole("button", { name: /save override/i }))

    await waitFor(() => {
      const write = requests.find((request) => request.method === "PUT")
      expect(write?.url).toContain(
        "/v1/organizations/me/pricing/11111111-1111-1111-1111-111111111111",
      )
      expect(write?.body).toMatchObject({ input_price_per_million: 1.25 })
      // Immutable on this endpoint, so the body must not carry it.
      expect(write?.body).not.toHaveProperty("model_key")
    })
  })

  // The endpoint requires a start on a replacement, so a cleared field would be
  // a 422 rather than a silent period move. Blocked here instead.
  it("blocks an edit whose start has been cleared", async () => {
    const requests = mockApi({
      overrides: [pricingOverride()],
      writeStatus: 200,
    })
    const user = userEvent.setup()

    await renderPage()

    await user.click(await screen.findByRole("button", { name: /edit/i }))
    await user.clear(await screen.findByLabelText(/applies from/i))

    expect(screen.getByText(/an edit needs a start/i)).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: /save override/i }),
    ).toBeDisabled()
    expect(requests.some((request) => request.method === "PUT")).toBe(false)
  })

  it("deletes an override after a confirmation", async () => {
    const requests = mockApi({
      overrides: [pricingOverride()],
      writeStatus: 204,
      writeBody: null,
    })
    const user = userEvent.setup()

    await renderPage()

    await user.click(await screen.findByRole("button", { name: /delete/i }))
    await user.click(
      await screen.findByRole("button", { name: /delete override/i }),
    )

    await waitFor(() => {
      expect(
        requests.some(
          (request) =>
            request.method === "DELETE" &&
            request.url.includes(
              "/v1/organizations/me/pricing/11111111-1111-1111-1111-111111111111",
            ),
        ),
      ).toBe(true)
    })
  })

  it("disables every write control for a member who cannot manage", async () => {
    mockApi({
      context: organizationContext({ role: "viewer" }),
      overrides: [pricingOverride()],
    })

    await renderPage()

    expect(
      await screen.findByRole("button", { name: /add override/i }),
    ).toBeDisabled()
    expect(await screen.findByRole("button", { name: /edit/i })).toBeDisabled()
    expect(
      await screen.findByRole("button", { name: /delete/i }),
    ).toBeDisabled()
    // And it says why, rather than only refusing.
    expect(screen.getByText(/only owners and admins/i)).toBeInTheDocument()
  })

  it("surfaces a server refusal", async () => {
    mockApi({
      overrides: [],
      writeStatus: 409,
      writeBody: {
        detail:
          "An override for 'openai:gpt-4o' already covers part of that period",
      },
    })
    const user = userEvent.setup()

    await renderPage()

    await user.click(
      await screen.findByRole("button", { name: /add override/i }),
    )
    await user.type(await screen.findByLabelText(/model key/i), "openai:gpt-4o")
    await user.type(screen.getByLabelText(/input, per 1m tokens/i), "3")
    await user.type(screen.getByLabelText(/output, per 1m tokens/i), "15")
    await user.click(screen.getByRole("button", { name: /^add override$/i }))

    expect(
      await screen.findByText(/already covers part of that period/i),
    ).toBeInTheDocument()
  })
})
