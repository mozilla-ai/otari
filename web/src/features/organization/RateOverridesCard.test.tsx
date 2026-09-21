import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { OrganizationContext, OrganizationPricingOverride } from "@/client"
import { RateOverridesCard } from "@/features/organization/RateOverridesCard"
import { API_ROOT } from "@/shared/api/client"
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
    unit: "tokens",
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
function catalogModel(id: string, deployment_managed = false) {
  return {
    id,
    object: "model",
    created: 0,
    owned_by: id.split(":")[0],
    pricing_source: "none",
    deployment_managed,
  }
}

function mockApi({
  context = organizationContext(),
  overrides = [] as OrganizationPricingOverride[],
  writeStatus = 201,
  writeBody = pricingOverride() as unknown,
  models = [] as string[],
  managedModels = [] as string[],
}: {
  context?: OrganizationContext
  overrides?: OrganizationPricingOverride[]
  writeStatus?: number
  writeBody?: unknown
  /** What GET /v1/models serves, which is where the model-key picker looks. */
  models?: string[]
  /** Catalog entries the deployment supplies the credential for. */
  managedModels?: string[]
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
    if (url.includes(`${API_ROOT}/organizations/me/pricing`)) {
      if (method === "GET") {
        // Honors the window, like the endpoint: a test that ignored it could
        // not tell a paged read from a read of everything.
        const params = new URL(url, "http://localhost").searchParams
        const skip = Number(params.get("skip") ?? 0)
        const limit = Number(params.get("limit") ?? 100)
        return jsonResponse({
          data: overrides.slice(skip, skip + limit),
          count: overrides.length,
        })
      }
      return jsonResponse(writeBody, writeStatus)
    }
    if (url.endsWith(`${API_ROOT}/models`)) {
      return jsonResponse({
        object: "list",
        data: [
          ...models.map((id) => catalogModel(id)),
          ...managedModels.map((id) => catalogModel(id, true)),
        ],
      })
    }
    return jsonResponse(context)
  })
  return requests
}

// The real router, per the frontend standards: the harness mounts what the app
// mounts, so a page that later grows a <Link> or URL state is already covered.
// Awaited, because the router resolves its first location asynchronously and a
// synchronous DOM read would race it.
function renderPage(url = "/organization/pricing") {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return renderWithRouter(
    <QueryClientProvider client={client}>
      <RateOverridesCard />
    </QueryClientProvider>,
    { url },
  )
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe("RateOverridesCard", () => {
  it("asks for a page rather than walking the overrides", async () => {
    // otari#1420: this table grows a row per model per period, so reading it
    // whole was a walk that lengthened for the life of the organization. The
    // endpoint already answers the tenancy envelope, so the window and the
    // total both come from the server.
    const requests = mockApi({
      overrides: Array.from({ length: 30 }, (_, index) =>
        pricingOverride({
          id: `11111111-1111-1111-1111-${String(index).padStart(12, "0")}`,
          model_key: `openai:model-${String(index).padStart(2, "0")}`,
        }),
      ),
    })

    await renderPage()

    const table = await screen.findByRole("grid", {
      name: /rate overrides/i,
    })
    // A header row and a page of 25, not all 30.
    await waitFor(() => {
      expect(within(table).getAllByRole("row")).toHaveLength(26)
    })
    expect(
      requests.some(
        (request) =>
          request.method === "GET" &&
          request.url.includes("/organizations/me/pricing?skip=0&limit=25"),
      ),
    ).toBe(true)
    // No second page was fetched to render the first.
    expect(
      requests.filter(
        (request) =>
          request.method === "GET" &&
          request.url.includes("/organizations/me/pricing?"),
      ),
    ).toHaveLength(1)
  })

  it("pages without reading the rest of the overrides", async () => {
    const user = userEvent.setup()
    const requests = mockApi({
      overrides: Array.from({ length: 30 }, (_, index) =>
        pricingOverride({
          id: `11111111-1111-1111-1111-${String(index).padStart(12, "0")}`,
          model_key: `openai:model-${String(index).padStart(2, "0")}`,
        }),
      ),
    })

    await renderPage()
    await screen.findByRole("grid", { name: /rate overrides/i })
    await user.click(
      screen.getByRole("button", { name: "Next page, rate overrides" }),
    )

    await waitFor(() => {
      expect(
        requests.some((request) =>
          request.url.includes("/organizations/me/pricing?skip=25&limit=25"),
        ),
      ).toBe(true)
    })
    const table = await screen.findByRole("grid", { name: /rate overrides/i })
    // The tail: five rows and the header.
    await waitFor(() => {
      expect(within(table).getAllByRole("row")).toHaveLength(6)
    })
  })

  it("steps back when a delete empties the page being shown", async () => {
    // The last row of the last page is the case: without the step-back the
    // table sits on a page the collection no longer reaches, showing nothing
    // while earlier pages still hold rows. The mock deletes for real, so the
    // refetch after the invalidation is what the component actually sees.
    const user = userEvent.setup()
    const live = Array.from({ length: 26 }, (_, index) =>
      pricingOverride({
        id: `11111111-1111-1111-1111-${String(index).padStart(12, "0")}`,
        model_key: `openai:model-${String(index).padStart(2, "0")}`,
      }),
    )
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input)
      const method = (init?.method ?? "GET").toUpperCase()
      if (url.includes(`${API_ROOT}/organizations/me/pricing`)) {
        if (method === "DELETE") {
          const id = url.split("/").pop() ?? ""
          live.splice(
            live.findIndex((row) => row.id === id),
            1,
          )
          return jsonResponse(null, 204)
        }
        const params = new URL(url, "http://localhost").searchParams
        const skip = Number(params.get("skip") ?? 0)
        const limit = Number(params.get("limit") ?? 100)
        return jsonResponse({
          data: live.slice(skip, skip + limit),
          count: live.length,
        })
      }
      if (url.endsWith(`${API_ROOT}/models`)) {
        return jsonResponse({ object: "list", data: [] })
      }
      return jsonResponse(organizationContext())
    })

    await renderPage()
    await screen.findByRole("grid", { name: /rate overrides/i })
    await user.click(
      screen.getByRole("button", { name: "Next page, rate overrides" }),
    )
    // Page two holds the twenty-sixth override on its own.
    expect(await screen.findByText("openai:model-25")).toBeInTheDocument()

    await user.click(await screen.findByRole("button", { name: /delete/i }))
    await user.click(
      await screen.findByRole("button", { name: /delete override/i }),
    )

    // Back on page one rather than stranded on an empty page two.
    expect(await screen.findByText("openai:model-00")).toBeInTheDocument()
  })

  it("lists the organization's overrides with their rates and period", async () => {
    mockApi({ overrides: [pricingOverride()] })

    await renderPage()

    expect(
      await screen.findByRole("heading", { name: /rate overrides/i }),
    ).toBeInTheDocument()
    expect(await screen.findByText("openai:gpt-4o")).toBeInTheDocument()
    expect(await screen.findByText("$2.50")).toBeInTheDocument()
    expect(await screen.findByText(/^From /)).toBeInTheDocument()
    expect(await screen.findByText("ACTIVE")).toBeInTheDocument()
  })

  it("shows an unset cache rate as absent rather than as zero", async () => {
    mockApi({ overrides: [pricingOverride()] })

    await renderPage()

    const row = (await screen.findByText("openai:gpt-4o")).closest(
      '[role="row"]',
    )
    expect(row).not.toBeNull()
    const cells = within(row as HTMLElement)

    // `formatCost` renders null as "$0.00", so the absent check has to sit in
    // front of it; an unset cache rate must read as "no rate stored", not as a
    // negotiated zero.
    expect(cells.queryByText("$0.00")).not.toBeInTheDocument()
    // Scoped to this row, and counted: page-scoped it would pass on the right
    // total while proving nothing about which cells are absent, since an
    // unrelated cell could supply or remove a match. Two is both cache columns.
    expect(cells.getAllByText("—")).toHaveLength(2)
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
      if (url.includes(`${API_ROOT}/organizations/me/pricing`)) {
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
      await screen.findByRole("combobox", { name: /model key/i }),
      "anthropic:claude-sonnet-5",
    )
    await user.type(screen.getByLabelText(/input, per 1m tokens/i), "3")
    await user.type(screen.getByLabelText(/output, per 1m tokens/i), "15")
    await user.click(screen.getByRole("button", { name: /^add override$/i }))

    await waitFor(() => {
      const write = requests.find(
        (request) =>
          request.method === "POST" &&
          request.url.includes(`${API_ROOT}/organizations/me/pricing`),
      )
      expect(write?.body).toMatchObject({
        model_key: "anthropic:claude-sonnet-5",
        input_price_per_million: 3,
        output_price_per_million: 15,
      })
    })
  })

  it("opens an add on the selector the catalog linked with", async () => {
    mockApi({ overrides: [] })

    await renderPage(
      "/organization/pricing?override=nebius%3Azai-org%2FGLM-5.3",
    )

    const dialog = await screen.findByRole("dialog")
    expect(
      within(dialog).getByRole("combobox", { name: /model key/i }),
    ).toHaveValue("nebius:zai-org/GLM-5.3")
  })

  it("does not open the linked editor for a member who cannot manage", async () => {
    mockApi({
      overrides: [],
      context: organizationContext({ role: "member" }),
    })

    await renderPage(
      "/organization/pricing?override=nebius%3Azai-org%2FGLM-5.3",
    )

    await screen.findByText(/no override yet/i)
    expect(screen.queryByRole("dialog")).toBeNull()
  })

  it("fills the model key from the catalog, so a rate is not stored under a typo", async () => {
    // The catalog rather than /v1/models/discoverable: this card answers to an
    // organization admin, who is refused the deployment-operator read.
    const requests = mockApi({
      overrides: [],
      models: ["anthropic:claude-sonnet-5"],
    })
    const user = userEvent.setup()

    await renderPage()

    await user.click(
      await screen.findByRole("button", { name: /add override/i }),
    )
    // The trigger, not the input: the field opens on typing, so that an
    // autofocused list does not hide the rest of the form from a screen reader.
    await user.click(
      await screen.findByRole("button", { name: /show suggestions/i }),
    )
    await user.click(
      await screen.findByRole("option", { name: "anthropic:claude-sonnet-5" }),
    )
    await user.type(screen.getByLabelText(/input, per 1m tokens/i), "3")
    await user.type(screen.getByLabelText(/output, per 1m tokens/i), "15")
    await user.click(screen.getByRole("button", { name: /^add override$/i }))

    await waitFor(() => {
      const write = requests.find(
        (request) =>
          request.method === "POST" &&
          request.url.includes(`${API_ROOT}/organizations/me/pricing`),
      )
      expect(write?.body).toMatchObject({
        model_key: "anthropic:claude-sonnet-5",
      })
    })
  })

  it("puts focus in the first rate when editing, not on the frame's Close", async () => {
    // The add path's first field is the model key, which carries `autoFocus`.
    // Editing replaces it with a read-only block, so without this the dialog
    // opened with focus on Close.
    mockApi({ overrides: [pricingOverride()] })
    const user = userEvent.setup()

    await renderPage()

    await user.click(await screen.findByRole("button", { name: /^edit$/i }))

    const dialog = await screen.findByRole("dialog")
    await waitFor(() =>
      expect(
        within(dialog).getByLabelText(/input, per 1m tokens/i),
      ).toHaveFocus(),
    )
  })

  it("does not greet the next open with the last attempt's refusal", async () => {
    // The create and replace mutations live inside the dialog, below the card's
    // key, so the remount that clears the draft clears the refusal with it.
    // Held in the card they outlived both: a blank form arrived under the
    // previous attempt's banner.
    mockApi({ overrides: [], writeStatus: 409 })
    const user = userEvent.setup()

    await renderPage()

    await user.click(
      await screen.findByRole("button", { name: /add override/i }),
    )
    await user.type(
      await screen.findByRole("combobox", { name: /model key/i }),
      "anthropic:claude-sonnet-5",
    )
    await user.type(screen.getByLabelText(/input, per 1m tokens/i), "3")
    await user.type(screen.getByLabelText(/output, per 1m tokens/i), "15")
    await user.click(screen.getByRole("button", { name: /^add override$/i }))

    expect(await screen.findByRole("alert")).toBeInTheDocument()

    // Out through the guard, then in again.
    await user.keyboard("{Escape}")
    await user.click(screen.getByRole("button", { name: "Discard" }))
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull())
    await user.click(screen.getByRole("button", { name: /add override/i }))

    const reopened = await screen.findByRole("dialog")
    expect(within(reopened).queryByRole("alert")).toBeNull()
    expect(
      within(reopened).getByRole("combobox", { name: /model key/i }),
    ).toHaveValue("")
  })

  it("refuses a model key with no provider prefix before sending it", async () => {
    const requests = mockApi({ overrides: [] })
    const user = userEvent.setup()

    await renderPage()

    await user.click(
      await screen.findByRole("button", { name: /add override/i }),
    )
    await user.type(
      await screen.findByRole("combobox", { name: /model key/i }),
      "gpt-4o",
    )
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
    await user.type(
      await screen.findByRole("combobox", { name: /model key/i }),
      "openai:gpt-4o",
    )
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
        `${API_ROOT}/organizations/me/pricing/11111111-1111-1111-1111-111111111111`,
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

  it("seeds each opener's dialog fresh, whichever one was used last", async () => {
    // Keying a dialog on an open counter is only right if every opener bumps
    // it. This card has two, Add and a row's Edit, and one that skipped the
    // bump would leave the previous open's rates in the fields. Rates are the
    // expensive case: an inherited figure is what a model is billed at.
    mockApi({ overrides: [pricingOverride()] })
    const user = userEvent.setup()

    await renderPage()

    await user.click(await screen.findByRole("button", { name: /edit/i }))
    const edited = await screen.findByLabelText(/input, per 1m tokens/i)
    expect(edited).not.toHaveValue("")
    await user.click(screen.getByRole("button", { name: /^cancel$/i }))

    await user.click(
      await screen.findByRole("button", { name: /add override/i }),
    )
    expect(await screen.findByLabelText(/input, per 1m tokens/i)).toHaveValue(
      "",
    )
    expect(
      await screen.findByRole("combobox", { name: /model key/i }),
    ).toHaveValue("")
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
              `${API_ROOT}/organizations/me/pricing/11111111-1111-1111-1111-111111111111`,
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
    await user.type(
      await screen.findByRole("combobox", { name: /model key/i }),
      "openai:gpt-4o",
    )
    await user.type(screen.getByLabelText(/input, per 1m tokens/i), "3")
    await user.type(screen.getByLabelText(/output, per 1m tokens/i), "15")
    await user.click(screen.getByRole("button", { name: /^add override$/i }))

    expect(
      await screen.findByText(/already covers part of that period/i),
    ).toBeInTheDocument()
  })

  // otari-ai#2095: the rates of a model the deployment supplies the credential
  // for are the deployment's, not a tenant's. The server refuses either way;
  // these cover the half that keeps the control from being offered.
  describe("a model the deployment supplies", () => {
    const tenant = organizationContext({ deployment_operator: false })

    it("blocks the save and says whose rate it is", async () => {
      mockApi({ context: tenant, managedModels: ["nebius_prod:llama-3"] })
      const user = userEvent.setup()

      await renderPage()

      await user.click(
        await screen.findByRole("button", { name: /add override/i }),
      )
      await user.type(
        await screen.findByRole("combobox", { name: /model key/i }),
        "nebius_prod:llama-3",
      )
      await user.type(screen.getByLabelText(/input, per 1m tokens/i), "0")
      await user.type(screen.getByLabelText(/output, per 1m tokens/i), "0")

      expect(
        await screen.findByText(/one of this deployment's own providers/i),
      ).toBeInTheDocument()
      expect(
        screen.getByRole("button", { name: /^add override$/i }),
      ).toBeDisabled()
    })

    it("leaves a model the organization supplies its own key for alone", async () => {
      mockApi({
        context: tenant,
        models: ["openai:gpt-4o"],
        managedModels: ["nebius_prod:llama-3"],
      })
      const user = userEvent.setup()

      await renderPage()

      await user.click(
        await screen.findByRole("button", { name: /add override/i }),
      )
      await user.type(
        await screen.findByRole("combobox", { name: /model key/i }),
        "openai:gpt-4o",
      )
      await user.type(screen.getByLabelText(/input, per 1m tokens/i), "3")
      await user.type(screen.getByLabelText(/output, per 1m tokens/i), "15")

      expect(
        screen.getByRole("button", { name: /^add override$/i }),
      ).toBeEnabled()
    })

    it("refuses to edit a row stored before the rule, and still allows deleting it", async () => {
      mockApi({
        context: tenant,
        overrides: [pricingOverride({ model_key: "nebius_prod:llama-3" })],
        managedModels: ["nebius_prod:llama-3"],
      })

      await renderPage()

      // Named by the reason rather than by the word "Edit": a disabled control
      // takes no focus, so its accessible name is where the refusal can be read.
      expect(
        await screen.findByRole("button", {
          name: /one of this deployment's own providers/i,
        }),
      ).toBeDisabled()
      expect(screen.getByRole("button", { name: /delete/i })).toBeEnabled()
    })

    // The dialog is rendered whether or not it is showing, so an ungated catalog
    // read inside it would fire for a reader and undo the card's own gate.
    it("asks for no catalog at all when the caller cannot edit", async () => {
      const requests = mockApi({
        context: organizationContext({ role: "member" }),
        overrides: [pricingOverride()],
      })

      await renderPage()
      expect(await screen.findByText("openai:gpt-4o")).toBeInTheDocument()

      expect(
        requests.filter((request) =>
          request.url.endsWith(`${API_ROOT}/models`),
        ),
      ).toEqual([])
    })

    it("leaves the deployment operator pricing its own models", async () => {
      mockApi({
        overrides: [pricingOverride({ model_key: "nebius_prod:llama-3" })],
        managedModels: ["nebius_prod:llama-3"],
      })

      await renderPage()

      expect(await screen.findByRole("button", { name: /edit/i })).toBeEnabled()
    })
  })
})
