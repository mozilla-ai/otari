import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { OrgProviderModel } from "@/client"
import { orgProviderKey, orgProviderModel } from "@/tests/fixtures"
import { renderWithRouter } from "@/tests/router"

import { ProviderModelsPanel } from "./ProviderModelsPanel"

const KEY = orgProviderKey({ name: "Production", provider: "openai" })

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

interface MockOpts {
  models?: OrgProviderModel[]
  /** What a refresh answers, which is where a provider's own refusal arrives. */
  refresh?: Record<string, unknown>
  listFails?: boolean
}

function mockApi(opts: MockOpts = {}) {
  const models = opts.models ?? [orgProviderModel()]
  const requests: { url: string; method: string }[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
    requests.push({ url, method })

    if (url.includes("/organizations/me/pricing/")) {
      return new Response(null, { status: 204 })
    }
    if (url.includes("/models/refresh") || url.includes("/pricing/refresh")) {
      return jsonResponse(
        opts.refresh ?? { added: [], repriced: [], count: models.length },
      )
    }
    if (url.includes("/available-models")) {
      return jsonResponse({ provider: "openai", models: ["gpt-4o-mini"] })
    }
    if (url.includes("/models") && method === "GET") {
      return opts.listFails
        ? jsonResponse({ detail: "Tenancy is unavailable" }, 500)
        : jsonResponse({ count: models.length, data: models })
    }
    // Every write answers with a row the panel re-reads through the invalidated
    // list, so one is enough for all of them.
    return jsonResponse(models[0] ?? orgProviderModel())
  })
  return requests
}

// Pagination is the page's, kept in the URL, so the panel takes it as props.
// Held here so a test that pages can read back what the panel asked for.
function renderPanel(
  canEdit = true,
  onEditRate = vi.fn(),
  pager: {
    page?: number
    pageSize?: number
    onPageChange?: (page: number) => void
    onPageSizeChange?: (size: number) => void
  } = {},
) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return renderWithRouter(
    <QueryClientProvider client={client}>
      <ProviderModelsPanel
        providerKey={KEY}
        canEdit={canEdit}
        onEditRate={onEditRate}
        page={pager.page ?? 0}
        pageSize={pager.pageSize ?? 25}
        onPageChange={pager.onPageChange ?? vi.fn()}
        onPageSizeChange={pager.onPageSizeChange ?? vi.fn()}
      />
    </QueryClientProvider>,
  )
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe("ProviderModelsPanel", () => {
  it("lists a model with its rate and where the rate came from", async () => {
    mockApi()
    await renderPanel()

    const table = await screen.findByRole("grid", {
      name: "Models on Production",
    })
    const row = within(table).getAllByRole("row")[1] as HTMLElement
    expect(within(row).getByText("gpt-4o")).toBeInTheDocument()
    expect(within(row).getByText("Default")).toBeInTheDocument()
  })

  it("names each rung of the ladder rather than calling them all the same", async () => {
    // The badge is the answer to "who decided this number", and the three rungs
    // are answered by three different people: the organization, the deployment,
    // and nobody.
    mockApi({
      models: [
        orgProviderModel({
          id: "a",
          model: "own",
          price_source: "organization",
        }),
        orgProviderModel({
          id: "b",
          model: "theirs",
          price_source: "deployment",
        }),
        orgProviderModel({
          id: "c",
          model: "nobody",
          price_source: null,
          input_price_per_million: null,
          output_price_per_million: null,
          enabled: false,
        }),
      ],
    })
    await renderPanel()

    expect(await screen.findByText("Your rate")).toBeInTheDocument()
    expect(screen.getByText("Deployment")).toBeInTheDocument()
    expect(screen.getByText("Unpriced")).toBeInTheDocument()
  })

  it("shows a dash for an absent rate, never a zero", async () => {
    // Zero is a real price: a model served for nothing costs nothing and spends
    // no budget, which is not the same as a rate nobody has set.
    mockApi({
      models: [
        orgProviderModel({
          input_price_per_million: null,
          output_price_per_million: null,
          price_source: null,
        }),
      ],
    })
    await renderPanel()

    const table = await screen.findByRole("grid", {
      name: "Models on Production",
    })
    expect(within(table).queryByText("$0")).toBeNull()
    expect(within(table).getAllByText("—").length).toBeGreaterThan(0)
  })

  it("withholds a model with its serving switch", async () => {
    const requests = mockApi()
    const user = userEvent.setup()
    await renderPanel()

    await user.click(
      await screen.findByRole("switch", { name: "Serve gpt-4o" }),
    )

    await waitFor(() => {
      expect(
        requests.some(
          (request) =>
            request.method === "PATCH" && request.url.includes("/models/"),
        ),
      ).toBe(true)
    })
  })

  it("will not offer to serve a model nothing prices", async () => {
    // The server refuses it too, for the reason the offer rule exists: a model
    // with no rate would be billed at nothing. The control says so rather than
    // taking a press and answering with a banner.
    mockApi({
      models: [
        orgProviderModel({
          model: "gpt-6-unreleased",
          price_source: null,
          input_price_per_million: null,
          output_price_per_million: null,
          enabled: false,
        }),
      ],
    })
    await renderPanel()

    expect(
      await screen.findByRole("switch", {
        name: "gpt-6-unreleased has no rate yet, so it cannot be served",
      }),
    ).toBeDisabled()
  })

  it("still lets an unpriced model be switched off", async () => {
    // One-directional, matching the server: a row that reached the served state
    // some other way has to stay withdrawable.
    mockApi({
      models: [
        orgProviderModel({
          model: "gpt-6-unreleased",
          price_source: null,
          input_price_per_million: null,
          output_price_per_million: null,
          enabled: true,
        }),
      ],
    })
    await renderPanel()

    expect(
      await screen.findByRole("switch", { name: "Serve gpt-6-unreleased" }),
    ).toBeEnabled()
  })

  it("says what each refresh did, per button", async () => {
    // Silence after a press reads as a button that did nothing, and the two
    // buttons asked different questions, so "nothing changed" has two wordings.
    mockApi({ refresh: { added: [], repriced: [], count: 1 } })
    const user = userEvent.setup()
    await renderPanel()

    await user.click(
      await screen.findByRole("button", { name: /Refresh models/ }),
    )
    // Queried by its sentence, not by the role: `TablePagination` is a status
    // region too, so the role alone matches two things here.
    expect(
      await screen.findByText("The provider lists nothing new."),
    ).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: /Refresh pricing/ }))
    expect(
      await screen.findByText("Every default rate is already current."),
    ).toBeInTheDocument()
  })

  it("reports a provider's own refusal beside a list that is still standing", async () => {
    mockApi({
      refresh: {
        added: [],
        repriced: [],
        count: 1,
        error: "the upstream refused the credential",
      },
    })
    const user = userEvent.setup()
    await renderPanel()

    await user.click(
      await screen.findByRole("button", { name: /Refresh models/ }),
    )

    expect(
      await screen.findByText("the upstream refused the credential"),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("grid", { name: "Models on Production" }),
    ).toBeInTheDocument()
  })

  it("reports a list that could not be read rather than an empty one", async () => {
    // Telling an admin their catalog is empty when the read failed is the one
    // answer worse than saying nothing.
    mockApi({ listFails: true })
    await renderPanel()

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Tenancy is unavailable",
    )
    expect(screen.queryByText("No models offered yet")).toBeNull()
  })

  it("offers to refresh when nothing is offered yet", async () => {
    mockApi({ models: [] })
    await renderPanel()

    expect(await screen.findByText("No models offered yet")).toBeInTheDocument()
  })

  it("withholds every write from a caller who cannot manage the organization", async () => {
    mockApi()
    await renderPanel(false)

    await screen.findByRole("grid", { name: "Models on Production" })
    expect(screen.queryByRole("button", { name: /Refresh models/ })).toBeNull()
    expect(screen.queryByRole("button", { name: /Add model/ })).toBeNull()
    expect(screen.getByRole("switch", { name: "Serve gpt-4o" })).toBeDisabled()
  })

  it("hands the rate editor up to the page, which owns the dialog", async () => {
    // Opened from a row and rendered above the table: a dialog mounted inside a
    // table cell closes with the row that opened it.
    const onEditRate = vi.fn()
    mockApi()
    const user = userEvent.setup()
    await renderPanel(true, onEditRate)

    await user.click(
      await screen.findByRole("button", { name: "Set your rate for gpt-4o" }),
    )

    expect(onEditRate).toHaveBeenCalledWith(
      expect.objectContaining({ model: "gpt-4o" }),
    )
  })

  it("clears a rate the organization set, and offers that only where there is one", async () => {
    // Without it an admin could set a rate and never go back to the default,
    // which is the state a seeded row is in and the one a refresh keeps current.
    const requests = mockApi({
      models: [
        orgProviderModel({
          id: "own",
          model: "priced-by-us",
          price_source: "organization",
          pricing_id: "99999999-9999-9999-9999-999999999999",
        }),
        orgProviderModel({
          id: "seeded",
          model: "priced-by-default",
          price_source: "defaults",
          pricing_id: null,
        }),
      ],
    })
    const user = userEvent.setup()
    await renderPanel()

    await screen.findByRole("grid", { name: "Models on Production" })
    expect(
      screen.queryByRole("button", {
        name: "Use the default rate for priced-by-default",
      }),
    ).toBeNull()

    await user.click(
      screen.getByRole("button", {
        name: "Use the default rate for priced-by-us",
      }),
    )
    await user.click(screen.getByRole("button", { name: "Use default" }))

    await waitFor(() => {
      expect(
        requests.some(
          (request) =>
            request.method === "DELETE" &&
            request.url.includes(
              "/organizations/me/pricing/99999999-9999-9999-9999-999999999999",
            ),
        ),
      ).toBe(true)
    })
  })

  it("confirms before it stops offering a model, and says the rate survives", async () => {
    mockApi()
    const user = userEvent.setup()
    await renderPanel()

    await user.click(
      await screen.findByRole("button", { name: "Stop offering gpt-4o" }),
    )

    // `alertdialog`, which is what `ConfirmDialog` renders: it interrupts to ask
    // rather than opening a form.
    const dialog = await screen.findByRole("alertdialog")
    expect(dialog).toHaveTextContent(/rate and its history stay/)
    expect(dialog).toHaveTextContent(/Serving switch/)
  })

  it("asks the provider what it serves only once the add form is open", async () => {
    // Answering dials the upstream, which can take the whole discovery timeout.
    const requests = mockApi()
    const user = userEvent.setup()
    await renderPanel()

    await screen.findByRole("grid", { name: "Models on Production" })
    expect(
      requests.some((request) => request.url.includes("/available-models")),
    ).toBe(false)

    await user.click(screen.getByRole("button", { name: /Add model/ }))

    // The available-models route specifically. The panel's own list read is
    // under the same prefix, so a predicate on the prefix alone passes whether
    // or not opening the form ever dials.
    await waitFor(() => {
      expect(
        requests.some((request) => request.url.includes("/available-models")),
      ).toBe(true)
    })
  })
})
