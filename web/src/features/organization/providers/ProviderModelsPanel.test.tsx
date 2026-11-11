import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { OrgProviderModel } from "@/client"
import type { KindFilter } from "@/features/models/offeredKinds"
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

function typeCounts(models: OrgProviderModel[]): Record<string, number> {
  const counts: Record<string, number> = {}
  for (const model of models) {
    const kind = model.type ?? "unknown"
    counts[kind] = (counts[kind] ?? 0) + 1
  }
  return counts
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
      // The kind asked for, as the gateway filters it; every kind is counted.
      const kind = new URL(url, "http://otari.test").searchParams.get("type")
      const shown = kind
        ? models.filter((model) => (model.type ?? "unknown") === kind)
        : models
      return opts.listFails
        ? jsonResponse({ detail: "Tenancy is unavailable" }, 500)
        : jsonResponse({
            count: shown.length,
            data: shown,
            type_counts: typeCounts(models),
          })
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
    kind?: KindFilter
    onKindChange?: (kind: KindFilter) => void
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
        kind={pager.kind ?? ""}
        onKindChange={pager.onKindChange ?? vi.fn()}
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

  it("names the provider a default was copied from when it is not the key's own", async () => {
    mockApi({
      models: [
        orgProviderModel({ id: "a", model: "own-listing" }),
        orgProviderModel({
          id: "b",
          model: "resold",
          default_from: "cortecs",
        }),
      ],
    })
    await renderPanel()

    expect(await screen.findByText("Default from cortecs")).toBeInTheDocument()
    expect(screen.getByText("Default")).toBeInTheDocument()
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

  it("names the unit of a rate that is not per token", async () => {
    // Stored per million like every rate, so a $2-per-1,000-searches price is
    // 2000 on the wire: shown bare under "Input / 1M" it would read as a token
    // price a thousand times too high.
    mockApi({
      models: [
        orgProviderModel({
          model: "rerank-v3.5",
          input_price_per_million: 2000,
          output_price_per_million: 0,
          price_source: "organization",
          unit: "request",
        }),
      ],
    })
    await renderPanel()

    const table = await screen.findByRole("grid", {
      name: "Models on Production",
    })
    expect(
      within(table).getByText(/Request: \$0\.002\/request/).parentElement,
    ).toHaveTextContent("$0.002")
    expect(within(table).queryByText("$2,000.00")).toBeNull()
  })

  it("offers one tab per kind of model, each with its count", async () => {
    mockApi({
      models: [
        orgProviderModel(),
        orgProviderModel({
          id: "v",
          model: "sora-2",
          type: "video",
          unit: "second",
          input_price_per_million: 100_000,
          output_price_per_million: null,
        }),
        orgProviderModel({ id: "u", model: "mystery", type: null }),
      ],
    })
    const onKindChange = vi.fn()
    await renderPanel(true, vi.fn(), { onKindChange })

    await screen.findByRole("grid", { name: "Models on Production" })
    expect(screen.getByRole("button", { name: "All 3" })).toHaveAttribute(
      "aria-pressed",
      "true",
    )
    for (const name of ["Text 1", "Video 1", "Unknown 1"]) {
      expect(screen.getByRole("button", { name })).toHaveAttribute(
        "aria-pressed",
        "false",
      )
    }

    await userEvent.click(screen.getByRole("button", { name: "Video 1" }))
    expect(onKindChange).toHaveBeenCalledWith("video")
  })

  it("asks the gateway for the kind shown", async () => {
    const requests = mockApi()
    await renderPanel(true, vi.fn(), { kind: "video" })

    await screen.findByRole("grid", { name: "Models on Production" })
    expect(requests.some((request) => request.url.includes("type=video"))).toBe(
      true,
    )
  })

  it("keeps one set of price columns and puts a video price under Output", async () => {
    // Every tab has the same columns. A token price is bare, per million
    // tokens; a video price is billed on what the model makes, so it sits
    // under Output and names its unit when hovered.
    mockApi({
      models: [
        orgProviderModel(),
        orgProviderModel({
          id: "v",
          model: "sora-2",
          type: "video",
          unit: "second",
          input_price_per_million: 100_000,
          output_price_per_million: null,
        }),
      ],
    })
    await renderPanel(true, vi.fn())

    const table = await screen.findByRole("grid", {
      name: "Models on Production",
    })
    const headers = within(table)
      .getAllByRole("columnheader")
      .map((header) => header.textContent)
    expect(headers).toEqual(
      expect.arrayContaining(["Input", "Output", "Cache read", "Cache write"]),
    )
    expect(within(table).getByText("$2.50")).toBeInTheDocument()
    expect(within(table).getByText("$10.00")).toBeInTheDocument()

    const video = within(table).getByText(/Video output: \$0\.10\/second/)
    expect(video.parentElement).toHaveTextContent("$0.10")
    const cells = within(
      within(table).getByRole("row", { name: /sora-2/ }),
    ).getAllByRole("gridcell")
    const output = cells.find((cell) => cell.contains(video))
    // The model's name is the row header, so Input is the first cell.
    expect(cells.indexOf(output as HTMLElement)).toBe(1)
  })

  it("lists only the kind its tab names", async () => {
    mockApi({
      models: [
        orgProviderModel(),
        orgProviderModel({ id: "v", model: "sora-2", type: "video" }),
      ],
    })
    await renderPanel(true, vi.fn(), { kind: "video" })

    const table = await screen.findByRole("grid", {
      name: "Models on Production",
    })
    expect(
      await within(table).findByRole("row", { name: /sora-2/ }),
    ).toBeInTheDocument()
    expect(within(table).queryByRole("row", { name: /gpt-4o/ })).toBeNull()
  })

  it("shows an empty table, not an empty catalog, under a kind with no models left", async () => {
    mockApi({ models: [] })
    await renderPanel(true, vi.fn(), { kind: "image" })

    expect(
      await screen.findByRole("button", { name: "Image 0" }),
    ).toBeInTheDocument()
    expect(screen.queryByText("No models offered yet")).toBeNull()
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
