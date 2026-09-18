import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { ActivityPage } from "@/features/activity/ActivityPage"
import { API_ROOT } from "@/shared/api/client"
import { entry, mockApi, renderPage } from "@/tests/activity"

afterEach(() => {
  vi.restoreAllMocks()
})

describe("ActivityPage bulk editing", () => {
  it("only lets imported rows be selected", async () => {
    mockApi({
      rows: [
        entry({ id: "gw", model: "gateway-model", counts_toward_budget: true }),
        entry({
          id: "imp",
          model: "imported-model",
          source: "claude_code",
          counts_toward_budget: false,
        }),
        entry({
          id: "gw-exempt",
          model: "exempt-model",
          source: "gateway",
          counts_toward_budget: false,
        }),
      ],
    })
    renderPage(<ActivityPage />)

    const gatewayRow = (await screen.findByText("gateway-model")).closest("tr")!
    const importedRow = screen.getByText("imported-model").closest("tr")!
    const exemptRow = screen.getByText("exempt-model").closest("tr")!
    expect(within(gatewayRow).getByRole("checkbox")).toBeDisabled()
    expect(within(importedRow).getByRole("checkbox")).toBeEnabled()
    // Traffic this gateway served on an exclude_from_budget key. Budget-exempt like
    // an import, so selecting on `counts_toward_budget` alone offered it, and the
    // delete then refused it and reported a smaller number than the dialog promised.
    expect(within(exemptRow).getByRole("checkbox")).toBeDisabled()
  })

  it("deletes the selected imported rows by id", async () => {
    const user = userEvent.setup()
    const { calls } = mockApi({
      rows: [
        entry({
          id: "imp-1",
          model: "imported-model",
          source: "claude_code",
          counts_toward_budget: false,
        }),
      ],
      total: 1,
    })
    renderPage(<ActivityPage />)

    const row = (await screen.findByText("imported-model")).closest("tr")!
    await user.click(within(row).getByRole("checkbox"))

    // Bulk bar appears with the page selection count.
    expect(await screen.findByText("1 selected")).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Delete" }))

    // Confirm in the dialog.
    const dialog = await screen.findByRole("alertdialog")
    await user.click(within(dialog).getByRole("button", { name: "Delete" }))

    await waitFor(() => {
      const del = calls.find(
        (c) => c.url.endsWith(`${API_ROOT}/usage`) && c.method === "DELETE",
      )
      expect(del).toBeTruthy()
      expect(del!.body).toContain("imp-1")
    })
  })

  it("carries the drill-down filters into an 'all matching' delete", async () => {
    // The count that sizes "select all N" is taken under the source/session/provider
    // scope, so the delete body has to repeat it. If it does not, the server
    // re-derives a wider set: omitting `source` alone widened the target from one
    // imported source to every imported row in the window.
    const user = userEvent.setup()
    const { calls } = mockApi({
      rows: [
        entry({
          id: "imp-1",
          model: "imported-model",
          source: "claude_code",
          counts_toward_budget: false,
        }),
      ],
      total: 5,
    })
    renderPage(
      <ActivityPage />,
      "/activity?source=claude_code&source_label=task-42&provider=anthropic&endpoint=external",
    )

    const row = (await screen.findByText("imported-model")).closest("tr")!
    await user.click(within(row).getByRole("checkbox"))
    await user.click(
      await screen.findByRole("button", { name: /Select all 5 matching/ }),
    )

    await user.click(screen.getByRole("button", { name: "Delete" }))
    const dialog = await screen.findByRole("alertdialog")
    await user.click(within(dialog).getByRole("button", { name: "Delete" }))

    await waitFor(() => {
      const del = calls.find(
        (c) => c.url.endsWith(`${API_ROOT}/usage`) && c.method === "DELETE",
      )
      expect(del).toBeTruthy()
      const body = JSON.parse(del!.body ?? "{}")
      expect(body.by_filter).toBe(true)
      expect(body.source).toBe("claude_code")
      expect(body.source_label).toBe("task-42")
      expect(body.provider).toBe("anthropic")
      expect(body.endpoint).toBe("external")
    })
  })

  it("carries the selected workspace into an 'all matching' delete", async () => {
    // The widest scope on the page, and the only one not set by a control on it:
    // the sidebar's switcher narrows the table, so the count that sizes "select
    // all N" is taken inside one workspace. A delete body that omits it is
    // re-derived server-side without the scope and destroys every other
    // workspace's imported rows from a view the operator had narrowed to one.
    const user = userEvent.setup()
    const workspaceId = "11111111-2222-3333-4444-555555555555"
    const { calls } = mockApi({
      rows: [
        entry({
          id: "imp-1",
          model: "imported-model",
          source: "claude_code",
          counts_toward_budget: false,
        }),
      ],
      total: 5,
      workspace: workspaceId,
    })
    renderPage(<ActivityPage />)

    const row = (await screen.findByText("imported-model")).closest("tr")!
    await user.click(within(row).getByRole("checkbox"))
    await user.click(
      await screen.findByRole("button", { name: /Select all 5 matching/ }),
    )

    await user.click(screen.getByRole("button", { name: "Delete" }))
    const dialog = await screen.findByRole("alertdialog")
    await user.click(within(dialog).getByRole("button", { name: "Delete" }))

    await waitFor(() => {
      const del = calls.find(
        (c) => c.url.endsWith(`${API_ROOT}/usage`) && c.method === "DELETE",
      )
      expect(del).toBeTruthy()
      const body = JSON.parse(del!.body ?? "{}")
      expect(body.by_filter).toBe(true)
      expect(body.workspace_id).toBe(workspaceId)
    })

    // The count the operator confirmed was taken under the same scope, which is
    // what makes "all 5 matching" mean the same set on both sides.
    const counts = calls.filter((c) =>
      c.url.includes(`${API_ROOT}/usage/count`),
    )
    expect(
      counts.some((c) => c.url.includes(`workspace_id=${workspaceId}`)),
    ).toBe(true)
  })

  it("carries a multi-value filter into an 'all matching' delete", async () => {
    // The dangerous case for repeatable filters: the operator confirms a count taken
    // over two models, so the delete body has to name both. A body that dropped the
    // extra value (or sent one of the two) would delete a different set than the
    // count promised, in the one direction that loses rows.
    const user = userEvent.setup()
    const { calls } = mockApi({
      rows: [
        entry({
          id: "imp-1",
          model: "imported-model",
          source: "claude_code",
          counts_toward_budget: false,
        }),
      ],
      total: 5,
    })
    renderPage(<ActivityPage />, "/activity?model=gpt-4o&model=claude-sonnet-5")

    const row = (await screen.findByText("imported-model")).closest("tr")!
    await user.click(within(row).getByRole("checkbox"))
    await user.click(
      await screen.findByRole("button", { name: /Select all 5 matching/ }),
    )

    await user.click(screen.getByRole("button", { name: "Delete" }))
    const dialog = await screen.findByRole("alertdialog")
    await user.click(within(dialog).getByRole("button", { name: "Delete" }))

    await waitFor(() => {
      const del = calls.find(
        (c) => c.url.endsWith(`${API_ROOT}/usage`) && c.method === "DELETE",
      )
      expect(del).toBeTruthy()
      const body = JSON.parse(del!.body ?? "{}")
      expect(body.by_filter).toBe(true)
      expect(body.model).toEqual(["gpt-4o", "claude-sonnet-5"])
    })

    // The count that sized "all matching" was scoped to the same two models.
    const counts = calls.filter((c) =>
      c.url.includes(`${API_ROOT}/usage/count`),
    )
    expect(
      counts.some(
        (c) =>
          c.url.includes("model=gpt-4o") &&
          c.url.includes("model=claude-sonnet-5"),
      ),
    ).toBe(true)
  })

  it("sets a manual price on the selected imported rows", async () => {
    const user = userEvent.setup()
    const { calls } = mockApi({
      rows: [
        entry({
          id: "imp-1",
          model: "imported-model",
          source: "claude_code",
          counts_toward_budget: false,
        }),
      ],
      total: 1,
    })
    renderPage(<ActivityPage />)

    const row = (await screen.findByText("imported-model")).closest("tr")!
    await user.click(within(row).getByRole("checkbox"))
    await user.click(screen.getByRole("button", { name: "Set price" }))

    const dialog = await screen.findByRole("dialog")
    await user.type(within(dialog).getByLabelText("Input $ / 1M"), "3")
    await user.type(within(dialog).getByLabelText("Output $ / 1M"), "15")
    await user.click(within(dialog).getByRole("button", { name: "Set price" }))

    await waitFor(() => {
      const priceCall = calls.find(
        (c) =>
          c.url.includes(`${API_ROOT}/usage/set-price`) && c.method === "POST",
      )
      expect(priceCall).toBeTruthy()
      expect(priceCall!.body).toContain("imp-1")
      expect(priceCall!.body).toContain('"input_price_per_million":3')
      expect(priceCall!.body).toContain('"output_price_per_million":15')
    })
  })

  it("hides the selection column when nothing on the page can be selected", async () => {
    // A gateway-only deployment has no imported rows, so every checkbox would
    // render disabled: a column of dead controls rather than an explanation.
    mockApi({
      rows: [
        entry({ id: "gw", model: "gateway-model", counts_toward_budget: true }),
      ],
    })
    renderPage(<ActivityPage />)

    await screen.findByText("gateway-model")
    expect(screen.queryAllByRole("checkbox")).toHaveLength(0)
  })

  it("prices the model from a request that carried no cost", async () => {
    const user = userEvent.setup()
    // A row stores the instance and the bare model separately, so the pricing
    // key has to be rebuilt from both: the model alone is prefix-less and the
    // dialog would (rightly) refuse it.
    const { calls } = mockApi({
      rows: [
        entry({
          id: "free",
          model: "mistral-small",
          provider: "vllm",
          cost: null,
        }),
      ],
    })
    renderPage(<ActivityPage />)

    const row = (await screen.findByText("mistral-small")).closest("tr")!
    await user.click(row)
    await user.click(screen.getByRole("button", { name: "Price this model" }))

    const dialog = await screen.findByRole("dialog")
    expect(
      within(dialog).getByRole("combobox", { name: "Model key" }),
    ).toHaveValue("vllm:mistral-small")
    await user.type(within(dialog).getByLabelText("Input $ / 1M"), "0.2")
    await user.type(within(dialog).getByLabelText("Output $ / 1M"), "0.6")
    // Trigger and submit say the same string, so this is scoped to the dialog.
    await user.click(
      within(dialog).getByRole("button", { name: "Price this model" }),
    )

    await waitFor(() => {
      const call = calls.find(
        (c) => c.url.includes(`${API_ROOT}/pricing`) && c.method === "POST",
      )
      expect(call).toBeTruthy()
      expect(JSON.parse(call!.body!)).toMatchObject({
        model_key: "vllm:mistral-small",
        input_price_per_million: 0.2,
        output_price_per_million: 0.6,
      })
    })
    // Setting the model's price must not rewrite what logged rows were billed.
    expect(
      calls.some((c) => c.url.includes(`${API_ROOT}/usage/set-price`)),
    ).toBe(false)
  })

  it("does not offer model pricing on a request that was costed", async () => {
    const user = userEvent.setup()
    mockApi({
      rows: [
        entry({ id: "paid", model: "gpt-4o", provider: "openai", cost: 0.5 }),
      ],
    })
    renderPage(<ActivityPage />)

    const row = (await screen.findByText("gpt-4o")).closest("tr")!
    await user.click(row)

    expect(screen.getByText("Request detail")).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Price this model" }),
    ).not.toBeInTheDocument()
  })

  it("treats a $0 cost as priced, not as a model needing a price", async () => {
    // cost=0 is a real price (a model priced at zero), which is why the backend
    // marks a row unpriced on cost IS NULL rather than on falsiness.
    const user = userEvent.setup()
    mockApi({
      rows: [
        entry({
          id: "free-model",
          model: "mistral-small",
          provider: "vllm",
          cost: 0,
        }),
      ],
    })
    renderPage(<ActivityPage />)

    const row = (await screen.findByText("mistral-small")).closest("tr")!
    await user.click(row)

    expect(screen.getByText("Request detail")).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Price this model" }),
    ).not.toBeInTheDocument()
  })

  it("prices a selector that never resolved from the row's model alone", async () => {
    // A selector the gateway could not resolve is logged with no provider and
    // the raw selector as the model, so it is already the key to price.
    const user = userEvent.setup()
    mockApi({
      rows: [
        entry({
          id: "unresolved",
          model: "vllm:mistral-small",
          provider: null,
          cost: null,
        }),
      ],
    })
    renderPage(<ActivityPage />)

    const row = (await screen.findByText("vllm:mistral-small")).closest("tr")!
    await user.click(row)
    await user.click(screen.getByRole("button", { name: "Price this model" }))

    const dialog = await screen.findByRole("dialog")
    expect(
      within(dialog).getByRole("combobox", { name: "Model key" }),
    ).toHaveValue("vllm:mistral-small")
  })
})
