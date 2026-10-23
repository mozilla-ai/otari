import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { HostedModel, HostedProvider } from "@/client"
import { HostedModelForm } from "@/features/hostedProviders/HostedModelForm"
import { API_ROOT } from "@/shared/api/client"
import { renderWithRouter } from "@/tests/router"

const provider = {
  id: "p1",
  name: "fireworks",
  provider: "fireworks",
} as HostedProvider

function model(overrides: Partial<HostedModel> = {}): HostedModel {
  return {
    id: "m1",
    provider: "fireworks",
    model: "accounts/fireworks/models/qwen3-reranker-8b",
    input_price_per_million: 0.2,
    output_price_per_million: 0,
    cache_read_price_per_million: null,
    cache_write_price_per_million: null,
    unit: "tokens",
    price_source: "deployment",
    enabled: true,
    created_at: "2026-10-13T00:00:00Z",
    updated_at: null,
    ...overrides,
  }
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })

function mockApi(answer: (body: unknown) => Response = (body) => json(body)) {
  const patches: unknown[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    if (
      init?.method === "PATCH" &&
      url.includes(`${API_ROOT}/hosted-providers/fireworks/models/m1`)
    ) {
      const body = JSON.parse(String(init.body))
      patches.push(body)
      return answer({ ...model(), ...body })
    }
    throw new Error(`unexpected request: ${url}`)
  })
  return patches
}

async function renderForm(editing: HostedModel) {
  const onSaved = vi.fn()
  await renderWithRouter(
    <QueryClientProvider
      client={
        new QueryClient({ defaultOptions: { queries: { retry: false } } })
      }
    >
      <HostedModelForm
        provider={provider}
        editing={editing}
        isOpen
        onClose={() => {}}
        onSaved={onSaved}
      />
    </QueryClientProvider>,
  )
  return { onSaved }
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe("HostedModelForm", () => {
  it("prices a model per request, entered per thousand and sent per million", async () => {
    const patches = mockApi()
    const user = userEvent.setup()
    const { onSaved } = await renderForm(model())

    expect(screen.getByRole("radio", { name: "Tokens" })).toBeChecked()
    expect(screen.getByLabelText(/Output, per 1M tokens/)).toBeInTheDocument()

    await user.click(screen.getByRole("radio", { name: "Requests" }))
    // A request price is one rate: the output and cache boxes go.
    expect(screen.queryByLabelText(/Output, per 1M tokens/)).toBeNull()
    expect(screen.queryByLabelText(/Cache read/)).toBeNull()

    const rate = screen.getByLabelText(/Price per 1,000 requests/)
    await user.clear(rate)
    await user.type(rate, "2")
    await user.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() => expect(onSaved).toHaveBeenCalled())
    expect(patches).toEqual([
      {
        input_price_per_million: 2000,
        output_price_per_million: 0,
        cache_read_price_per_million: null,
        cache_write_price_per_million: null,
        unit: "requests",
      },
    ])
  })

  it("opens a per-image price per thousand images", async () => {
    mockApi()
    await renderForm(model({ unit: "images", input_price_per_million: 40_000 }))
    expect(screen.getByRole("radio", { name: "Images" })).toBeChecked()
    expect(screen.getByLabelText(/Price per 1,000 images/)).toHaveValue("40")
  })

  it("holds a rate that does not parse back from the save", async () => {
    const patches = mockApi()
    const user = userEvent.setup()
    await renderForm(model())

    const input = screen.getByLabelText(/Input, per 1M tokens/)
    await user.clear(input)
    await user.type(input, "-1")

    expect(
      await screen.findByText("A non-negative number of US dollars."),
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled()
    expect(patches).toEqual([])
  })

  it("reports a refused save inside the dialog", async () => {
    mockApi(() => json({ detail: "The price could not be stored." }, 422))
    const user = userEvent.setup()
    const { onSaved } = await renderForm(model())

    await user.click(screen.getByRole("button", { name: "Save" }))

    expect(
      await screen.findByText(/The price could not be stored/),
    ).toBeInTheDocument()
    expect(onSaved).not.toHaveBeenCalled()
  })

  it("offers a decimal keyboard on every rate and opens a reprice on the first", async () => {
    mockApi()
    const user = userEvent.setup()
    await renderForm(model())

    const input = screen.getByLabelText(/Input, per 1M tokens/)
    await waitFor(() => expect(input).toHaveFocus())
    for (const label of [/Input, per 1M/, /Output, per 1M/, /Cache read/]) {
      expect(screen.getByLabelText(label)).toHaveAttribute(
        "inputmode",
        "decimal",
      )
    }

    // The swapped-in field does not take focus from the unit just picked.
    await user.click(screen.getByRole("radio", { name: "Requests" }))
    expect(screen.getByLabelText(/Price per 1,000 requests/)).not.toHaveFocus()
  })
})
