import { screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { entry, mockApi, renderPage } from "@/tests/activity"
import { flushRouter } from "@/tests/router"
import { RequestDetail } from "./RequestDetail"

afterEach(() => {
  vi.restoreAllMocks()
})

describe("RequestDetail", () => {
  it("shows the stored error verbatim, under a source-neutral heading", async () => {
    mockApi()
    renderPage(
      <RequestDetail
        entry={entry({
          status: "error",
          status_code: 402,
          error_message: "no pricing configured for openai:gpt-4o",
        })}
        onPriceModel={null}
      />,
    )
    await flushRouter()

    expect(screen.getByText("Error (402)")).toBeInTheDocument()
    expect(
      screen.getByText("no pricing configured for openai:gpt-4o"),
    ).toBeInTheDocument()
  })

  it("spells out the billed total beside the provider-reported one", async () => {
    mockApi()
    renderPage(
      <RequestDetail
        entry={entry({
          prompt_tokens: 400,
          completion_tokens: 200,
          // An additive-convention row: the provider reports the cache buckets
          // outside its own total, so the billed figure is the larger one.
          total_tokens: 600,
          billing_meters: {
            total_input_tokens: 1000,
            cache_read_tokens: 600,
            completion_tokens: 200,
          },
        })}
        onPriceModel={null}
      />,
    )
    await flushRouter()

    expect(screen.getByText("Total tokens")).toBeInTheDocument()
    expect(screen.getByText("600")).toBeInTheDocument()
    expect(screen.getByText("Billed tokens")).toBeInTheDocument()
    expect(
      screen.getByTitle(/the tokens this request was priced on/),
    ).toHaveTextContent("1,200")
  })

  it("names a row's tool calls and what they cost", async () => {
    mockApi()
    renderPage(
      <RequestDetail
        entry={entry({
          billing_meters: {
            tools: { web_search: { billed: 3, errors: 1, unit_rate: 0.01 } },
          },
        })}
        onPriceModel={null}
      />,
    )
    await flushRouter()

    expect(screen.getByText("web search ×3, 1 failed")).toBeInTheDocument()
    expect(screen.getByText("$0.03")).toBeInTheDocument()
  })

  it("says a tool is unpriced rather than reporting it as free", async () => {
    mockApi()
    renderPage(
      <RequestDetail
        entry={entry({
          billing_meters: { tools: { web_search: { billed: 3 } } },
        })}
        onPriceModel={null}
      />,
    )
    await flushRouter()

    expect(screen.getByText("unpriced")).toBeInTheDocument()
  })

  it("offers to price the model behind an uncosted row, keyed on instance:model", async () => {
    const user = userEvent.setup()
    const onPriceModel = vi.fn()
    mockApi()
    renderPage(
      <RequestDetail
        entry={entry({ cost: null, provider: "openai", model: "gpt-4o" })}
        onPriceModel={onPriceModel}
      />,
    )
    await flushRouter()

    await user.click(screen.getByRole("button", { name: "Price this model" }))
    expect(onPriceModel).toHaveBeenCalledWith("openai:gpt-4o")
  })

  it("still explains an uncosted row to a caller who cannot price it", async () => {
    // Pricing a model is a deployment-wide write, so offering the button to a
    // tenant would be offering a refusal; the sentence is still theirs to read.
    mockApi()
    renderPage(
      <RequestDetail entry={entry({ cost: null })} onPriceModel={null} />,
    )
    await flushRouter()

    expect(
      screen.queryByRole("button", { name: "Price this model" }),
    ).not.toBeInTheDocument()
    expect(
      screen.getByText(/A deployment operator sets one/),
    ).toBeInTheDocument()
  })

  it("treats a zero cost as a real price, not as uncosted", async () => {
    mockApi()
    renderPage(
      <RequestDetail entry={entry({ cost: 0 })} onPriceModel={vi.fn()} />,
    )
    await flushRouter()

    expect(
      screen.queryByRole("button", { name: "Price this model" }),
    ).not.toBeInTheDocument()
  })

  it("renders the routing plan only for a routed request", async () => {
    mockApi()
    const { unmount } = renderPage(
      <RequestDetail
        entry={entry({ policy_name: null })}
        onPriceModel={null}
      />,
    )
    await flushRouter()
    expect(screen.queryByText(/Routing plan/)).not.toBeInTheDocument()
    unmount()

    mockApi()
    renderPage(
      <RequestDetail
        entry={entry({ policy_name: "cheap-first" })}
        onPriceModel={null}
      />,
    )
    await flushRouter()
    expect(await screen.findByText(/Routing plan/)).toBeInTheDocument()
  })

  it("prints a charge line through the rate format its shape carries", async () => {
    mockApi()
    renderPage(
      <RequestDetail
        entry={entry({
          pricing_breakdown: [
            {
              meter: "web_search",
              units: 2,
              unit_rate: 0.01,
              cost: 0.02,
            },
            {
              meter: "completion_tokens",
              units: 200,
              rate_per_million: 10,
              cost: 0.002,
            },
          ],
        })}
        onPriceModel={null}
      />,
    )
    await flushRouter()

    expect(screen.getByText("Billed meters")).toBeInTheDocument()
    expect(screen.getByText(/200 at \$10.00 \/ 1M/)).toBeInTheDocument()
    expect(screen.getByText(/2 at .* each/)).toBeInTheDocument()
  })
})
