import { act, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { useState } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { InFlightRequest, InFlightResponse } from "@/client"
import { mockApi, renderPage } from "@/tests/activity"
import { flushRouter } from "@/tests/router"
import { InFlightControl } from "./InFlightControl"

afterEach(() => {
  vi.restoreAllMocks()
})

function request(overrides: Partial<InFlightRequest> = {}): InFlightRequest {
  return {
    id: "run-1",
    model: "gpt-4o",
    provider: "openai",
    user_id: "alice",
    api_key_id: null,
    policy_name: null,
    endpoint: "/v1/chat/completions",
    elapsed_ms: 4_000,
    started_at: new Date().toISOString(),
    ...overrides,
  }
}

function payload(overrides: Partial<InFlightResponse> = {}): InFlightResponse {
  return { requests: [request()], total: 1, ...overrides }
}

describe("InFlightControl", () => {
  it("reports the count and opens the list behind it", async () => {
    const user = userEvent.setup()
    mockApi()
    renderPage(<InFlightControl data={payload()} updatedAt={Date.now()} />)
    await flushRouter()

    await user.click(screen.getByRole("button", { name: /1 in flight/ }))

    expect(await screen.findByText("gpt-4o")).toBeInTheDocument()
    // Said in the popover rather than in the trigger's label: the endpoint takes
    // no filters, so the count is not narrowed by the page's.
    expect(
      screen.getByText(/Not narrowed by the filters above/),
    ).toBeInTheDocument()
  })

  it("names the policy beside the model for a routed request", async () => {
    const user = userEvent.setup()
    mockApi()
    renderPage(
      <InFlightControl
        data={payload({ requests: [request({ policy_name: "cheap-first" })] })}
        updatedAt={Date.now()}
      />,
    )
    await flushRouter()

    await user.click(screen.getByRole("button", { name: /1 in flight/ }))
    expect(await screen.findByText(/cheap-first/)).toBeInTheDocument()
  })

  it("renders nothing on an idle gateway", async () => {
    mockApi()
    const { container } = renderPage(
      <InFlightControl
        data={payload({ requests: [], total: 0 })}
        updatedAt={Date.now()}
      />,
    )
    await flushRouter()

    expect(container).toBeEmptyDOMElement()
  })

  it("keeps an opened list open once the last request lands", async () => {
    // The whole reason the open state is controlled here rather than left to
    // DialogTrigger: unmounting would discard it under an operator who is
    // reading the list.
    // The open popover `aria-hidden`s the rest of the page, so the poll that
    // settles the last request is driven from outside the tree rather than
    // through a control the operator could no longer reach either.
    let settle = () => {}
    function Settling() {
      const [data, setData] = useState(payload())
      settle = () => setData(payload({ requests: [], total: 0 }))
      return <InFlightControl data={data} updatedAt={Date.now()} />
    }
    const user = userEvent.setup()
    mockApi()
    renderPage(<Settling />)
    await flushRouter()

    await user.click(screen.getByRole("button", { name: /1 in flight/ }))
    await screen.findByText("gpt-4o")
    await act(async () => {
      settle()
    })

    // Still open, and still the operator's to close.
    expect(screen.getByRole("dialog", { name: "In flight" })).toBeVisible()
    expect(
      await screen.findByText(/Nothing running right now/),
    ).toBeInTheDocument()
    // `hidden`, because the open popover `aria-hidden`s the page the trigger
    // sits on; the control is what stayed mounted, which is the point here.
    expect(
      screen.getByRole("button", { name: /0 in flight/, hidden: true }),
    ).toBeInTheDocument()
  })

  it("says how many further requests the endpoint's cap left out", async () => {
    const user = userEvent.setup()
    mockApi()
    renderPage(
      <InFlightControl data={payload({ total: 3 })} updatedAt={Date.now()} />,
    )
    await flushRouter()

    await user.click(screen.getByRole("button", { name: /3 in flight/ }))
    expect(
      await screen.findByText(
        /2 further requests are in flight beyond the 1 listed/,
      ),
    ).toBeInTheDocument()
  })
})
