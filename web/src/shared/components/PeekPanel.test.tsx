import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactNode } from "react"
import { describe, expect, it, vi } from "vitest"

import { PeekPanel } from "./PeekPanel"

function renderPanel(
  props: Partial<Parameters<typeof PeekPanel>[0]> = {},
  body: ReactNode = <p>Body</p>,
) {
  const onClose = vi.fn()
  const onPrevious = vi.fn()
  const onNext = vi.fn()
  render(
    <PeekPanel
      label="Request detail"
      kind="Request"
      noun="request"
      recordKey="r1"
      onClose={onClose}
      onPrevious={onPrevious}
      onNext={onNext}
      {...props}
    >
      {body}
    </PeekPanel>,
  )
  return { onClose, onPrevious, onNext }
}

describe("PeekPanel", () => {
  it("opens as a named region with its kind, identity and body, focused", async () => {
    renderPanel({ identity: <span>r1-id</span> })
    const panel = screen.getByRole("complementary", { name: "Request detail" })
    expect(panel).toHaveTextContent("Request")
    expect(panel).toHaveTextContent("r1-id")
    expect(panel).toHaveTextContent("Body")
    await waitFor(() => expect(panel).toHaveFocus())
  })

  it("closes from the Close button, after the exit", async () => {
    const { onClose } = renderPanel()
    await userEvent.click(screen.getByRole("button", { name: "Close" }))
    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1))
  })

  it("closes on Escape", async () => {
    const { onClose } = renderPanel()
    await userEvent.keyboard("{Escape}")
    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1))
  })

  it("leaves Escape to a field", async () => {
    const { onClose } = renderPanel({}, <input aria-label="Search" />)
    await userEvent.click(screen.getByRole("textbox", { name: "Search" }))
    await userEvent.keyboard("{Escape}")
    await new Promise((resolve) => setTimeout(resolve, 300))
    expect(onClose).not.toHaveBeenCalled()
  })

  it("walks with the buttons and with K and J", async () => {
    const { onPrevious, onNext } = renderPanel()
    await userEvent.click(
      screen.getByRole("button", { name: "Previous request" }),
    )
    await userEvent.click(screen.getByRole("button", { name: "Next request" }))
    expect(onPrevious).toHaveBeenCalledTimes(1)
    expect(onNext).toHaveBeenCalledTimes(1)
    await userEvent.keyboard("k")
    await userEvent.keyboard("j")
    expect(onPrevious).toHaveBeenCalledTimes(2)
    expect(onNext).toHaveBeenCalledTimes(2)
  })

  it("disables the walk controls at the ends of the list", async () => {
    renderPanel({ onPrevious: undefined, onNext: undefined })
    expect(
      screen.getByRole("button", { name: "Previous request" }),
    ).toBeDisabled()
    expect(screen.getByRole("button", { name: "Next request" })).toBeDisabled()
    await userEvent.keyboard("kj")
  })

  it("toggles between Expand and Collapse", async () => {
    renderPanel()
    await userEvent.click(screen.getByRole("button", { name: "Expand" }))
    expect(screen.getByRole("button", { name: "Collapse" })).toBeVisible()
  })
})
