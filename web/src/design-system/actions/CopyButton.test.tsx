import { act, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { useRef } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { CopyButton } from "@/design-system/actions/CopyButton"

describe("CopyButton", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("writes the value to the clipboard and confirms over the icon", async () => {
    const user = userEvent.setup()
    render(<CopyButton value="anthropic:claude-opus-4" label="model id" />)

    // Nothing is shown until a copy happens: this reports an event, so it must
    // not open on hover the way a hint tooltip would.
    await user.hover(screen.getByRole("button", { name: "Copy model id" }))
    expect(screen.queryByText("Copied!")).not.toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Copy model id" }))

    expect(await navigator.clipboard.readText()).toBe("anthropic:claude-opus-4")
    expect(await screen.findByText("Copied!")).toBeInTheDocument()
  })

  it("says the copy was blocked rather than claiming one when no path works", async () => {
    // The Clipboard API refuses and jsdom has no document.execCommand, so both
    // paths in copyToClipboard are exhausted.
    const user = userEvent.setup()
    vi.spyOn(navigator.clipboard, "writeText").mockRejectedValue(
      new Error("not a secure context"),
    )
    render(<CopyButton value="openai:gpt-4o" label="model id" />)

    await user.click(screen.getByRole("button", { name: "Copy model id" }))

    expect(await screen.findByText(/Copy blocked/)).toBeInTheDocument()
    expect(screen.queryByText("Copied!")).not.toBeInTheDocument()
  })

  it("keeps the confirmation out of the cell, so it cannot reflow the row", async () => {
    const user = userEvent.setup()
    const { container } = render(
      <CopyButton value="openai:gpt-4o" label="model id" />,
    )
    // `sr-only` is clipped and taken out of flow, so the announcement below is
    // not part of what the cell lays out; everything else here would be.
    const laidOut = () => {
      const clone = container.cloneNode(true) as HTMLElement
      for (const node of clone.querySelectorAll(".sr-only")) node.remove()
      return clone.textContent
    }
    expect(laidOut()).toBe("")

    await user.click(screen.getByRole("button", { name: "Copy model id" }))
    await screen.findByText("Copied!")

    // The confirmation is an overlay, not a sibling of the id it copied.
    expect(laidOut()).toBe("")
  })

  it("announces the outcome, which a press-opened tooltip does not", async () => {
    const user = userEvent.setup()
    const { container } = render(
      <CopyButton value="openai:gpt-4o" label="model id" />,
    )
    const live = container.querySelector("[aria-live]") as HTMLElement
    expect(live.textContent).toBe("")

    await user.click(screen.getByRole("button", { name: "Copy model id" }))

    await waitFor(() =>
      expect(live.textContent).toBe("Copied model id to clipboard."),
    )
  })

  it("announces a blocked copy rather than staying silent", async () => {
    const user = userEvent.setup()
    vi.spyOn(navigator.clipboard, "writeText").mockRejectedValue(
      new Error("not a secure context"),
    )
    const { container } = render(
      <CopyButton value="openai:gpt-4o" label="model id" />,
    )

    await user.click(screen.getByRole("button", { name: "Copy model id" }))

    const live = container.querySelector("[aria-live]") as HTMLElement
    await waitFor(() =>
      expect(live.textContent).toBe(
        "Could not copy model id. Select the value and press Ctrl/Cmd-C.",
      ),
    )
  })

  it("clears the confirmation on its own", async () => {
    // The dismissal is jumped, not waited out. On real timers this test spent
    // 1.5s asleep, which was the second-slowest case in the suite; the deadlock
    // the previous note described (the clipboard write is a promise, and fake
    // timers stall userEvent's own waits) is what `advanceTimers` exists for,
    // and the Activity page's live-traffic cases already used it.
    vi.useFakeTimers({ shouldAdvanceTime: true })
    try {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime })
      render(<CopyButton value="openai:gpt-4o" label="model id" />)

      await user.click(screen.getByRole("button", { name: "Copy model id" }))
      expect(await screen.findByText("Copied!")).toBeInTheDocument()

      // The component's own 1.5s reset for a successful copy. Advanced inside
      // `act` because firing it sets state, and the assertion below reads the
      // paint that follows rather than polling for it.
      await act(async () => {
        vi.advanceTimersByTime(1_500)
      })
      expect(screen.queryByText("Copied!")).not.toBeInTheDocument()
    } finally {
      vi.useRealTimers()
    }
  })

  it("selects a field it was given only after a failed attempt", async () => {
    const user = userEvent.setup()
    // Both paths refused: the async API throws, and the legacy fallback fails on
    // its own because jsdom defines no `document.execCommand`. The attempt is
    // recorded so the ordering can be asserted, not just the end state.
    const order: string[] = []
    vi.spyOn(navigator.clipboard, "writeText").mockImplementation(() => {
      order.push("attempt")
      return Promise.reject(new Error("not a secure context"))
    })

    function Harness() {
      const ref = useRef<HTMLInputElement | null>(null)
      return (
        <>
          <input ref={ref} readOnly defaultValue="otari-verify=abc" />
          <CopyButton
            value="otari-verify=abc"
            label="TXT record"
            selectOnFailure={ref}
          />
        </>
      )
    }
    render(<Harness />)
    const field = screen.getByDisplayValue(
      "otari-verify=abc",
    ) as HTMLInputElement
    field.addEventListener("focus", () => order.push("focus"))

    await user.click(screen.getByRole("button", { name: "Copy TXT record" }))

    await waitFor(() => expect(document.activeElement).toBe(field))
    expect(field.selectionStart).toBe(0)
    expect(field.selectionEnd).toBe("otari-verify=abc".length)
    // The order is the point: the legacy path restores the selection and focus
    // it found on its way out, so selecting before the attempt is undone by it.
    expect(order).toEqual(["attempt", "focus"])
  })

  it("leaves focus alone when no field was given", async () => {
    const user = userEvent.setup()
    vi.spyOn(navigator.clipboard, "writeText").mockRejectedValue(
      new Error("not a secure context"),
    )
    render(<CopyButton value="openai:gpt-4o" label="model id" />)
    const button = screen.getByRole("button", { name: "Copy model id" })

    await user.click(button)
    await screen.findByText(/Copy blocked/)

    // The default: a table cell has no field to select, so nothing is moved.
    expect(document.activeElement).toBe(button)
  })
})
