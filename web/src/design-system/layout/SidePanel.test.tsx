import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"
import { SidePanel } from "./SidePanel"

describe("SidePanel", () => {
  it("walks the list and closes from its own controls", async () => {
    const onClose = vi.fn()
    const onNext = vi.fn()
    render(
      <SidePanel
        isOpen
        onClose={onClose}
        label="Session"
        heading="abc"
        onNext={onNext}
      >
        <p>Body</p>
      </SidePanel>,
    )

    const panel = await screen.findByRole("dialog", { name: "Session" })
    expect(panel).toHaveTextContent("Body")
    // The first record has nothing above it.
    expect(screen.getByRole("button", { name: "Previous" })).toBeDisabled()
    await userEvent.click(screen.getByRole("button", { name: "Next" }))
    expect(onNext).toHaveBeenCalledOnce()
    await userEvent.click(screen.getByRole("button", { name: "Close" }))
    expect(onClose).toHaveBeenCalledOnce()
  })

  it("closes on Escape", async () => {
    const onClose = vi.fn()
    render(
      <SidePanel isOpen onClose={onClose} label="Session" heading="abc">
        <p>Body</p>
      </SidePanel>,
    )

    await screen.findByRole("dialog", { name: "Session" })
    await userEvent.keyboard("{Escape}")

    expect(onClose).toHaveBeenCalledOnce()
  })
})
