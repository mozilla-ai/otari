import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { useState } from "react"
import { describe, expect, it } from "vitest"

import { DisclosureRow } from "./DisclosureRow"

function Harness() {
  const [isOpen, setIsOpen] = useState(false)
  return (
    <DisclosureRow
      label="Configure search tools"
      help="None configured, so POST /api/v1/search refuses every request."
      trailing={<span>0 tools</span>}
      isOpen={isOpen}
      onToggle={() => setIsOpen((open) => !open)}
    >
      <p>The panel</p>
    </DisclosureRow>
  )
}

describe("DisclosureRow", () => {
  it("opens and closes from the pointer, reporting its state", async () => {
    const user = userEvent.setup()
    render(<Harness />)

    const row = screen.getByRole("button", { name: /Configure search tools/ })
    expect(row).toHaveAttribute("aria-expanded", "false")
    // Present but hidden, not unmounted: `aria-controls` has to name an element
    // that exists even while the row is closed.
    expect(screen.getByText("The panel")).not.toBeVisible()

    await user.click(row)
    expect(row).toHaveAttribute("aria-expanded", "true")
    expect(screen.getByText("The panel")).toBeVisible()

    await user.click(row)
    expect(row).toHaveAttribute("aria-expanded", "false")
    expect(screen.getByText("The panel")).not.toBeVisible()
  })

  it("opens from the keyboard on Enter and on Space", async () => {
    const user = userEvent.setup()
    render(<Harness />)
    const row = screen.getByRole("button", { name: /Configure search tools/ })

    await user.tab()
    expect(row).toHaveFocus()

    await user.keyboard("{Enter}")
    expect(row).toHaveAttribute("aria-expanded", "true")

    await user.keyboard(" ")
    expect(row).toHaveAttribute("aria-expanded", "false")
  })

  it("points at the panel it controls", async () => {
    const user = userEvent.setup()
    render(<Harness />)
    const row = screen.getByRole("button", { name: /Configure search tools/ })

    // Named before it is opened, which is the point: a collapsed row must not
    // point `aria-controls` at an id nothing has.
    const panelId = row.getAttribute("aria-controls")
    expect(panelId).toBeTruthy()
    expect(document.getElementById(String(panelId))).toHaveTextContent(
      "The panel",
    )

    await user.click(row)
    expect(document.getElementById(String(panelId))).toBeVisible()
  })

  it("makes the whole row the target, not the chevron", () => {
    // A chevron is a 16px glyph. Making it the control puts a 16px target in a
    // 44px row, which is the mistake this component exists to prevent.
    render(<Harness />)
    const row = screen.getByRole("button", { name: /Configure search tools/ })

    expect(row.className).toContain("w-full")
    expect(row.className).toContain("min-h-11")
    expect(screen.queryAllByRole("button")).toHaveLength(1)
  })
})
