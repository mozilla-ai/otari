import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"

import { Toggle } from "./Toggle"

describe("Toggle", () => {
  it("draws off as an outline on the page ground", () => {
    render(<Toggle label="Allow" isSelected={false} onChange={vi.fn()} />)
    const toggle = screen.getByRole("switch", { name: "Allow" })
    expect(toggle).toHaveAttribute("aria-checked", "false")
    expect(toggle).toHaveClass("border-control-border", "bg-background")
    expect(toggle).not.toHaveClass("bg-control-indicator")
  })

  it("fills the whole track when on, so the states differ in area and not only in the knob", () => {
    render(<Toggle label="Allow" isSelected onChange={vi.fn()} />)
    const toggle = screen.getByRole("switch", { name: "Allow" })
    expect(toggle).toHaveAttribute("aria-checked", "true")
    expect(toggle).toHaveClass(
      "border-control-indicator",
      "bg-control-indicator",
    )
    expect(toggle.firstElementChild).toHaveClass("bg-accent-glyph")
  })

  it("reports the next value when pressed", async () => {
    const onChange = vi.fn()
    render(<Toggle label="Allow" isSelected={false} onChange={onChange} />)
    await userEvent.setup().click(screen.getByRole("switch"))
    expect(onChange).toHaveBeenCalledWith(true)
  })
})
