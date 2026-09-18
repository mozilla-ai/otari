import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"

import { ModeToggle } from "./ModeToggle"

describe("ModeToggle", () => {
  it("shows the current mode as the selected one", () => {
    render(<ModeToggle label="On failure" value="monitor" onChange={vi.fn()} />)
    expect(screen.getByRole("button", { name: "monitor" })).toHaveAttribute(
      "aria-pressed",
      "true",
    )
    expect(screen.getByRole("button", { name: "block" })).toHaveAttribute(
      "aria-pressed",
      "false",
    )
  })

  it("reports the mode that was pressed", async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(
      <ModeToggle label="On failure" value="monitor" onChange={onChange} />,
    )
    await user.click(screen.getByRole("button", { name: "block" }))
    expect(onChange).toHaveBeenCalledWith("block")
  })

  it("reports the mode already selected, so the caller decides", async () => {
    // Unlike ScopePicker's tabs, which guard against a press that would discard
    // a selection, this one has nothing to lose and stays a plain report.
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(<ModeToggle label="On failure" value="block" onChange={onChange} />)
    await user.click(screen.getByRole("button", { name: "block" }))
    expect(onChange).toHaveBeenCalledWith("block")
  })

  it("renders the hint only when there is one", () => {
    const { rerender } = render(
      <ModeToggle label="On failure" value="block" onChange={vi.fn()} />,
    )
    expect(screen.queryByText("Blocks the request.")).not.toBeInTheDocument()

    rerender(
      <ModeToggle
        label="On failure"
        hint="Blocks the request."
        value="block"
        onChange={vi.fn()}
      />,
    )
    expect(screen.getByText("Blocks the request.")).toBeInTheDocument()
  })
})
