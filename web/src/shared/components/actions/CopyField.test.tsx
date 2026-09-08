import { fireEvent, render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"
import { CopyableValue } from "@/shared/components/actions/CopyField"

describe("CopyableValue", () => {
  it("copies the value, which need not be what is displayed", async () => {
    const user = userEvent.setup()
    render(
      <CopyableValue value="openai:gpt-4o-2024-11-20" label="model id">
        gpt-4o-2024-11-20
      </CopyableValue>,
    )

    expect(screen.getByText("gpt-4o-2024-11-20")).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Copy model id" }))
    expect(await navigator.clipboard.readText()).toBe(
      "openai:gpt-4o-2024-11-20",
    )
  })

  it("keeps a row press from starting on the value, so a drag can highlight it", () => {
    // The whole reason highlighting an id in a table used to fail: react-aria's
    // row press toggles selection on pointer down, and that re-render lands
    // mid-drag and discards the browser's nascent selection (#478). The value
    // stops the pointer sequence from reaching the row.
    const onRowPointerDown = vi.fn()
    const onRowMouseDown = vi.fn()
    render(
      // biome-ignore lint/a11y/noStaticElementInteractions: a stand-in for the table row whose handlers this test proves are not reached
      <div onPointerDown={onRowPointerDown} onMouseDown={onRowMouseDown}>
        <CopyableValue
          value="anthropic:claude-opus-4-5-20251101"
          label="model id"
        />
      </div>,
    )

    const value = screen.getByText("anthropic:claude-opus-4-5-20251101")
    fireEvent.pointerDown(value, {
      pointerId: 1,
      pointerType: "mouse",
      button: 0,
    })
    fireEvent.mouseDown(value, { button: 0 })

    expect(onRowPointerDown).not.toHaveBeenCalled()
    expect(onRowMouseDown).not.toHaveBeenCalled()
    // Selectable in its own right, so an inherited `user-select: none` from a
    // press elsewhere in the row cannot suppress it.
    expect(value.className).toContain("select-text")
  })
})
