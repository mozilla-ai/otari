import { render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { useState } from "react"
import { describe, expect, it, vi } from "vitest"

import { MultiSelect } from "./MultiSelect"

const PEOPLE = [
  { id: "operator", label: "Operator" },
  { id: "alice", label: "alice@example.com" },
  { id: "pat", label: "pat@example.com (Pat Okafor)" },
  { id: "priya", label: "priya@example.com (Priya Raghavan)" },
  { id: "parity", label: "parity-heavy@example.com" },
]

const LABEL = "Assign to people (optional)"

function Live({ initial = [] }: { initial?: string[] }) {
  const [value, setValue] = useState<string[]>(initial)
  return (
    <MultiSelect
      label={LABEL}
      options={PEOPLE}
      value={value}
      onChange={setValue}
      searchPlaceholder="Search people…"
      countNoun="people assigned"
    />
  )
}

const field = () => screen.getByLabelText(LABEL)
/** The chip row, which is a real list named after the field. */
const chips = () => screen.getByRole("list", { name: `${LABEL}, selected` })

describe("MultiSelect", () => {
  it("names the field by its visible label", () => {
    render(<Live />)
    expect(field()).toBeInTheDocument()
    expect(field()).toHaveAttribute("role", "combobox")
  })

  it("leaves the field where it was when a pick is made", async () => {
    // The bug this exists for: chips above the field grew the block upward and
    // pushed the control out from under the pointer. Measured as the field's
    // own position in the DOM order of its container, which is what moving it
    // would change.
    const user = userEvent.setup()
    render(<Live />)

    await user.click(field())
    await user.click(screen.getByRole("option", { name: /Operator/ }))
    await user.click(screen.getByRole("option", { name: /alice/ }))

    expect(within(chips()).getAllByRole("listitem")).toHaveLength(2)
    // Every chip comes AFTER the field in document order, which is the whole
    // fix: chips above it grew the block upward and moved the control.
    for (const chip of within(chips()).getAllByRole("listitem")) {
      expect(field().compareDocumentPosition(chip)).toBe(
        Node.DOCUMENT_POSITION_FOLLOWING,
      )
    }
  })

  it("keeps a selected option in the list, checked, and toggles it off", async () => {
    // The other half: a picked person used to vanish from the list, so the list
    // only ever showed who was left rather than who was in.
    const user = userEvent.setup()
    render(<Live />)

    await user.click(field())
    const before = screen
      .getAllByRole("option")
      .map((option) => option.textContent)

    await user.click(screen.getByRole("option", { name: /Operator/ }))

    const operator = screen.getByRole("option", { name: /Operator/ })
    expect(operator).toHaveAttribute("aria-selected", "true")
    // Order is the same list, in the same order: no selected-first re-sort,
    // which would move rows under the pointer between two presses.
    expect(
      screen.getAllByRole("option").map((option) => option.textContent),
    ).toEqual(before)

    await user.click(operator)
    expect(screen.getByRole("option", { name: /Operator/ })).toHaveAttribute(
      "aria-selected",
      "false",
    )
    expect(screen.queryByRole("list", { name: /selected/ })).toBeNull()
  })

  it("closes the popover on Escape without letting the key travel further", async () => {
    // Inside a FormDialog, an Escape that reaches the dialog arms its
    // unsaved-changes guard. Dismissing a list must not do that, so the handler
    // stops the event rather than only acting on it.
    const onKeyDown = vi.fn()
    const user = userEvent.setup()
    render(
      // biome-ignore lint/a11y/noStaticElementInteractions: a stand-in for the dialog that would catch this key
      <div onKeyDown={onKeyDown}>
        <Live />
      </div>,
    )

    await user.click(field())
    expect(screen.getByRole("listbox")).toBeInTheDocument()

    await user.keyboard("{Escape}")
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument()
    expect(onKeyDown).not.toHaveBeenCalled()
  })

  it("filters on the query and counts the matches against the whole selection", async () => {
    const user = userEvent.setup()
    render(<Live initial={["operator"]} />)

    await user.click(field())
    await user.type(field(), "pa")

    // "pa" reaches two of the five: pat@ and parity-heavy@.
    expect(screen.getAllByRole("option")).toHaveLength(2)
    expect(
      within(screen.getByRole("listbox")).getByText(/Pat Okafor/),
    ).toBeInTheDocument()
    // None of the two matches is selected, and one person is assigned overall.
    expect(screen.getByText(/0 of 2 matches selected/)).toBeInTheDocument()
    expect(screen.getByText(/1 people assigned/)).toBeInTheDocument()
  })

  it("removes the last chip on Backspace with an empty query", async () => {
    const user = userEvent.setup()
    render(<Live initial={["operator", "alice"]} />)

    await user.click(field())
    await user.keyboard("{Backspace}")

    expect(within(chips()).getAllByRole("listitem")).toHaveLength(1)
    expect(within(chips()).getByText("Operator")).toBeInTheDocument()
  })

  it("moves the active option with the arrows and toggles it with Enter", async () => {
    const user = userEvent.setup()
    render(<Live />)

    await user.click(field())
    await user.keyboard("{ArrowDown}")
    await user.keyboard("{Enter}")

    expect(screen.getByRole("option", { name: /alice/ })).toHaveAttribute(
      "aria-selected",
      "true",
    )
  })
})
