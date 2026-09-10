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
      countNoun={{ one: "person assigned", other: "people assigned" }}
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

  it("swallows Enter while the list is open with nothing to toggle", async () => {
    // The input sits inside a real form, so an Enter that falls through submits
    // it: type a query that matches nobody, press Enter, and the enclosing
    // dialog would create the object with none of this field's work in it.
    const onSubmit = vi.fn((event: React.FormEvent) => event.preventDefault())
    const user = userEvent.setup()
    render(
      <form onSubmit={onSubmit}>
        <Live />
      </form>,
    )

    await user.click(field())
    await user.type(field(), "zzz")
    expect(screen.queryAllByRole("option")).toHaveLength(0)
    await user.keyboard("{Enter}")

    expect(onSubmit).not.toHaveBeenCalled()
  })

  it("leaves the list closed when a chip is removed", async () => {
    // The chips sit outside the popover, so they are pressable while it is
    // closed. Refocusing the input on every toggle sprang the list open over
    // the chip row, one extra Escape per removal.
    const user = userEvent.setup()
    render(<Live initial={["operator", "alice"]} />)

    await user.click(field())
    await user.keyboard("{Escape}")
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Remove Operator" }))

    expect(within(chips()).getAllByRole("listitem")).toHaveLength(1)
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument()
  })

  it("renders at most `maxVisible` matches and says so", async () => {
    const many = Array.from({ length: 8 }, (_, index) => ({
      id: `user-${index}`,
      label: `person-${index}@example.com`,
    }))
    const user = userEvent.setup()
    render(
      <MultiSelect
        label={LABEL}
        options={many}
        value={[]}
        onChange={() => {}}
        maxVisible={3}
        countNoun={{ one: "person assigned", other: "people assigned" }}
      />,
    )

    await user.click(field())
    expect(screen.getAllByRole("option")).toHaveLength(3)
    // Two clauses, not one nested sentence: what is shown, then what is
    // picked among it.
    expect(screen.getByText(/Showing 3 of 8/)).toBeInTheDocument()
    expect(screen.getByText(/0 of 3 selected here/)).toBeInTheDocument()
  })

  it("names its description on the control, and its error in that line's place", async () => {
    // One rung, the way `Field` does it: the error replaces the description
    // rather than adding a row, so going invalid moves nothing.
    const { rerender } = render(
      <MultiSelect
        label={LABEL}
        description="Everyone selected is held to this budget."
        options={PEOPLE}
        value={[]}
        onChange={() => {}}
      />,
    )

    const described = () => field().getAttribute("aria-describedby") ?? ""
    expect(described().split(" ").filter(Boolean)).toHaveLength(1)
    expect(document.getElementById(described())?.textContent).toContain(
      "held to this budget",
    )

    rerender(
      <MultiSelect
        label={LABEL}
        description="Everyone selected is held to this budget."
        options={PEOPLE}
        value={[]}
        onChange={() => {}}
        isInvalid
        errorMessage="Pick at least one person."
      />,
    )

    expect(described().split(" ").filter(Boolean)).toHaveLength(1)
    expect(document.getElementById(described())?.textContent).toContain(
      "Pick at least one person",
    )
    // Replaced, not joined: the description is gone from the DOM.
    expect(screen.queryByText(/held to this budget/)).toBeNull()
  })

  it("keeps a live region mounted so a change is announced", () => {
    // A region inserted with its content already in it announces nothing, so
    // one that exists only while the popover is open is silent on the first
    // count and absent when a chip is removed from the closed state.
    render(<Live />)
    const live = document.querySelector('[aria-live="polite"]')
    expect(live).not.toBeNull()
    expect(live?.textContent).toBe("0 people assigned")
  })

  it("pluralizes the count noun", async () => {
    const user = userEvent.setup()
    render(<Live />)

    await user.click(field())
    await user.click(screen.getByRole("option", { name: /Operator/ }))

    expect(screen.getAllByText(/1 person assigned/).length).toBeGreaterThan(0)
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
    expect(screen.getByText(/0 of 2 selected here/)).toBeInTheDocument()
    // Singular, which is the whole reason the noun is a pair.
    expect(screen.getAllByText(/1 person assigned/).length).toBeGreaterThan(0)
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
