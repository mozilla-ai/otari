import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it } from "vitest"

import { moveRowFocus } from "@/features/traces/rowFocus"

function Rows({ names }: { names: string[] }) {
  return (
    <div data-trace-rows="">
      {names.map((name, index) => (
        <button
          key={name}
          type="button"
          data-trace-row=""
          tabIndex={index === 0 ? 0 : -1}
          onKeyDown={moveRowFocus}
        >
          {name}
        </button>
      ))}
    </div>
  )
}

const row = (name: string) => screen.getByRole("button", { name })

describe("moveRowFocus", () => {
  it("moves focus down and up one row at a time", async () => {
    const user = userEvent.setup()
    render(<Rows names={["first", "second", "third"]} />)
    row("first").focus()

    await user.keyboard("{ArrowDown}")
    expect(row("second")).toHaveFocus()
    await user.keyboard("{ArrowUp}")
    expect(row("first")).toHaveFocus()
  })

  it("jumps to either end with Home and End", async () => {
    const user = userEvent.setup()
    render(<Rows names={["first", "second", "third"]} />)
    row("second").focus()

    await user.keyboard("{End}")
    expect(row("third")).toHaveFocus()
    await user.keyboard("{Home}")
    expect(row("first")).toHaveFocus()
  })

  it("stays on the edge row rather than wrapping", async () => {
    const user = userEvent.setup()
    render(<Rows names={["first", "second"]} />)

    row("first").focus()
    await user.keyboard("{ArrowUp}")
    expect(row("first")).toHaveFocus()

    row("second").focus()
    await user.keyboard("{ArrowDown}")
    expect(row("second")).toHaveFocus()
  })

  it("leaves every other key to the row", async () => {
    const user = userEvent.setup()
    render(<Rows names={["first", "second"]} />)
    row("first").focus()

    await user.keyboard("{ArrowRight}")
    expect(row("first")).toHaveFocus()
  })

  it("does nothing for a row outside a row list", async () => {
    const user = userEvent.setup()
    render(
      <button type="button" data-trace-row="" onKeyDown={moveRowFocus}>
        alone
      </button>,
    )
    row("alone").focus()

    await user.keyboard("{ArrowDown}")
    expect(row("alone")).toHaveFocus()
  })
})
