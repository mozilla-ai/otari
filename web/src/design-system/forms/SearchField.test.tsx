import { act, render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { startTransition, useState } from "react"
import { describe, expect, it } from "vitest"

import { SearchField } from "./SearchField"

// A caller that commits what was typed the way the router commits a search
// param: in a transition, after the keystroke's own event has finished.
function TransitionCaller({ initial = "" }: { initial?: string }) {
  const [value, setValue] = useState(initial)
  return (
    <>
      <SearchField
        label="Search"
        value={value}
        onChange={(next) => startTransition(() => setValue(next))}
      />
      <button type="button" onClick={() => setValue("")}>
        Reset elsewhere
      </button>
      <output aria-label="Committed">{value}</output>
    </>
  )
}

describe("SearchField", () => {
  it("keeps the caret where it was when the caller commits later", async () => {
    render(<TransitionCaller />)
    const user = userEvent.setup()
    const field = screen.getByRole("searchbox", { name: "Search" })

    await user.type(field, "deepseek")
    await user.keyboard("{ArrowLeft>4/}XY")

    expect(field).toHaveValue("deepXYseek")
    expect(screen.getByLabelText("Committed")).toHaveTextContent("deepXYseek")
  })

  it("empties when the caller clears the value from elsewhere", async () => {
    render(<TransitionCaller initial="deepseek" />)
    const user = userEvent.setup()
    const field = screen.getByRole("searchbox", { name: "Search" })
    expect(field).toHaveValue("deepseek")

    await user.click(screen.getByRole("button", { name: "Reset elsewhere" }))

    expect(field).toHaveValue("")
  })

  it("empties from its own clear button and reports the empty value", async () => {
    render(<TransitionCaller initial="deepseek" />)
    const user = userEvent.setup()

    await user.click(screen.getByRole("button", { name: "Close" }))

    expect(screen.getByRole("searchbox", { name: "Search" })).toHaveValue("")
    await act(async () => {})
    expect(screen.getByLabelText("Committed")).toHaveTextContent("")
  })
})
