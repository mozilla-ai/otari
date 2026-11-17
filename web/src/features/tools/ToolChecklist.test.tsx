import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { useState } from "react"
import { describe, expect, it, vi } from "vitest"

import { ToolChecklist } from "@/features/tools/ToolChecklist"

const TOOLS = ["create_issue", "create_pr", "list_issues"]

function Harness({
  initial = [],
  isKnown = true,
  allowPatterns = true,
  onCheck,
  isDisabled,
}: {
  initial?: string[]
  isKnown?: boolean
  allowPatterns?: boolean
  onCheck?: () => void
  isDisabled?: boolean
}) {
  const [selected, setSelected] = useState<string[]>(initial)
  return (
    <>
      <ToolChecklist
        label="Rule tools"
        tools={isKnown ? TOOLS : []}
        descriptions={{ create_issue: "Open a new issue." }}
        isKnown={isKnown}
        selected={selected}
        onChange={setSelected}
        allowPatterns={allowPatterns}
        onCheck={onCheck}
        isDisabled={isDisabled}
      />
      <output aria-label="Selected">{selected.join(",")}</output>
    </>
  )
}

const selected = () => screen.getByLabelText("Selected").textContent

describe("ToolChecklist", () => {
  it("searches a tool named like an object's own machinery without crashing", async () => {
    const user = userEvent.setup()
    render(
      <ToolChecklist
        label="Rule tools"
        tools={["constructor", "toString"]}
        isKnown
        selected={[]}
        onChange={() => {}}
      />,
    )

    await user.type(screen.getByRole("searchbox"), "con")
    const list = await screen.findByRole("list", { name: "Rule tools" })
    expect(within(list).getAllByRole("listitem")).toHaveLength(1)
    expect(within(list).getByText("constructor")).toBeVisible()
  })

  it("lists the checked tools with their descriptions, and selects all or none", async () => {
    const user = userEvent.setup()
    render(<Harness />)

    const list = screen.getByRole("list", { name: "Rule tools" })
    expect(within(list).getByText("Open a new issue.")).toBeVisible()
    await user.click(screen.getByRole("button", { name: "Select all" }))
    expect(selected()).toBe(TOOLS.join(","))
    expect(screen.getByText("3 of 3 tools selected")).toBeVisible()
    await user.click(screen.getByRole("button", { name: "Select none" }))
    expect(selected()).toBe("")
  })

  it("narrows the list by search, and Select all takes only what it shows", async () => {
    const user = userEvent.setup()
    render(<Harness />)

    await user.type(screen.getByRole("searchbox"), "create")
    await waitFor(() =>
      expect(
        within(screen.getByRole("list", { name: "Rule tools" })).queryByText(
          "list_issues",
        ),
      ).not.toBeInTheDocument(),
    )
    await user.click(screen.getByRole("button", { name: "Select all" }))
    expect(selected()).toBe("create_issue,create_pr")
  })

  it("takes a typed pattern, says how many tools it matches, and flags one that matches none", async () => {
    const user = userEvent.setup()
    render(<Harness initial={["delete_*"]} />)

    const typed = screen.getByRole("list", {
      name: "Rule tools: typed entries",
    })
    expect(within(typed).getByText("Matches no known tool")).toBeVisible()

    await user.type(
      screen.getByLabelText("Add a tool name or pattern"),
      "create_*",
    )
    expect(screen.getByText("Matches 2 of the server's 3 tools.")).toBeVisible()
    await user.click(screen.getByRole("button", { name: "Add" }))
    expect(selected()).toBe("delete_*,create_*")
    expect(within(typed).getByText("Matches 2 known tools")).toBeVisible()
  })

  it("refuses a pattern where only names go", async () => {
    const user = userEvent.setup()
    render(<Harness allowPatterns={false} />)

    await user.type(screen.getByLabelText("Add a tool name"), "create_*")
    expect(screen.getByText("Only whole tool names go here.")).toBeVisible()
    expect(screen.getByRole("button", { name: "Add" })).toBeDisabled()
  })

  it("asks for a check before there is anything to pick from, and still takes typed names", async () => {
    const onCheck = vi.fn()
    const user = userEvent.setup()
    render(<Harness isKnown={false} onCheck={onCheck} />)

    expect(
      screen.getByText(/Check the server to pick from its tools/),
    ).toBeVisible()
    await user.click(screen.getByRole("button", { name: "Check" }))
    expect(onCheck).toHaveBeenCalled()
    await user.type(
      screen.getByLabelText("Add a tool name or pattern"),
      "list_*",
    )
    await user.click(screen.getByRole("button", { name: "Add" }))
    expect(selected()).toBe("list_*")
  })

  it("starts no check while the checklist is disabled", () => {
    render(<Harness isKnown={false} onCheck={vi.fn()} isDisabled />)

    expect(screen.getByRole("button", { name: "Check" })).toBeDisabled()
  })
})
