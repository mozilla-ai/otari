import { render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"

import { McpGrantDialog } from "@/features/tools/McpGrantDialog"
import type { GrantUser } from "@/features/tools/mcpGrantSubjects"

const USERS: GrantUser[] = [
  { userId: "u-ada", label: "Ada", keys: [] },
  { userId: "u-bob", label: "Bob", keys: [] },
]

function renderDialog(
  props: Partial<Parameters<typeof McpGrantDialog>[0]> = {},
) {
  const onOpenChange = vi.fn()
  render(
    <McpGrantDialog
      isOpen
      onOpenChange={onOpenChange}
      users={USERS}
      tools={["read_file"]}
      isKnown
      readTools={[]}
      isPending={false}
      error={undefined}
      onSubmit={() => {}}
      {...props}
    />,
  )
  return { onOpenChange }
}

describe("McpGrantDialog", () => {
  it("asks before discarding a swapped pick, even when the count is unchanged", async () => {
    const user = userEvent.setup()
    const { onOpenChange } = renderDialog({ initialUserIds: ["u-ada"] })

    await user.click(
      screen.getByRole("checkbox", { name: "Ada, all their keys" }),
    )
    await user.click(
      screen.getByRole("checkbox", { name: "Bob, all their keys" }),
    )
    await user.click(screen.getByRole("button", { name: "Close" }))

    expect(onOpenChange).not.toHaveBeenCalled()
    expect(screen.getByRole("dialog")).toHaveTextContent("Unsaved changes")
  })

  it("closes without asking when the preselected pick is untouched", async () => {
    const user = userEvent.setup()
    const { onOpenChange } = renderDialog({ initialUserIds: ["u-ada"] })

    await user.click(screen.getByRole("button", { name: "Close" }))

    expect(onOpenChange).toHaveBeenCalledWith(false)
  })

  it("shows why a check started from the dialog could not run", async () => {
    const onDismissCheckError = vi.fn()
    const user = userEvent.setup()
    renderDialog({
      checkError: "The check could not run. Try again.",
      onDismissCheckError,
    })

    const notice = screen.getByRole("status")
    expect(notice).toHaveTextContent("The check could not run. Try again.")
    await user.click(within(notice).getByRole("button", { name: "Dismiss" }))
    expect(onDismissCheckError).toHaveBeenCalled()
  })
})
