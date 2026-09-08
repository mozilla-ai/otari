import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"
import { EmptyState } from "@/shared/components/feedback/EmptyState"

describe("EmptyState", () => {
  it("renders the title as a heading with its description", () => {
    render(
      <EmptyState
        title="No budgets yet"
        description="Create one to cap spending."
      />,
    )
    expect(
      screen.getByRole("heading", { name: "No budgets yet" }),
    ).toBeInTheDocument()
    expect(screen.getByText("Create one to cap spending.")).toBeInTheDocument()
  })

  it("fires onAction when the call to action is pressed", async () => {
    const user = userEvent.setup()
    const onAction = vi.fn()
    render(
      <EmptyState
        title="No API keys yet"
        actionLabel="Create your first key"
        onAction={onAction}
      />,
    )
    await user.click(
      screen.getByRole("button", { name: "Create your first key" }),
    )
    expect(onAction).toHaveBeenCalledOnce()
  })

  it("omits the action entirely for a purely informational empty state", () => {
    render(
      <EmptyState
        title="No usage yet"
        description="Spend appears here once traffic flows."
      />,
    )
    expect(screen.queryByRole("button")).not.toBeInTheDocument()
  })

  it("disables the call to action and suppresses onAction when isActionDisabled is set", async () => {
    const user = userEvent.setup()
    const onAction = vi.fn()
    render(
      <EmptyState
        title="Welcome"
        actionLabel="Add your first provider"
        onAction={onAction}
        isActionDisabled
      />,
    )
    const button = screen.getByRole("button", {
      name: "Add your first provider",
    })
    expect(button).toBeDisabled()
    await user.click(button)
    expect(onAction).not.toHaveBeenCalled()
  })
})
