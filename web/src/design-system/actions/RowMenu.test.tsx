import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { FiEdit2, FiUserMinus } from "react-icons/fi"
import { describe, expect, it, vi } from "vitest"
import { RowMenu } from "@/design-system/actions/RowMenu"

describe("RowMenu", () => {
  it("names a disabled action's reason on the trigger, where a keyboard reaches it", async () => {
    const user = userEvent.setup()
    render(
      <RowMenu
        label="Actions for Wei Zhang"
        actions={[
          {
            id: "edit",
            label: "Edit…",
            hint: "Their role and the models they may call",
            icon: FiEdit2,
            run: vi.fn(),
          },
          {
            id: "remove",
            label: "Remove…",
            hint: "This is the last active owner. Promote someone else first.",
            icon: FiUserMinus,
            isDisabled: true,
            run: vi.fn(),
          },
        ]}
      />,
    )
    await user.tab()
    expect(
      screen.getByRole("button", {
        name: "Actions for Wei Zhang. Remove… This is the last active owner. Promote someone else first.",
      }),
    ).toHaveFocus()
  })

  it("keeps the plain name when nothing is refused", async () => {
    render(
      <RowMenu
        label="Actions for Wei Zhang"
        actions={[{ id: "edit", label: "Edit…", icon: FiEdit2, run: vi.fn() }]}
      />,
    )
    expect(
      screen.getByRole("button", { name: "Actions for Wei Zhang" }),
    ).toBeInTheDocument()
  })

  it("runs the chosen action and hands focus back to its trigger", async () => {
    const user = userEvent.setup()
    const run = vi.fn()
    render(
      <RowMenu
        label="Actions for Wei Zhang"
        actions={[{ id: "edit", label: "Edit…", icon: FiEdit2, run }]}
      />,
    )
    const trigger = screen.getByRole("button", {
      name: "Actions for Wei Zhang",
    })
    await user.click(trigger)
    await user.click(await screen.findByRole("menuitem", { name: "Edit…" }))
    expect(run).toHaveBeenCalledOnce()
    await waitFor(() => expect(trigger).toHaveFocus())
  })

  it("draws nothing when the row has no actions", () => {
    const { container } = render(
      <RowMenu label="Actions for Wei Zhang" actions={[]} />,
    )
    expect(container).toBeEmptyDOMElement()
  })
})
