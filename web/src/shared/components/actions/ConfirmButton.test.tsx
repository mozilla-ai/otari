import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"
import { ConfirmButton } from "@/shared/components/actions/ConfirmButton"

describe("ConfirmButton", () => {
  it("hands focus to the confirm and back on cancel", async () => {
    const user = userEvent.setup()
    render(
      <ConfirmButton confirmLabel="Delete permanently" onConfirm={() => {}}>
        Delete
      </ConfirmButton>,
    )

    const trigger = screen.getByRole("button", { name: "Delete" })
    await user.click(trigger)

    // Arming unmounts the trigger, which drops focus to <body> unless the pair
    // that replaces it takes it: this is the delete button of a table row.
    await waitFor(() =>
      expect(document.activeElement).toBe(
        screen.getByRole("button", { name: "Delete permanently" }),
      ),
    )

    await user.click(screen.getByRole("button", { name: "Cancel" }))
    await waitFor(() =>
      expect(document.activeElement).toBe(
        screen.getByRole("button", { name: "Delete" }),
      ),
    )
  })

  it("leaves focus alone until it is armed", () => {
    render(
      <>
        <button type="button">elsewhere</button>
        <ConfirmButton confirmLabel="Delete permanently" onConfirm={() => {}}>
          Delete
        </ConfirmButton>
      </>,
    )
    const elsewhere = screen.getByRole("button", { name: "elsewhere" })
    elsewhere.focus()

    // Mounting a row of these must not pull focus out of whatever the operator
    // was doing.
    expect(document.activeElement).toBe(elsewhere)
  })

  it("confirms through the second press", async () => {
    const onConfirm = vi.fn()
    const user = userEvent.setup()
    render(
      <ConfirmButton confirmLabel="Delete permanently" onConfirm={onConfirm}>
        Delete
      </ConfirmButton>,
    )

    await user.click(screen.getByRole("button", { name: "Delete" }))
    expect(onConfirm).not.toHaveBeenCalled()
    await user.click(screen.getByRole("button", { name: "Delete permanently" }))
    expect(onConfirm).toHaveBeenCalledTimes(1)
  })
})
