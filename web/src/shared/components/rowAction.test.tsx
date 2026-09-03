import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"
import { ConfirmRowAction, RowAction, RowActionRow } from "./surface"

/**
 * The rule the docstring spends a paragraph on: danger ink marks the armed
 * state and nothing else. It was got wrong at six sites on the first pass, so
 * it is the one thing here worth a test that fails loudly, and it is only
 * reachable from this component now.
 */
describe("RowAction", () => {
  it("is muted at rest even when it offers a destructive action", () => {
    render(<RowAction onPress={() => undefined}>Remove</RowAction>)
    const action = screen.getByRole("button", { name: "Remove" })
    // Token checks, not substring: a variant like `hover:text-muted` contains
    // "text-muted", so a substring assertion would pass for a control that
    // never carries the ink at rest.
    expect([...action.classList]).toContain("text-muted")
    expect(action.className).not.toContain("text-danger")
  })

  it("takes danger ink only once armed", () => {
    render(
      <RowAction isDanger onPress={() => undefined}>
        Remove
      </RowAction>,
    )
    const action = screen.getByRole("button", { name: "Remove" })
    expect([...action.classList]).toContain("text-danger")
    // Not both: the armed state replaces the resting ink rather than layering
    // over it, so whichever the browser resolved last cannot decide the color.
    expect(action.className).not.toContain("text-muted")
  })

  it("presses", async () => {
    const user = userEvent.setup()
    const onPress = vi.fn()
    render(<RowAction onPress={onPress}>Edit</RowAction>)
    await user.click(screen.getByRole("button", { name: "Edit" }))
    expect(onPress).toHaveBeenCalledTimes(1)
  })

  it("does not press while disabled", async () => {
    const user = userEvent.setup()
    const onPress = vi.fn()
    render(
      <RowAction isDisabled onPress={onPress}>
        Edit
      </RowAction>,
    )
    await user.click(screen.getByRole("button", { name: "Edit" }))
    expect(onPress).not.toHaveBeenCalled()
  })

  it("lets the accessible name carry what the visible label cannot", () => {
    // A disabled control takes no focus, so a tooltip reaches a pointer and
    // nothing else; the row it acts on and the reason it is refused have to be
    // in the name or they are not anywhere.
    render(
      <RowAction
        isDisabled
        ariaLabel="Remove alice@example.com (the last owner cannot be removed)"
        onPress={() => undefined}
      >
        Remove
      </RowAction>,
    )
    expect(
      screen.getByRole("button", {
        name: "Remove alice@example.com (the last owner cannot be removed)",
      }),
    ).toBeDisabled()
    expect(screen.queryByRole("button", { name: "Remove" })).toBeNull()
  })
})

describe("RowActionRow", () => {
  it("sets the 16px pitch the actions repeat within", () => {
    // Named here because it is the constraint on anything that later grows the
    // actions' hit area: a target may not exceed the pitch it repeats within,
    // and this is where that pitch is decided.
    render(
      <RowActionRow>
        <RowAction onPress={() => undefined}>Edit</RowAction>
      </RowActionRow>,
    )
    const lane = screen.getByRole("button", { name: "Edit" })
      .parentElement as HTMLElement
    expect([...lane.classList]).toContain("gap-4")
    expect([...lane.classList]).toContain("justify-end")
  })
})

/**
 * Asking twice, in text rather than in buttons. The cycle is the whole
 * component, and every step of it has a way to be wrong that a reader would not
 * see: a resting Remove already red, a Cancel that confirms, a confirm that
 * fires twice while the first one is still in flight.
 */
describe("ConfirmRowAction", () => {
  const setup = (props: Partial<{ isPending: boolean }> = {}) => {
    const onConfirm = vi.fn()
    render(
      <ConfirmRowAction
        confirmLabel="Confirm remove"
        onConfirm={onConfirm}
        {...props}
      >
        Remove
      </ConfirmRowAction>,
    )
    return { onConfirm, user: userEvent.setup() }
  }

  it("offers one muted action at rest", () => {
    setup()
    const trigger = screen.getByRole("button", { name: "Remove" })
    expect([...trigger.classList]).toContain("text-muted")
    expect(screen.queryByRole("button", { name: "Cancel" })).toBeNull()
  })

  it("does not confirm on the first press", async () => {
    const { onConfirm, user } = setup()
    await user.click(screen.getByRole("button", { name: "Remove" }))
    expect(onConfirm).not.toHaveBeenCalled()
  })

  it("arms into the confirmation and a plain Cancel", async () => {
    const { user } = setup()
    await user.click(screen.getByRole("button", { name: "Remove" }))

    const confirm = screen.getByRole("button", { name: "Confirm remove" })
    expect([...confirm.classList]).toContain("text-danger")
    // Cancel is the ordinary way out and stays muted; two danger controls side
    // by side would make the safe one look like the destructive one.
    const cancel = screen.getByRole("button", { name: "Cancel" })
    expect([...cancel.classList]).toContain("text-muted")
    expect(cancel.className).not.toContain("text-danger")
    // The resting label is gone, so there is no second, unarmed way through.
    expect(screen.queryByRole("button", { name: "Remove" })).toBeNull()
  })

  it("confirms on the second press", async () => {
    const { onConfirm, user } = setup()
    await user.click(screen.getByRole("button", { name: "Remove" }))
    await user.click(screen.getByRole("button", { name: "Confirm remove" }))
    expect(onConfirm).toHaveBeenCalledTimes(1)
  })

  it("disarms on Cancel without confirming", async () => {
    const { onConfirm, user } = setup()
    await user.click(screen.getByRole("button", { name: "Remove" }))
    await user.click(screen.getByRole("button", { name: "Cancel" }))
    expect(onConfirm).not.toHaveBeenCalled()
    expect(screen.getByRole("button", { name: "Remove" })).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Confirm remove" })).toBeNull()
  })

  it("refuses a second confirm while the first is in flight", async () => {
    // The armed state is the one place a double click lands on the same control
    // twice, because the label does not move under the pointer between presses.
    const { onConfirm, user } = setup({ isPending: true })
    await user.click(screen.getByRole("button", { name: "Remove" }))
    const confirm = screen.getByRole("button", { name: "Confirm remove" })
    expect(confirm).toBeDisabled()
    await user.click(confirm)
    expect(onConfirm).not.toHaveBeenCalled()
    // Cancel is held too: backing out mid-request would leave the row saying
    // one thing while the request said another.
    expect(screen.getByRole("button", { name: "Cancel" })).toBeDisabled()
  })
})
