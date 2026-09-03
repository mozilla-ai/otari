import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it } from "vitest"
import { ConfirmRowAction } from "./surface"

/**
 * Where focus goes, which is the half of this control nothing was testing.
 *
 * Arming swaps the trigger for a Confirm/Cancel pair and cancelling swaps them
 * back, and a swap that unmounts the focused element sends focus to `<body>`:
 * a keyboard user loses their place mid-gesture and a screen reader announces
 * nothing, on the most destructive control in every table.
 *
 * Measured rather than assumed, because only half of that is true here. On ARM
 * React reconciles the single trigger against the pair's first child, reuses
 * the same `<button>` node, and focus rides along onto Confirm without anything
 * managing it. On CANCEL focus is on the second button, that node has no
 * counterpart at rest, it unmounts, and focus lands on `<body>`. So the live
 * defect is cancel-only, and arm works by an accident of reconciliation that
 * nothing states. Both are pinned below, the arm one precisely because it is
 * accidental: giving those two actions `key`s would separate the nodes and
 * break it silently.
 *
 * The twelve tests beside this one cover what the component does and all of
 * them pass with focus management deleted, which is how this survived three
 * audits. Asserting the arm/confirm wiring is not the same as asserting where
 * the caret ends up.
 *
 * Scope is arm and cancel. The post-confirm case, where the row unmounts along
 * with the focused Confirm button, is a real gap and a separate one: it needs a
 * ruling on where focus should land that no shared behavior currently answers.
 */
describe("ConfirmRowAction focus", () => {
  const setup = () => {
    render(
      <>
        <button type="button">before</button>
        <ConfirmRowAction confirmLabel="Confirm remove" onConfirm={() => {}}>
          Remove
        </ConfirmRowAction>
      </>,
    )
    return userEvent.setup()
  }

  it("hands focus to Confirm when the action arms", async () => {
    const user = await setup()
    await user.click(screen.getByRole("button", { name: "Remove" }))
    expect(screen.getByRole("button", { name: "Confirm remove" })).toHaveFocus()
  })

  it("reuses the trigger's own node as Confirm, which is why arming keeps focus", async () => {
    // The mechanism behind the assertion above, pinned separately so that if a
    // later edit keys these two actions apart, this fails with the reason
    // rather than leaving the focus test to fail with the symptom.
    const user = await setup()
    const trigger = screen.getByRole("button", { name: "Remove" })
    await user.click(trigger)
    expect(screen.getByRole("button", { name: "Confirm remove" })).toBe(trigger)
  })

  it("hands focus back to the trigger when the action is cancelled", async () => {
    // Backing out has to return the caret to where the gesture started, or the
    // way out of a destructive action costs a keyboard user their place too.
    const user = await setup()
    await user.click(screen.getByRole("button", { name: "Remove" }))
    await user.click(screen.getByRole("button", { name: "Cancel" }))
    expect(screen.getByRole("button", { name: "Remove" })).toHaveFocus()
  })

  it("never lands on the body", async () => {
    // The failure this whole file exists for, asserted directly rather than
    // inferred: at no point in the cycle is `document.body` the active element.
    const user = await setup()
    await user.click(screen.getByRole("button", { name: "Remove" }))
    expect(document.activeElement).not.toBe(document.body)
    await user.click(screen.getByRole("button", { name: "Cancel" }))
    expect(document.activeElement).not.toBe(document.body)
  })

  it("takes no focus on mount, before anything has been armed", async () => {
    // Nothing has been unmounted yet, so there is nothing to restore, and a
    // table of ten of these would otherwise fight over the caret on every
    // render. This is the guard that makes the two assertions above safe to
    // want.
    await setup()
    // Asserted at rest rather than after a click: the effect runs once on
    // mount, so a restore that fired there would grab the caret before anyone
    // had touched the page, and a table of ten of these would fight over it.
    expect(screen.getByRole("button", { name: "Remove" })).not.toHaveFocus()
    expect(document.activeElement).toBe(document.body)
  })
})
