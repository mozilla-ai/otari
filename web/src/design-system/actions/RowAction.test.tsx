import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { FiArchive, FiTrash2 } from "react-icons/fi"
import { describe, expect, it, vi } from "vitest"
import { ConfirmRowAction } from "@/design-system/actions/ConfirmRowAction"
import { RowAction, RowActionRow } from "@/design-system/actions/RowAction"

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
    // `text-caption` carries the muted ink itself, which is why the resting
    // variant no longer names a color: repeating it is what
    // `foundation.test.ts` bans.
    expect([...action.classList]).toContain("text-caption")
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

/**
 * The glyph form, whose whole risk is the word going missing with the text. A
 * control with no name is anonymous to a screen reader and unreachable by
 * speech input, so `label` being required is the design and these are what
 * hold it: the name is there, the glyph is not read twice, and the refusal
 * still says why.
 */
describe("RowAction wearing a glyph", () => {
  it("takes its accessible name from the label the glyph replaced", () => {
    render(
      <RowAction icon={FiTrash2} label="Delete" onPress={() => undefined} />,
    )
    const action = screen.getByRole("button", { name: "Delete" })
    // The glyph is decoration once the button is named, so it must not add a
    // second reading of the same word.
    const glyph = action.querySelector("svg")
    expect(glyph).not.toBeNull()
    expect(glyph?.getAttribute("aria-hidden")).toBe("true")
  })

  it("reaches the 44px touch floor a 32px glyph does not", () => {
    // jsdom lays nothing out, so the device is what is asserted: 32px of
    // visual and a 6px bleed each way, which is under half `RowActionRow`'s
    // 16px pitch and so cannot overlap the action beside it.
    render(
      <RowAction icon={FiTrash2} label="Delete" onPress={() => undefined} />,
    )
    const classes = [
      ...screen.getByRole("button", { name: "Delete" }).classList,
    ]
    expect(classes).toContain("size-8")
    expect(classes).toContain("before:-inset-1.5")
  })

  it("lets ariaLabel carry the row and the reason, as the text form does", () => {
    render(
      <RowAction
        icon={FiTrash2}
        label="Delete"
        ariaLabel="Delete Default workspace (an organization keeps one workspace)"
        isDisabled
        onPress={() => undefined}
      />,
    )
    const action = screen.getByRole("button", {
      name: "Delete Default workspace (an organization keeps one workspace)",
    })
    expect(action).toBeDisabled()
    // A disabled control takes neither hover nor focus, so react-aria's
    // tooltip never opens on one: without this the glyph is nameless to a
    // pointer, which is the one reader the accessible name does not reach.
    expect(action).toHaveAttribute(
      "title",
      "Delete Default workspace (an organization keeps one workspace)",
    )
  })

  it("leaves the native tooltip off an action that can be pressed", () => {
    // Two tooltips over one control, the browser's and the product's, open on
    // the same hover and say the same thing.
    render(
      <RowAction icon={FiTrash2} label="Delete" onPress={() => undefined} />,
    )
    expect(screen.getByRole("button", { name: "Delete" })).not.toHaveAttribute(
      "title",
    )
  })

  it("is one control, and the one the tooltip is wired to", async () => {
    // The tooltip's trigger is the button itself rather than a wrapper around
    // it. HeroUI's default trigger is a div it reports as a button, which
    // around a real button doubles both the announcement and the tab stop, and
    // gives every `getByRole("button", { name })` in the e2e suite two matches.
    const { container } = render(
      <RowAction icon={FiTrash2} label="Delete" onPress={() => undefined} />,
    )
    expect(screen.getAllByRole("button")).toHaveLength(1)
    expect(container.firstElementChild?.tagName).toBe("BUTTON")
    expect(container.firstElementChild).toHaveAttribute(
      "data-slot",
      "tooltip-trigger",
    )
    // The trigger's behavior, not its presentation. `.tooltip__trigger` sets
    // `display: inline-block`, which un-centers the glyph, and answers
    // `:focus-visible` with a box-shadow ring after zeroing `outline-style`,
    // which motion-and-access.md forbids outright. Spreading the props
    // wholesale is what puts it back, so its absence is the assertion.
    expect(container.firstElementChild?.className).not.toContain(
      "tooltip__trigger",
    )
  })

  it("presses", async () => {
    const user = userEvent.setup()
    const onPress = vi.fn()
    render(<RowAction icon={FiTrash2} label="Delete" onPress={onPress} />)
    await user.click(screen.getByRole("button", { name: "Delete" }))
    expect(onPress).toHaveBeenCalledTimes(1)
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
    expect([...trigger.classList]).toContain("text-caption")
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
    expect([...cancel.classList]).toContain("text-caption")
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

  it("arms a glyph trigger into the words that name the consequence", async () => {
    // The glyph form of the trigger, which is what a lane of icons uses. The
    // armed half stays text in both forms: it is the step that has to say what
    // the next press costs, and no glyph says it.
    const user = userEvent.setup()
    const onConfirm = vi.fn()
    render(
      <ConfirmRowAction
        icon={FiArchive}
        label="Archive"
        confirmLabel="Archive permanently"
        onConfirm={onConfirm}
      />,
    )
    await user.click(screen.getByRole("button", { name: "Archive" }))
    const confirm = screen.getByRole("button", { name: "Archive permanently" })
    expect([...confirm.classList]).toContain("text-danger")
    expect(confirm.querySelector("svg")).toBeNull()
    expect(screen.getByRole("button", { name: "Cancel" })).toBeInTheDocument()
    await user.click(confirm)
    expect(onConfirm).toHaveBeenCalledTimes(1)
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
