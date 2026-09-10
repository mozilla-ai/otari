import { render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"

import { FormDialog } from "./FormDialog"

const base = {
  title: "New key",
  submitLabel: "Create key",
  isPending: false,
  onSubmit: vi.fn(),
  onOpenChange: vi.fn(),
}

const withField = (
  <label>
    Key name
    <input name="name" />
  </label>
)

/**
 * A textarea, which is where Cmd/Ctrl+Enter is the only submit gesture: Enter
 * on its own is a newline there, and the browser's implicit form submission
 * does not fire from a multi-line control.
 */
const withTextArea = (
  <label>
    Instructions
    <textarea name="instructions" />
  </label>
)

describe("FormDialog", () => {
  it("renders nothing while closed", () => {
    render(
      <FormDialog {...base} isOpen={false}>
        {withField}
      </FormDialog>,
    )
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
  })

  it("is a dialog rather than an alertdialog, because a form is not an alert", () => {
    render(
      <FormDialog {...base} isOpen>
        {withField}
      </FormDialog>,
    )
    expect(screen.getByRole("dialog")).toBeInTheDocument()
    expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument()
  })

  it("names the object in the title and the action on the submit button", () => {
    render(
      <FormDialog {...base} isOpen description="Shown once.">
        {withField}
      </FormDialog>,
    )
    const dialog = screen.getByRole("dialog")
    expect(dialog).toHaveTextContent("New key")
    expect(dialog).toHaveTextContent("Shown once.")
    expect(
      screen.getByRole("button", { name: "Create key" }),
    ).toBeInTheDocument()
  })

  it("gives the close control a name rather than leaving it a glyph", () => {
    render(
      <FormDialog {...base} isOpen>
        {withField}
      </FormDialog>,
    )
    expect(screen.getByRole("button", { name: "Close" })).toBeInTheDocument()
  })

  it("submits when the primary is pressed", async () => {
    const onSubmit = vi.fn()
    const user = userEvent.setup()
    render(
      <FormDialog {...base} isOpen onSubmit={onSubmit}>
        {withField}
      </FormDialog>,
    )
    await user.click(screen.getByRole("button", { name: "Create key" }))
    expect(onSubmit).toHaveBeenCalledTimes(1)
  })

  it("submits on Cmd/Ctrl+Enter from a textarea, where Enter is a newline", async () => {
    const onSubmit = vi.fn()
    const user = userEvent.setup()
    render(
      <FormDialog {...base} isOpen onSubmit={onSubmit}>
        {withTextArea}
      </FormDialog>,
    )
    const field = screen.getByLabelText("Instructions")
    await user.click(field)
    // Enter alone belongs to the control. Asserting this first is what keeps
    // the case below honest: from a single-line input the browser's own
    // implicit submission fires and the assertion passes with the handler
    // deleted, which is a green that proves nothing.
    await user.keyboard("{Enter}")
    expect(onSubmit).not.toHaveBeenCalled()
    expect(field).toHaveValue("\n")
    await user.keyboard("{Meta>}{Enter}{/Meta}")
    expect(onSubmit).toHaveBeenCalledTimes(1)
  })

  it("runs the form's validation on Cmd/Ctrl+Enter rather than going around it", async () => {
    const onSubmit = vi.fn()
    const user = userEvent.setup()
    render(
      <FormDialog {...base} isOpen onSubmit={onSubmit}>
        <label>
          Instructions
          <textarea name="instructions" required />
        </label>
      </FormDialog>,
    )
    await user.click(screen.getByLabelText("Instructions"))
    // Required and empty. The shortcut goes through requestSubmit, so the
    // browser refuses it exactly as it refuses the button; calling onSubmit
    // directly would hand an invalid form to the mutation.
    await user.keyboard("{Meta>}{Enter}{/Meta}")
    expect(onSubmit).not.toHaveBeenCalled()
    await user.keyboard("a")
    await user.keyboard("{Meta>}{Enter}{/Meta}")
    expect(onSubmit).toHaveBeenCalledTimes(1)
  })

  it("submits on Enter from a single-line field, the way every form does", async () => {
    const onSubmit = vi.fn()
    const user = userEvent.setup()
    render(
      <FormDialog {...base} isOpen onSubmit={onSubmit}>
        {withField}
      </FormDialog>,
    )
    await user.click(screen.getByLabelText("Key name"))
    await user.keyboard("{Enter}")
    expect(onSubmit).toHaveBeenCalledTimes(1)
  })

  it("closes on Cancel and on the close control when nothing is dirty", async () => {
    const onOpenChange = vi.fn()
    const user = userEvent.setup()
    render(
      <FormDialog {...base} isOpen onOpenChange={onOpenChange}>
        {withField}
      </FormDialog>,
    )
    await user.click(screen.getByRole("button", { name: "Cancel" }))
    expect(onOpenChange).toHaveBeenLastCalledWith(false)
    await user.click(screen.getByRole("button", { name: "Close" }))
    expect(onOpenChange).toHaveBeenCalledTimes(2)
  })

  describe("while the mutation is in flight", () => {
    it("keeps the submit label in the DOM, so the button keeps its width", () => {
      render(
        <FormDialog {...base} isOpen isPending>
          {withField}
        </FormDialog>,
      )
      // The spinner replaces the label in place rather than beside it: the
      // label element is still laid out, only transparent. A spinner appended
      // next to a removed label is the shape this asserts against, and it
      // would change the button's resting width mid-submit.
      //
      // `opacity-0` and not `invisible` for a second reason this cannot see:
      // `visibility: hidden` takes the label out of the accessibility tree,
      // which would leave the button nameless mid-submit. jsdom loads no
      // stylesheet, so `getByRole` here would find it either way.
      const submit = screen.getByRole("button", { name: /Create key/ })
      const label = within(submit).getByText("Create key")
      expect(label).toHaveClass("opacity-0")
    })

    it("says the submit is busy rather than refused, and blocks its own press", () => {
      render(
        <FormDialog {...base} isOpen isPending>
          {withField}
        </FormDialog>,
      )
      // Disabled is one treatment in this product, at 0.4 opacity, and it has
      // to read as denied. A submit in flight is working, so it keeps its fill
      // and stops the press itself.
      const submit = screen.getByRole("button", { name: /Create key/ })
      expect(submit).not.toBeDisabled()
      expect(submit).not.toHaveAttribute("data-disabled")
      expect(submit).not.toHaveAttribute("aria-disabled", "true")
      expect(submit).toHaveClass("pointer-events-none")
      // On the form, not the button: react-aria filters unrecognized ARIA off
      // a Button, so an `aria-busy` there reaches no DOM node.
      expect(submit.closest("form")).toHaveAttribute("aria-busy", "true")
    })

    it("disables Cancel and the close control, which genuinely are refused", () => {
      render(
        <FormDialog {...base} isOpen isPending>
          {withField}
        </FormDialog>,
      )
      expect(screen.getByRole("button", { name: "Cancel" })).toBeDisabled()
      expect(screen.getByRole("button", { name: "Close" })).toBeDisabled()
    })

    it("does not submit again when the primary is pressed", async () => {
      const onSubmit = vi.fn()
      const user = userEvent.setup()
      render(
        <FormDialog {...base} isOpen isPending onSubmit={onSubmit}>
          {withField}
        </FormDialog>,
      )
      await user.click(screen.getByRole("button", { name: /Create key/ }))
      expect(onSubmit).not.toHaveBeenCalled()
    })
  })

  describe("the dirty guard", () => {
    it("holds the dialog open and swaps the footer instead of closing", async () => {
      const onOpenChange = vi.fn()
      const user = userEvent.setup()
      render(
        <FormDialog {...base} isOpen isDirty onOpenChange={onOpenChange}>
          {withField}
        </FormDialog>,
      )
      await user.click(screen.getByRole("button", { name: "Close" }))
      expect(onOpenChange).not.toHaveBeenCalled()
      expect(screen.getByRole("dialog")).toHaveTextContent("Unsaved changes")
      expect(
        screen.getByRole("button", { name: "Keep editing" }),
      ).toBeInTheDocument()
      expect(
        screen.getByRole("button", { name: "Discard" }),
      ).toBeInTheDocument()
    })

    it("guards inside the one dialog, replacing its actions rather than stacking", async () => {
      const user = userEvent.setup()
      render(
        <FormDialog {...base} isOpen isDirty>
          {withField}
        </FormDialog>,
      )
      await user.click(screen.getByRole("button", { name: "Close" }))
      // The guard's controls are in the dialog that was already open, and the
      // actions they replaced are gone. A second dialog would leave both sets
      // on the page and two dialog nodes in it.
      const dialog = screen.getByRole("dialog")
      expect(screen.getAllByRole("dialog")).toHaveLength(1)
      expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument()
      expect(
        within(dialog).getByRole("button", { name: "Keep editing" }),
      ).toBeInTheDocument()
      expect(
        within(dialog).queryByRole("button", { name: "Cancel" }),
      ).not.toBeInTheDocument()
      expect(
        within(dialog).queryByRole("button", { name: "Create key" }),
      ).not.toBeInTheDocument()
    })

    it("refuses a keyboard submit while it holds the footer", async () => {
      // The guard has taken the footer, so the submit control is not on screen.
      // A keyboard submit under it would run the mutation from a footer whose
      // only actions are Keep editing and Discard, and `isPending` would never
      // become visible because the guard owns the footer for its duration.
      const onSubmit = vi.fn()
      const user = userEvent.setup()
      render(
        <FormDialog {...base} isOpen isDirty onSubmit={onSubmit}>
          {withField}
        </FormDialog>,
      )
      await user.click(screen.getByRole("button", { name: "Close" }))
      expect(screen.getByRole("dialog")).toHaveTextContent("Unsaved changes")

      await user.click(screen.getByLabelText("Key name"))
      await user.keyboard("{Enter}")
      expect(onSubmit).not.toHaveBeenCalled()
      await user.keyboard("{Meta>}{Enter}{/Meta}")
      await user.keyboard("{Control>}{Enter}{/Control}")
      expect(onSubmit).not.toHaveBeenCalled()

      // And it submits again once the guard is put away, so what is blocked is
      // the guarded state rather than the gesture.
      await user.click(screen.getByRole("button", { name: "Keep editing" }))
      await user.click(screen.getByLabelText("Key name"))
      await user.keyboard("{Enter}")
      expect(onSubmit).toHaveBeenCalledTimes(1)
    })

    it("closes on Discard", async () => {
      const onOpenChange = vi.fn()
      const user = userEvent.setup()
      render(
        <FormDialog {...base} isOpen isDirty onOpenChange={onOpenChange}>
          {withField}
        </FormDialog>,
      )
      await user.click(screen.getByRole("button", { name: "Close" }))
      await user.click(screen.getByRole("button", { name: "Discard" }))
      expect(onOpenChange).toHaveBeenCalledWith(false)
    })

    it("returns the footer to its actions on Keep editing", async () => {
      const user = userEvent.setup()
      render(
        <FormDialog {...base} isOpen isDirty>
          {withField}
        </FormDialog>,
      )
      await user.click(screen.getByRole("button", { name: "Close" }))
      await user.click(screen.getByRole("button", { name: "Keep editing" }))
      expect(
        screen.getByRole("button", { name: "Create key" }),
      ).toBeInTheDocument()
      expect(screen.getByRole("dialog")).not.toHaveTextContent(
        "Unsaved changes",
      )
    })
  })

  it("returns a scrolled body to the top when a request fails", () => {
    // The banner mounts at the top of the body, which in a scrolled `lg` body
    // is above the fold: `role="alert"` reaches a screen reader, and a sighted
    // operator watches the submit finish and sees nothing change.
    //
    // jsdom lays nothing out, so `scrollTop` here is only what a test sets. It
    // still answers the question this covers, which is whether the effect
    // writes it, and the assertion fails with the effect removed.
    const { rerender } = render(
      <FormDialog {...base} isOpen size="lg">
        {withField}
      </FormDialog>,
    )
    const body = screen
      .getByRole("dialog")
      .querySelector("form > div") as HTMLElement
    body.scrollTop = 240
    expect(body.scrollTop).toBe(240)

    rerender(
      <FormDialog {...base} isOpen size="lg" error={new Error("Refused.")}>
        {withField}
      </FormDialog>,
    )
    expect(body.scrollTop).toBe(0)
  })

  it("mounts a request failure at the top of the body, above the fields", () => {
    render(
      <FormDialog {...base} isOpen error={new Error("Domain already claimed.")}>
        {withField}
      </FormDialog>,
    )
    const banner = screen.getByRole("alert")
    expect(banner).toHaveTextContent("Domain already claimed.")
    expect(
      banner.compareDocumentPosition(screen.getByLabelText("Key name")),
    ).toBe(Node.DOCUMENT_POSITION_FOLLOWING)
  })

  it("puts the tab row between the header and the body", () => {
    render(
      <FormDialog
        {...base}
        isOpen
        size="lg"
        tabs={<button type="button">Known</button>}
      >
        {withField}
      </FormDialog>,
    )
    const tab = screen.getByRole("button", { name: "Known" })
    expect(tab.compareDocumentPosition(screen.getByLabelText("Key name"))).toBe(
      Node.DOCUMENT_POSITION_FOLLOWING,
    )
  })
})
