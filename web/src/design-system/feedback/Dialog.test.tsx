import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"

import { Button } from "../actions/Button"
import { Dialog } from "./Dialog"

function open(props: Partial<React.ComponentProps<typeof Dialog>> = {}) {
  const onOpenChange = vi.fn()
  render(
    <Dialog
      isOpen
      onOpenChange={onOpenChange}
      title="Send your first request"
      {...props}
    >
      <p>the body</p>
    </Dialog>,
  )
  return onOpenChange
}

describe("Dialog", () => {
  it("names itself by its title and describes itself by its description", async () => {
    // What a screen reader announces on open is the dialog, so the sentence
    // under the title has to be the dialog's description rather than a
    // paragraph that happens to sit there.
    open({ description: "It lands in Default workspace." })

    const dialog = await screen.findByRole("dialog", {
      name: "Send your first request",
    })
    expect(dialog).toHaveAccessibleDescription("It lands in Default workspace.")
  })

  it("closes from the close control", async () => {
    const user = userEvent.setup()
    const onOpenChange = open()

    await user.click(await screen.findByRole("button", { name: "Close" }))

    expect(onOpenChange).toHaveBeenCalledWith(false)
  })

  it("offers no close control when it cannot be dismissed", async () => {
    // A control that refuses is worse than none: the footer's action is the
    // way out of a frame whose content cannot be recovered.
    open({
      isDismissable: false,
      actions: <Button>I have copied it</Button>,
    })

    await screen.findByRole("dialog")
    expect(screen.queryByRole("button", { name: "Close" })).toBeNull()
  })

  it("renders no footer when it is given no controls and no caption", async () => {
    open()

    const dialog = await screen.findByRole("dialog")
    expect(dialog.querySelector("footer")).toBeNull()
  })

  it("puts the caption and the controls in one footer", async () => {
    open({
      footerStart: <p>Skipping keeps the key.</p>,
      actions: <Button>Skip</Button>,
    })

    await screen.findByRole("dialog")
    expect(screen.getByText("Skipping keeps the key.")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Skip" })).toBeInTheDocument()
  })

  it("drops the close control in the centered shape, and shows the mark", async () => {
    // A close control in the corner of a celebration is the only asymmetry on
    // the screen, and the footer already carries the way out.
    open({
      align: "center",
      mark: <span data-testid="mark" />,
      actions: <Button>Dismiss</Button>,
    })

    await screen.findByRole("dialog")
    expect(screen.getByTestId("mark")).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Close" })).toBeNull()
  })

  it("wears the announcement type role only when asked", async () => {
    // `text-display-sub` is reserved for one thing per page, so an ordinary
    // frame must not take it.
    const { unmount } = render(
      <Dialog isOpen onOpenChange={() => {}} title="Ordinary">
        <p>body</p>
      </Dialog>,
    )
    // By role: the hidden trigger slot carries the same string, so a text
    // query matches it as well as the heading.
    expect(
      await screen.findByRole("heading", { name: "Ordinary" }),
    ).toHaveClass("text-heading")
    unmount()

    render(
      <Dialog isOpen onOpenChange={() => {}} isAnnouncement title="First run">
        <p>body</p>
      </Dialog>,
    )
    expect(
      await screen.findByRole("heading", { name: "First run" }),
    ).toHaveClass("text-display-sub")
  })
})
