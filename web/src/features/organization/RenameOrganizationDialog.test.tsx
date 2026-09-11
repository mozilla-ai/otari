import { render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { useState } from "react"
import { describe, expect, it, vi } from "vitest"

import {
  RenameOrganizationDialog,
  type RenameOrganizationDialogProps,
} from "@/features/organization/RenameOrganizationDialog"

function renderDialog(overrides: Partial<RenameOrganizationDialogProps> = {}) {
  const onSubmit = vi.fn()
  render(
    <RenameOrganizationDialog
      isOpen
      onOpenChange={() => {}}
      currentName="Default Organization"
      isPending={false}
      onSubmit={onSubmit}
      {...overrides}
    />,
  )
  return { onSubmit }
}

function dialog() {
  return within(screen.getByRole("dialog"))
}

function newNameField() {
  return dialog().getByRole("textbox", { name: /New name/ })
}

describe("RenameOrganizationDialog", () => {
  it("shows the name being replaced alongside the field that replaces it", () => {
    renderDialog()

    expect(dialog().getByText("Current name")).toBeInTheDocument()
    expect(dialog().getByText("Default Organization")).toBeInTheDocument()
    expect(newNameField()).toHaveValue("Default Organization")
  })

  it("submits the trimmed name", async () => {
    const user = userEvent.setup()
    const { onSubmit } = renderDialog()

    await user.clear(newNameField())
    await user.type(newNameField(), "  Platform  ")
    await user.click(
      dialog().getByRole("button", { name: "Change organization name" }),
    )

    expect(onSubmit).toHaveBeenCalledWith("Platform")
  })

  it("refuses a name that is unchanged or blank", async () => {
    const user = userEvent.setup()
    renderDialog()

    const confirm = dialog().getByRole("button", {
      name: "Change organization name",
    })
    expect(confirm).toBeDisabled()

    await user.clear(newNameField())
    expect(confirm).toBeDisabled()

    // Whitespace is not a name, and trimming is what the submit sends.
    await user.type(newNameField(), "   ")
    expect(confirm).toBeDisabled()

    await user.clear(newNameField())
    await user.type(newNameField(), "Platform")
    expect(confirm).toBeEnabled()
  })

  it("keeps Cancel out of reach while the rename is in flight", () => {
    renderDialog({ isPending: true })

    expect(dialog().getByRole("button", { name: "Cancel" })).toBeDisabled()
  })

  it("reports a rejected rename inside the dialog", () => {
    renderDialog({ error: new Error("Name already taken") })

    expect(dialog().getByRole("alert")).toHaveTextContent("Name already taken")
  })

  it("starts each open from the name as it stands, not from the abandoned draft", async () => {
    // The dialog seeds on mount and the caller remounts it on each open, which
    // is what clears an abandoned draft: on the way in, so the fields survive
    // the closing animation. The harness keys it the way the page does.
    const user = userEvent.setup()

    function Harness() {
      const [isOpen, setIsOpen] = useState(true)
      const [openCount, setOpenCount] = useState(0)
      return (
        <>
          <button
            type="button"
            onClick={() => {
              setOpenCount((count) => count + 1)
              setIsOpen(true)
            }}
          >
            Reopen
          </button>
          <RenameOrganizationDialog
            key={openCount}
            isOpen={isOpen}
            onOpenChange={setIsOpen}
            currentName="Default Organization"
            isPending={false}
            onSubmit={() => {}}
          />
        </>
      )
    }
    render(<Harness />)

    await user.clear(newNameField())
    await user.type(newNameField(), "Abandoned")
    // A changed name is dirty, so leaving goes through the guard.
    await user.click(dialog().getByRole("button", { name: "Cancel" }))
    await user.click(screen.getByRole("button", { name: "Discard" }))
    await user.click(screen.getByRole("button", { name: "Reopen" }))

    expect(newNameField()).toHaveValue("Default Organization")
  })
})
