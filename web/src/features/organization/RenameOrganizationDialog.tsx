import { useEffect, useState } from "react"

import { Button } from "@/design-system/actions/Button"
import { Dialog } from "@/design-system/feedback/Dialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { Field } from "@/design-system/forms/Field"

export interface RenameOrganizationDialogProps {
  isOpen: boolean
  onOpenChange: (isOpen: boolean) => void
  /** The name being replaced, kept in front of the caller while they type. */
  currentName: string
  isPending: boolean
  /** A rejected attempt, reported inside the dialog. */
  error?: unknown
  /** Receives the trimmed name; the dialog never submits an empty one. */
  onSubmit: (name: string) => void
}

/**
 * The rename, behind a dialog rather than an input sitting open on the page.
 *
 * The organization's name is read by every screen that names it, and a box open
 * on the settings page is one stray keystroke from changing it. The dialog is
 * the deliberate step; showing the current name beside the new one is the other
 * half, because a guard that does not say what is being replaced only slows the
 * mistake down.
 */
export function RenameOrganizationDialog({
  isOpen,
  onOpenChange,
  currentName,
  isPending,
  error,
  onSubmit,
}: RenameOrganizationDialogProps) {
  const [draft, setDraft] = useState(currentName)

  // The dialog stays mounted across close and reopen, so the draft is reseeded
  // each time it opens: an abandoned edit, or a switch to another organization
  // while this was shut, must not be what the next confirm sends.
  useEffect(() => {
    if (isOpen) setDraft(currentName)
  }, [isOpen, currentName])

  const trimmed = draft.trim()
  const isUnchanged = trimmed === currentName

  return (
    <Dialog
      isOpen={isOpen}
      onOpenChange={onOpenChange}
      heading="Change organization name"
      footer={
        <>
          <Button isDisabled={isPending} onPress={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            variant="primary"
            isDisabled={isUnchanged || trimmed === ""}
            isPending={isPending}
            onPress={() => onSubmit(trimmed)}
          >
            Change name
          </Button>
        </>
      }
    >
      <dl className="flex flex-col gap-1">
        <dt className="text-muted">Current name</dt>
        <dd className="text-emphasis">{currentName}</dd>
      </dl>
      <Field
        label="New name"
        value={draft}
        onChange={setDraft}
        isRequired
        autoFocus
        description="What this deployment's tenant is called across the dashboard. The slug does not follow a rename."
        reserveMessage
      />
      <ErrorBanner error={error} />
    </Dialog>
  )
}
