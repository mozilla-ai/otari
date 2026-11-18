import { useState } from "react"

import { Button } from "@/design-system/actions/Button"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { Section } from "@/design-system/layout/Section"
import { useUpdateAvatar } from "@/shared/api/auth"

import { AvatarFace } from "./AvatarFace"
import { AvatarPickerDialog } from "./AvatarPickerDialog"
import { useOwnAvatar } from "./avatarChoice"
import { face } from "./faces"

/**
 * The face drawn for you on the account control and in every member list of
 * your organization. Everybody starts with one picked at random; this is where
 * you choose another.
 */
export function AvatarCard() {
  const current = useOwnAvatar()
  const update = useUpdateAvatar()
  const [isOpen, setIsOpen] = useState(false)

  return (
    <Section
      aria-labelledby="account-avatar-title"
      className="border-t border-border pt-6 pb-5"
      contentClassName="flex flex-col gap-4"
    >
      <h2 id="account-avatar-title" className="text-title">
        Avatar
      </h2>
      <p className="max-w-3xl text-sm text-muted">
        The face drawn beside your name in the sidebar and wherever your
        organization lists its members. You started with one at random; pick any
        other here.
      </p>
      <div className="flex flex-wrap items-center gap-3">
        {current === null ? (
          <span aria-hidden className="size-10 bg-surface-alt" />
        ) : (
          <AvatarFace id={current} className="size-10" />
        )}
        <Button
          isDisabled={current === null}
          onPress={() => setIsOpen(true)}
          aria-label={
            current === null
              ? "Change avatar"
              : `Change avatar, currently ${face(current).label}`
          }
        >
          Change
        </Button>
      </div>
      <ErrorBanner error={update.error} />
      {isOpen && current !== null ? (
        <AvatarPickerDialog
          isOpen
          onOpenChange={setIsOpen}
          initial={current}
          onSave={(id) => update.mutate(id)}
        />
      ) : null}
    </Section>
  )
}
