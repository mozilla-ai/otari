import { useState } from "react"

import { Button } from "@/design-system/actions/Button"
import { Section } from "@/design-system/layout/Section"
import { useSessionCaller } from "@/shared/api/auth"

import { AvatarFace } from "./AvatarFace"
import { AvatarPickerDialog } from "./AvatarPickerDialog"
import { saveAvatarChoice, useAvatar } from "./avatarChoice"
import { face } from "./faces"

/**
 * The face drawn for you on the account control. Everybody starts with one
 * picked at random; this is where you choose another.
 */
export function AvatarCard() {
  const userId = useSessionCaller()?.user_id
  const current = useAvatar(userId)
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
        The face drawn beside your name at the foot of the sidebar. You started
        with one at random; pick any other here. Kept in this browser for now.
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
      {isOpen && userId && current !== null ? (
        <AvatarPickerDialog
          isOpen
          onOpenChange={setIsOpen}
          initial={current}
          onSave={(id) => saveAvatarChoice(userId, id)}
        />
      ) : null}
    </Section>
  )
}
