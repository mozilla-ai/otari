/**
 * A key's secret, shown once: where the Create API key dialog ends, and in
 * the dialog a regenerated secret arrives in.
 */

import type { RefObject } from "react"
import { useState } from "react"
import type { CreateKeyResponse } from "@/client"
import {
  CopyField,
  concealedFingerprint,
} from "@/design-system/actions/CopyField"
import { keyLabel } from "@/features/keys/keyList"
import { secretCaption } from "@/features/keys/secretCaption"

export function KeySecretStep({
  announce,
  result,
  memberLabels,
  secretRef,
}: {
  /** The dialog's title, repeated as the alert's own name. */
  announce: string
  result: CreateKeyResponse
  memberLabels: ReadonlyMap<string, string>
  secretRef: RefObject<HTMLInputElement | HTMLTextAreaElement | null>
}) {
  const secret = result.key
  const concealedSecret = concealedFingerprint(
    result.key_prefix ?? undefined,
    result.key_suffix ?? undefined,
  )
  const [isSecretRevealed, setIsSecretRevealed] = useState(false)

  return (
    // `alert`: this is the one message that is gone forever if it is missed.
    <div role="alert" aria-label={announce} className="flex flex-col gap-4">
      <p className="text-sm text-muted">
        Copy this key now. For security it is shown only once and cannot be
        retrieved later. If you lose it, use Regenerate to issue a new secret.
      </p>
      <div className="flex flex-col gap-1">
        <p className="truncate text-emphasis">{keyLabel(result)}</p>
        <CopyField
          label="Secret key"
          value={secret}
          concealed={concealedSecret}
          isRevealed={isSecretRevealed}
          onRevealChange={setIsSecretRevealed}
          fieldRef={secretRef}
        />
        <p className="text-caption">{secretCaption(result, memberLabels)}</p>
      </div>
    </div>
  )
}
