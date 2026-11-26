import { useEffect, useRef } from "react"
import type { CreateKeyResponse } from "@/client"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { KeySecretStep } from "@/features/keys/KeySecretStep"

/**
 * The secret a regenerate produced, in the frame a created key's arrives in.
 * No form step and no "Create another": the key already exists and this is
 * its new secret.
 */
export function RegeneratedSecretDialog({
  title,
  result,
  memberLabels,
  onClose,
}: {
  title: string
  result: CreateKeyResponse
  memberLabels: ReadonlyMap<string, string>
  onClose: () => void
}) {
  const secretRef = useRef<HTMLInputElement | HTMLTextAreaElement | null>(null)
  useEffect(() => {
    secretRef.current?.focus()
  }, [])
  return (
    <FormDialog
      isOpen
      onOpenChange={onClose}
      size="lg"
      // A stray backdrop click here loses the key forever.
      isDismissable={false}
      title={title}
      submitLabel="I’ve saved this key"
      onSubmit={onClose}
      isPending={false}
    >
      <KeySecretStep
        announce={title}
        result={result}
        memberLabels={memberLabels}
        secretRef={secretRef}
      />
    </FormDialog>
  )
}
