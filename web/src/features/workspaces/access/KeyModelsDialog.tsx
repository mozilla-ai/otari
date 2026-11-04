import { useState } from "react"

import { ModelSelectionDialog } from "@/features/workspaces/access/ModelSelectionDialog"
import { useModelVendors } from "@/features/workspaces/access/modelVendors"
import {
  useAddWorkspaceProviderKeyModel,
  useRemoveWorkspaceProviderKeyModel,
} from "@/shared/api/workspaces"

/**
 * Which models of one organization key a workspace may call.
 *
 * The key's list is stored a model at a time, so a save adds what was picked
 * and removes what was dropped. Every model is the empty list.
 */
export function KeyModelsDialog({
  workspaceId,
  keyId,
  keyName,
  provider,
  allowed,
  served,
  validateEntry,
  onClose,
}: {
  workspaceId: string
  keyId: string
  keyName: string
  /** The key's provider, or empty for a key whose provider this caller cannot read. */
  provider: string
  allowed: readonly string[]
  /** What the catalog lists for the provider; empty where it lists nothing, and names are typed. */
  served: readonly string[]
  validateEntry: (name: string) => string | undefined
  onClose: () => void
}) {
  const add = useAddWorkspaceProviderKeyModel()
  const remove = useRemoveWorkspaceProviderKeyModel()
  const [isSaving, setIsSaving] = useState(false)
  const [error, setError] = useState<unknown>(null)
  const vendors = useModelVendors(provider, provider !== "")

  const save = async (next: string[] | null) => {
    const wanted = new Set(next ?? [])
    setIsSaving(true)
    setError(null)
    try {
      for (const model of next ?? []) {
        if (!allowed.includes(model)) {
          await add.mutateAsync({ workspaceId, keyId, model })
        }
      }
      for (const model of allowed) {
        if (!wanted.has(model)) {
          await remove.mutateAsync({ workspaceId, keyId, model })
        }
      }
      onClose()
    } catch (caught) {
      setError(caught)
    } finally {
      setIsSaving(false)
    }
  }

  return (
    <ModelSelectionDialog
      isOpen
      onClose={onClose}
      title={`Models on ${keyName}`}
      models={served.length > 0 ? served : null}
      vendors={vendors}
      initial={allowed.length > 0 ? allowed : null}
      isPending={isSaving}
      error={error}
      validateEntry={validateEntry}
      onSave={(next) => void save(next)}
    />
  )
}
