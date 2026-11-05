import { useState } from "react"

import { ModelSelectionDialog } from "@/features/workspaces/access/ModelSelectionDialog"
import { useProviderCatalog } from "@/features/workspaces/access/modelVendors"
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
  validateEntry,
  onClose,
}: {
  workspaceId: string
  keyId: string
  keyName: string
  /** The key's provider, or empty for a key whose provider this caller cannot read. */
  provider: string
  allowed: readonly string[]
  validateEntry: (name: string) => string | undefined
  onClose: () => void
}) {
  const add = useAddWorkspaceProviderKeyModel()
  const remove = useRemoveWorkspaceProviderKeyModel()
  const [isSaving, setIsSaving] = useState(false)
  const [error, setError] = useState<unknown>(null)
  // The provider's models as the catalog lists them, by the bare name the list
  // stores. Where it lists none, the names are typed instead.
  const catalog = useProviderCatalog(provider, provider !== "")

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

  // Held until the catalog answers, so the dialog does not open on a typed
  // list and switch to checkboxes under the caller.
  if (catalog.isPending) return null

  return (
    <ModelSelectionDialog
      isOpen
      onClose={onClose}
      title={`Models on ${keyName}`}
      models={catalog.models.length > 0 ? catalog.models : null}
      vendors={catalog.vendors}
      initial={allowed.length > 0 ? allowed : null}
      isPending={isSaving}
      error={error}
      validateEntry={validateEntry}
      onSave={(next) => void save(next)}
    />
  )
}
