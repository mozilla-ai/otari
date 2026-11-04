import type { WorkspaceHostedProvider } from "@/client"
import { ModelSelectionDialog } from "@/features/workspaces/access/ModelSelectionDialog"
import { useProviderCatalog } from "@/features/workspaces/access/modelVendors"
import { useSetWorkspaceHostedProvider } from "@/shared/api/workspaces"

/** Which models of one hosted provider a workspace may call: one stored rule, saved at once. */
export function HostedModelsDialog({
  workspaceId,
  row,
  onClose,
}: {
  workspaceId: string
  row: WorkspaceHostedProvider
  onClose: () => void
}) {
  const save = useSetWorkspaceHostedProvider()
  const { vendors } = useProviderCatalog(row.provider, true)
  return (
    <ModelSelectionDialog
      isOpen
      onClose={onClose}
      title={`Models on ${row.provider}`}
      models={row.served_models}
      vendors={vendors}
      initial={row.models}
      isPending={save.isPending}
      error={save.error}
      onSave={(models) =>
        save.mutate(
          {
            workspaceId,
            provider: row.provider,
            body: { enabled: row.enabled, models },
          },
          { onSuccess: onClose },
        )
      }
    />
  )
}
