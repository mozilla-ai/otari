import { useState } from "react"
import { FiRefreshCw } from "react-icons/fi"

import { Button } from "@/design-system/actions/Button"
import { CatalogUpdateDialog } from "@/features/hostedProviders/CatalogUpdateDialog"
import { useHostedCatalogUpdate } from "@/shared/api/hostedProviders"
import { useDeploymentOperator } from "@/shared/api/organizations"
import { useSurfaces } from "@/shared/hooks/useDeployment"

/**
 * Whether the caller may run the catalog update: a deployment operator on a
 * deployment that serves Hosted providers. Strict rather than `isOperator`,
 * because the update's API answers a non-operator 404, so a failed read must
 * not offer it.
 */
export function useCanUpdateCatalog(): boolean {
  const isSurfaced = useSurfaces()
  const { answer } = useDeploymentOperator()
  return answer === "operator" && isSurfaced("hosted_providers")
}

/**
 * The catalog update, offered beside an organization key's refreshes: the same
 * preview and confirm as on Hosted providers, without the cleanup, which
 * deletes deployment prices and belongs on that page. What it moved shows in
 * the refreshed models below.
 */
export function CatalogUpdateButton() {
  const update = useHostedCatalogUpdate()
  const [isOpen, setIsOpen] = useState(false)
  const [openCount, setOpenCount] = useState(0)
  return (
    <>
      <Button
        size="sm"
        variant="ghost"
        className="min-h-11 md:min-h-9"
        isPending={update.isPending}
        onPress={() => {
          update.reset()
          setOpenCount((n) => n + 1)
          setIsOpen(true)
        }}
      >
        <FiRefreshCw aria-hidden className="size-4" />
        Update catalog
      </Button>
      <CatalogUpdateDialog
        key={openCount}
        opening={openCount}
        isOpen={isOpen}
        onClose={() => setIsOpen(false)}
        update={update}
        offerCleanup={false}
      />
    </>
  )
}
