import { useState } from "react"

import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { Checkbox } from "@/design-system/forms/Checkbox"
import { useDeletePricing } from "@/shared/api/pricing"
import { useDeleteSearchTool } from "@/shared/api/tools"

import type { WebToolRow } from "./webTools"

// Removing a tool leaves two things keyed by its name: its price, and its place
// in the search tool access lists of API keys and users. A later tool of the
// same name would inherit both. The price is offered for deletion here, ticked;
// the access lists are only named, since nothing here can edit them.
export function RemoveWebToolDialog({
  tool,
  isDefault,
  onClose,
}: {
  tool: WebToolRow
  /** Whether a default setting names the tool. */
  isDefault: boolean
  onClose: () => void
}) {
  const remove = useDeleteSearchTool()
  const deletePrice = useDeletePricing()
  // Not ticked for a tool that overrides config.yml's: the entry it gives way
  // to has the same name, so the same price.
  const [alsoPrice, setAlsoPrice] = useState(!tool.overridesConfig)

  const confirm = async () => {
    try {
      await remove.mutateAsync(tool.name)
      if (alsoPrice) {
        await deletePrice.mutateAsync({ kind: tool.kind, modelKey: tool.name })
      }
      onClose()
    } catch {
      // Shown in the dialog through the mutations' errors.
    }
  }

  return (
    <ConfirmDialog
      isOpen
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
      heading={`Remove ${tool.name}?`}
      body={
        <div className="flex flex-col gap-3">
          <p>
            {tool.name} and its stored key are removed
            {tool.kind === "search"
              ? ", and a call that names it is refused"
              : ""}
            .
            {isDefault
              ? ` It is the ${tool.kind === "search" ? "model's search" : "fetch"} default, so the default goes back to ${tool.kind === "search" ? "automatic" : "builtin_fetch"}.`
              : ""}
            {tool.overridesConfig
              ? " The config.yml tool of the same name serves again."
              : ""}
          </p>
          {tool.kind === "search" ? (
            <p>
              API keys and users that list {tool.name} in their search tool
              access keep the name, so a new tool named {tool.name} would be
              open to them.
            </p>
          ) : null}
          <Checkbox
            isSelected={alsoPrice}
            onChange={setAlsoPrice}
            isDisabled={remove.isPending || deletePrice.isPending}
          >
            <span className="flex flex-col gap-0.5">
              <span>Also delete its price</span>
              <span className="text-caption">
                Requests already billed keep what they cost.
              </span>
            </span>
          </Checkbox>
        </div>
      }
      confirmLabel="Remove permanently"
      isPending={remove.isPending || deletePrice.isPending}
      error={remove.error ?? deletePrice.error}
      onConfirm={() => void confirm()}
    />
  )
}
