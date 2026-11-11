import { useState } from "react"

import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { Checkbox } from "@/design-system/forms/Checkbox"
import {
  type useHostedCatalogUpdate,
  useHostedCatalogUpdatePreview,
} from "@/shared/api/hostedProviders"

import {
  catalogPreviewOutcome,
  catalogRefreshFailures,
  modelsPreviewOutcome,
  pricePreviewOutcome,
} from "./refreshOutcome"

/**
 * What a catalog update is about to do: the price changes, the models each
 * provider newly lists, and, when the cleanup is ticked, the models it removes,
 * named before anything is removed.
 */
function CatalogUpdatePreview({
  preview,
  cleanup,
  onCleanupChange,
}: {
  preview: ReturnType<typeof useHostedCatalogUpdatePreview>
  cleanup: boolean
  onCleanupChange: (cleanup: boolean) => void
}) {
  const body = () => {
    if (preview.error) {
      // The confirm refuses to fire in this state, so the banner is the answer.
      return <ErrorBanner error={preview.error} />
    }
    if (preview.isPending || preview.isFetching) {
      return <p>Asking models.dev and every provider what changed…</p>
    }
    const { catalog, prices } = preview.data
    const failures = catalogRefreshFailures(catalog)
    return (
      <>
        <p>{pricePreviewOutcome(preview.data)}</p>
        {prices?.needs_review ? (
          <p className="text-caption text-warning">
            Unusual: {prices.review_reason}. A scheduled update would wait for a
            person.
          </p>
        ) : null}
        <p>{modelsPreviewOutcome(catalog)}</p>
        {failures.map((failure) => (
          <p key={failure.provider} className="text-caption">
            {failure.message}
          </p>
        ))}
        {cleanup ? (
          <>
            <p>{catalogPreviewOutcome(catalog)}</p>
            {catalog.removed.length > 0 ? (
              <ul className="max-h-48 overflow-y-auto font-mono text-caption">
                {catalog.removed.map((modelKey) => (
                  <li key={modelKey}>{modelKey}</li>
                ))}
              </ul>
            ) : null}
            {catalog.removed_count > catalog.removed.length ? (
              <p className="text-caption">
                … and {catalog.removed_count - catalog.removed.length} more.
              </p>
            ) : null}
            {catalog.kept.map((group) => (
              <p key={group.reason} className="text-caption">
                Keeping {group.count} {group.count === 1 ? "model" : "models"}:{" "}
                {group.reason}.
              </p>
            ))}
          </>
        ) : null}
      </>
    )
  }
  return (
    <div className="flex flex-col gap-3">
      {body()}
      <Checkbox isSelected={cleanup} onChange={onCleanupChange}>
        Also remove the prices of models no provider serves
      </Checkbox>
      {cleanup ? (
        <p className="text-caption">
          Removing a model takes its rate history and the overrides of
          organizations with no key of their own for the provider. Requests
          already billed keep the cost they were billed at.
        </p>
      ) : null}
    </div>
  )
}

/**
 * The catalog update's confirm: a preview of what it would do, then the update.
 *
 * `opening` counts the openings, and the caller remounts this per opening
 * (`key={opening}`), so each one asks afresh and starts with the cleanup off.
 * The caller owns `update`, so it can report what the update did after this
 * closes.
 */
export function CatalogUpdateDialog({
  opening,
  isOpen,
  onClose,
  update,
}: {
  opening: number
  isOpen: boolean
  onClose: () => void
  update: ReturnType<typeof useHostedCatalogUpdate>
}) {
  const [cleanup, setCleanup] = useState(false)
  // Asked only while the dialog is open, because answering dials every
  // configured provider, and asked afresh on every opening.
  const preview = useHostedCatalogUpdatePreview(isOpen ? opening : 0, cleanup)
  // The confirm waits for a preview this opening read successfully and shows:
  // never a previous opening's, a failed or an in-flight one, because the
  // update applies the prices that preview named.
  const isPreviewShown = preview.isSuccess && !preview.isFetching

  return (
    <ConfirmDialog
      isOpen={isOpen}
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
      heading="Update the catalog?"
      body={
        <CatalogUpdatePreview
          preview={preview}
          cleanup={cleanup}
          onCleanupChange={setCleanup}
        />
      }
      confirmLabel={cleanup ? "Update and remove" : "Update catalog"}
      // Danger only once it deletes: without the cleanup it only adds and reprices.
      confirmVariant={cleanup ? "danger" : "primary"}
      // The update's own pending state only: the preview dials every provider,
      // and folding it in would leave Cancel dead for that whole fan-out.
      isPending={update.isPending}
      isConfirmDisabled={!isPreviewShown}
      error={update.error}
      onConfirm={() => {
        // Nothing is confirmed against a preview nobody has seen.
        if (!isPreviewShown) return
        update.mutate(
          { digest: preview.data.prices?.digest ?? null, cleanup },
          { onSuccess: onClose },
        )
      }}
    />
  )
}
