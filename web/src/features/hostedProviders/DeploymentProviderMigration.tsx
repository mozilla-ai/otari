import { useState } from "react"

import type { DeploymentProviderMigrationRow } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import {
  useApplyDeploymentProviderMigration,
  useDeploymentProviderMigrationPreview,
  useDiscardDeploymentProvider,
} from "@/shared/api/hostedProviders"

// Startup moves every stored provider it can, so the preview lists only the
// ones it could not, each with the reason. The report and the banner are
// independent so a row can be moved again once its cause is fixed, without
// reloading the page.

const OUTCOME_LABELS: Record<
  DeploymentProviderMigrationRow["outcome"],
  string
> = {
  migrate: "Moves here",
  migrated: "Moved",
  conflict: "Not moved: name taken",
  unreadable: "Not moved: key unreadable",
  invalid: "Not moved",
  failed: "Not moved: failed",
}

// Copied, but its original entry was kept for review: it listed no models, or
// changed during the move. Its notes say which.
function isRetained(row: DeploymentProviderMigrationRow) {
  return row.outcome === "migrated" && !row.source_removed
}

function outcomeLabel(row: DeploymentProviderMigrationRow) {
  return isRetained(row)
    ? "Copied, original entry kept"
    : OUTCOME_LABELS[row.outcome]
}

// A row nothing will move: it serves nothing, so the operator may discard it
// once its cause is settled on this page or in config.yml.
function isDiscardable(row: DeploymentProviderMigrationRow) {
  return row.outcome === "migrated"
    ? !row.source_removed
    : row.outcome !== "migrate"
}

function MigrationRows({
  rows,
  onDiscard,
}: {
  rows: DeploymentProviderMigrationRow[]
  /** Offers Discard on each row that cannot move; absent, none is offered. */
  onDiscard?: (instance: string) => void
}) {
  return (
    <ul className="flex max-h-72 flex-col gap-2 overflow-y-auto">
      {rows.map((row) => (
        <li key={row.instance} className="flex flex-col gap-0.5">
          <span className="flex flex-wrap items-baseline gap-2">
            <code>{row.instance}</code>
            {row.provider && row.provider !== row.instance ? (
              <span className="text-caption">({row.provider})</span>
            ) : null}
            <span
              className={
                (row.outcome === "migrate" || row.outcome === "migrated") &&
                !isRetained(row)
                  ? "text-caption text-success"
                  : "text-caption text-warning"
              }
            >
              {outcomeLabel(row)}
              {row.outcome === "migrated" && row.models !== null
                ? `, ${row.models} ${row.models === 1 ? "model" : "models"}`
                : ""}
            </span>
            {onDiscard && isDiscardable(row) ? (
              <Button
                size="sm"
                variant="ghost"
                aria-label={`Discard ${row.instance}`}
                onPress={() => onDiscard(row.instance)}
              >
                Discard
              </Button>
            ) : null}
          </span>
          {row.detail ? (
            <span className="text-caption">{row.detail}</span>
          ) : null}
          {(row.notes ?? []).map((note) => (
            <span key={note} className="text-caption">
              {note}
            </span>
          ))}
        </li>
      ))}
    </ul>
  )
}

export function DeploymentProviderMigration() {
  const preview = useDeploymentProviderMigrationPreview(true)
  const apply = useApplyDeploymentProviderMigration()
  const discard = useDiscardDeploymentProvider()
  const [isOpen, setIsOpen] = useState(false)
  const [discarding, setDiscarding] = useState("")
  const [discarded, setDiscarded] = useState<string[]>([])

  const rows = preview.data?.rows ?? []
  const reviewed = rows
    .filter((row) => row.outcome === "migrate")
    .map((row) => ({
      instance: row.instance,
      updated_at: row.updated_at ?? null,
    }))
  const movable = reviewed.length
  // Read again on opening, and confirmed only against that read, so the rows
  // sent are the rows on screen rather than ones another operator has since
  // changed.
  const isPreviewCurrent = preview.isSuccess && !preview.isFetching
  const report = apply.data?.rows ?? []
  const moved = report.filter((row) => row.source_removed).length
  const retained = report.filter(isRetained).length
  const openDiscard = (instance: string) => {
    discard.reset()
    setDiscarding(instance)
  }

  return (
    <>
      <ErrorBanner error={preview.error} />
      {apply.data ? (
        <InfoBanner tone="info">
          <p role="status">
            Moved {moved} of {report.length} stored providers.
            {retained > 0
              ? ` ${retained} copied, with the original kept for review; each says why.`
              : ""}
          </p>
          <div className="mt-2">
            <MigrationRows
              rows={apply.data.rows.filter(
                (row) => !discarded.includes(row.instance),
              )}
              onDiscard={openDiscard}
            />
          </div>
        </InfoBanner>
      ) : null}
      {rows.length > 0 ? (
        <InfoBanner tone="info">
          <p className="flex flex-wrap items-center gap-2">
            <span>
              {rows.length === 1
                ? "1 stored provider could not be moved here at startup."
                : `${rows.length} stored providers could not be moved here at startup.`}{" "}
              {movable > 0
                ? "Review the reasons, then move the ones that can move now."
                : "None can move yet; review why."}
            </span>
            <Button
              size="sm"
              onPress={() => {
                void preview.refetch()
                setIsOpen(true)
              }}
            >
              Review the move
            </Button>
          </p>
          <div className="mt-2">
            <MigrationRows rows={rows} onDiscard={openDiscard} />
          </div>
        </InfoBanner>
      ) : null}
      <ConfirmDialog
        isOpen={discarding !== ""}
        onOpenChange={(open) => {
          if (!open) setDiscarding("")
        }}
        heading={`Discard ${discarding}?`}
        body={
          <p>
            Removes the stored provider for good. It serves nothing now, so no
            request changes.
          </p>
        }
        confirmLabel="Discard"
        isPending={discard.isPending}
        error={discard.error}
        onConfirm={() =>
          discard.mutate(discarding, {
            onSuccess: () => {
              setDiscarded((current) => [...current, discarding])
              setDiscarding("")
            },
          })
        }
      />
      <ConfirmDialog
        isOpen={isOpen}
        onOpenChange={(open) => {
          if (!open) setIsOpen(false)
        }}
        heading="Move stored providers to hosted providers?"
        body={
          <div className="flex flex-col gap-3">
            <p>
              Each becomes a hosted provider of the same name that serves any
              model a request names, so requests naming it keep working, and its
              original entry is removed. An organization's own key for the
              provider is now asked first.
            </p>
            <MigrationRows rows={rows} />
          </div>
        }
        confirmLabel={`Move ${movable} ${movable === 1 ? "provider" : "providers"}`}
        confirmVariant="primary"
        isConfirmDisabled={movable === 0 || !isPreviewCurrent}
        isPending={apply.isPending}
        // A failed re-read holds the confirm, so its reason shows here too.
        error={apply.error ?? preview.error}
        onConfirm={() => {
          if (!isPreviewCurrent) return
          apply.mutate(reviewed, { onSuccess: () => setIsOpen(false) })
        }}
      />
    </>
  )
}
