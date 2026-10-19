import { useState } from "react"

import type { DeploymentProviderMigrationRow } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import {
  useApplyDeploymentProviderMigration,
  useDeploymentProviderMigrationPreview,
} from "@/shared/api/hostedProviders"

// The report and the banner are independent so a row that failed can be moved
// again once its cause is fixed, without reloading the page.

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

// Copied, but its deployment provider still serves until the copy offers a model.
function isRetained(row: DeploymentProviderMigrationRow) {
  return row.outcome === "migrated" && !row.source_removed
}

function outcomeLabel(row: DeploymentProviderMigrationRow) {
  return isRetained(row)
    ? "Copied, still served from the Deployment providers page"
    : OUTCOME_LABELS[row.outcome]
}

function MigrationRows({ rows }: { rows: DeploymentProviderMigrationRow[] }) {
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
  const [isOpen, setIsOpen] = useState(false)

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

  return (
    <>
      <ErrorBanner error={preview.error} />
      {apply.data ? (
        <InfoBanner tone="info">
          <p role="status">
            Moved {moved} of {report.length} deployment providers.
            {retained > 0
              ? ` ${retained} copied and still served from the Deployment providers page until ${retained === 1 ? "its" : "their"} models are offered here.`
              : ""}
          </p>
          <div className="mt-2">
            <MigrationRows rows={apply.data.rows} />
          </div>
        </InfoBanner>
      ) : null}
      {rows.length > 0 ? (
        <InfoBanner tone="info">
          <p className="flex flex-wrap items-center gap-2">
            <span>
              {rows.length === 1
                ? "1 deployment provider is still stored on the Deployment providers page."
                : `${rows.length} deployment providers are still stored on the Deployment providers page.`}{" "}
              {movable > 0
                ? "Move them here to keep serving them once that page goes."
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
        </InfoBanner>
      ) : null}
      <ConfirmDialog
        isOpen={isOpen}
        onOpenChange={(open) => {
          if (!open) setIsOpen(false)
        }}
        heading="Move deployment providers to hosted providers?"
        body={
          <div className="flex flex-col gap-3">
            <p>
              Each becomes a hosted provider of the same name, so requests
              naming it keep working, and is removed from the Deployment
              providers page once its models are offered here. An organization's
              own key for the provider is now asked first.
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
