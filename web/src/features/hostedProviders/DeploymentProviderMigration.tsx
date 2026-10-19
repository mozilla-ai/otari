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

// Moving the deployment providers stored through the dashboard onto hosted
// providers of the same name, so `instance:model` keeps working. The preview is
// read whenever the page is open, because it reads the stored rows only; the
// banner appears only while there is something to move. Applying asks each
// provider what it serves, as adding one does, so the confirm can take a while.
// The last move's report stays beside the banner, which the refreshed preview
// keeps offering while any deployment provider is left, so one that failed can
// be moved again once its cause is fixed.

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
                row.outcome === "migrate" || row.outcome === "migrated"
                  ? "text-caption text-success"
                  : "text-caption text-warning"
              }
            >
              {OUTCOME_LABELS[row.outcome]}
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
  const movable = rows.filter((row) => row.outcome === "migrate").length

  return (
    <>
      <ErrorBanner error={preview.error} />
      {apply.data ? (
        <InfoBanner tone="info">
          <p role="status">
            Moved{" "}
            {apply.data.rows.filter((row) => row.outcome === "migrated").length}{" "}
            of {apply.data.rows.length} deployment providers.
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
            <Button size="sm" onPress={() => setIsOpen(true)}>
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
        isConfirmDisabled={movable === 0}
        isPending={apply.isPending}
        error={apply.error}
        onConfirm={() =>
          apply.mutate(undefined, { onSuccess: () => setIsOpen(false) })
        }
      />
    </>
  )
}
