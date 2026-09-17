import { useState } from "react"
import { FiActivity, FiEdit2, FiTrash2 } from "react-icons/fi"

import type { StoredGuardrail } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { RowAction, RowActionRow } from "@/design-system/actions/RowAction"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { Toggle } from "@/design-system/forms/Toggle"
import { Badge } from "@/design-system/indicators/Badge"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import { isDeploymentOperator } from "@/features/organization/roles"
import { GuardrailDefinitionDialog } from "@/features/tools/GuardrailDefinitionDialog"
import { GuardrailTestDialog } from "@/features/tools/GuardrailTestDialog"
import {
  findGuardrail,
  operationLabel,
} from "@/features/tools/guardrailOperations"
import { useOrganizationContext } from "@/shared/api/organizations"
import {
  useBuiltInGuardrailCatalog,
  useDeleteGuardrailDefinition,
  useGuardrailDefinitions,
  useUpdateGuardrailDefinition,
} from "@/shared/api/tools"
import { docsSourceHref } from "@/shared/helpers/docs"

// The guardrails this deployment defines, each built and run inside the gateway
// rather than sent to a separate service. An enabled row checks every request
// from the workspaces it covers, whether the caller asked for it or not, and its
// name is also what a caller may put in a guardrail entry's `profile` field.
//
// Shaped as the providers page is, and for the same reason: this is a list of
// credentialed things an operator adds, edits and removes, not a set of
// settings. Add opens a dialog over the page, the table is the destination, and
// a credential is encrypted at rest and never returned.

/** What a row is doing to traffic right now, and the switch that stops it. */
function StatusCell({ guardrail }: { guardrail: StoredGuardrail }) {
  const update = useUpdateGuardrailDefinition()
  // "Running" would no longer say enough: a row can run and refuse nothing.
  const state = !guardrail.enabled
    ? "Paused"
    : guardrail.mode === "block"
      ? "Blocking"
      : "Monitoring"
  return (
    <div className="flex items-center gap-3">
      <Toggle
        isSelected={guardrail.enabled}
        isDisabled={update.isPending}
        label={`Run ${guardrail.name}`}
        onChange={(next) =>
          update.mutate({
            name: guardrail.name,
            body: { enabled: next, expected_updated_at: guardrail.updated_at },
          })
        }
      />
      <span className="text-caption text-subtle">{state}</span>
    </div>
  )
}

/** How many workspaces a definition covers, in as few words as say it. */
function scopeLabel(guardrail: StoredGuardrail): string {
  if (guardrail.applies_to_all_workspaces) return "Every workspace"
  const count = guardrail.workspace_ids?.length ?? 0
  if (count === 0) return "No workspaces"
  return count === 1 ? "1 workspace" : `${count} workspaces`
}

export function GuardrailsPage() {
  const context = useOrganizationContext()
  const isOperator = isDeploymentOperator(context.data)
  // Operator-only, as every route behind this page is, so the reads are held
  // until the role is known rather than firing and catching the 403.
  const catalog = useBuiltInGuardrailCatalog(isOperator)
  const stored = useGuardrailDefinitions(isOperator)
  const remove = useDeleteGuardrailDefinition()

  const [adding, setAdding] = useState(false)
  const [openCount, setOpenCount] = useState(0)
  const [editing, setEditing] = useState<string | undefined>()
  const [testing, setTesting] = useState<string | undefined>()
  const [pendingDelete, setPendingDelete] = useState<string | undefined>()

  const known = catalog.data?.guardrails ?? []
  const rows = stored.data ?? []
  // The row's own copy is what a dialog seeds from, so it has to be the fresh
  // one after a save rather than the object the button was pressed with.
  const editingRow = rows.find((row) => row.name === editing)
  const testingRow = rows.find((row) => row.name === testing)
  const loading = context.isPending || stored.isLoading

  const columns: DataTableColumn<StoredGuardrail>[] = [
    {
      id: "name",
      header: "Profile",
      isRowHeader: true,
      cell: (row) => (
        <div className="flex flex-col gap-0.5">
          <code className="font-mono text-xs">{row.name}</code>
          {row.decryptable ? null : (
            <span className="text-caption text-warning">
              Credentials unreadable: check OTARI_SECRET_KEY
            </span>
          )}
          {/* Enabled and not built means its checks do not run, which is the one
              state that would otherwise be invisible: the row reads as healthy
              while traffic goes past it unchecked. */}
          {row.enabled && !row.loaded && row.decryptable ? (
            <span className="text-caption text-warning">
              Failed to build: it is checking nothing. See the gateway log.
            </span>
          ) : null}
        </div>
      ),
    },
    {
      id: "guardrail",
      header: "Guardrail",
      cell: (row) => {
        // A row whose class this build no longer ships still reads: it falls
        // back to the wire name rather than to a blank cell.
        const spec = findGuardrail(known, row.guardrail_name)
        return (
          <div className="flex flex-col gap-0.5">
            <span>{spec?.display_name ?? row.guardrail_name}</span>
            <span className="text-caption text-subtle">
              {spec?.vendor ?? "Not in this build"}
            </span>
          </div>
        )
      },
    },
    {
      id: "checks",
      header: "Checks",
      cell: (row) => {
        const spec = findGuardrail(known, row.guardrail_name)
        if (!spec) return <span className="text-caption text-subtle">—</span>
        // Every operation it detects, not only its headline one, which is the
        // same rule the picker that chose it follows.
        return (
          <div className="flex flex-wrap gap-1">
            {spec.categories.map((category) => (
              <Badge key={category} tone="muted">
                {operationLabel(category)}
              </Badge>
            ))}
          </div>
        )
      },
    },
    {
      id: "scope",
      header: "Where",
      cell: (row) => (
        <span className="text-caption text-subtle">{scopeLabel(row)}</span>
      ),
    },
    {
      id: "status",
      header: "Status",
      cell: (row) => <StatusCell guardrail={row} />,
    },
    {
      id: "actions",
      header: "Actions",
      align: "end",
      cell: (row) => (
        <RowActionRow>
          <RowAction
            icon={FiActivity}
            label="Test"
            ariaLabel={`Test ${row.name}`}
            // A row whose credentials cannot be read cannot be built, so there
            // is nothing to run; Edit still recovers it.
            isDisabled={!row.decryptable}
            onPress={() => setTesting(row.name)}
          />
          <RowAction
            icon={FiEdit2}
            label="Edit"
            ariaLabel={`Edit ${row.name}`}
            onPress={() => {
              setAdding(false)
              setEditing(row.name)
            }}
          />
          <RowAction
            icon={FiTrash2}
            label="Remove"
            ariaLabel={`Remove ${row.name}`}
            onPress={() => setPendingDelete(row.name)}
          />
        </RowActionRow>
      ),
    },
  ]

  return (
    <div className="flex flex-col gap-6">
      <PageIntro
        title="Guardrails"
        docsHref={docsSourceHref("guardrails.md")}
        action={
          // Absent rather than disabled on a build that ships none: a disabled
          // control has to carry its reason, and there is no action to explain.
          isOperator && known.length > 0 ? (
            <Button
              variant="primary"
              onPress={() => {
                setEditing(undefined)
                setOpenCount((count) => count + 1)
                setAdding(true)
              }}
            >
              Add guardrail
            </Button>
          ) : null
        }
      >
        Add checks that run on your LLM requests: prompt injection, personal
        data, harmful content and more. An enabled guardrail checks every
        request from the workspaces you choose, without the caller asking for
        it. Credentials are encrypted at rest.
      </PageIntro>

      <ErrorBanner error={context.error ?? catalog.error ?? stored.error} />

      {!isOperator && !context.isPending ? (
        <InfoBanner tone="info">
          Guardrails are configured by the deployment operator, because a
          definition stores a vendor credential for the whole deployment.
        </InfoBanner>
      ) : null}

      {/* Keyed on the open count, so each open remounts a blank form. Clearing
          the draft on close instead would blank the fields while the dialog is
          still animating away. */}
      {adding ? (
        <GuardrailDefinitionDialog
          key={openCount}
          isOpen={adding}
          onClose={() => setAdding(false)}
          onCreated={() => setAdding(false)}
          guardrails={known}
          takenNames={rows.map((row) => row.name)}
        />
      ) : null}
      {editingRow ? (
        // Keyed on the row, so one row's draft cannot reach another's dialog.
        <GuardrailDefinitionDialog
          key={editingRow.name}
          isOpen
          onClose={() => setEditing(undefined)}
          guardrails={known}
          takenNames={[]}
          editing={editingRow}
        />
      ) : null}
      {testingRow ? (
        <GuardrailTestDialog
          key={testingRow.name}
          isOpen
          onClose={() => setTesting(undefined)}
          guardrail={testingRow}
        />
      ) : null}

      {isOperator ? (
        <TableScrollFrame className="otari-guardrails-table">
          <DataTable
            ariaLabel="Guardrails"
            columns={columns}
            rows={rows}
            getRowKey={(row) => row.name}
            isLoading={loading}
            emptyContent={
              known.length === 0
                ? "This build ships no guardrails it can run itself."
                : "No guardrails yet. Add one to check traffic without running a second service."
            }
          />
        </TableScrollFrame>
      ) : null}

      <ConfirmDialog
        isOpen={pendingDelete !== undefined}
        // Cleared on the way out: a refusal otherwise sits on the mutation and
        // greets the next row's confirm as if that row had failed.
        onOpenChange={(open) => {
          if (open) return
          setPendingDelete(undefined)
          remove.reset()
        }}
        heading="Remove guardrail"
        body={
          pendingDelete
            ? `${pendingDelete} and the credentials stored with it are removed, and the requests it was checking stop being checked. A request that still names it as a profile is refused, so update any callers that use it.`
            : null
        }
        confirmLabel="Remove permanently"
        isPending={remove.isPending}
        error={remove.error}
        onConfirm={() => {
          if (!pendingDelete) return
          remove.mutate(pendingDelete, {
            onSuccess: () => setPendingDelete(undefined),
          })
        }}
      />
    </div>
  )
}
