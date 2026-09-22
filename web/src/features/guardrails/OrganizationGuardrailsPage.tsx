import { useState } from "react"
import { FiEdit2, FiPause, FiPlay, FiTrash2 } from "react-icons/fi"

import type {
  OrganizationGuardrail,
  OrganizationGuardrailDefinition,
} from "@/client"
import { Button } from "@/design-system/actions/Button"
import { RowAction, RowActionRow } from "@/design-system/actions/RowAction"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import {
  countServedUnchecked,
  mandateConsequence,
} from "@/features/guardrails/buildState"
import { DefinitionDialog } from "@/features/guardrails/DefinitionDialog"
import {
  DefinitionStatus,
  MandateConsequenceMark,
  ServedUncheckedBanner,
} from "@/features/guardrails/GuardrailStatus"
import {
  credentialsLabel,
  definitionChecks,
  guardrailLabel,
  ifItCantRunLabel,
  mandatesOn,
  runsOnLabel,
} from "@/features/guardrails/guardrailRows"
import { MandateDialog } from "@/features/guardrails/MandateDialog"
import { scopeLabel } from "@/features/guardrails/WorkspaceScope"
import { canManage } from "@/features/organization/roles"
import {
  useBuiltInGuardrailCatalog,
  useDeleteOrganizationGuardrailDefinition,
  useOrganizationGuardrailDefinitions,
} from "@/shared/api/guardrails"
import { useOrganizationContext } from "@/shared/api/organizations"
import {
  useDeleteOrganizationGuardrail,
  useGuardrailProfiles,
  useOrganizationGuardrails,
  useUpdateOrganizationGuardrail,
} from "@/shared/api/tools"
import { useWorkspaces } from "@/shared/api/workspaces"

// The organization's guardrails: what each check is, and where each one runs.
//
// Two tables because a definition and a mandate have different keys: one
// definition can be mandated under several profiles, and a mandate may run on
// a service instead of a definition. Definitions come first, since nothing can
// be mandated onto one before it exists.
//
// Not the workspace rail's `/tools/guardrails`, which sets the deployment's own
// guardrails service. Rows here are keyed on the organization, which is why a
// hosted deployment publishes this page too.

type Open = { kind: "" } | { kind: "definition" | "mandate"; editingId: string }

export function OrganizationGuardrailsPage() {
  const context = useOrganizationContext()
  // Gates the reads as well as the writes: these rows name the endpoints this
  // gateway connects to and which of them carry a credential, and the server
  // refuses a member the list.
  const canEdit = canManage(context.data)
  const definitions = useOrganizationGuardrailDefinitions(canEdit)
  const mandates = useOrganizationGuardrails(canEdit)
  const builtInCatalog = useBuiltInGuardrailCatalog(canEdit)
  const remoteCatalog = useGuardrailProfiles(canEdit)
  const workspaces = useWorkspaces()
  const removeDefinition = useDeleteOrganizationGuardrailDefinition()
  const updateMandate = useUpdateOrganizationGuardrail()
  const removeMandate = useDeleteOrganizationGuardrail()

  const [open, setOpen] = useState<Open>({ kind: "" })
  // Bumped per open and used as the dialog's key, so each open starts from the
  // stored row or from blank and nothing is cleared in front of the reader.
  const [openCount, setOpenCount] = useState(0)
  const [pendingDefinitionDelete, setPendingDefinitionDelete] =
    useState<OrganizationGuardrailDefinition>()
  const [pendingMandateDelete, setPendingMandateDelete] =
    useState<OrganizationGuardrail>()

  const openDialog = (kind: "definition" | "mandate", editingId = "") => {
    setOpenCount((count) => count + 1)
    setOpen({ kind, editingId })
  }
  const close = () => setOpen({ kind: "" })

  const defined = definitions.data ?? []
  const entries = mandates.data ?? []
  const known = workspaces.data ?? []
  const editingDefinition =
    open.kind === "definition"
      ? defined.find((row) => row.id === open.editingId)
      : undefined
  const editingMandate =
    open.kind === "mandate"
      ? entries.find((row) => row.id === open.editingId)
      : undefined
  const blockingDelete = pendingDefinitionDelete
    ? mandatesOn(pendingDefinitionDelete, entries)
    : []

  const definitionColumns: DataTableColumn<OrganizationGuardrailDefinition>[] =
    [
      {
        id: "name",
        header: "Name",
        isRowHeader: true,
        cell: (row) => (
          <span className="font-medium text-foreground">{row.name}</span>
        ),
      },
      {
        id: "guardrail",
        header: "Guardrail",
        cell: (row) => (
          <span className="text-muted">
            {guardrailLabel(row, builtInCatalog.data)}
          </span>
        ),
      },
      {
        id: "checks",
        header: "Checks",
        cell: (row) => (
          <span className="text-muted">
            {definitionChecks(row, builtInCatalog.data)}
          </span>
        ),
      },
      {
        id: "credentials",
        header: "Credentials",
        cell: (row) => (
          <span className="text-muted">{credentialsLabel(row)}</span>
        ),
      },
      {
        id: "status",
        header: "Status",
        cell: (row) => <DefinitionStatus definition={row} />,
      },
      {
        id: "actions",
        header: "Actions",
        align: "end",
        cell: (row) => (
          <RowActionRow>
            <RowAction
              icon={FiEdit2}
              label="Edit"
              ariaLabel={`Edit ${row.name}`}
              onPress={() => openDialog("definition", row.id)}
            />
            <RowAction
              icon={FiTrash2}
              label="Remove"
              ariaLabel={`Remove ${row.name}`}
              onPress={() => setPendingDefinitionDelete(row)}
            />
          </RowActionRow>
        ),
      },
    ]

  const mandateColumns: DataTableColumn<OrganizationGuardrail>[] = [
    {
      id: "profile",
      header: "Profile",
      isRowHeader: true,
      // What a mandate on a broken guardrail is doing to requests stays on the
      // row: Pause and Resume say only whether the mandate is on.
      cell: (row) => (
        <div className="flex flex-col gap-1">
          <code className="font-mono text-foreground">{row.profile}</code>
          <MandateConsequenceMark
            consequence={mandateConsequence(row, defined)}
          />
        </div>
      ),
    },
    {
      id: "runs_on",
      header: "Runs on",
      cell: (row) => (
        <span className="text-muted">{runsOnLabel(row, defined)}</span>
      ),
    },
    {
      id: "mode",
      header: "Mode",
      cell: (row) => (
        <span className="text-muted">
          {row.mode === "block" ? "Block" : "Monitor"}
        </span>
      ),
    },
    {
      // In the table, not only behind Edit: it decides whether a failure is
      // silent.
      id: "on_unavailable",
      header: "If it can't run",
      cell: (row) => (
        <span className="text-muted">{ifItCantRunLabel(row)}</span>
      ),
    },
    {
      id: "runs_in",
      header: "Runs in",
      cell: (row) => (
        <span className="text-muted">{scopeLabel(row, known)}</span>
      ),
    },
    {
      id: "actions",
      header: "Actions",
      align: "end",
      cell: (row) => (
        <RowActionRow>
          {/* First, and the row's status as well as its control: Pause on a
              running mandate, Resume on a paused one. */}
          <RowAction
            icon={row.enabled ? FiPause : FiPlay}
            label={row.enabled ? "Pause" : "Resume"}
            ariaLabel={`${row.enabled ? "Pause" : "Resume"} ${row.profile}`}
            // Only the row being written, so one in-flight press does not
            // freeze the rest of the table.
            isDisabled={
              updateMandate.isPending &&
              updateMandate.variables?.guardrailId === row.id
            }
            onPress={() =>
              updateMandate.mutate({
                guardrailId: row.id,
                body: { enabled: !row.enabled },
              })
            }
          />
          <RowAction
            icon={FiEdit2}
            label="Edit"
            ariaLabel={`Edit ${row.profile}`}
            onPress={() => openDialog("mandate", row.id)}
          />
          <RowAction
            icon={FiTrash2}
            label="Remove"
            ariaLabel={`Remove ${row.profile}`}
            onPress={() => setPendingMandateDelete(row)}
          />
        </RowActionRow>
      ),
    },
  ]

  const isLoading = context.isPending || definitions.isLoading

  return (
    <div className="flex flex-col">
      <PageIntro
        title="Guardrails"
        action={
          canEdit ? (
            <div className="flex flex-wrap gap-2">
              <Button onPress={() => openDialog("definition")}>
                Set up guardrail
              </Button>
              <Button variant="primary" onPress={() => openDialog("mandate")}>
                Mandate a guardrail
              </Button>
            </div>
          ) : null
        }
      >
        Checks that run on every request from the workspaces you choose, whether
        the caller asked for them or not. Set up a guardrail for Otari to run
        itself, then say where it runs. A guardrail can only make fewer requests
        succeed.
      </PageIntro>

      <ErrorBanner
        error={
          context.error ??
          definitions.error ??
          mandates.error ??
          workspaces.error ??
          updateMandate.error
        }
      />

      {/* Held back until the role has answered, so an admin's first paint does
          not tell them they may not do this. */}
      {context.data && !canEdit ? (
        <InfoBanner>
          Organization guardrails are set by an owner or admin of the
          organization.
        </InfoBanner>
      ) : null}

      {canEdit ? (
        <ServedUncheckedBanner count={countServedUnchecked(entries, defined)} />
      ) : null}

      {canEdit || context.isPending ? (
        <div className="flex flex-col gap-8">
          <section className="flex flex-col gap-3">
            <h2 className="text-title">Guardrails you have configured</h2>
            <TableScrollFrame className="otari-guardrail-definitions-table">
              <DataTable
                ariaLabel="Guardrails you have configured"
                columns={definitionColumns}
                rows={defined}
                getRowKey={(row) => row.id}
                isLoading={isLoading}
                emptyContent="None yet. Set up a guardrail to have Otari run a vendor's check itself."
              />
            </TableScrollFrame>
          </section>
          <section className="flex flex-col gap-3">
            <h2 className="text-title">Where they run</h2>
            <TableScrollFrame className="otari-guardrail-mandates-table">
              <DataTable
                ariaLabel="Where they run"
                columns={mandateColumns}
                rows={entries}
                getRowKey={(row) => row.id}
                isLoading={context.isPending || mandates.isLoading}
                emptyContent="Nothing mandated, so only the guardrails a caller asks for run."
              />
            </TableScrollFrame>
          </section>
        </div>
      ) : null}

      {canEdit ? (
        <>
          <DefinitionDialog
            key={`definition-${openCount}`}
            isOpen={open.kind === "definition"}
            onClose={close}
            definition={editingDefinition}
            catalog={builtInCatalog.data}
            isCatalogPending={!builtInCatalog.isFetched}
            takenNames={defined.map((row) => row.name)}
            onSaved={() => {}}
          />
          <MandateDialog
            key={`mandate-${openCount}`}
            isOpen={open.kind === "mandate"}
            onClose={close}
            mandate={editingMandate}
            definitions={defined}
            // `isFetched` rather than `isPending`: an errored read returns to
            // pending when its observers remount.
            isDefinitionsSettled={definitions.isFetched}
            builtInCatalog={builtInCatalog.data}
            remoteCatalog={remoteCatalog.data}
            isRemoteCatalogPending={!remoteCatalog.isFetched}
            workspaces={known}
            onSetUpDefinition={() => openDialog("definition")}
            onSaved={() => {}}
          />
        </>
      ) : null}

      <ConfirmDialog
        isOpen={pendingDefinitionDelete !== undefined}
        onOpenChange={(isOpen) => {
          if (isOpen) return
          setPendingDefinitionDelete(undefined)
          removeDefinition.reset()
        }}
        heading="Remove guardrail"
        // Said before the click: the link is RESTRICT, so a mandated definition
        // is refused, and the server's 409 stays the answer for a mandate added
        // since this page read the list.
        body={
          pendingDefinitionDelete
            ? blockingDelete.length > 0
              ? `${pendingDefinitionDelete.name} is still mandated by ${blockingDelete.map((row) => row.profile).join(", ")}, so it cannot be removed. Stop mandating it first, or switch it off to stop it everywhere at once.`
              : `${pendingDefinitionDelete.name} and the credentials it holds are removed for good. Nothing mandates it, so no request changes.`
            : null
        }
        confirmLabel="Remove permanently"
        isPending={removeDefinition.isPending}
        error={removeDefinition.error}
        onConfirm={() => {
          if (!pendingDefinitionDelete) return
          removeDefinition.mutate(pendingDefinitionDelete.id, {
            onSuccess: () => setPendingDefinitionDelete(undefined),
          })
        }}
      />

      <ConfirmDialog
        isOpen={pendingMandateDelete !== undefined}
        onOpenChange={(isOpen) => {
          if (isOpen) return
          setPendingMandateDelete(undefined)
          removeMandate.reset()
        }}
        heading="Remove mandate"
        // The stored mode: this describes what is in force, and a monitoring
        // guardrail never blocked anything.
        body={
          pendingMandateDelete
            ? pendingMandateDelete.mode === "block"
              ? `${pendingMandateDelete.profile} stops running on every request it covers${pendingMandateDelete.has_credential ? ", and its stored credential is removed with it" : ""}. Requests it would have blocked are served.`
              : `${pendingMandateDelete.profile} stops running on every request it covers${pendingMandateDelete.has_credential ? ", and its stored credential is removed with it" : ""}. Requests it would have recorded go unchecked.`
            : null
        }
        confirmLabel="Remove permanently"
        isPending={removeMandate.isPending}
        error={removeMandate.error}
        onConfirm={() => {
          if (!pendingMandateDelete) return
          removeMandate.mutate(pendingMandateDelete.id, {
            onSuccess: () => setPendingMandateDelete(undefined),
          })
        }}
      />
    </div>
  )
}
