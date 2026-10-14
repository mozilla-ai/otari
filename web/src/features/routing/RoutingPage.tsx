import { Button } from "@heroui/react"
import { Link, useNavigate } from "@tanstack/react-router"
import { useCallback, useMemo, useRef, useState } from "react"
import { FiEdit2, FiEye, FiList, FiTrash2 } from "react-icons/fi"

import { CopyableValue } from "@/design-system/actions/CopyField"
import { RowAction, RowActionRow } from "@/design-system/actions/RowAction"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { EmptyState } from "@/design-system/feedback/EmptyState"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import { DocsLink } from "@/design-system/navigation/DocsLink"
import { RouterReadiness } from "@/features/routing/RouterReadiness"
import {
  useDeleteAlias,
  useDeleteOrganizationAlias,
  useDeleteOrganizationRoutingPolicy,
  useDeleteRoutingPolicy,
} from "@/shared/api/routing"
import { docsSourceHref } from "@/shared/helpers/docs"
import { useUrlValue } from "@/shared/helpers/urlState"
import { useDeployment } from "@/shared/hooks/useDeployment"

import { CreatePolicyDialog } from "./CreatePolicyDialog"
import { BackupsCell, MainModelCell } from "./PolicyCells"
import type { Intent } from "./policyDraft"
import {
  findCandidates,
  findFallthroughTarget,
  findRouterBackend,
  isEditableInForm,
  KNN_BACKEND,
  type RoutingRow,
  rowKeyOf,
} from "./policyModel"
import { useRoutingRows } from "./useRoutingRows"

export function RoutingPage() {
  const navigate = useNavigate()
  const {
    rows,
    isLoading: isListLoading,
    error,
    isOperator,
    canEdit,
    writeWorkspaceId,
  } = useRoutingRows()
  // Only a gateway that serves requests itself compiles a policy. A hybrid one
  // takes its attempt plan from its control plane, and a hosted control plane
  // serves no requests, so a policy stored there is not consulted.
  const { deployment_type } = useDeployment()
  const isStandalone = deployment_type === "standalone"
  const deletePolicy = useDeleteRoutingPolicy()
  const deleteAlias = useDeleteAlias()
  const deleteOrgPolicy = useDeleteOrganizationRoutingPolicy()
  const deleteOrgAlias = useDeleteOrganizationAlias()
  // A deep link may pre-fill a new fallback with ?target=provider:model.
  const initialTarget = useUrlValue("target")
  const [adding, setAdding] = useState(initialTarget !== "")
  // Where the create dialog opens: a form, or the choice of one.
  const [startAt, setStartAt] = useState<Intent | undefined>(
    initialTarget === "" ? undefined : "fallback",
  )
  // Where focus goes when nothing else claims it: creating the first policy
  // unmounts the empty state whose button opened the dialog.
  const createButtonRef = useRef<HTMLButtonElement | null>(null)
  const [createCount, setCreateCount] = useState(0)
  const openCreate = (intent?: Intent) => {
    setStartAt(intent)
    setCreateCount((n) => n + 1)
    setAdding(true)
  }
  const closeCreate = () => setAdding(false)
  const [pendingDelete, setPendingDelete] = useState<RoutingRow>()
  // Readiness opens inline under its own row, next to the control that opened it.
  const [expanded, setExpanded] = useState<string>()
  // `adding` is seeded from ?target= before the membership context settles, so
  // the role is applied here: a member on that link gets the read-only page.
  const isAdding = adding && canEdit

  // Stable, because the columns below are memoized on it for DataTable.
  const openPolicy = useCallback(
    (row: RoutingRow) =>
      void navigate({
        to: "/routing/$name",
        params: { name: row.name },
        search: row.user_id ? { user: row.user_id } : {},
      }),
    [navigate],
  )

  // Stable so DataTable's row cache holds; see its docstring.
  const renderDetail = useCallback(
    (row: RoutingRow) => (
      <RouterReadiness
        policyName={row.name}
        candidates={findCandidates(row.spec)}
        defaultTarget={findFallthroughTarget(row.spec)}
        backend={findRouterBackend(row.spec) ?? KNN_BACKEND}
        scopedUserId={row.user_id ?? null}
        onClose={() => setExpanded(undefined)}
      />
    ),
    [],
  )

  const columns = useMemo<DataTableColumn<RoutingRow>[]>(() => {
    const base: DataTableColumn<RoutingRow>[] = [
      {
        id: "name",
        header: "Policy",
        isRowHeader: true,
        cell: (policy) => (
          <CopyableValue value={policy.name} label="policy name" />
        ),
      },
      {
        id: "main",
        header: "Main model",
        cell: (row) => <MainModelCell row={row} />,
      },
      {
        id: "backups",
        header: "Backups",
        cell: (row) => <BackupsCell row={row} />,
      },
      {
        id: "scope",
        header: "Applies to",
        cell: (policy) =>
          (policy.user_id ?? null) === null ? (
            <span className="text-muted">Every caller</span>
          ) : (
            <CopyableValue value={policy.user_id ?? ""} label="user id" />
          ),
      },
    ]
    // No actions column for a caller who cannot act: every affordance in it is
    // either a write or the Examples panel, whose read is operator-only.
    if (!canEdit) return base
    base.push({
      id: "actions",
      header: "",
      cell: (policy) => {
        // Teaching is data, not configuration, so it is offered even for a policy
        // defined in config.yml: an operator can score examples for a policy they
        // cannot edit here, and without this that policy could never route.
        // "Examples" rather than "Router": on a Routing page full of routing
        // policies, "Router" names the thing rather than what opens, and the count
        // of scored examples is the one number in there that changes.
        //
        // Three outcomes, not two. Operator-only within an actions column an
        // admin now also gets, because the panel reads `/routing/status`,
        // which is deployment-wide: an admin sees no readiness at all. Then an
        // em dash where a policy has no readiness to report, rather than an
        // empty cell, since a fallback chain has nothing to learn and that
        // absence is worth stating and is not the same as zero examples. Then
        // the control, for a backend that learns.
        const readiness = !isOperator ? null : findRouterBackend(
            policy.spec,
          ) !== KNN_BACKEND ? (
          <span className="text-muted">—</span>
        ) : (
          <RowAction
            icon={FiList}
            label={expanded === rowKeyOf(policy) ? "Hide examples" : "Examples"}
            onPress={() =>
              setExpanded((current) =>
                current === rowKeyOf(policy) ? undefined : rowKeyOf(policy),
              )
            }
          />
        )
        const isConfig = policy.source === "config"
        // A policy this dashboard cannot edit without loss opens read-only.
        const canOpenToEdit = !isConfig && isEditableInForm(policy.spec)
        return (
          <RowActionRow>
            {readiness}
            {isConfig ? (
              <span className="text-xs text-subtle">set in config.yml</span>
            ) : null}
            <RowAction
              icon={canOpenToEdit ? FiEdit2 : FiEye}
              label={canOpenToEdit ? "Edit" : "View"}
              onPress={() => openPolicy(policy)}
            />
            {isConfig ? null : (
              <RowAction
                icon={FiTrash2}
                label="Delete"
                onPress={() => setPendingDelete(policy)}
              />
            )}
          </RowActionRow>
        )
      },
    })
    return base
  }, [canEdit, expanded, isOperator, openPolicy])

  // Which of the four delete surfaces a row goes to. Both name the workspace the
  // row lives in; only the deployment-wide pair carries the user scope, which
  // the tenant surface has no rows in.
  const deleteWorkspaceFor = (row: RoutingRow) =>
    row.workspace_id ?? writeWorkspaceId
  const deleteMutationFor = (row: RoutingRow) =>
    isOperator
      ? row.kind === "alias"
        ? deleteAlias
        : deletePolicy
      : row.kind === "alias"
        ? deleteOrgAlias
        : deleteOrgPolicy
  // Resolved for the pending row alone, not as a chain over all four: a refusal
  // stays on its mutation until the next call, so reading every one of them
  // would report the last row's failure over this row's confirm.
  const pendingDeleteMutation = pendingDelete
    ? deleteMutationFor(pendingDelete)
    : undefined

  return (
    <div className="flex flex-col gap-6">
      <PageIntro
        title="Routing"
        action={
          canEdit ? (
            <Button
              ref={createButtonRef}
              // Visible while the dialog is open: the dialog is over the page,
              // so there is nothing for hiding this to prevent.
              variant="primary"
              onPress={() => openCreate()}
            >
              Create policy
            </Button>
          ) : undefined
        }
      >
        {/* Three readings of the same page, because what a caller may do here
            differs: an operator sees the deployment-wide capabilities, an
            admin sees what their own workspace writes reach, and a member is
            told who manages these rather than offered a control they would
            be refused. */}
        Model names your callers send as{" "}
        <code className="font-mono text-caption">model</code>. Each policy picks
        the model that serves a request and the backups tried, in order, when it
        fails.{" "}
        {isOperator
          ? null
          : canEdit
            ? "What you create here applies in the selected workspace, and can name any model your organization has a provider key for."
            : "These are the ones in force in your workspaces; your organization's admins manage them."}
      </PageIntro>

      {/* The reads only. Every delete on this page reports inside its own
          confirm dialog, which is where the operator is looking. */}
      <ErrorBanner error={error} />

      {isStandalone ? null : (
        <InfoBanner>
          Policies are a standalone feature: they apply only on a gateway that
          serves requests itself. A hybrid gateway receives its attempt plan
          from the connected control plane.
        </InfoBanner>
      )}

      {/* Mounted while closed so the frame plays its exit with the content
          intact, and keyed on the open counter so the draft is fresh on the way
          in rather than cleared on the way out. See feedback.md. */}
      <CreatePolicyDialog
        key={createCount}
        isOpen={isAdding}
        startAt={startAt}
        initialTarget={initialTarget}
        // The empty state's buttons are gone by the time this closes, since
        // creating one is what makes the page non-empty. The heading's action
        // survives.
        returnFocusRef={createButtonRef}
        isDeploymentWide={isOperator}
        workspaceId={writeWorkspaceId}
        onClose={closeCreate}
      />

      {rows.length === 0 && !isListLoading ? (
        canEdit ? (
          <EmptyState
            title="No routing policies yet"
            description="A policy gives callers one model name with backups behind it, so a provider outage does not become a failed request."
            actionLabel="Create fallback"
            onAction={() => openCreate("fallback")}
            secondaryActionLabel="Other policy types"
            onSecondaryAction={() => openCreate()}
          >
            {/* Catalog-ID fallback is compiled by a standalone gateway only. */}
            {isStandalone ? (
              <p className="max-w-prose text-caption">
                Tip: the same model on several providers already falls back
                automatically when callers send its catalog ID.{" "}
                <Link to="/models" className="text-link hover:underline">
                  Browse models
                </Link>{" "}
                or{" "}
                <DocsLink
                  href={docsSourceHref("models.md", "catalog-spellings")}
                >
                  read how
                </DocsLink>
                .
              </p>
            ) : null}
          </EmptyState>
        ) : (
          <EmptyState title="No routing policies yet">
            <p className="text-sm text-muted">
              Policies that apply in your workspaces will be listed here once
              your organization's admins define them.
            </p>
          </EmptyState>
        )
      ) : (
        <TableScrollFrame className="otari-routing-table">
          <DataTable
            ariaLabel="Routing policies"
            columns={columns}
            rows={rows}
            getRowKey={rowKeyOf}
            detailKey={expanded}
            renderDetail={renderDetail}
            isLoading={isListLoading}
            emptyContent="No routing policies yet."
          />
        </TableScrollFrame>
      )}

      <ConfirmDialog
        isOpen={pendingDelete !== undefined}
        // Cleared on the way out: a refusal otherwise sits on the mutation and
        // greets the next row's confirm as if that row had failed.
        onOpenChange={(open) => {
          if (open) return
          setPendingDelete(undefined)
          deletePolicy.reset()
          deleteAlias.reset()
          deleteOrgPolicy.reset()
          deleteOrgAlias.reset()
        }}
        heading={
          pendingDelete?.kind === "alias" ? "Delete alias" : "Delete policy"
        }
        body={
          pendingDelete
            ? `${pendingDelete.name} stops resolving. A request that still sends it as its model is refused, so update the callers that name it.`
            : null
        }
        confirmLabel={
          pendingDelete?.kind === "alias" ? "Delete alias" : "Delete policy"
        }
        isPending={pendingDeleteMutation?.isPending ?? false}
        error={pendingDeleteMutation?.error}
        onConfirm={() => {
          if (!pendingDelete) return
          const onSuccess = () => setPendingDelete(undefined)
          const rowWorkspace = deleteWorkspaceFor(pendingDelete)
          if (isOperator) {
            const deployment = {
              name: pendingDelete.name,
              userId: pendingDelete.user_id,
              workspaceId: rowWorkspace,
            }
            if (pendingDelete.kind === "alias")
              deleteAlias.mutate(deployment, { onSuccess })
            else deletePolicy.mutate(deployment, { onSuccess })
            return
          }
          // Unreachable: Delete is rendered behind `canEdit`, which for a
          // tenant admin already requires a selected workspace, and a stored row
          // always carries its own.
          if (rowWorkspace === null) return
          const scoped = { name: pendingDelete.name, workspaceId: rowWorkspace }
          if (pendingDelete.kind === "alias")
            deleteOrgAlias.mutate(scoped, { onSuccess })
          else deleteOrgPolicy.mutate(scoped, { onSuccess })
        }}
      />
    </div>
  )
}
