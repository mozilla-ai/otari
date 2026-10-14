import { Button } from "@heroui/react"
import { Link } from "@tanstack/react-router"
import { useCallback, useMemo, useRef, useState } from "react"
import { FiEdit2, FiList, FiTrash2 } from "react-icons/fi"

import type { AliasResponse, PolicySpec } from "@/client"
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
import { canManage } from "@/features/organization/roles"
import { RouterReadiness } from "@/features/routing/RouterReadiness"
import {
  useDeploymentOperator,
  useOrganizationContext,
} from "@/shared/api/organizations"
import {
  useAliases,
  useDeleteAlias,
  useDeleteOrganizationAlias,
  useDeleteOrganizationRoutingPolicy,
  useDeleteRoutingPolicy,
  useRoutingPolicies,
} from "@/shared/api/routing"
import { docsSourceHref } from "@/shared/helpers/docs"
import { useUrlValue } from "@/shared/helpers/urlState"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"
import { useDeployment } from "@/shared/hooks/useDeployment"

import { BackupsCell, MainModelCell } from "./PolicyCells"
import { PolicyForm } from "./PolicyForm"
import {
  findCandidates,
  findFallthroughTarget,
  findRouterBackend,
  KNN_BACKEND,
  normalizeBackend,
  PRIORITY_BACKEND,
  type RoutingRow,
  WEIGHTED_BACKEND,
} from "./policyModel"

/** Present an alias as the one-target policy it is. */
function aliasAsRow(alias: AliasResponse): RoutingRow {
  return {
    kind: "alias",
    name: alias.name,
    spec: { select: [{ default: alias.target }] },
    source: alias.source,
    user_id: alias.user_id,
    // Carried, not dropped: an edit or a delete goes back to the workspace the
    // row lives in, and a row without it falls back to the *selected* workspace,
    // which for an admin whose list spans the organization is a different one.
    workspace_id: alias.workspace_id,
    is_dynamic: false,
    created_at: alias.created_at,
    updated_at: alias.updated_at,
  }
}

// Scope is part of the identity, so it is part of the row key: the same policy
// name can exist globally and per user, and keying on the name alone would
// collapse those rows into one. Same reasoning (and encoding) as the alias table.
const rowKeyOf = (row: RoutingRow): string =>
  JSON.stringify([row.kind, row.user_id, row.name])

/** Whether this form can represent a spec without losing part of it.
 *
 *  The editor reconstructs a spec from four pieces of state, so anything it does
 *  not model (a `user_id`/`key_id` condition, a comparator other than `gte`, a
 *  `budget_remaining_usd` threshold, a router entry) would be silently dropped on
 *  save. Offering Edit on such a policy would quietly destroy the operator's
 *  config, so those are shown read-only until the form covers them. Refusing to
 *  edit is recoverable; a silent lossy save is not.
 */
function isEditableInForm(spec: PolicySpec): boolean {
  // The form re-emits `select` as conditions, then the router, then the default.
  // Selection is order-sensitive server-side (the first matching entry wins), so a
  // spec whose router sits *before* its conditions would come back with different
  // behavior than it went in with. Refusing to edit is recoverable; a silent
  // semantic change on Save is not.
  const routerIndex = spec.select.findIndex(
    (entry) => entry.router !== undefined,
  )
  const lastConditionIndex = spec.select.reduce(
    (last, entry, index) => (entry.when !== undefined ? index : last),
    -1,
  )
  if (
    routerIndex !== -1 &&
    lastConditionIndex !== -1 &&
    routerIndex < lastConditionIndex
  )
    return false
  return spec.select.every((entry) => {
    if (entry.default !== undefined) return entry.when === undefined
    // A router entry is editable: the form models the backend, its pool and (for
    // the weighted backend) the split, which is the whole entry. An unknown backend
    // is still shown read-only, because saving it back through one of these
    // controls would silently rewrite it as a backend the operator did not choose.
    if (entry.router !== undefined) {
      if ((entry.candidates?.length ?? 0) === 0) return false
      const backend = normalizeBackend(entry.router)
      if (backend === KNN_BACKEND || backend === PRIORITY_BACKEND) return true
      // A weighted entry without weights cannot be saved back (the API refuses it),
      // so the form would have to invent a split. Read-only says so instead.
      return (
        backend === WEIGHTED_BACKEND &&
        Object.keys(entry.weights ?? {}).length > 0
      )
    }
    const when = entry.when
    if (when === undefined || entry.target === undefined) return false
    const keys = Object.keys(when)
    return (
      keys.length === 1 &&
      keys[0] === "budget_used_pct" &&
      when.budget_used_pct?.gte !== undefined
    )
  })
}

export function RoutingPage() {
  // Scoped to the selected workspace, like keys and usage (otari-ai#2087).
  // Resolution is workspace-keyed (`services/policy_store`), so a stored policy
  // decides the traffic of one workspace and belongs on that workspace's page.
  // Left unscoped, the deployment-wide list showed an operator every tenant's
  // rows and an admin every workspace of theirs, neither of which is what the
  // page claims to be. Config-file entries have no workspace and are listed
  // whatever the selection, being in force in all of them.
  //
  // Which of the two surfaces answers depends on who is signed in
  // (otari-ai#1942, otari-ai#1969), and `useRoutingScope` makes that choice once
  // for both lists: an operator reads the deployment-wide pair, anyone else the
  // tenant-scoped `/organizations/me/` pair. Both wait for the context to settle
  // rather than taking "not yet an operator" as "member".
  const organization = useOrganizationContext()
  const { isOperator, isSettled: isContextSettled } = useDeploymentOperator()
  // The switcher is seeded from the caller's own memberships, not the
  // organization's whole list (otari-ai#1969), so this is null only for somebody
  // who belongs to no workspace: they have nothing of their own to see and
  // nowhere to write, and are shown the config entries on a read-only page.
  const { selected: selectedWorkspace } = useSelectedWorkspace()
  // Only a gateway that serves requests itself compiles a policy. A hybrid one
  // takes its attempt plan from its control plane, and a hosted control plane
  // serves no requests, so a policy stored there is not consulted.
  const { deployment_type } = useDeployment()
  const isStandalone = deployment_type === "standalone"
  const workspaceId = selectedWorkspace?.workspace_id
  const policies = useRoutingPolicies(workspaceId)
  const aliases = useAliases(workspaceId)
  const deletePolicy = useDeleteRoutingPolicy()
  const deleteAlias = useDeleteAlias()
  const deleteOrgPolicy = useDeleteOrganizationRoutingPolicy()
  const deleteOrgAlias = useDeleteOrganizationAlias()
  // Where a create lands, on either surface. An operator's used to omit it and
  // land in the deployment's default workspace, which is a row saved from one
  // workspace's page and listed on another's. A write to an existing row uses
  // that row's own workspace instead (`deleteWorkspaceFor` below, and the Edit
  // form's `workspaceId`), so a list still being refetched through a switch
  // cannot move a row between workspaces.
  const writeWorkspaceId = workspaceId ?? null
  const canEdit =
    isOperator || (canManage(organization.data) && writeWorkspaceId !== null)
  // A deep link may pre-fill the add form with ?target=provider:model.
  const initialTarget = useUrlValue("target")
  const [adding, setAdding] = useState(initialTarget !== "")
  // Where focus goes when nothing else claims it. React Aria restores focus to
  // whatever had it when the dialog opened, which is right for the heading's own
  // button and wrong for the empty state's: creating the first policy fills the
  // table, so the empty state unmounts and the node react-aria stored is gone,
  // leaving focus on `document.body` where the next Tab starts at the top of the
  // document. `FormDialog` checks for that when the frame is actually gone.
  const createButtonRef = useRef<HTMLButtonElement | null>(null)
  const [createCount, setCreateCount] = useState(0)
  const openCreate = () => {
    setEditing(undefined)
    setCreateCount((n) => n + 1)
    setAdding(true)
  }
  const closeCreate = () => setAdding(false)
  const [editing, setEditing] = useState<RoutingRow>()
  const [pendingDelete, setPendingDelete] = useState<RoutingRow>()
  // Readiness opens inline under its own row (DataTable's accordion), because it
  // describes one policy and the operator clicked that policy. A card above the
  // table would put the panel nowhere near the control that opened it.
  const [expanded, setExpanded] = useState<string>()
  // `adding` is seeded from ?target= before the membership context settles, so
  // the role is applied here rather than in the initializer: gating the
  // initializer would drop an operator's deep link, since `isOperator` is still
  // false at the moment it runs. A member arriving on that link gets the
  // read-only empty state instead of a form whose only outcome is a refusal.
  const isAdding = adding && canEdit

  // Aliases and policies are listed together: an alias is the one-target case,
  // and this page is the only place either is managed.
  const rows: RoutingRow[] = [
    ...(policies.data ?? []).map((policy) => ({
      ...policy,
      kind: "policy" as const,
    })),
    ...(aliases.data ?? []).map(aliasAsRow),
  ].sort(
    (a, b) =>
      a.name.localeCompare(b.name) ||
      (a.user_id ?? "").localeCompare(b.user_id ?? ""),
  )
  // The context counts as loading too: until it settles, neither list has been
  // asked, and an empty table would read as "no policies" rather than "not yet".
  const isListLoading =
    !isContextSettled || policies.isLoading || aliases.isLoading

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
        return policy.source === "config" ? (
          <RowActionRow>
            {readiness}
            <span className="text-xs text-subtle">set in config.yml</span>
          </RowActionRow>
        ) : (
          <RowActionRow>
            {readiness}
            {isEditableInForm(policy.spec) ? (
              <RowAction
                icon={FiEdit2}
                label="Edit"
                onPress={() => {
                  // `setAdding(false)` cannot fire while the create dialog is
                  // open (its backdrop covers the table and `ariaHideOutside`
                  // takes the rows out of the accessibility tree), so this is
                  // belt and braces for a future surface that reaches a row
                  // without going through the modal.
                  setAdding(false)
                  setEditing(policy)
                }}
              />
            ) : (
              <span className="max-w-xs text-xs text-muted">
                Uses options this form cannot show yet. Edit it through the API
                so nothing is lost.
              </span>
            )}
            <RowAction
              icon={FiTrash2}
              label="Delete"
              onPress={() => setPendingDelete(policy)}
            />
          </RowActionRow>
        )
      },
    })
    return base
  }, [canEdit, expanded, isOperator])

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
              onPress={openCreate}
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
      <ErrorBanner error={policies.error ?? aliases.error} />

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
      <PolicyForm
        key={createCount}
        existing={null}
        initialTarget={initialTarget}
        isOpen={isAdding}
        // The empty state's "Create fallback" is gone by the time
        // this closes, since creating one is what makes the page non-empty, so
        // the frame's own restore has nothing to land on. The heading's action
        // survives.
        returnFocusRef={createButtonRef}
        isDeploymentWide={isOperator}
        workspaceId={writeWorkspaceId}
        onClose={closeCreate}
      />
      {editing !== undefined ? (
        <PolicyForm
          // Keyed on the row: the fields seed from `existing` once, through
          // mount-only state, so without this a second row's Edit would open
          // with the first row's draft and save it under the second one's name.
          key={rowKeyOf(editing)}
          existing={editing}
          isDeploymentWide={isOperator}
          workspaceId={editing.workspace_id ?? writeWorkspaceId}
          onClose={() => setEditing(undefined)}
        />
      ) : null}

      {rows.length === 0 && !isListLoading ? (
        canEdit ? (
          <EmptyState
            title="No routing policies yet"
            description="A policy gives callers one model name with backups behind it, so a provider outage does not become a failed request."
            actionLabel="Create fallback"
            onAction={openCreate}
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
