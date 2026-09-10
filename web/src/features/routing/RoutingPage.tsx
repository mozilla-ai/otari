import { Button } from "@heroui/react"
import { useCallback, useMemo, useRef, useState } from "react"

import type { AliasResponse, PolicySpec, RoutingPolicyResponse } from "@/client"
import { CopyableValue } from "@/design-system/actions/CopyField"
import { RowAction, RowActionRow } from "@/design-system/actions/RowAction"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { EmptyState } from "@/design-system/feedback/EmptyState"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { Dot } from "@/design-system/indicators/Dot"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import { canManage, isDeploymentOperator } from "@/features/organization/roles"
import { RouterReadiness } from "@/features/routing/RouterReadiness"
import { useOrganizationContext } from "@/shared/api/organizations"
import {
  useAliases,
  useDeleteAlias,
  useDeleteOrganizationAlias,
  useDeleteOrganizationRoutingPolicy,
  useDeleteRoutingPolicy,
  useOrganizationAliases,
  useOrganizationRoutingPolicies,
  useRoutingPolicies,
} from "@/shared/api/routing"
import { useUrlValue } from "@/shared/helpers/urlState"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"

import { PolicyForm } from "./PolicyForm"
import {
  candidatesOf,
  defaultTargetOf,
  KNN_BACKEND,
  normalizedBackend,
  type RoutingRow,
  routerBackendOf,
  sharesOf,
  WEIGHTED_BACKEND,
  weightsOf,
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
      const backend = normalizedBackend(entry.router)
      if (backend === KNN_BACKEND) return true
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

/** What to call the backend that decides, for a chip or a one-line summary.
 *
 *  Named per backend rather than "Dynamic", because the backend's name is what tells
 *  the reader what to do next (teach it, or move the shares). A backend this build
 *  does not know gets the neutral word: it is routed, and claiming it learns would be
 *  a guess about a backend added after this line was written.
 */
function routerLabelOf(spec: PolicySpec): string {
  const backend = routerBackendOf(spec)
  if (backend === WEIGHTED_BACKEND) return "Weighted"
  if (backend === KNN_BACKEND) return "Learned"
  return "Routed"
}

/** One line summarising what a policy serves, for the table. */
function servesSummary(policy: RoutingPolicyResponse): string {
  const chain = policy.spec.on_failure ?? []
  const pool = candidatesOf(policy.spec)
  if (pool.length > 0 && routerBackendOf(policy.spec) === WEIGHTED_BACKEND) {
    // The split shape, not the model names: two provider:model strings do not fit a
    // table cell, and the shares are what distinguishes one weighted policy from
    // another. The pool is spelled out in the editor and in explain.
    const declared = weightsOf(policy.spec)
    const target = defaultTargetOf(policy.spec)
    const full = pool.includes(target) ? pool : [...pool, target]
    const split = sharesOf(full.map((selector) => declared[selector] ?? 0))
      .map((share) => `${Math.round(share)}%`)
      .join(" / ")
    return `Weighted · ${split} across ${full.length} models`
  }
  if (pool.length > 0) {
    return `${routerLabelOf(policy.spec)} · ${pool.length} candidates, ${defaultTargetOf(policy.spec)} by default`
  }
  if (policy.is_dynamic) {
    const total = 1 + chain.length
    return `Chosen per request · ${total} candidate${total === 1 ? "" : "s"}`
  }
  const target = defaultTargetOf(policy.spec)
  return chain.length > 0 ? `${target}  +${chain.length} on failure` : target
}

// ---------------------------------------------------------------------------
// Editor
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

/** The affirmative kind mark: an accent dot and the word, in mono. */
function KindMark({ label }: { label: string }) {
  return (
    <span className="flex items-center gap-2 text-mono-caption text-foreground">
      <Dot className="bg-accent" />
      {label.toUpperCase()}
    </span>
  )
}

export function RoutingPage() {
  // Deliberately unscoped, unlike keys and usage. The gateway stores every
  // policy and alias in the default workspace on purpose, because resolution
  // reads a process-wide name-keyed cache: one stored elsewhere would be listed
  // as scoped while it resolved for everyone. Filtering this list by the
  // selected workspace would therefore show an empty page while those policies
  // were live for that workspace's traffic, and hide a policy the moment it was
  // created. Scope this when resolution is scoped, not before.
  //
  // Which list is asked depends on who is signed in (otari-ai#1942): an
  // operator reads the deployment-wide management view, and anyone else reads
  // the tenant-scoped `/organizations/me/*` pair. Both reads wait for the
  // context to settle rather than taking "not yet an operator" as "member", so
  // an operator's page does not fire a read it is about to drop.
  const organization = useOrganizationContext()
  const isOperator = isDeploymentOperator(organization.data)
  const isContextSettled =
    organization.data !== undefined || organization.isError
  const policies = useRoutingPolicies(isOperator)
  const memberPolicies = useOrganizationRoutingPolicies(
    isContextSettled && !isOperator,
  )
  const aliases = useAliases(isOperator)
  const memberAliases = useOrganizationAliases(isContextSettled && !isOperator)
  const deletePolicy = useDeleteRoutingPolicy()
  const deleteAlias = useDeleteAlias()
  const deleteOrgPolicy = useDeleteOrganizationRoutingPolicy()
  const deleteOrgAlias = useDeleteOrganizationAlias()
  // Where a tenant admin's write lands, and null for an operator, who writes
  // deployment-wide. The tenant surface requires the workspace named, so an
  // admin who belongs to none has nowhere to write and is shown the read-only
  // page: the switcher is seeded from their own memberships, not the
  // organization's whole list (otari-ai#1969).
  const { selected: selectedWorkspace } = useSelectedWorkspace()
  const writeWorkspaceId = isOperator
    ? null
    : (selectedWorkspace?.workspace_id ?? null)
  const canEdit =
    isOperator || (canManage(organization.data) && writeWorkspaceId !== null)
  // An admin's list spans every workspace of the organization, not just the
  // selected one, so a write to an existing row goes back to the workspace that
  // row lives in (`rowWorkspace` below, and the Edit form's `workspaceId`).
  // Using the selection would create a second policy of the same name in the
  // selected workspace and leave the edited one untouched.
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
    setEditing(null)
    setCreateCount((n) => n + 1)
    setAdding(true)
  }
  const closeCreate = () => setAdding(false)
  const [editing, setEditing] = useState<RoutingRow | null>(null)
  const [pendingDelete, setPendingDelete] = useState<RoutingRow>()
  // Readiness opens inline under its own row (DataTable's accordion), because it
  // describes one policy and the operator clicked that policy. A card above the
  // table would put the panel nowhere near the control that opened it.
  const [expanded, setExpanded] = useState<string | null>(null)
  // `adding` is seeded from ?target= before the membership context settles, so
  // the role is applied here rather than in the initializer: gating the
  // initializer would drop an operator's deep link, since `isOperator` is still
  // false at the moment it runs. A member arriving on that link gets the
  // read-only empty state instead of a form whose only outcome is a refusal.
  const isAdding = adding && canEdit

  // Aliases and policies are listed together: an alias is the one-target case,
  // and this page is the only place either is managed.
  //
  // Both operator lists are read through `isOperator` rather than relied on to
  // be empty because their hooks are disabled: a disabled query still hands
  // back whatever sits in the cache under its key, so a caller who was an
  // operator earlier in the session would keep seeing the deployment-wide rows
  // after being demoted. The gate belongs where the data is rendered.
  const rows: RoutingRow[] = [
    ...((isOperator ? policies.data : memberPolicies.data) ?? []).map(
      (policy) => ({
        ...policy,
        kind: "policy" as const,
      }),
    ),
    ...((isOperator ? aliases.data : memberAliases.data) ?? []).map(aliasAsRow),
  ].sort(
    (a, b) =>
      a.name.localeCompare(b.name) ||
      (a.user_id ?? "").localeCompare(b.user_id ?? ""),
  )
  // The context counts as loading too: until it settles, neither list has been
  // asked, and an empty table would read as "no policies" rather than "not yet".
  const isListLoading =
    !isContextSettled ||
    (isOperator
      ? policies.isLoading || aliases.isLoading
      : memberPolicies.isLoading || memberAliases.isLoading)

  // Stable so DataTable's row cache holds; see its docstring.
  const renderDetail = useCallback(
    (row: RoutingRow) => (
      <RouterReadiness
        policyName={row.name}
        candidates={candidatesOf(row.spec)}
        defaultTarget={defaultTargetOf(row.spec)}
        backend={routerBackendOf(row.spec) ?? KNN_BACKEND}
        scopedUserId={row.user_id ?? null}
        onClose={() => setExpanded(null)}
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
        id: "serves",
        header: "Serves",
        cell: (policy) => (
          <div className="flex items-center gap-2">
            <span className="text-sm text-foreground">
              {servesSummary(policy)}
            </span>
            {/* The kind of routing, as an affirmative mark: a fallback chain or
                a learned router is a decision somebody made about this policy,
                where a plain single-target policy is just the default shape. */}
            {candidatesOf(policy.spec).length > 0 ? (
              <KindMark label={routerLabelOf(policy.spec)} />
            ) : policy.is_dynamic ? (
              <KindMark label="Dynamic" />
            ) : null}
          </div>
        ),
      },
      {
        id: "guards",
        header: "Guards",
        cell: (policy) => {
          const guardrails = policy.spec.guardrails ?? []
          // An em dash: no guardrails is an absence, not a zero.
          if (guardrails.length === 0)
            return <span className="text-muted">—</span>
          return (
            <span className="text-body">
              {guardrails
                .map((guardrail) => `${guardrail.profile} (${guardrail.mode})`)
                .join(", ")}
            </span>
          )
        },
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
      {
        id: "source",
        header: "Source",
        cell: (row) => (
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-2 text-mono-caption text-muted">
              <Dot
                className={
                  row.source === "config" ? "bg-text-subtle" : "bg-accent"
                }
              />
              {row.source.toUpperCase()}
            </span>
            {row.kind === "alias" ? (
              <span className="text-mono-overline text-subtle">alias</span>
            ) : null}
          </div>
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
        const readiness = !isOperator ? null : routerBackendOf(policy.spec) !==
          KNN_BACKEND ? (
          <span className="text-muted">—</span>
        ) : (
          <RowAction
            onPress={() =>
              setExpanded((current) =>
                current === rowKeyOf(policy) ? null : rowKeyOf(policy),
              )
            }
          >
            {expanded === rowKeyOf(policy) ? "Hide examples" : "Examples"}
          </RowAction>
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
                onPress={() => {
                  // `setAdding(false)` cannot fire while the create dialog is
                  // open (its backdrop covers the table and `ariaHideOutside`
                  // takes the rows out of the accessibility tree), so this is
                  // belt and braces for a future surface that reaches a row
                  // without going through the modal.
                  setAdding(false)
                  setEditing(policy)
                }}
              >
                Edit
              </RowAction>
            ) : (
              <span className="max-w-xs text-xs text-muted">
                Uses options this form cannot show yet. Edit it through the API
                so nothing is lost.
              </span>
            )}
            <RowAction onPress={() => setPendingDelete(policy)}>
              Delete
            </RowAction>
          </RowActionRow>
        )
      },
    })
    return base
  }, [canEdit, expanded, isOperator])

  // Which of the four delete surfaces a row goes to. The tenant one names the
  // workspace and has no user scope; the deployment-wide one defaults the
  // workspace and keeps it.
  const deleteWorkspaceFor = (row: RoutingRow) =>
    isOperator ? null : (row.workspace_id ?? writeWorkspaceId)
  const deleteMutationFor = (row: RoutingRow) =>
    deleteWorkspaceFor(row) !== null
      ? row.kind === "alias"
        ? deleteOrgAlias
        : deleteOrgPolicy
      : row.kind === "alias"
        ? deleteAlias
        : deletePolicy
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
        {isOperator
          ? "Named models your callers send as `model`. A policy decides which real model serves each request, what is tried if that fails, and which guardrails always run. It can also split traffic across providers by weight, or let a router learn which prompts a cheaper model handles just as well."
          : canEdit
            ? "Named models your callers send as `model`. A policy decides which real model serves each request, what is tried if that fails, and which guardrails always run. What you create here applies in the selected workspace, and can name any model your organization has a provider key for."
            : "Named models your callers send as `model`. A policy decides which real model serves each request, what is tried if that fails, and which guardrails always run. These are the ones in force in your workspaces; your organization's admins manage them."}
      </PageIntro>

      {/* The reads only. Every delete on this page reports inside its own
          confirm dialog, which is where the operator is looking. */}
      <ErrorBanner
        error={
          policies.error ??
          memberPolicies.error ??
          aliases.error ??
          memberAliases.error
        }
      />

      {/* Mounted while closed so the frame plays its exit with the content
          intact, and keyed on the open counter so the draft is fresh on the way
          in rather than cleared on the way out. See feedback.md. */}
      <PolicyForm
        key={createCount}
        existing={null}
        initialTarget={initialTarget}
        isOpen={isAdding}
        // The empty state's "Create your first policy" is gone by the time
        // this closes, since creating one is what makes the page non-empty, so
        // the frame's own restore has nothing to land on. The heading's action
        // survives.
        returnFocusRef={createButtonRef}
        workspaceId={writeWorkspaceId}
        onClose={closeCreate}
      />
      {editing !== null ? (
        <PolicyForm
          // Keyed on the row: the fields seed from `existing` once, through
          // mount-only state, so without this a second row's Edit would open
          // with the first row's draft and save it under the second one's name.
          key={rowKeyOf(editing)}
          existing={editing}
          workspaceId={
            isOperator ? null : (editing.workspace_id ?? writeWorkspaceId)
          }
          onClose={() => setEditing(null)}
        />
      ) : null}

      {rows.length === 0 && !isListLoading ? (
        canEdit ? (
          <EmptyState
            title="No routing policies yet"
            actionLabel="Create your first policy"
            onAction={openCreate}
          >
            <ol className="flex list-decimal flex-col gap-1 pl-5 text-sm text-muted">
              <li>
                Create a policy and point it at the model that should normally
                serve.
              </li>
              <li>
                Add a fallback chain so a provider outage does not become a
                failed request.
              </li>
              <li>
                Or split the traffic across two providers by weight, and move
                the shares as you learn.
              </li>
              <li>
                Or let a router choose per request between a cheap and a strong
                model, then teach it with a few scored examples.
              </li>
              <li>Have your callers send the policy name as their `model`.</li>
            </ol>
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
          if (rowWorkspace !== null) {
            const scoped = {
              name: pendingDelete.name,
              workspaceId: rowWorkspace,
            }
            if (pendingDelete.kind === "alias")
              deleteOrgAlias.mutate(scoped, { onSuccess })
            else deleteOrgPolicy.mutate(scoped, { onSuccess })
            return
          }
          const deployment = {
            name: pendingDelete.name,
            userId: pendingDelete.user_id,
          }
          if (pendingDelete.kind === "alias")
            deleteAlias.mutate(deployment, { onSuccess })
          else deletePolicy.mutate(deployment, { onSuccess })
        }}
      />
    </div>
  )
}
