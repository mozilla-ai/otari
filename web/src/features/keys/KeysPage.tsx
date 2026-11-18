import { useNavigate } from "@tanstack/react-router"
import { type ReactNode, useEffect, useRef, useState } from "react"
import { FiPlus } from "react-icons/fi"
import type { ApiKey, CreateKeyResponse } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import {
  PAGE_SIZE_OPTIONS,
  TablePagination,
} from "@/design-system/data/TablePagination"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { Skeleton } from "@/design-system/feedback/Skeleton"
import { useIsMobile } from "@/design-system/hooks/useIsMobile"
import { DismissChip } from "@/design-system/indicators/DismissChip"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import { FilterSelect } from "@/design-system/navigation/FilterSelect"
import { limitLabel } from "@/features/budgets/organizationBudget"
import { EditKeyForm } from "@/features/keys/EditKeyForm"
import { EndUsersDialog } from "@/features/keys/EndUsersDialog"
import { FederationRulesDialog } from "@/features/keys/FederationRulesDialog"
import {
  ceilingLabel,
  ceilingUsage,
  useKeyBudgets,
} from "@/features/keys/KeyBudgetField"
import { KeyOwner } from "@/features/keys/KeyOwner"
import { KeyRowMenu } from "@/features/keys/KeyRowMenu"
import { KeyStateMark } from "@/features/keys/KeyStateMark"
import {
  type Caller,
  canAddWorkloadIdentity,
  compareRows,
  countLine,
  type KeyMenuId,
  type KeyRow,
  keyAccess,
  keyLabel,
  keyMenu,
  keyRow,
  keyState,
  listsOtherPeoplesKeys,
  parseStateFilter,
  rowsInState,
  STATE_FILTERS,
} from "@/features/keys/keyList"
import { useOpenCreateKey } from "@/features/keys/openCreateKey"
import { RegeneratedSecretDialog } from "@/features/keys/RegeneratedSecretDialog"
import {
  useCanCreateKey,
  useIsOrganizationManager,
} from "@/features/organization/access"
import {
  useCurrentOwnerId,
  useMemberAttributionLabels,
} from "@/features/organization/attribution"
import { perWords } from "@/features/spend/limits"
import { windowOf } from "@/features/spend/period"
import {
  useDeleteKey,
  useKeys,
  useRotateKey,
  useUpdateKey,
} from "@/shared/api/apiKeys"
import { useApiKeyCeilings } from "@/shared/api/budgets"
import {
  useDeploymentOperator,
  useOrganizationContext,
} from "@/shared/api/organizations"
import { useUsageSummary } from "@/shared/api/usage"
import { useWorkspaces } from "@/shared/api/workspaces"
import {
  formatDate,
  formatRelative,
  formatTokens,
  formatUsd,
} from "@/shared/helpers/format"
import { useUrlState } from "@/shared/helpers/urlState"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"
import { useDeployment } from "@/shared/hooks/useDeployment"

/** A row of the table: a key, or one of the placeholders that hold its height while the list loads. */
interface TableRow {
  id: string
  row?: KeyRow
}

const LOADING_ROWS: TableRow[] = [0, 1, 2, 3].map((n) => ({
  id: `loading-${n}`,
}))
// Staggered so four loading rows do not read as one block.
const LOADING_NAME_WIDTHS = [
  "w-[13.75rem]",
  "w-[11.25rem]",
  "w-[15rem]",
  "w-40",
]

// `owner` is set by a person's page, which links here with the id their keys
// are issued under. `state` opens on Active.
const URL_DEFAULTS = {
  owner: "",
  state: "active",
  page: "0",
  size: String(PAGE_SIZE_OPTIONS[0]),
} as const

const placeholder = (width: string) => (
  <Skeleton className={`h-2.5 max-w-full ${width}`} />
)

const UNKNOWN = <span className="text-mono-caption text-subtle">—</span>

/** Where a read the table leans on stands: still coming, refused, or in hand. */
type Read = "pending" | "failed" | "ready"

function readOf(query: { data?: unknown; isError: boolean }): Read {
  if (query.data !== undefined) return "ready"
  return query.isError ? "failed" : "pending"
}

/** A figure that waits on a read: a bar while it comes, a dash if it failed. */
function unread(read: Read, width: string) {
  return read === "pending" ? placeholder(width) : UNKNOWN
}

function KeyName({ apiKey, isActive }: { apiKey: ApiKey; isActive: boolean }) {
  const name = keyLabel(apiKey)
  return (
    <span className="otari-key-cell flex min-w-0 items-baseline gap-2.5">
      <span
        className={`otari-key-name truncate text-body ${isActive ? "" : "text-muted"}`}
        title={name}
      >
        {name}
      </span>
      {apiKey.key_suffix ? (
        <span className="shrink-0 text-mono-caption text-subtle">
          …{apiKey.key_suffix}
        </span>
      ) : null}
      {apiKey.is_service_key ? (
        <span className="shrink-0 text-mono-overline text-muted">
          Service key
        </span>
      ) : null}
    </span>
  )
}

const NEVER = <span className="text-mono-caption text-subtle">Never</span>

function columnsFor({
  showOwner,
  budgetOf,
  budgetRead,
  usageRead,
  renderMenu,
}: {
  showOwner: boolean
  /** The key's own cap, "$50.00 a month"; absent for who cannot read caps. */
  budgetOf?: (keyId: string) => string | null
  budgetRead: Read
  usageRead: Read
  renderMenu: (row: KeyRow) => ReactNode
}): DataTableColumn<TableRow>[] {
  return [
    {
      id: "name",
      header: "Key",
      isRowHeader: true,
      cell: ({ id, row }) =>
        row ? (
          <KeyName apiKey={row.key} isActive={row.state === "active"} />
        ) : (
          placeholder(LOADING_NAME_WIDTHS[Number(id.slice(-1))])
        ),
    },
    {
      id: "state",
      header: "State",
      cell: ({ row }) =>
        row ? <KeyStateMark state={row.state} /> : placeholder("w-16"),
    },
    ...(showOwner
      ? [
          {
            id: "owner",
            header: "Owner",
            cell: ({ row }: TableRow) =>
              row ? <KeyOwner userId={row.key.user_id} /> : placeholder("w-30"),
          } satisfies DataTableColumn<TableRow>,
        ]
      : []),
    {
      id: "used",
      header: "Last used",
      cell: ({ row }) =>
        !row ? (
          // The third lane's bar, which is Owner's when there is one.
          showOwner ? null : (
            placeholder("w-18")
          )
        ) : row.key.last_used_at ? (
          <span className="text-mono-caption text-muted">
            {formatRelative(row.key.last_used_at)}
          </span>
        ) : (
          NEVER
        ),
    },
    {
      id: "expires",
      header: "Expires",
      cell: ({ row }) =>
        !row ? null : row.key.expires_at ? (
          <span
            className="text-mono-caption text-muted"
            title={new Date(row.key.expires_at).toLocaleString()}
          >
            {formatDate(row.key.expires_at)}
          </span>
        ) : (
          NEVER
        ),
    },
    ...(budgetOf
      ? [
          {
            id: "budget",
            header: "Budget",
            align: "end",
            cell: ({ row }: TableRow) => {
              if (!row) return null
              if (budgetRead !== "ready") return unread(budgetRead, "w-16")
              const budget = budgetOf(row.key.id)
              return budget ? (
                <span className="whitespace-nowrap text-mono-caption text-muted">
                  {budget}
                </span>
              ) : (
                <span className="text-mono-caption text-subtle">None</span>
              )
            },
          } satisfies DataTableColumn<TableRow>,
        ]
      : []),
    {
      id: "spend",
      header: "Spent this month",
      align: "end",
      cell: ({ row }) =>
        !row ? null : row.spend === null ? (
          unread(usageRead, "w-12")
        ) : (
          <span
            className={`text-mono-caption ${row.state === "active" ? "text-foreground" : "text-muted"}`}
          >
            {formatUsd(row.spend)}
          </span>
        ),
    },
    {
      id: "tokens",
      header: "Tokens",
      align: "end",
      cell: ({ row }) =>
        !row ? null : row.tokens === null ? (
          unread(usageRead, "w-10")
        ) : (
          <span className="text-mono-caption text-foreground">
            {formatTokens(row.tokens)}
          </span>
        ),
    },
    {
      id: "actions",
      header: <span className="sr-only">Actions</span>,
      align: "end",
      cell: ({ row }) => (row ? renderMenu(row) : placeholder("w-8")),
    },
  ]
}

/**
 * Every API key the caller may read, one line each, with its state, when it
 * was last used, when it expires, and what it spent this month. The State
 * filter opens on Active; keys that stopped working are a pick away. Each
 * row's menu holds exactly the changes the server accepts from the caller.
 *
 * The deployment operator reads every key through `/keys`, with an Owner
 * column; anyone else reads their own, and an owner or admin the
 * organization's service keys too.
 */
export function KeysPage() {
  const [now] = useState(() => Date.now())
  const month = windowOf("month", now)
  const url = useUrlState(URL_DEFAULTS)
  const owner = url.get("owner")
  const filter = parseStateFilter(url.get("state"))
  const page = Math.max(0, url.getNumber("page"))
  const rawSize = url.getNumber("size")
  const pageSize = PAGE_SIZE_OPTIONS.reduce((best, option) =>
    Math.abs(option - rawSize) < Math.abs(best - rawSize) ? option : best,
  )
  const isMobile = useIsMobile()
  const navigate = useNavigate()

  const { isOperator } = useDeploymentOperator()
  const context = useOrganizationContext().data
  const me = useCurrentOwnerId()
  const deployment = useDeployment()
  const { selected: workspace, select } = useSelectedWorkspace()
  const canCreate = useCanCreateKey()
  const isManager = useIsOrganizationManager()
  const openCreateKey = useOpenCreateKey()
  const keys = useKeys()
  const filterRef = useRef<HTMLDivElement>(null)
  const {
    canCap,
    budgets: capBudgets,
    ceilings: keyCeilings,
  } = useKeyBudgets(isOperator)
  // The same read `useKeyBudgets` makes, for whether it answered.
  const ceilingsRead = useApiKeyCeilings(
    isOperator ? "deployment" : "organization",
    canCap,
  )
  const budgetRead = readOf(ceilingsRead)
  const owners = useMemberAttributionLabels()
  const workspaces = useWorkspaces()
  // Over the caller's scope, as `useKeys` lists: the operator's spans the
  // deployment, so another organization's key reads its own spend.
  const usage = useUsageSummary({ start_date: month.start }, "day", ["api_key"])
  const updateKey = useUpdateKey()
  const rotateKey = useRotateKey()
  const deleteKey = useDeleteKey()

  const [editing, setEditing] = useState<ApiKey>()
  const [pendingRegenerate, setPendingRegenerate] = useState<ApiKey>()
  const [regenerated, setRegenerated] = useState<{
    title: string
    result: CreateKeyResponse
  }>()
  const [pendingDelete, setPendingDelete] = useState<ApiKey>()
  const [federating, setFederating] = useState<ApiKey>()
  const [listingEndUsers, setListingEndUsers] = useState<ApiKey>()

  // A menu closes on choice, so a dialog opened from one has no trigger of its
  // own to give focus back to. The row's menu trigger is remembered instead,
  // and the State filter stands in once that row has left the list.
  const opener = useRef<HTMLElement | null>(null)
  const isDialogOpen = Boolean(
    editing ||
      pendingRegenerate ||
      regenerated ||
      pendingDelete ||
      federating ||
      listingEndUsers,
  )
  const wasDialogOpen = useRef(false)
  useEffect(() => {
    if (isDialogOpen) {
      wasDialogOpen.current = true
      return
    }
    if (!wasDialogOpen.current) return
    wasDialogOpen.current = false
    if (opener.current?.isConnected) opener.current.focus()
    else filterRef.current?.querySelector("button")?.focus()
  }, [isDialogOpen])

  const caller: Caller = {
    isOperator,
    role: context?.role,
    workspaceRoles: new Map(
      (context?.workspace_memberships ?? []).map((membership) => [
        membership.workspace_id,
        membership.role,
      ]),
    ),
  }
  const canFederate = canAddWorkloadIdentity(
    caller,
    deployment.federation_member_rules,
  )
  const organizationName = context?.organization.name
  const teamNames = new Map(
    (workspaces.data ?? []).map((one) => [one.id, one.name]),
  )
  const isViewer = context?.role === "viewer" || workspace?.role === "viewer"

  const allKeys = keys.data ?? []
  const isLoading = keys.isPending && !keys.data
  // Decided while loading too, so the lanes do not move when the rows land:
  // only the operator is listed other people's keys.
  const showOwner = isLoading ? isOperator : listsOtherPeoplesKeys(allKeys, me)
  // Anyone but the operator is listed their own keys alone, so another
  // person's never come back however the URL asks.
  const ownerHidden = !isOperator && owner !== "" && owner !== me
  const usageByKey = usage.data
    ? new Map((usage.data.by_api_key ?? []).map((one) => [one.key, one]))
    : null
  const rows: KeyRow[] = allKeys
    .filter((key) => !owner || key.user_id === owner)
    .map((key) => keyRow(key, usageByKey, now))
  const listed = rowsInState(rows, filter).sort(compareRows)
  const lastPage = Math.max(0, Math.ceil(listed.length / pageSize) - 1)
  const shownPage = Math.min(page, lastPage)
  const pageRows = listed.slice(
    shownPage * pageSize,
    (shownPage + 1) * pageSize,
  )

  const run = (key: ApiKey, id: KeyMenuId) => {
    if (id === "edit") setEditing(key)
    if (id === "disable" || id === "enable") {
      updateKey.mutate(
        { id: key.id, body: { is_active: id === "enable" } },
        {
          onSuccess: () => {
            // The row leaves a list filtered to the state it left, and focus
            // on its menu would go with it. The State filter is where the
            // list says which keys it holds, so focus moves there.
            const next = keyState({ ...key, is_active: id === "enable" }, now)
            if (filter !== "all" && filter !== next) {
              filterRef.current?.querySelector("button")?.focus()
            }
          },
        },
      )
    }
    if (id === "end-users") setListingEndUsers(key)
    if (id === "workload-identity") setFederating(key)
    if (id === "regenerate") setPendingRegenerate(key)
    if (id === "delete") setPendingDelete(key)
    if (id === "sessions") {
      // Sessions lists one workspace, so it opens on the key's own.
      select(key.workspace_id)
      void navigate({
        to: "/sessions",
        search: { start_date: month.start, api_key_id: key.id },
      })
    }
  }

  const isPending = updateKey.isPending || rotateKey.isPending
  const ceilingOf = (keyId: string) =>
    keyCeilings.find((one) => one.scope_id === keyId)
  const budgetLine = (keyId: string) => {
    if (budgetRead !== "ready") return "—"
    const ceiling = ceilingOf(keyId)
    return ceiling
      ? `${ceilingUsage(ceiling)} (${ceilingLabel(ceiling, capBudgets)})`
      : "None"
  }
  const renderMenu = (row: KeyRow) => {
    const access = keyAccess(row.key, caller)
    return (
      // Focus anywhere in the menu, its portaled items included, marks this
      // row's trigger as the one a dialog opened from it gives focus back to.
      // biome-ignore lint/a11y/noStaticElementInteractions: the handler only remembers where focus was; there is no action to expose
      <span
        className="contents"
        onFocus={(event) => {
          opener.current = event.currentTarget.querySelector("button")
        }}
      >
        <KeyRowMenu
          apiKey={row.key}
          items={keyMenu({
            key: row.key,
            access,
            isOperator,
            canFederate,
            isPending,
          })}
          isReadOnly={access === "read"}
          teamName={teamNames.get(row.key.workspace_id)}
          organizationName={organizationName}
          budget={canCap ? budgetLine(row.key.id) : undefined}
          onAction={(id) => run(row.key, id)}
        />
      </span>
    )
  }
  const budgetOf = (keyId: string) => {
    const ceiling = ceilingOf(keyId)
    if (!ceiling) return null
    if (ceiling.max_budget === null) return limitLabel(ceiling)
    // How often it resets is the budget's to say; one the list does not hold
    // (a deployment budget on an organization's key) shows its amount alone.
    const per = perWords(
      capBudgets.find((each) => each.budget_id === ceiling.budget_id) ??
        ceiling,
    )
    return per
      ? `${formatUsd(ceiling.max_budget)} ${per}`
      : formatUsd(ceiling.max_budget)
  }
  const columns = columnsFor({
    showOwner,
    budgetOf: canCap ? budgetOf : undefined,
    budgetRead,
    usageRead: readOf(usage),
    renderMenu,
  })

  const ownerName = owners.get(owner) ?? "This person"
  const emptyContent = ownerHidden ? (
    <span className="flex w-full flex-col gap-0.5 text-left">
      <span className="text-body">
        {ownerName}’s keys are not listed to you.
      </span>
      <span className="text-caption">
        Each person sees and changes their own keys.
        {/* Home lists every key's spend for an owner or admin, and only a
            member's own for anyone else. */}
        {isManager ? " Their spend per key is on Home, under API keys." : null}
      </span>
    </span>
  ) : rows.length === 0 ? (
    <span className="flex w-full flex-wrap items-center justify-between gap-3 text-left">
      {canCreate ? (
        <>
          <span className="flex flex-col gap-0.5">
            <span className="text-body">No API keys yet.</span>
            <span className="text-caption">
              A key is how an agent or an app reaches the gateway. It takes a
              minute.
            </span>
          </span>
          <Button
            variant="ghost"
            onPress={() => openCreateKey("api_keys_empty")}
          >
            Create API key
          </Button>
        </>
      ) : (
        <span className="text-body">
          {isViewer
            ? "No API keys. A viewer cannot create one."
            : "No API keys."}
        </span>
      )}
    </span>
  ) : (
    <span className="flex w-full flex-wrap items-baseline gap-3 text-left">
      <span className="text-body">No {filter} keys.</span>
      <button
        type="button"
        onClick={() => url.patch({ state: "all", page: 0 })}
        className="min-h-11 text-body text-link underline"
      >
        Show every state
      </button>
    </span>
  )

  return (
    <div className="flex flex-col">
      <PageIntro
        title="API keys"
        action={
          canCreate ? (
            <Button
              variant="primary"
              className="max-sm:w-full"
              onPress={() => openCreateKey("api_keys_page")}
            >
              <FiPlus aria-hidden className="size-4" />
              Create API key
            </Button>
          ) : undefined
        }
      />

      <div className="flex flex-col gap-5">
        <div className="flex flex-col gap-2">
          <div className="otari-toolbar flex items-end gap-3">
            <div ref={filterRef} className="max-md:min-w-0 max-md:flex-1">
              <FilterSelect
                label="State"
                value={filter}
                onChange={(value) => url.patch({ state: value, page: 0 })}
                options={STATE_FILTERS}
                fullWidth={isMobile}
              />
            </div>
            <span className="ml-auto shrink-0 pb-1.5 text-caption tabular-nums">
              {isLoading || ownerHidden || rows.length === 0
                ? null
                : countLine(rows, filter)}
            </span>
          </div>
          {owner ? (
            <div className="otari-toolbar flex flex-wrap items-center gap-2 pointer-coarse:gap-y-5">
              <DismissChip
                label="Owner"
                value={owners.get(owner) ?? owner}
                onDismiss={() => url.patch({ owner: "", page: 0 })}
                dismissLabel="Show every key"
              />
            </div>
          ) : null}
        </div>

        {/* Whenever a read failed, a refetch over cached rows included. */}
        <ErrorBanner
          error={
            keys.error ?? usage.error ?? (canCap ? ceilingsRead.error : null)
          }
        />
        <ErrorBanner error={updateKey.error} />

        <TableScrollFrame className="otari-keys-table">
          <DataTable
            ariaLabel="API keys"
            columns={columns}
            rows={
              isLoading
                ? LOADING_ROWS
                : ownerHidden
                  ? []
                  : pageRows.map((row) => ({ id: row.key.id, row }))
            }
            getRowKey={(row) => row.id}
            emptyContent={keys.isError && !keys.data ? "" : emptyContent}
            mobileLayout="cards"
          />
        </TableScrollFrame>

        {listed.length > PAGE_SIZE_OPTIONS[0] && !ownerHidden ? (
          <TablePagination
            page={shownPage}
            pageSize={pageSize}
            total={listed.length}
            rowsOnPage={pageRows.length}
            onPageChange={(next) => url.patch({ page: next })}
            onPageSizeChange={(size) => url.patch({ size, page: 0 })}
            pageSizeOptions={PAGE_SIZE_OPTIONS}
            label="API keys"
          />
        ) : null}
      </div>

      {/* Keyed on the key: the fields seed from it on mount only, so the next
          Edit opens on its own key's values rather than the last one's. */}
      {editing ? (
        <EditKeyForm
          key={editing.id}
          isOpen
          isDeploymentWide={isOperator}
          apiKey={editing}
          onClose={() => setEditing(undefined)}
        />
      ) : null}

      {federating ? (
        <FederationRulesDialog
          apiKey={federating}
          onClose={() => setFederating(undefined)}
        />
      ) : null}

      {listingEndUsers ? (
        <EndUsersDialog
          apiKey={listingEndUsers}
          // An organization's service key reaches only its owners and admins,
          // who read its budgets; a member's operator-made key shows ids.
          canReadBudgets={
            isOperator || listingEndUsers.is_organization_service_key
          }
          onClose={() => setListingEndUsers(undefined)}
        />
      ) : null}

      {regenerated ? (
        <RegeneratedSecretDialog
          title={regenerated.title}
          result={regenerated.result}
          memberLabels={owners}
          onClose={() => {
            setRegenerated(undefined)
            // A later regenerate must never flash this secret.
            rotateKey.reset()
          }}
        />
      ) : null}

      <ConfirmDialog
        isOpen={pendingRegenerate !== undefined}
        onOpenChange={(open) => {
          if (open) return
          setPendingRegenerate(undefined)
          rotateKey.reset()
        }}
        heading={
          pendingRegenerate
            ? `Regenerate ${keyLabel(pendingRegenerate)}?`
            : "Regenerate this API key?"
        }
        body={
          pendingRegenerate ? (
            <>
              Regenerate the secret for{" "}
              <strong>{keyLabel(pendingRegenerate)}</strong>? The current secret
              stops working immediately, with no grace period.
            </>
          ) : null
        }
        confirmLabel="Regenerate"
        isPending={rotateKey.isPending}
        error={rotateKey.error}
        onConfirm={() => {
          const target = pendingRegenerate
          if (!target) return
          rotateKey.mutate(target.id, {
            onSuccess: (result) => {
              setPendingRegenerate(undefined)
              setRegenerated({
                title: `New secret for ${keyLabel(target)}`,
                result,
              })
            },
          })
        }}
      />

      <ConfirmDialog
        isOpen={pendingDelete !== undefined}
        // Cleared on the way out, so a refusal does not greet the next row's confirm.
        onOpenChange={(open) => {
          if (open) return
          setPendingDelete(undefined)
          deleteKey.reset()
        }}
        heading={
          pendingDelete
            ? `Delete ${keyLabel(pendingDelete)}?`
            : "Delete this API key?"
        }
        body={
          pendingDelete ? (
            <>
              Permanently delete <strong>{keyLabel(pendingDelete)}</strong>?
              This removes the key and unlinks its usage history. Cannot be
              undone.
            </>
          ) : null
        }
        confirmLabel="Delete permanently"
        isPending={deleteKey.isPending}
        error={deleteKey.error}
        onConfirm={() => {
          if (!pendingDelete) return
          deleteKey.mutate(pendingDelete.id, {
            onSuccess: () => {
              // Its row is about to leave the list, trigger and all.
              opener.current = null
              setPendingDelete(undefined)
            },
          })
        }}
      />
    </div>
  )
}
