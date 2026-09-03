// `Chip` is still reached by the organization-owned marker, which arrived with
// main's own-budgets work and is the one chip left on this page. The redesign
// took chips off everything it rebuilt, so this is an inconsistency rather than
// a decision: see the note in the PR body.
import { Button, Chip, Spinner } from "@heroui/react"
import { useEffect, useMemo, useState } from "react"
import type {
  Budget,
  BudgetResetLog,
  CreateBudgetRequest,
  User,
} from "@/client"
import { isDeploymentOperator } from "@/features/organization/roles"
import { UserMultiSelect } from "@/features/users/UserMultiSelect"
import {
  useAllWorkspaceBudgetDefaults,
  useBudgetResetLogs,
  useBudgets,
  useCreateBudget,
  useDeleteBudget,
  useOrganizationContext,
  useUpdateBudget,
  useUpdateUser,
  useUsers,
  useWorkspaces,
} from "@/shared/api/hooks"
import { BulkActionBar } from "@/shared/components/BulkActionBar"
import { ConfirmDialog } from "@/shared/components/ConfirmDialog"
import { DataTable, type DataTableColumn } from "@/shared/components/DataTable"
import { Field } from "@/shared/components/Field"
import {
  PageIntro,
  RowAction,
  RowActionRow,
  Section,
  Segmented,
  SpendMeter,
  spendState,
  TableScrollFrame,
} from "@/shared/components/surface"
import {
  CopyableValue,
  EmptyState,
  ErrorBanner,
  InfoBanner,
  PageLoading,
} from "@/shared/components/ui"
import {
  resolveSelectedIds,
  useTableSelection,
} from "@/shared/helpers/tableSelection"
import { OrganizationBudgetsPage } from "./OrganizationBudgetsPage"
import { hasNoLimit, limitLabel } from "./organizationBudget"

// ---------- formatting ----------

const usd = new Intl.NumberFormat(undefined, {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 2,
})

function formatUSD(value: number): string {
  return usd.format(value)
}

const DAY = 86_400
const HOUR = 3_600

// Named periods the picker offers; `formatDuration` reuses them so an exact match
// reads as "Daily" rather than "86400s".
const PERIOD_PRESETS: { label: string; seconds: number | null }[] = [
  { label: "No reset", seconds: null },
  { label: "Daily", seconds: DAY },
  { label: "Weekly", seconds: 7 * DAY },
  { label: "Monthly", seconds: 30 * DAY },
]

function formatDuration(seconds: number | null): string {
  if (seconds === null) return "No reset"
  const preset = PERIOD_PRESETS.find((p) => p.seconds === seconds)
  if (preset) return preset.label
  if (seconds % DAY === 0) return `Every ${seconds / DAY} days`
  if (seconds % HOUR === 0) return `Every ${seconds / HOUR} hours`
  return `Every ${seconds}s`
}

function absolute(iso: string | null): string {
  if (!iso) return "—"
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? "—" : d.toLocaleString()
}

// The segment that opens the custom-days field. A sentinel rather than a number,
// because "custom" is not a duration and any real one it borrowed would collide
// with a preset the day someone added it.
const CUSTOM_PERIOD = "custom"

// ---------- limit + period inputs ----------

// A non-negative dollar amount, empty for "unlimited". Parsed leniently; the
// caller decides what an empty or invalid value means.
function parseLimit(raw: string): { value: number | null; valid: boolean } {
  const trimmed = raw.trim()
  if (trimmed === "") return { value: null, valid: true }
  const n = Number(trimmed)
  if (!Number.isFinite(n) || n < 0) return { value: null, valid: false }
  return { value: n, valid: true }
}

// Whole-day string for a duration, or "" when it is not a whole number of days,
// so the custom field speaks the same unit an operator thinks in.
function daysString(seconds: number | null): string {
  return seconds !== null && seconds % DAY === 0 ? String(seconds / DAY) : ""
}

function PeriodPicker({
  value,
  onChange,
  onInvalidChange,
}: {
  value: number | null
  onChange: (seconds: number | null) => void
  // Reports whether the custom field currently holds an invalid entry, so the
  // form can block Save (an invalid entry emits null, which would otherwise
  // clear the committed period on save).
  onInvalidChange?: (invalid: boolean) => void
}) {
  const isPreset = PERIOD_PRESETS.some((p) => p.seconds === value)
  const [custom, setCustom] = useState(!isPreset)
  // The custom field's own draft, so an in-progress, not-yet-valid entry (e.g.
  // "1.5") stays on screen to be flagged rather than being coerced. It is seeded
  // on mount and reset only by an explicit action here (a preset click), never
  // from `value`: the only thing that changes `value` in place is this component's
  // own onChange, so reseeding from it would wipe the invalid entry on the very
  // null we emit for it, before the operator can read the error. Editing a
  // different budget remounts the form (it is keyed), reseeding from the new value.
  const [draft, setDraft] = useState(() => daysString(value))

  const trimmedDays = draft.trim()
  const daysValue = Number(trimmedDays)
  // Whole days only: a fractional, non-positive, or non-finite entry is rejected
  // outright (surfaced below and left unsaved) rather than silently rounded, so
  // 1.5 never becomes 2. isSafeInteger also rules out an overflowing day count.
  const invalidDays =
    trimmedDays !== "" && (!Number.isSafeInteger(daysValue) || daysValue <= 0)

  // Surface validity to the form so Save is gated on it (like the limit field).
  useEffect(() => {
    onInvalidChange?.(invalidDays)
  }, [invalidDays, onInvalidChange])

  return (
    <div className="flex flex-col gap-2">
      <span className="text-sm font-medium text-foreground">Reset period</span>
      {/* A segmented control rather than a row of buttons. These are the
          alternatives for one field, not five things to do, and filling the
          chosen one primary said the opposite: it put the submit button's own
          treatment on a value, two controls apart from the real submit button
          wearing the same fill. */}
      <Segmented
        label="Reset period"
        // `String(null)` rather than a blank: "No reset" IS a preset here, and
        // its seconds are null, so collapsing null to "" would leave the group
        // with nothing selected on a form that has always opened on it.
        value={custom ? CUSTOM_PERIOD : String(value)}
        options={[
          ...PERIOD_PRESETS.map((preset) => ({
            value: String(preset.seconds),
            label: preset.label,
          })),
          { value: CUSTOM_PERIOD, label: "Custom" },
        ]}
        onChange={(next) => {
          if (next === CUSTOM_PERIOD) {
            setCustom(true)
            return
          }
          setCustom(false)
          // Keep the (hidden) custom draft in step, so reopening Custom shows
          // the preset's day count rather than a stale earlier entry.
          const seconds = next === "null" ? null : Number(next)
          setDraft(daysString(seconds))
          onChange(seconds)
        }}
      />
      {custom ? (
        <div className="flex items-end gap-2">
          <Field
            label="Every N days"
            value={draft}
            onChange={(raw) => {
              setDraft(raw)
              const n = Number(raw.trim())
              // Reject a non-integer or non-positive value instead of rounding it;
              // it is held as null (unsaved) until the operator types whole days.
              onChange(
                raw.trim() === "" || !Number.isSafeInteger(n) || n <= 0
                  ? null
                  : n * DAY,
              )
            }}
            placeholder="14"
            description={
              invalidDays ? (
                <span className="text-danger">
                  Enter a whole number of days.
                </span>
              ) : (
                "Whole days between resets."
              )
            }
          />
        </div>
      ) : null}
      <span className="text-caption">
        Spend returns to zero each period. A user&rsquo;s clock starts when the
        budget is assigned to them.
      </span>
    </div>
  )
}

// ---------- create / edit forms (inline cards, matching KeysPage) ----------

function BudgetForm({
  title,
  submitLabel,
  initial,
  error,
  isPending,
  onSubmit,
  onClose,
  assignUsers,
  assignedUserIds,
  assignmentNote,
}: {
  title: string
  submitLabel: string
  initial: {
    name: string | null
    max_budget: number | null
    budget_duration_sec: number | null
  }
  error: unknown
  isPending: boolean
  onSubmit: (body: CreateBudgetRequest, userIds: string[]) => void
  onClose: () => void
  // Offer a multiselect to assign this budget to existing users on save. Given
  // on both create and edit: assignment used to be reachable per-person on the
  // Users page, and that page is gone, so this is the only place a budget is
  // attached to a person.
  assignUsers?: User[]
  // Who already holds this budget, so edit opens with them selected rather than
  // reading as an empty assignment that would clear them on save.
  assignedUserIds?: string[]
  // Why the multiselect is absent, where its absence is a rule rather than a
  // failed read. The failed-roster case is explained by the page's banner.
  assignmentNote?: string
}) {
  const [name, setName] = useState(initial.name ?? "")
  const [limit, setLimit] = useState(
    initial.max_budget === null ? "" : String(initial.max_budget),
  )
  const [durationSec, setDurationSec] = useState<number | null>(
    initial.budget_duration_sec,
  )
  const [periodInvalid, setPeriodInvalid] = useState(false)
  const [userIds, setUserIds] = useState<string[]>(assignedUserIds ?? [])

  const parsed = parseLimit(limit)
  const canSubmit = !isPending && parsed.valid && !periodInvalid

  const submit = () => {
    if (!canSubmit) return
    // Send name as null (not "") when blank so it clears to unnamed on the wire.
    onSubmit(
      {
        name: name.trim() || null,
        max_budget: parsed.value,
        budget_duration_sec: durationSec,
      },
      userIds,
    )
  }

  return (
    <Section
      className="border-y border-border py-5"
      contentClassName="flex flex-col gap-4"
    >
      {/* A heading, not a styled div: this panel is a section of the page and
          the type role is what it looks like, not what it is. */}
      <h2 className="text-title">{title}</h2>
      <ErrorBanner error={error} />
      <Field
        label="Name (optional)"
        value={name}
        onChange={setName}
        autoFocus
        placeholder="team-free-tier"
        description="A label to recognize this budget later."
      />
      <Field
        label="Spending limit (USD)"
        value={limit}
        onChange={setLimit}
        placeholder="100.00"
        description={
          parsed.valid ? (
            "The most a single user on this budget may spend per period. Leave blank for no limit."
          ) : (
            <span className="text-danger">
              Enter a non-negative number, or leave blank for no limit.
            </span>
          )
        }
      />
      <PeriodPicker
        value={durationSec}
        onChange={setDurationSec}
        onInvalidChange={setPeriodInvalid}
      />
      {assignUsers ? (
        <UserMultiSelect
          label="Assign to people (optional)"
          description="Everyone selected is held to this budget, each with their own allowance rather than a shared pool."
          value={userIds}
          onChange={setUserIds}
          users={assignUsers}
        />
      ) : assignmentNote ? (
        // Why there is nobody to assign, rather than an absent control the
        // reader has to account for. The prop is supplied by the caller and
        // without this branch it would be passed and never rendered.
        <p className="text-caption">{assignmentNote}</p>
      ) : null}
      <div className="flex gap-2">
        <Button variant="primary" isDisabled={!canSubmit} onPress={submit}>
          {isPending ? "Saving…" : submitLabel}
        </Button>
        <Button variant="ghost" isDisabled={isPending} onPress={onClose}>
          Cancel
        </Button>
      </div>
    </Section>
  )
}

// ---------- aggregate usage indicator ----------

// `max_budget` is a per-user cap and users share a budget, so the honest budget
// wide number is spend summed across assigned users against the total they are
// collectively allowed (cap × users). A bar only when both are meaningful.
function UsageCell({ budget }: { budget: Budget }) {
  if (budget.user_count === 0) {
    return <span className="text-caption">No users assigned</span>
  }
  const spent = budget.total_spend
  if (budget.max_budget === null) {
    return (
      <span className="text-xs text-foreground">
        {formatUSD(spent)} spent
        {/* "dollar", because this cell is a spend bar and the budget may still
            cap tokens or requests: the Limit column beside it names those, and
            an unqualified "no limit" here contradicts it. */}
        <span className="text-muted"> · no dollar limit</span>
      </span>
    )
  }
  const allocated = budget.max_budget * budget.user_count
  const state = spendState(spent, allocated)
  return (
    <div className="flex min-w-[140px] flex-col gap-1">
      <div className="flex items-baseline justify-between gap-2 text-xs">
        {/* The one number in this product that changes color, and only in the
            state that has already gone past the limit. */}
        <span className={state === "over" ? "text-danger" : "text-foreground"}>
          {formatUSD(spent)}
        </span>
        <span className="text-muted">of {formatUSD(allocated)}</span>
      </div>
      <SpendMeter
        spent={spent}
        allocated={allocated}
        ariaLabel="Aggregate spend against total allocation"
      />
    </div>
  )
}

// ---------- reset history drill-down ----------

function ResetHistory({ budgetId }: { budgetId: string }) {
  const logs = useBudgetResetLogs(budgetId)

  if (logs.isLoading) {
    return (
      <div className="flex items-center gap-2 px-4 py-4 text-sm text-muted">
        {/* Decorative: the text beside it is the more specific announcement. */}
        <Spinner size="sm" aria-hidden="true" /> Loading reset history…
      </div>
    )
  }
  if (logs.error) {
    return (
      <div className="px-4 py-4">
        <ErrorBanner error={logs.error} />
      </div>
    )
  }
  const rows = logs.data ?? []
  if (rows.length === 0) {
    return (
      <div className="px-4 py-4 text-sm text-muted">
        No resets recorded yet for this budget.
      </div>
    )
  }
  return (
    <div className="overflow-x-auto px-4 py-3">
      <table className="w-full border-collapse text-xs">
        <thead className="text-left text-muted">
          <tr>
            <th scope="col" className="py-1.5 pr-4 font-medium">
              User
            </th>
            <th scope="col" className="py-1.5 pr-4 font-medium">
              Spend cleared
            </th>
            <th scope="col" className="py-1.5 pr-4 font-medium">
              Reset at
            </th>
            <th scope="col" className="py-1.5 font-medium">
              Next reset
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((log: BudgetResetLog) => (
            <tr key={log.id} className="border-t border-border">
              <td className="py-1.5 pr-4">
                <code>{log.user_id ?? "—"}</code>
              </td>
              <td className="py-1.5 pr-4 text-foreground">
                {formatUSD(log.previous_spend)}
              </td>
              <td className="py-1.5 pr-4 text-muted">
                {absolute(log.reset_at)}
              </td>
              <td className="py-1.5 text-muted">
                {absolute(log.next_reset_at)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ---------- onboarding ----------

// ---------- inline confirm (names the target, no modal) ----------

function InlineDelete({
  label,
  isPending,
  onConfirm,
}: {
  label: string
  isPending: boolean
  onConfirm: () => void
}) {
  const [armed, setArmed] = useState(false)

  if (!armed) {
    return <RowAction onPress={() => setArmed(true)}>Delete</RowAction>
  }
  // Armed, this one keeps its sentence: a budget's deletion has a consequence
  // the label cannot carry, and a row action that only said "Delete permanently"
  // would be asking for a decision without the fact behind it. The tinted box is
  // gone with every other box on the row; the words and the danger ink are what
  // is left, and they were doing the work anyway.
  return (
    <span className="flex flex-col items-end gap-1 text-right">
      <span className="max-w-xs text-xs text-muted">
        Delete <strong className="font-medium">{label}</strong>? Users keep
        their spend but lose this limit. Cannot be undone.
      </span>
      <span className="flex items-center gap-4">
        <RowAction isDanger isDisabled={isPending} onPress={onConfirm}>
          Delete permanently
        </RowAction>
        <RowAction isDisabled={isPending} onPress={() => setArmed(false)}>
          Cancel
        </RowAction>
      </span>
    </span>
  )
}

// ---------- page ----------

// A short, stable fingerprint for a budget id (its leading segment), shown when a
// budget has no name and used as a fallback label.
// Stable row-key getter so DataTable's per-row cache holds across re-renders.
const getBudgetRowKey = (b: Budget): string => b.budget_id

function shortId(budgetId: string): string {
  return budgetId.split("-")[0]
}

function budgetLabel(budget: Budget): string {
  return budget.name ?? shortId(budget.budget_id)
}

// Whose budget a row is: a tenant's carries an organization, the deployment's own
// carries none. `/v1/users` refuses to cap a gateway user at a tenant's
// (otari#881), so the page marks the row and withholds the assignment control
// rather than offering a save the API answers 404.
function isOrganizationOwned(budget: Budget): boolean {
  return budget.organization_id !== null
}

/**
 * The deployment's own budgets page, which is what an operator sees.
 *
 * Deployment-wide end to end: `/v1/budgets`, the gateway's `users` table, and
 * every workspace's member default, all behind `require_deployment_operator`.
 * Unchanged by otari-ai#1943, which added the tenant-scoped page beside it
 * rather than reshaping this one, because the two surfaces answer to different
 * callers and read different rows.
 */
function DeploymentBudgetsPage() {
  const budgets = useBudgets()
  const users = useUsers()
  const workspaces = useWorkspaces()
  const workspaceIds = useMemo(
    () => (workspaces.data ?? []).map((workspace) => workspace.id),
    [workspaces.data],
  )
  const workspaceDefaults = useAllWorkspaceBudgetDefaults(workspaceIds)
  const createBudget = useCreateBudget()
  const updateBudget = useUpdateBudget()
  const deleteBudget = useDeleteBudget()
  const updateUser = useUpdateUser()

  const [addOpen, setAddOpen] = useState(false)
  const [editing, setEditing] = useState<string | null>(null)
  const [historyOpen, setHistoryOpen] = useState<string | null>(null)
  const [assignmentError, setAssignmentError] = useState<Error | null>(null)
  const [pendingAssignments, setPendingAssignments] = useState<{
    budgetId: string
    userIds: string[]
  } | null>(null)
  const [assigningUsers, setAssigningUsers] = useState(false)
  const selection = useTableSelection()
  const [bulkDeleteOpen, setBulkDeleteOpen] = useState(false)
  const [bulkError, setBulkError] = useState<unknown>(undefined)
  const [bulkPending, setBulkPending] = useState(false)

  const rows = budgets.data ?? []
  // The roster is part of the edit form's seed: `assignedUserIds` is read on
  // mount, and the form is keyed by budget id so it does not reseed when `users`
  // resolves. Opening Edit against an empty roster would therefore start the
  // multiselect empty and read as "detach everyone" on save, so the table waits
  // for both reads rather than only for the budgets one.
  //
  // Success, not merely "not loading": `isLoading` goes false on a rejection
  // too, and a failed roster would then open a form claiming nobody holds this
  // budget. The assignment controls are withheld instead, and the banner above
  // says why.
  const loading = budgets.isLoading || users.isLoading
  const rosterReady = users.isSuccess

  // Which workspaces hand out each budget. A budget may be the default for
  // several, and a workspace may narrow one to a provider, which is named here
  // because the workspace form only offers the unnarrowed one.
  const defaultFor = useMemo(() => {
    const names = new Map(
      (workspaces.data ?? []).map((workspace) => [
        workspace.id,
        workspace.name,
      ]),
    )
    const byBudget = new Map<string, string[]>()
    for (const { workspaceId, default: row } of workspaceDefaults.data) {
      const name = names.get(workspaceId) ?? workspaceId.slice(0, 8)
      const label = row.provider_key_id
        ? `${name} (${row.provider_key_id})`
        : name
      byBudget.set(row.budget_id, [
        ...(byBudget.get(row.budget_id) ?? []),
        label,
      ])
    }
    return byBudget
  }, [workspaces.data, workspaceDefaults.data])
  const editingBudget = rows.find((b) => b.budget_id === editing) ?? null
  const historyBudget = rows.find((b) => b.budget_id === historyOpen) ?? null
  const showOnboarding = !loading && rows.length === 0 && !addOpen
  const selectableKeys = rows.map((b) => b.budget_id)
  const selectedIds = resolveSelectedIds(selection.selectedKeys, selectableKeys)

  const onBulkDelete = async () => {
    setBulkPending(true)
    setBulkError(undefined)
    try {
      for (const id of selectedIds) {
        await deleteBudget.mutateAsync(id)
      }
      selection.clear()
      setBulkDeleteOpen(false)
    } catch (error) {
      setBulkError(error)
    } finally {
      setBulkPending(false)
    }
  }

  // Memoized on the values the cells actually read so DataTable's per-row
  // cache holds across selection clicks (see the DataTable docstring).
  // historyOpen is a real dependency: it drives the History button label, so
  // toggling it must invalidate the cached rows.
  const columns = useMemo<DataTableColumn<Budget>[]>(
    () => [
      {
        id: "budget",
        header: "Budget",
        isRowHeader: true,
        cell: (b) => (
          <div className="flex flex-col gap-0.5">
            <span className="font-medium text-foreground">
              {b.name ?? <span className="text-muted">(unnamed)</span>}
            </span>
            {isOrganizationOwned(b) ? (
              // Not a warning: the budget is a real one this page may still
              // relabel and refigure, it is simply not one a gateway user can be
              // held to. The mirror of the tenant page's "Set at the deployment
              // level".
              <Chip className="self-start" size="sm" variant="secondary">
                Owned by an organization
              </Chip>
            ) : null}
            {/* Only a prefix is rendered, so the id an API call needs is not on the
              page in full; the copy hands over the whole thing. */}
            <CopyableValue value={b.budget_id} label="budget id">
              <code className="text-mono-micro" title={b.budget_id}>
                {shortId(b.budget_id)}
              </code>
            </CopyableValue>
          </div>
        ),
      },
      {
        id: "limit",
        header: "Limit (per user)",
        cell: (b) => (
          <span className={hasNoLimit(b) ? "text-muted" : undefined}>
            {limitLabel(b)}
          </span>
        ),
      },
      {
        id: "reset",
        header: "Reset",
        cell: (b) => (
          <span className="text-muted">
            {formatDuration(b.budget_duration_sec)}
          </span>
        ),
      },
      {
        id: "users",
        header: "People",
        cell: (b) => <span className="text-muted">{b.user_count}</span>,
      },
      {
        id: "default-for",
        header: "Default for",
        cell: (b) => {
          const holders = defaultFor.get(b.budget_id)
          if (!holders || holders.length === 0) {
            return <span className="text-caption">&mdash;</span>
          }
          // Names in prose, not chips, as everywhere else a row lists what
          // points at it.
          return (
            <div className="flex flex-wrap items-center gap-x-2 text-xs">
              {holders.map((holder, index) => (
                <span key={holder} className="flex items-center gap-2">
                  {index > 0 ? (
                    <span aria-hidden className="text-subtle">
                      ·
                    </span>
                  ) : null}
                  <span className="text-foreground">{holder}</span>
                </span>
              ))}
            </div>
          )
        },
      },
      { id: "usage", header: "Usage", cell: (b) => <UsageCell budget={b} /> },
      {
        id: "actions",
        header: "Actions",
        align: "end",
        cell: (b) => (
          <RowActionRow>
            <RowAction
              onPress={() =>
                setHistoryOpen((current) =>
                  current === b.budget_id ? null : b.budget_id,
                )
              }
            >
              {historyOpen === b.budget_id ? "Hide history" : "History"}
            </RowAction>
            <RowAction
              onPress={() => {
                setAddOpen(false)
                setEditing(b.budget_id)
              }}
            >
              Edit
            </RowAction>
            <InlineDelete
              label={budgetLabel(b)}
              isPending={deleteBudget.isPending}
              onConfirm={() => deleteBudget.mutate(b.budget_id)}
            />
          </RowActionRow>
        ),
      },
    ],
    [historyOpen, deleteBudget.isPending, deleteBudget.mutate, defaultFor],
  )

  /**
   * Reconcile who holds this budget.
   *
   * Both directions, because this is now the only place a budget is attached to
   * a person: deselecting someone on the edit form has to detach them, which an
   * assign-only pass would silently ignore while the form reported success.
   * Returns whether everything landed, so the caller decides what to close.
   */
  const assignUsers = async (
    budgetId: string,
    userIds: string[],
    previousUserIds: string[] = [],
  ): Promise<boolean> => {
    const added = userIds.filter((id) => !previousUserIds.includes(id))
    const removed = previousUserIds.filter((id) => !userIds.includes(id))
    if (added.length === 0 && removed.length === 0) {
      setPendingAssignments(null)
      return true
    }
    setAssigningUsers(true)
    setAssignmentError(null)
    const targets = [
      ...added.map((id) => ({ id, budgetId: budgetId as string | null })),
      ...removed.map((id) => ({ id, budgetId: null })),
    ]
    const results = await Promise.allSettled(
      targets.map(({ id, budgetId: value }) =>
        updateUser.mutateAsync({ id, body: { budget_id: value } }),
      ),
    )
    setAssigningUsers(false)

    const failedUserIds = results.flatMap((result, index) =>
      result.status === "rejected" ? [targets[index].id] : [],
    )
    if (failedUserIds.length > 0) {
      // Only the additions are worth retrying as a set; a failed detach is
      // re-attempted by saving again with the same selection.
      setPendingAssignments({
        budgetId,
        userIds: failedUserIds.filter((id) => added.includes(id)),
      })
      setAssignmentError(
        new Error(
          `The budget was saved, but these people were not updated: ${failedUserIds.join(", ")}. Retry to try again.`,
        ),
      )
      return false
    }

    setPendingAssignments(null)
    return true
  }

  // Create the budget, then (optionally) attach it to the chosen users. The
  // per-user PATCH sets each user's reset clock. Failed assignments stay in the
  // form so a retry never creates a duplicate budget.
  const createAndAssign = (body: CreateBudgetRequest, userIds: string[]) => {
    if (pendingAssignments) {
      void assignUsers(pendingAssignments.budgetId, pendingAssignments.userIds)
      return
    }

    setAssignmentError(null)
    createBudget.mutate(body, {
      onSuccess: async (budget: Budget) => {
        if (
          userIds.length > 0 &&
          !(await assignUsers(budget.budget_id, userIds))
        ) {
          return
        }
        setAddOpen(false)
      },
    })
  }

  return (
    <div className="flex flex-col">
      <PageIntro
        title="Budgets"
        action={
          addOpen || showOnboarding ? null : (
            <Button
              variant="primary"
              onPress={() => {
                setEditing(null)
                setAssignmentError(null)
                setPendingAssignments(null)
                setAddOpen(true)
              }}
            >
              Create budget
            </Button>
          )
        }
      >
        Define spending limits and reset schedules. Assign a budget to users to
        enforce it.
      </PageIntro>

      <ErrorBanner
        error={
          budgets.error ??
          createBudget.error ??
          updateBudget.error ??
          deleteBudget.error ??
          updateUser.error ??
          // Without this a failed roster silently withholds the assignment
          // control and the "Default for" column, with nothing saying why.
          users.error ??
          workspaces.error ??
          workspaceDefaults.error
        }
      />

      <InfoBanner>
        Assign a budget to people when you create it, or from this page&rsquo;s
        Edit action later. Each row&rsquo;s usage aggregates the spend of
        everyone currently on that budget.
      </InfoBanner>

      {showOnboarding ? (
        <EmptyState
          title="No budgets yet"
          description="A budget caps how much a user may spend and, optionally, resets that spend on a schedule. Create one, then assign it to users to enforce a limit."
          actionLabel="Create your first budget"
          onAction={() => {
            setEditing(null)
            setAssignmentError(null)
            setPendingAssignments(null)
            setAddOpen(true)
          }}
        />
      ) : null}

      {addOpen ? (
        <BudgetForm
          title="Create budget"
          submitLabel={
            pendingAssignments ? "Retry assignments" : "Create budget"
          }
          initial={{ name: null, max_budget: null, budget_duration_sec: null }}
          error={createBudget.error ?? assignmentError}
          isPending={createBudget.isPending || assigningUsers}
          assignUsers={users.data ?? []}
          onSubmit={createAndAssign}
          onClose={() => {
            setAssignmentError(null)
            setPendingAssignments(null)
            setAddOpen(false)
          }}
        />
      ) : null}
      {/* Key on the row id so switching which budget is edited remounts the form,
          its fields seed from `initial` on mount only. */}
      {editingBudget ? (
        <BudgetForm
          key={editingBudget.budget_id}
          title={`Edit budget ${budgetLabel(editingBudget)}`}
          submitLabel="Save changes"
          initial={{
            name: editingBudget.name,
            max_budget: editingBudget.max_budget,
            budget_duration_sec: editingBudget.budget_duration_sec,
          }}
          error={updateBudget.error ?? assignmentError}
          isPending={updateBudget.isPending || assigningUsers}
          assignUsers={
            rosterReady && !isOrganizationOwned(editingBudget)
              ? (users.data ?? [])
              : undefined
          }
          assignmentNote={
            isOrganizationOwned(editingBudget)
              ? "This budget belongs to an organization, so people here cannot be held to it. Its limit and reset are still the deployment's to change."
              : undefined
          }
          assignedUserIds={(users.data ?? [])
            .filter((u) => u.budget_id === editingBudget.budget_id)
            .map((u) => u.user_id)}
          onSubmit={(body, userIds) =>
            updateBudget.mutate(
              { id: editingBudget.budget_id, body },
              {
                onSuccess: async () => {
                  const held = (users.data ?? [])
                    .filter((u) => u.budget_id === editingBudget.budget_id)
                    .map((u) => u.user_id)
                  if (
                    await assignUsers(editingBudget.budget_id, userIds, held)
                  ) {
                    setEditing(null)
                  }
                },
              },
            )
          }
          onClose={() => setEditing(null)}
        />
      ) : null}

      {selectedIds.length > 0 ? (
        <BulkActionBar
          selectedCount={selectedIds.length}
          allMatching={false}
          matchingTotal={null}
          canSelectAllMatching={false}
          onSelectAllMatching={() => {}}
          onClear={selection.clear}
        >
          <Button
            size="sm"
            variant="danger"
            onPress={() => setBulkDeleteOpen(true)}
          >
            Delete
          </Button>
        </BulkActionBar>
      ) : null}

      {/* Suppress the table (and its own empty message) while the onboarding
          panel owns the empty state, so a fresh gateway shows one call to action,
          not a panel stacked over a redundant "no rows" table. */}
      {showOnboarding ? null : (
        <TableScrollFrame className="otari-budgets-table">
          <DataTable
            ariaLabel="Budgets"
            columns={columns}
            rows={rows}
            getRowKey={getBudgetRowKey}
            isLoading={loading}
            emptyContent="No budgets yet. Create one to cap spending."
            selectionMode="multiple"
            selectedKeys={selection.selectedKeys}
            onSelectionChange={selection.onSelectionChange}
          />
        </TableScrollFrame>
      )}

      {historyBudget ? (
        <Section className="border-y border-border">
          <div className="flex items-center justify-between border-b border-border py-2">
            <span className="text-sm font-medium text-foreground">
              Reset history — {budgetLabel(historyBudget)}
            </span>
            <Button
              size="sm"
              variant="ghost"
              onPress={() => setHistoryOpen(null)}
            >
              Close
            </Button>
          </div>
          <ResetHistory budgetId={historyBudget.budget_id} />
        </Section>
      ) : null}

      <ConfirmDialog
        isOpen={bulkDeleteOpen}
        onOpenChange={setBulkDeleteOpen}
        heading="Delete budgets"
        body={`Delete ${selectedIds.length} ${selectedIds.length === 1 ? "budget" : "budgets"}? Users on ${
          selectedIds.length === 1 ? "it" : "them"
        } will no longer be capped.`}
        confirmLabel="Delete"
        isPending={bulkPending}
        error={bulkError}
        onConfirm={onBulkDelete}
      />
    </div>
  )
}

/**
 * Spend & budgets, picking the page for whoever is asking.
 *
 * One route with two pages behind it, the way Routing does since otari#867: an
 * operator gets the deployment's budgets, and an organization owner or admin
 * gets their own organization's, which is the surface otari-ai#1943 added. Not
 * two rail rows, because it is one destination in the design and one row in the
 * roles matrix; and not one merged page, because the two read different tables
 * and every deployment-wide read here answers 403 to a tenant.
 *
 * Held until the context lands rather than defaulting to one of them. Guessing
 * would either flash the operator page at an admin and swap it, or fire the
 * deployment-wide reads as an admin and paint their refusals; the shell already
 * waits on this query, so the cost is a spinner that is usually already over.
 */
export function BudgetsPage() {
  const organization = useOrganizationContext()

  if (organization.isPending && !organization.data) {
    return <PageLoading label="Loading spend and budgets…" />
  }
  // Fails towards the operator page on an errored context, matching what the
  // rail does with this row: it is the page that was here before, its reads say
  // in their own words when they are refused, and an admin seeing that is a
  // worse-looking version of a page they can still reach, where the reverse
  // would hide the deployment's budgets from the one caller who owns them.
  if (organization.data && !isDeploymentOperator(organization.data)) {
    return <OrganizationBudgetsPage organization={organization.data} />
  }
  return <DeploymentBudgetsPage />
}
