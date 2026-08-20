import { Button, Chip } from "@heroui/react"
import { useState } from "react"

import type {
  CreateWorkspaceBudgetDefaultRequest,
  WorkspaceBudgetDefault,
} from "@/client"
import {
  useCreateWorkspaceBudgetDefault,
  useDeleteWorkspaceBudgetDefault,
  useStoredProviders,
  useUpdateWorkspaceBudgetDefault,
  useWorkspaceBudgetDefaults,
} from "@/shared/api/hooks"
import { ConfirmDialog } from "@/shared/components/ConfirmDialog"
import { Field } from "@/shared/components/Field"
import { ErrorBanner, FilterSelect, InfoBanner } from "@/shared/components/ui"

// A workspace-level template for the per-member `scoped_budgets` ceiling
// (`WorkspaceBudgetDefault` on the wire, `WorkspaceMemberBudgetPolicy*` in the
// generated client). Materialization into a concrete, spend-tracking ceiling
// happens server-side on creation and on join; this panel only manages the
// template.
//
// Deliberately simpler than the Spend & budgets page's form (no reset-period
// presets): a workspace default is a template, not something an operator
// tunes as often as a per-user budget.

const DAY = 86_400
const HOUR = 3_600

// A non-negative dollar amount, empty for "unlimited". Parsed leniently; the
// caller decides what an empty or invalid value means.
function parseLimit(raw: string): { value: number | null; valid: boolean } {
  const trimmed = raw.trim()
  if (trimmed === "") return { value: null, valid: true }
  const n = Number(trimmed)
  if (!Number.isFinite(n) || n < 0) return { value: null, valid: false }
  return { value: n, valid: true }
}

// Whole days, empty for "never resets".
function parseDays(raw: string): { seconds: number | null; valid: boolean } {
  const trimmed = raw.trim()
  if (trimmed === "") return { seconds: null, valid: true }
  const n = Number(trimmed)
  if (!Number.isSafeInteger(n) || n <= 0) return { seconds: null, valid: false }
  return { seconds: n * DAY, valid: true }
}

function daysString(seconds: number | null): string {
  return seconds !== null && seconds % DAY === 0 ? String(seconds / DAY) : ""
}

function providerLabel(providerKeyId: string | null): string {
  return providerKeyId ?? "All providers"
}

// The reset period as the row displays it. The API accepts any positive
// number of seconds, not just whole days (the edit form's own field is
// days-only, which is why it has to special-case this same value, see
// `daysTouched` below); dividing by `DAY` unconditionally here would print a
// fraction for one, e.g. "/ 0.041666666666666664 days" for 3,600.
function formatDuration(seconds: number): string {
  if (seconds % DAY === 0) return `${seconds / DAY} days`
  if (seconds % HOUR === 0) return `${seconds / HOUR} hours`
  return `${seconds}s`
}

const usd = new Intl.NumberFormat(undefined, {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 2,
})

function DefaultForm({
  title,
  submitLabel,
  initial,
  providerLocked,
  providerOptions,
  error,
  isPending,
  onSubmit,
  onClose,
}: {
  title: string
  submitLabel: string
  initial: {
    name: string | null
    provider_key_id: string | null
    max_budget: number | null
    budget_duration_sec: number | null
  }
  // The scope is fixed on edit (see `WorkspaceBudgetDefaultService.update_default`):
  // changing it would move the template to a different identity, which is a
  // delete and a create, not an update.
  providerLocked: boolean
  providerOptions: string[]
  error: unknown
  isPending: boolean
  onSubmit: (body: CreateWorkspaceBudgetDefaultRequest) => void
  onClose: () => void
}) {
  const [name, setName] = useState(initial.name ?? "")
  const [providerKeyId, setProviderKeyId] = useState(
    initial.provider_key_id ?? "",
  )
  const [limit, setLimit] = useState(
    initial.max_budget === null ? "" : String(initial.max_budget),
  )
  const [days, setDays] = useState(daysString(initial.budget_duration_sec))
  // Whether the operator has actually edited the days field, as opposed to it
  // merely rendering empty because the stored value isn't a whole number of
  // days (the API accepts any positive number of seconds; this field only
  // offers days). Without this, opening such a default shows an empty field
  // and saving unrelated changes (the name, say) would read that emptiness as
  // "clear the reset period" and silently drop the value nothing here ever
  // touched.
  const [daysTouched, setDaysTouched] = useState(false)

  const parsedLimit = parseLimit(limit)
  const parsedDays = parseDays(days)
  const canSubmit = !isPending && parsedLimit.valid && parsedDays.valid

  return (
    <div className="flex flex-col gap-4 rounded-lg border border-border bg-surface-alt p-4">
      <div className="text-sm font-semibold text-foreground">{title}</div>
      <ErrorBanner error={error} />
      <Field
        label="Name (optional)"
        value={name}
        onChange={setName}
        autoFocus
        placeholder="Default member budget"
        description="A label to recognize this default later."
      />
      <FilterSelect
        label="Provider"
        value={providerKeyId}
        disabled={providerLocked}
        onChange={setProviderKeyId}
        options={[
          { value: "", label: "All providers" },
          ...providerOptions.map((instance) => ({
            value: instance,
            label: instance,
          })),
        ]}
      />
      <Field
        label="Spending limit (USD)"
        value={limit}
        onChange={setLimit}
        placeholder="50.00"
        description={
          parsedLimit.valid ? (
            "The most a member gets from this default per period. Leave blank for no limit."
          ) : (
            <span className="text-danger">
              Enter a non-negative number, or leave blank for no limit.
            </span>
          )
        }
      />
      <Field
        label="Reset every N days (optional)"
        value={days}
        onChange={(value) => {
          setDaysTouched(true)
          setDays(value)
        }}
        placeholder="30"
        description={
          !daysTouched &&
          days === "" &&
          initial.budget_duration_sec !== null ? (
            <span className="text-warning">
              Currently set to {initial.budget_duration_sec}s, which isn&rsquo;t
              a whole number of days and can&rsquo;t be shown here. Left alone,
              it is kept as-is; enter a day count to replace it.
            </span>
          ) : parsedDays.valid ? (
            "Leave blank for a limit that never resets."
          ) : (
            <span className="text-danger">
              Enter a whole number of days, or leave blank.
            </span>
          )
        }
      />
      <div className="flex gap-2">
        <Button
          variant="primary"
          isDisabled={!canSubmit}
          onPress={() =>
            onSubmit({
              name: name.trim() || null,
              provider_key_id: providerKeyId.trim() || null,
              max_budget: parsedLimit.value,
              // Untouched and blank only because the value can't be shown as
              // days: keep it exactly as it was rather than reading the
              // display gap as "clear it".
              budget_duration_sec: daysTouched
                ? parsedDays.seconds
                : initial.budget_duration_sec,
            })
          }
        >
          {isPending ? "Saving…" : submitLabel}
        </Button>
        <Button variant="ghost" isDisabled={isPending} onPress={onClose}>
          Cancel
        </Button>
      </div>
    </div>
  )
}

export function WorkspaceBudgetDefaultsPanel({
  workspaceId,
  workspaceName,
  canManageWorkspace,
}: {
  workspaceId: string
  workspaceName: string
  canManageWorkspace: boolean
}) {
  const defaults = useWorkspaceBudgetDefaults(workspaceId)
  const providers = useStoredProviders()
  const create = useCreateWorkspaceBudgetDefault()
  const update = useUpdateWorkspaceBudgetDefault()
  const remove = useDeleteWorkspaceBudgetDefault()

  const [addOpen, setAddOpen] = useState(false)
  const [editing, setEditing] = useState<string | null>(null)
  const [deleting, setDeleting] = useState<WorkspaceBudgetDefault | null>(null)

  const rows = defaults.data ?? []
  const providerOptions = (providers.data ?? []).map((p) => p.instance)
  const editingDefault = rows.find((d) => d.id === editing) ?? null

  return (
    <div className="flex flex-col gap-4 p-4">
      <div className="flex items-center justify-between">
        <div className="text-sm font-semibold text-foreground">
          Budget defaults for {workspaceName}
        </div>
        {canManageWorkspace && !addOpen && !editingDefault ? (
          <Button size="sm" variant="primary" onPress={() => setAddOpen(true)}>
            Add default
          </Button>
        ) : null}
      </div>
      <ErrorBanner
        error={defaults.error ?? create.error ?? update.error ?? remove.error}
      />
      <InfoBanner>
        A default is a template: creating one gives every current active member
        a matching budget right away, and a member who joins later gets one too.
        Editing or deleting a default never rewrites a budget it already handed
        out; members keep what they were given.
      </InfoBanner>

      {defaults.isLoading ? (
        <p className="text-sm text-muted">Loading budget defaults…</p>
      ) : rows.length === 0 && !addOpen ? (
        <p className="text-sm text-muted">
          This workspace has no budget defaults yet.
        </p>
      ) : (
        <ul className="flex flex-col gap-2">
          {rows.map((row) =>
            editingDefault?.id === row.id ? (
              <li key={row.id}>
                <DefaultForm
                  title={`Edit default (${providerLabel(row.provider_key_id)})`}
                  submitLabel="Save changes"
                  initial={row}
                  providerLocked
                  providerOptions={providerOptions}
                  error={update.error}
                  isPending={update.isPending}
                  onSubmit={(body) =>
                    update.mutate(
                      {
                        workspaceId,
                        defaultId: row.id,
                        body: {
                          name: body.name,
                          max_budget: body.max_budget,
                          budget_duration_sec: body.budget_duration_sec,
                        },
                      },
                      { onSuccess: () => setEditing(null) },
                    )
                  }
                  onClose={() => setEditing(null)}
                />
              </li>
            ) : (
              <li
                key={row.id}
                className="flex flex-wrap items-center gap-3 rounded-lg border border-border bg-surface-alt px-3 py-2"
              >
                <span className="text-sm text-foreground">
                  {row.name ?? "(unnamed default)"}
                </span>
                <Chip size="sm" color="default">
                  {providerLabel(row.provider_key_id)}
                </Chip>
                <span className="text-sm text-muted">
                  {row.max_budget === null
                    ? "No limit"
                    : usd.format(row.max_budget)}
                  {row.budget_duration_sec
                    ? ` / ${formatDuration(row.budget_duration_sec)}`
                    : ""}
                </span>
                {canManageWorkspace ? (
                  <span className="ml-auto flex items-center gap-2">
                    <Button
                      size="sm"
                      variant="ghost"
                      isDisabled={addOpen}
                      onPress={() => {
                        setAddOpen(false)
                        setEditing(row.id)
                      }}
                    >
                      Edit
                    </Button>
                    <Button
                      size="sm"
                      variant="danger-soft"
                      onPress={() => setDeleting(row)}
                    >
                      Delete
                    </Button>
                  </span>
                ) : null}
              </li>
            ),
          )}
        </ul>
      )}

      {addOpen ? (
        <DefaultForm
          title="Add budget default"
          submitLabel="Add default"
          initial={{
            name: null,
            provider_key_id: null,
            max_budget: null,
            budget_duration_sec: null,
          }}
          providerLocked={false}
          providerOptions={providerOptions}
          error={create.error}
          isPending={create.isPending}
          onSubmit={(body) =>
            create.mutate(
              { workspaceId, body },
              { onSuccess: () => setAddOpen(false) },
            )
          }
          onClose={() => setAddOpen(false)}
        />
      ) : null}

      <ConfirmDialog
        isOpen={deleting !== null}
        onOpenChange={(open) => {
          if (!open) setDeleting(null)
        }}
        heading="Delete budget default"
        body={
          <>
            Delete <strong>{deleting?.name ?? "(unnamed default)"}</strong>?
            Members who already got a budget from it keep it; a member joining
            afterwards will not get one from this default.
          </>
        }
        confirmLabel="Delete default"
        isPending={remove.isPending}
        error={remove.error}
        onConfirm={() => {
          if (deleting) {
            remove.mutate(
              { workspaceId, defaultId: deleting.id },
              { onSuccess: () => setDeleting(null) },
            )
          }
        }}
      />
    </div>
  )
}
