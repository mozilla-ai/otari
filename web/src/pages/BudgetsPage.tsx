import { Button, Card, Spinner } from "@heroui/react";
import { Fragment, useState } from "react";

import {
  useBudgetResetLogs,
  useBudgets,
  useCreateBudget,
  useDeleteBudget,
  useUpdateBudget,
  useUpdateUser,
  useUsers,
} from "@/api/hooks";
import type { Budget, BudgetResetLog, CreateBudgetRequest, User } from "@/api/types";
import { Field } from "@/components/Field";
import { LoadingRow, Table, TableMessage, Td, Th, THead, Tr } from "@/components/Table";
import { ErrorBanner, InfoBanner, PageHeader } from "@/components/ui";
import { UserMultiSelect } from "@/components/UserMultiSelect";

// ---------- formatting ----------

const usd = new Intl.NumberFormat(undefined, { style: "currency", currency: "USD", maximumFractionDigits: 2 });

function formatUSD(value: number): string {
  return usd.format(value);
}

const DAY = 86_400;
const HOUR = 3_600;

// Named periods the picker offers; `formatDuration` reuses them so an exact match
// reads as "Daily" rather than "86400s".
const PERIOD_PRESETS: { label: string; seconds: number | null }[] = [
  { label: "No reset", seconds: null },
  { label: "Daily", seconds: DAY },
  { label: "Weekly", seconds: 7 * DAY },
  { label: "Monthly", seconds: 30 * DAY },
];

function formatDuration(seconds: number | null): string {
  if (seconds === null) return "No reset";
  const preset = PERIOD_PRESETS.find((p) => p.seconds === seconds);
  if (preset) return preset.label;
  if (seconds % DAY === 0) return `Every ${seconds / DAY} days`;
  if (seconds % HOUR === 0) return `Every ${seconds / HOUR} hours`;
  return `Every ${seconds}s`;
}

function absolute(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "—" : d.toLocaleString();
}

// ---------- limit + period inputs ----------

// A non-negative dollar amount, empty for "unlimited". Parsed leniently; the
// caller decides what an empty or invalid value means.
function parseLimit(raw: string): { value: number | null; valid: boolean } {
  const trimmed = raw.trim();
  if (trimmed === "") return { value: null, valid: true };
  const n = Number(trimmed);
  if (Number.isNaN(n) || n < 0) return { value: null, valid: false };
  return { value: n, valid: true };
}

// seconds shown as a whole number of days when it divides evenly, so the custom
// field speaks the same unit an operator thinks in.
function PeriodPicker({ value, onChange }: { value: number | null; onChange: (seconds: number | null) => void }) {
  const isPreset = PERIOD_PRESETS.some((p) => p.seconds === value);
  const [custom, setCustom] = useState(!isPreset);
  const customDays = value !== null && value % DAY === 0 ? String(value / DAY) : "";

  return (
    <div className="flex flex-col gap-2">
      <span className="text-sm font-medium text-[var(--otari-ink)]">Reset period</span>
      <div className="flex flex-wrap gap-2">
        {PERIOD_PRESETS.map((preset) => (
          <Button
            key={preset.label}
            size="sm"
            variant={!custom && value === preset.seconds ? "primary" : "outline"}
            onPress={() => {
              setCustom(false);
              onChange(preset.seconds);
            }}
          >
            {preset.label}
          </Button>
        ))}
        <Button size="sm" variant={custom ? "primary" : "outline"} onPress={() => setCustom(true)}>
          Custom
        </Button>
      </div>
      {custom ? (
        <div className="flex items-end gap-2">
          <Field
            label="Every N days"
            value={customDays}
            onChange={(raw) => {
              const n = Number(raw.trim());
              onChange(raw.trim() === "" || Number.isNaN(n) || n <= 0 ? null : Math.round(n) * DAY);
            }}
            placeholder="14"
            description="Whole days between resets."
          />
        </div>
      ) : null}
      <span className="text-xs text-[var(--otari-muted)]">
        Spend returns to zero each period. A user&rsquo;s clock starts when the budget is assigned to them.
      </span>
    </div>
  );
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
}: {
  title: string;
  submitLabel: string;
  initial: { name: string | null; max_budget: number | null; budget_duration_sec: number | null };
  error: unknown;
  isPending: boolean;
  onSubmit: (body: CreateBudgetRequest, userIds: string[]) => void;
  onClose: () => void;
  // When provided (create only), offer an optional multiselect to assign the new
  // budget to existing users on save. Omitted for edit, where membership is
  // managed per-user on the Users page.
  assignUsers?: User[];
}) {
  const [name, setName] = useState(initial.name ?? "");
  const [limit, setLimit] = useState(initial.max_budget === null ? "" : String(initial.max_budget));
  const [durationSec, setDurationSec] = useState<number | null>(initial.budget_duration_sec);
  const [userIds, setUserIds] = useState<string[]>([]);

  const parsed = parseLimit(limit);

  const submit = () => {
    if (isPending || !parsed.valid) return;
    // Send name as null (not "") when blank so it clears to unnamed on the wire.
    onSubmit({ name: name.trim() || null, max_budget: parsed.value, budget_duration_sec: durationSec }, userIds);
  };

  return (
    <Card>
      <Card.Content className="flex flex-col gap-4 p-5">
        <div className="text-sm font-semibold text-[var(--otari-ink)]">{title}</div>
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
              <span className="text-red-700">Enter a non-negative number, or leave blank for no limit.</span>
            )
          }
        />
        <PeriodPicker value={durationSec} onChange={setDurationSec} />
        {assignUsers ? (
          <UserMultiSelect
            label="Assign to users (optional)"
            description="Attach this budget to existing users now. You can also manage assignments later on the Users page."
            value={userIds}
            onChange={setUserIds}
            users={assignUsers}
          />
        ) : null}
        <div className="flex gap-2">
          <Button variant="primary" isDisabled={isPending || !parsed.valid} onPress={submit}>
            {isPending ? "Saving…" : submitLabel}
          </Button>
          <Button variant="ghost" onPress={onClose}>
            Cancel
          </Button>
        </div>
      </Card.Content>
    </Card>
  );
}

// ---------- aggregate usage indicator ----------

// `max_budget` is a per-user cap and users share a budget, so the honest budget
// wide number is spend summed across assigned users against the total they are
// collectively allowed (cap × users). A bar only when both are meaningful.
function UsageCell({ budget }: { budget: Budget }) {
  if (budget.user_count === 0) {
    return <span className="text-xs text-[var(--otari-muted)]">No users assigned</span>;
  }
  const spent = budget.total_spend;
  if (budget.max_budget === null) {
    return (
      <span className="text-xs text-[var(--otari-ink)]">
        {formatUSD(spent)} spent<span className="text-[var(--otari-muted)]"> · no limit</span>
      </span>
    );
  }
  const allocated = budget.max_budget * budget.user_count;
  const pct = allocated > 0 ? Math.min(100, (spent / allocated) * 100) : 0;
  const over = spent > allocated;
  return (
    <div className="flex min-w-[140px] flex-col gap-1">
      <div className="flex items-baseline justify-between gap-2 text-xs">
        <span className="text-[var(--otari-ink)]">{formatUSD(spent)}</span>
        <span className="text-[var(--otari-muted)]">of {formatUSD(allocated)}</span>
      </div>
      <div
        className="h-1.5 w-full overflow-hidden rounded-full bg-[var(--otari-line)]"
        role="progressbar"
        aria-valuenow={Math.round(pct)}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label="Aggregate spend against total allocation"
      >
        <div
          className={`h-full rounded-full ${over ? "bg-red-500" : "bg-[var(--otari-brand)]"}`}
          style={{ width: `${Math.max(pct, over ? 100 : 2)}%` }}
        />
      </div>
    </div>
  );
}

// ---------- reset history drill-down ----------

function ResetHistory({ budgetId }: { budgetId: string }) {
  const logs = useBudgetResetLogs(budgetId);

  if (logs.isLoading) {
    return (
      <div className="flex items-center gap-2 px-4 py-4 text-sm text-[var(--otari-muted)]">
        <Spinner size="sm" /> Loading reset history…
      </div>
    );
  }
  if (logs.error) {
    return (
      <div className="px-4 py-4">
        <ErrorBanner error={logs.error} />
      </div>
    );
  }
  const rows = logs.data ?? [];
  if (rows.length === 0) {
    return <div className="px-4 py-4 text-sm text-[var(--otari-muted)]">No resets recorded yet for this budget.</div>;
  }
  return (
    <div className="overflow-x-auto px-4 py-3">
      <table className="w-full border-collapse text-xs">
        <thead className="text-left text-[var(--otari-muted)]">
          <tr>
            <th className="py-1.5 pr-4 font-medium">User</th>
            <th className="py-1.5 pr-4 font-medium">Spend cleared</th>
            <th className="py-1.5 pr-4 font-medium">Reset at</th>
            <th className="py-1.5 font-medium">Next reset</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((log: BudgetResetLog) => (
            <tr key={log.id} className="border-t border-[var(--otari-line)]">
              <td className="py-1.5 pr-4">
                <code>{log.user_id ?? "—"}</code>
              </td>
              <td className="py-1.5 pr-4 text-[var(--otari-ink)]">{formatUSD(log.previous_spend)}</td>
              <td className="py-1.5 pr-4 text-[var(--otari-muted)]">{absolute(log.reset_at)}</td>
              <td className="py-1.5 text-[var(--otari-muted)]">{absolute(log.next_reset_at)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---------- onboarding ----------

function OnboardingPanel({ onCreate }: { onCreate: () => void }) {
  return (
    <Card>
      <Card.Content className="flex flex-col gap-4 p-6">
        <div>
          <h2 className="text-lg font-semibold text-[var(--otari-ink)]">No budgets yet</h2>
          <p className="mt-1 text-sm text-[var(--otari-muted)]">
            A budget caps how much a user may spend and, optionally, resets that spend on a schedule. Create one, then
            assign it to users to enforce a limit.
          </p>
        </div>
        <div>
          <Button variant="primary" onPress={onCreate}>
            Create your first budget
          </Button>
        </div>
      </Card.Content>
    </Card>
  );
}

// ---------- inline confirm (names the target, no modal) ----------

function InlineDelete({ label, isPending, onConfirm }: { label: string; isPending: boolean; onConfirm: () => void }) {
  const [armed, setArmed] = useState(false);

  if (!armed) {
    return (
      <Button size="sm" variant="danger-soft" onPress={() => setArmed(true)}>
        Delete
      </Button>
    );
  }
  return (
    <div className="flex flex-col items-end gap-1.5 rounded-lg border border-amber-200 bg-amber-50 p-2 text-right">
      <span className="max-w-xs text-xs text-amber-800">
        Delete <strong>{label}</strong>? Users keep their spend but lose this limit. Cannot be undone.
      </span>
      <span className="inline-flex gap-1">
        <Button size="sm" variant="danger" isDisabled={isPending} onPress={onConfirm}>
          Delete permanently
        </Button>
        <Button size="sm" variant="ghost" isDisabled={isPending} onPress={() => setArmed(false)}>
          Cancel
        </Button>
      </span>
    </div>
  );
}

// ---------- page ----------

// A short, stable fingerprint for a budget id (its leading segment), shown when a
// budget has no name and used as a fallback label.
function shortId(budgetId: string): string {
  return budgetId.split("-")[0];
}

function budgetLabel(budget: Budget): string {
  return budget.name ?? shortId(budget.budget_id);
}

export function BudgetsPage() {
  const budgets = useBudgets();
  const users = useUsers();
  const createBudget = useCreateBudget();
  const updateBudget = useUpdateBudget();
  const deleteBudget = useDeleteBudget();
  const updateUser = useUpdateUser();

  const [addOpen, setAddOpen] = useState(false);
  const [editing, setEditing] = useState<string | null>(null);
  const [historyOpen, setHistoryOpen] = useState<string | null>(null);

  const rows = budgets.data ?? [];
  const loading = budgets.isLoading;
  const editingBudget = rows.find((b) => b.budget_id === editing) ?? null;
  const showOnboarding = !loading && rows.length === 0 && !addOpen;

  // Create the budget, then (optionally) attach it to the chosen users. The
  // per-user PATCH is what sets each user's reset clock, so it is reused rather
  // than duplicated here; allSettled so one failed assignment does not abort the
  // rest (any failure surfaces via updateUser.error).
  const createAndAssign = (body: CreateBudgetRequest, userIds: string[]) => {
    createBudget.mutate(body, {
      onSuccess: async (budget: Budget) => {
        if (userIds.length > 0) {
          await Promise.allSettled(
            userIds.map((id) => updateUser.mutateAsync({ id, body: { budget_id: budget.budget_id } })),
          );
        }
        setAddOpen(false);
      },
    });
  };

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Budgets"
        description="Define spending limits and reset schedules. Assign a budget to users to enforce it."
        action={
          addOpen || showOnboarding ? null : (
            <Button
              variant="primary"
              onPress={() => {
                setEditing(null);
                setAddOpen(true);
              }}
            >
              Create budget
            </Button>
          )
        }
      />

      <ErrorBanner error={budgets.error ?? createBudget.error ?? updateBudget.error ?? deleteBudget.error ?? updateUser.error} />

      <InfoBanner>
        Assign a budget to users when you create it, or later from the Users page. Each row&rsquo;s usage aggregates the
        spend of the users currently on that budget.
      </InfoBanner>

      {showOnboarding ? (
        <OnboardingPanel
          onCreate={() => {
            setEditing(null);
            setAddOpen(true);
          }}
        />
      ) : null}

      {addOpen ? (
        <BudgetForm
          title="Create budget"
          submitLabel="Create budget"
          initial={{ name: null, max_budget: null, budget_duration_sec: null }}
          error={createBudget.error}
          isPending={createBudget.isPending}
          assignUsers={users.data ?? []}
          onSubmit={createAndAssign}
          onClose={() => setAddOpen(false)}
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
          error={updateBudget.error}
          isPending={updateBudget.isPending}
          onSubmit={(body) =>
            updateBudget.mutate({ id: editingBudget.budget_id, body }, { onSuccess: () => setEditing(null) })
          }
          onClose={() => setEditing(null)}
        />
      ) : null}

      <Table>
        <THead>
          <tr>
            <Th>Budget</Th>
            <Th>Limit (per user)</Th>
            <Th>Reset</Th>
            <Th>Users</Th>
            <Th>Usage</Th>
            <Th className="text-right">Actions</Th>
          </tr>
        </THead>
        <tbody>
          {loading ? (
            <LoadingRow colSpan={6} />
          ) : rows.length === 0 ? (
            <TableMessage colSpan={6}>No budgets yet. Create one to cap spending.</TableMessage>
          ) : (
            rows.map((b) => (
              <Fragment key={b.budget_id}>
                <Tr
                  selected={editing === b.budget_id}
                  onClick={() => {
                    setAddOpen(false);
                    setEditing(b.budget_id);
                  }}
                >
                  <Td className="font-medium text-[var(--otari-ink)]">
                    <div className="flex flex-col gap-0.5">
                      <span>{b.name ?? <span className="text-[var(--otari-muted)]">(unnamed)</span>}</span>
                      <code className="text-[11px] text-[var(--otari-muted)]" title={b.budget_id}>
                        {shortId(b.budget_id)}
                      </code>
                    </div>
                  </Td>
                  <Td>{b.max_budget === null ? <span className="text-[var(--otari-muted)]">Unlimited</span> : formatUSD(b.max_budget)}</Td>
                  <Td className="text-[var(--otari-muted)]">{formatDuration(b.budget_duration_sec)}</Td>
                  <Td className="text-[var(--otari-muted)]">{b.user_count}</Td>
                  <Td>
                    <UsageCell budget={b} />
                  </Td>
                  <Td>
                    {/* Row click opens Edit; the action buttons stop propagation so
                        they fire their own handler instead of also opening Edit. */}
                    <div className="flex items-center justify-end gap-1.5" onClick={(e) => e.stopPropagation()}>
                      <Button
                        size="sm"
                        variant="ghost"
                        onPress={() =>
                          setHistoryOpen((current) => (current === b.budget_id ? null : b.budget_id))
                        }
                      >
                        {historyOpen === b.budget_id ? "Hide history" : "History"}
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        onPress={() => {
                          setAddOpen(false);
                          setEditing(b.budget_id);
                        }}
                      >
                        Edit
                      </Button>
                      <InlineDelete
                        label={budgetLabel(b)}
                        isPending={deleteBudget.isPending}
                        onConfirm={() => deleteBudget.mutate(b.budget_id)}
                      />
                    </div>
                  </Td>
                </Tr>
                {historyOpen === b.budget_id ? (
                  <tr className="border-b border-[var(--otari-line)] bg-[var(--otari-bg)]">
                    <td colSpan={6}>
                      <ResetHistory budgetId={b.budget_id} />
                    </td>
                  </tr>
                ) : null}
              </Fragment>
            ))
          )}
        </tbody>
      </Table>
    </div>
  );
}
