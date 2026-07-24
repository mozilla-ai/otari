import { AlertDialog, Button, Card, Chip } from "@heroui/react";
import { type ReactNode, useEffect, useState } from "react";

import { useBudgets, useCreateUser, useDeleteUser, useUpdateUser, useUsers } from "@/api/hooks";
import type { Budget, CreateUserRequest, UpdateUserRequest, User } from "@/api/types";
import { BulkActionBar } from "@/components/BulkActionBar";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { DataTable, type DataTableColumn } from "@/components/DataTable";
import { Field } from "@/components/Field";
import { accessLabel, ModelScopeControl } from "@/components/ModelScopeControl";
import { ErrorBanner, FilterSelect, PageHeader } from "@/components/ui";
import { resolveSelectedIds, useTableSelection } from "@/lib/tableSelection";

// ---------- formatting ----------

const usd = new Intl.NumberFormat(undefined, { style: "currency", currency: "USD", maximumFractionDigits: 4 });

function formatUSD(value: number): string {
  return usd.format(value);
}

// A user auto-created to own a single API key (id "apikey-..."), rather than one
// an operator named. Shown as such so the list is not full of opaque virtual ids.
const isVirtualUser = (userId: string): boolean => userId.startsWith("apikey-");

function shortId(budgetId: string): string {
  return budgetId.split("-")[0];
}

function budgetLabel(budget: Budget): string {
  return budget.name ?? shortId(budget.budget_id);
}

// ---------- budget picker ----------

function BudgetSelect({
  value,
  onChange,
  budgets,
}: {
  value: string | null;
  onChange: (budgetId: string | null) => void;
  budgets: Budget[];
}) {
  return (
    <div className="flex flex-col gap-1">
      <label htmlFor="user-budget" className="text-sm font-medium text-[var(--otari-ink)]">
        Budget
      </label>
      <select
        id="user-budget"
        value={value ?? ""}
        onChange={(e) => onChange(e.target.value || null)}
        className="w-full rounded-lg border border-[var(--otari-line)] bg-[var(--otari-bg)] px-3 py-2 text-sm text-[var(--otari-ink)]"
      >
        <option value="">No budget (unlimited)</option>
        {budgets.map((b) => (
          <option key={b.budget_id} value={b.budget_id}>
            {budgetLabel(b)}
            {b.max_budget === null ? " · no limit" : ` · ${formatUSD(b.max_budget)}`}
          </option>
        ))}
      </select>
      <span className="text-xs text-[var(--otari-muted)]">
        The spending limit this user is held to. Manage budgets on the Budgets page.
      </span>
    </div>
  );
}

// ---------- create / edit forms (inline cards, matching KeysPage) ----------

function CreateUserForm({ onClose }: { onClose: () => void }) {
  const create = useCreateUser();
  const budgets = useBudgets();
  const [userId, setUserId] = useState("");
  const [alias, setAlias] = useState("");
  const [budgetId, setBudgetId] = useState<string | null>(null);
  const [allowedModels, setAllowedModels] = useState<string[] | null>(null);
  const [scopeValid, setScopeValid] = useState(true);

  const submit = () => {
    if (create.isPending || !scopeValid || userId.trim() === "") return;
    const body: CreateUserRequest = {
      user_id: userId.trim(),
      alias: alias.trim() || null,
      budget_id: budgetId,
      allowed_models: allowedModels,
    };
    create.mutate(body, { onSuccess: onClose });
  };

  return (
    <Card>
      <Card.Content className="flex flex-col gap-4 p-5">
        <div className="text-sm font-semibold text-[var(--otari-ink)]">Create user</div>
        <ErrorBanner error={create.error} />
        <div className="grid gap-4 sm:grid-cols-2">
          <Field
            label="User ID"
            value={userId}
            onChange={setUserId}
            placeholder="alice@example.com"
            isRequired
            autoFocus
            description="The identifier callers send as the `user` field; spend and budgets track against it."
          />
          <Field label="Alias (optional)" value={alias} onChange={setAlias} placeholder="Alice" />
        </div>
        <BudgetSelect value={budgetId} onChange={setBudgetId} budgets={budgets.data ?? []} />
        <ModelScopeControl
          title="Model access (default for this user's keys)"
          description="The models this user's keys may list and call by default. A key can narrow this, but never exceed it."
          initial={null}
          onChange={(value, valid) => {
            setAllowedModels(value);
            setScopeValid(valid);
          }}
        />
        <div className="flex gap-2">
          <Button
            variant="primary"
            isDisabled={create.isPending || !scopeValid || userId.trim() === ""}
            onPress={submit}
          >
            {create.isPending ? "Creating…" : "Create user"}
          </Button>
          <Button variant="ghost" onPress={onClose}>
            Cancel
          </Button>
        </div>
      </Card.Content>
    </Card>
  );
}

function EditUserForm({ user, onClose }: { user: User; onClose: () => void }) {
  const update = useUpdateUser();
  const budgets = useBudgets();
  const [alias, setAlias] = useState(user.alias ?? "");
  const [budgetId, setBudgetId] = useState<string | null>(user.budget_id);
  const [allowedModels, setAllowedModels] = useState<string[] | null>(user.allowed_models);
  const [scopeValid, setScopeValid] = useState(true);

  const submit = () => {
    if (update.isPending || !scopeValid) return;
    const body: UpdateUserRequest = {
      alias: alias.trim() || null,
      budget_id: budgetId,
      allowed_models: allowedModels,
    };
    update.mutate({ id: user.user_id, body }, { onSuccess: onClose });
  };

  return (
    <Card>
      <Card.Content className="flex flex-col gap-4 p-5">
        <div className="text-sm font-semibold text-[var(--otari-ink)]">
          Edit <code>{user.user_id}</code>
        </div>
        <ErrorBanner error={update.error} />
        <Field label="Alias" value={alias} onChange={setAlias} placeholder="Alice" />
        <BudgetSelect value={budgetId} onChange={setBudgetId} budgets={budgets.data ?? []} />
        <ModelScopeControl
          title="Model access (default for this user's keys)"
          description="The models this user's keys may list and call by default. A key can narrow this, but never exceed it."
          initial={user.allowed_models}
          onChange={(value, valid) => {
            setAllowedModels(value);
            setScopeValid(valid);
          }}
        />
        <div className="flex gap-2">
          <Button variant="primary" isDisabled={update.isPending || !scopeValid} onPress={submit}>
            {update.isPending ? "Saving…" : "Save changes"}
          </Button>
          <Button variant="ghost" onPress={onClose}>
            Cancel
          </Button>
        </div>
      </Card.Content>
    </Card>
  );
}

// ---------- inline confirm ----------

function InlineConfirm({
  trigger,
  message,
  confirmLabel,
  isPending,
  onConfirm,
}: {
  trigger: string;
  message: ReactNode;
  confirmLabel: string;
  isPending: boolean;
  onConfirm: () => void;
}) {
  const [armed, setArmed] = useState(false);
  if (!armed) {
    return (
      <Button size="sm" variant="danger-soft" onPress={() => setArmed(true)}>
        {trigger}
      </Button>
    );
  }
  return (
    <div className="flex flex-col items-end gap-1.5 rounded-lg border border-amber-200 bg-amber-50 p-2 text-right">
      <span className="max-w-xs text-xs text-amber-800">{message}</span>
      <span className="inline-flex gap-1">
        <Button size="sm" variant="danger" isDisabled={isPending} onPress={onConfirm}>
          {confirmLabel}
        </Button>
        <Button size="sm" variant="ghost" isDisabled={isPending} onPress={() => setArmed(false)}>
          Cancel
        </Button>
      </span>
    </div>
  );
}

// ---------- status + access chips ----------

function StatusChip({ user }: { user: User }) {
  return user.blocked ? (
    <Chip size="sm" color="warning">
      Blocked
    </Chip>
  ) : (
    <Chip size="sm" color="accent">
      Active
    </Chip>
  );
}

function AccessChip({ allowed }: { allowed: string[] | null }) {
  const { text, tone } = accessLabel(allowed);
  const cls =
    tone === "danger"
      ? "text-red-700 font-medium"
      : tone === "muted"
        ? "text-[var(--otari-muted)]"
        : "text-[var(--otari-brand-dark)] font-medium";
  const title = allowed && allowed.length > 0 ? allowed.join(", ") : undefined;
  return (
    <span className={`text-xs ${cls}`} title={title}>
      {text}
    </span>
  );
}

// ---------- onboarding ----------

function OnboardingPanel({ onCreate }: { onCreate: () => void }) {
  return (
    <Card>
      <Card.Content className="flex flex-col gap-4 p-6">
        <div>
          <h2 className="text-lg font-semibold text-[var(--otari-ink)]">No users yet</h2>
          <p className="mt-1 text-sm text-[var(--otari-muted)]">
            A user owns API keys and carries the budget and default model access those keys inherit. Create a user here,
            then issue its keys on the API keys page.
          </p>
        </div>
        <div>
          <Button variant="primary" onPress={onCreate}>
            Create your first user
          </Button>
        </div>
      </Card.Content>
    </Card>
  );
}

// Assign one budget to a set of selected users at once.
function AssignBudgetDialog({
  isOpen,
  onOpenChange,
  budgets,
  count,
  isPending,
  error,
  onAssign,
}: {
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
  budgets: Budget[];
  count: number;
  isPending: boolean;
  error: unknown;
  onAssign: (budgetId: string) => void;
}) {
  const [budgetId, setBudgetId] = useState("");
  // Reset the picker each time the dialog opens so a prior selection is not inherited.
  useEffect(() => {
    if (isOpen) {
      setBudgetId("");
    }
  }, [isOpen]);
  return (
    <AlertDialog isOpen={isOpen} onOpenChange={onOpenChange}>
      {isOpen ? (
        <AlertDialog.Backdrop>
          <AlertDialog.Container placement="center" size="md">
            <AlertDialog.Dialog>
              <AlertDialog.Header>
                <AlertDialog.Heading>Assign budget</AlertDialog.Heading>
              </AlertDialog.Header>
              <AlertDialog.Body className="flex flex-col gap-4">
                <p className="text-sm text-[var(--otari-muted)]">
                  Assign a budget to {count} selected {count === 1 ? "user" : "users"}.
                </p>
                <FilterSelect
                  label="Budget"
                  value={budgetId}
                  onChange={setBudgetId}
                  options={[
                    { value: "", label: "Select a budget…" },
                    ...budgets.map((b) => ({ value: b.budget_id, label: budgetLabel(b) })),
                  ]}
                />
                <ErrorBanner error={error} />
              </AlertDialog.Body>
              <AlertDialog.Footer>
                <Button variant="ghost" isDisabled={isPending} onPress={() => onOpenChange(false)}>
                  Cancel
                </Button>
                <Button variant="primary" isDisabled={!budgetId} isPending={isPending} onPress={() => onAssign(budgetId)}>
                  Assign
                </Button>
              </AlertDialog.Footer>
            </AlertDialog.Dialog>
          </AlertDialog.Container>
        </AlertDialog.Backdrop>
      ) : null}
    </AlertDialog>
  );
}

// ---------- page ----------

export function UsersPage() {
  const users = useUsers();
  const budgets = useBudgets();
  const updateUser = useUpdateUser();
  const deleteUser = useDeleteUser();

  const [addOpen, setAddOpen] = useState(false);
  const [editing, setEditing] = useState<string | null>(null);
  const [showVirtual, setShowVirtual] = useState(false);

  const allRows = users.data ?? [];
  const loading = users.isLoading;
  // Virtual users are per-key shadows the API auto-creates; they clutter the list
  // of people/teams you actually manage, so hide them behind a toggle.
  const virtualCount = allRows.filter((u) => isVirtualUser(u.user_id)).length;
  const rows = showVirtual ? allRows : allRows.filter((u) => !isVirtualUser(u.user_id));
  const editingUser = allRows.find((u) => u.user_id === editing) ?? null;
  const showOnboarding = !loading && rows.length === 0 && !addOpen;

  const budgetById = new Map((budgets.data ?? []).map((b) => [b.budget_id, b]));
  const selection = useTableSelection();
  const [bulkDeleteOpen, setBulkDeleteOpen] = useState(false);
  const [assignOpen, setAssignOpen] = useState(false);
  const [bulkError, setBulkError] = useState<unknown>(undefined);
  const [bulkPending, setBulkPending] = useState(false);

  const selectableKeys = rows.map((u) => u.user_id);
  const selectedIds = resolveSelectedIds(selection.selectedKeys, selectableKeys);

  const setBlocked = (u: User, blocked: boolean) =>
    updateUser.mutate({ id: u.user_id, body: { blocked } });

  const runBulk = async (action: (id: string) => Promise<unknown>, onDone: () => void) => {
    setBulkPending(true);
    setBulkError(undefined);
    try {
      for (const id of selectedIds) {
        await action(id);
      }
      selection.clear();
      onDone();
    } catch (error) {
      setBulkError(error);
    } finally {
      setBulkPending(false);
    }
  };

  const columns: DataTableColumn<User>[] = [
    {
      id: "user",
      header: "User",
      isRowHeader: true,
      cell: (u) => (
        <div className="flex flex-col gap-0.5">
          <span className="inline-flex items-center gap-1.5">
            <code className="text-xs font-medium text-[var(--otari-ink)]">{u.user_id}</code>
            {isVirtualUser(u.user_id) ? (
              <Chip size="sm" color="default">
                virtual
              </Chip>
            ) : null}
          </span>
          {u.alias ? <span className="text-xs text-[var(--otari-muted)]">{u.alias}</span> : null}
        </div>
      ),
    },
    { id: "status", header: "Status", cell: (u) => <StatusChip user={u} /> },
    {
      id: "budget",
      header: "Budget",
      cell: (u) =>
        u.budget_id ? (
          <span className="text-[var(--otari-muted)]" title={u.budget_id}>
            {budgetById.get(u.budget_id) ? budgetLabel(budgetById.get(u.budget_id)!) : shortId(u.budget_id)}
          </span>
        ) : (
          <span className="text-[var(--otari-muted)]">—</span>
        ),
    },
    {
      id: "spend",
      header: "Spend",
      cell: (u) => (
        <span className="text-[var(--otari-muted)]">
          {formatUSD(u.spend)}
          {u.reserved > 0 ? <span> (+{formatUSD(u.reserved)} held)</span> : null}
        </span>
      ),
    },
    { id: "access", header: "Model access", cell: (u) => <AccessChip allowed={u.allowed_models} /> },
    {
      id: "actions",
      header: "Actions",
      align: "end",
      cell: (u) => (
        <div className="flex items-center justify-end gap-1.5">
          <Button
            size="sm"
            variant="outline"
            isDisabled={updateUser.isPending}
            onPress={() => setBlocked(u, !u.blocked)}
          >
            {u.blocked ? "Unblock" : "Block"}
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onPress={() => {
              setAddOpen(false);
              setEditing(u.user_id);
            }}
          >
            Edit
          </Button>
          <InlineConfirm
            trigger="Delete"
            confirmLabel="Delete user"
            isPending={deleteUser.isPending}
            message={
              <>
                Delete <strong>{u.user_id}</strong>? This deactivates its API keys and hides the user; usage history is
                preserved.
              </>
            }
            onConfirm={() => deleteUser.mutate(u.user_id)}
          />
        </div>
      ),
    },
  ];

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Users"
        description="People and teams that own API keys. Set each one's budget and default model access here; issue their keys on the API keys page."
        action={
          addOpen ? null : (
            <Button
              variant="primary"
              onPress={() => {
                setEditing(null);
                setAddOpen(true);
              }}
            >
              Create user
            </Button>
          )
        }
      />

      <ErrorBanner error={users.error ?? updateUser.error ?? deleteUser.error} />

      {showOnboarding ? (
        <OnboardingPanel
          onCreate={() => {
            setEditing(null);
            setAddOpen(true);
          }}
        />
      ) : null}

      {virtualCount > 0 ? (
        <label className="flex w-fit items-center gap-2 text-xs text-[var(--otari-muted)]">
          <input type="checkbox" checked={showVirtual} onChange={(e) => setShowVirtual(e.target.checked)} />
          Show auto-created (virtual) users ({virtualCount})
        </label>
      ) : null}

      {addOpen ? <CreateUserForm onClose={() => setAddOpen(false)} /> : null}
      {/* Key on the user id so switching which user is edited remounts the form:
          its fields seed from `user` on mount only. */}
      {editingUser ? <EditUserForm key={editingUser.user_id} user={editingUser} onClose={() => setEditing(null)} /> : null}

      {selectedIds.length > 0 ? (
        <BulkActionBar
          selectedCount={selectedIds.length}
          allMatching={false}
          matchingTotal={null}
          canSelectAllMatching={false}
          onSelectAllMatching={() => {}}
          onClear={selection.clear}
        >
          <Button size="sm" variant="primary" onPress={() => setAssignOpen(true)}>
            Assign budget
          </Button>
          <Button size="sm" variant="danger" onPress={() => setBulkDeleteOpen(true)}>
            Delete
          </Button>
        </BulkActionBar>
      ) : null}

      <DataTable
        ariaLabel="Users"
        columns={columns}
        rows={rows}
        getRowKey={(u) => u.user_id}
        isLoading={loading}
        emptyContent="No users yet. Create one, or create an API key to auto-create one."
        selectionMode="multiple"
        selectedKeys={selection.selectedKeys}
        onSelectionChange={selection.onSelectionChange}
      />

      <ConfirmDialog
        isOpen={bulkDeleteOpen}
        onOpenChange={setBulkDeleteOpen}
        heading="Delete users"
        body={`Delete ${selectedIds.length} ${
          selectedIds.length === 1 ? "user" : "users"
        }? This deactivates their API keys and hides them; usage history is preserved.`}
        confirmLabel="Delete"
        isPending={bulkPending}
        error={bulkError}
        onConfirm={() => runBulk((id) => deleteUser.mutateAsync(id), () => setBulkDeleteOpen(false))}
      />

      <AssignBudgetDialog
        isOpen={assignOpen}
        onOpenChange={setAssignOpen}
        budgets={budgets.data ?? []}
        count={selectedIds.length}
        isPending={bulkPending}
        error={bulkError}
        onAssign={(budgetId) =>
          runBulk((id) => updateUser.mutateAsync({ id, body: { budget_id: budgetId } }), () => setAssignOpen(false))
        }
      />
    </div>
  );
}
