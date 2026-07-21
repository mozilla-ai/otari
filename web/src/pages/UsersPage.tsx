import { Button, Card, Chip } from "@heroui/react";
import { type ReactNode, useState } from "react";

import { useBudgets, useCreateUser, useDeleteUser, useUpdateUser, useUsers } from "@/api/hooks";
import type { Budget, CreateUserRequest, UpdateUserRequest, User } from "@/api/types";
import { Field } from "@/components/Field";
import { accessLabel, ModelScopeControl } from "@/components/ModelScopeControl";
import { LoadingRow, Table, TableMessage, Td, Th, THead, Tr } from "@/components/Table";
import { ErrorBanner, PageHeader } from "@/components/ui";

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

  const setBlocked = (u: User, blocked: boolean) =>
    updateUser.mutate({ id: u.user_id, body: { blocked } });

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

      <Table>
        <THead>
          <tr>
            <Th>User</Th>
            <Th>Status</Th>
            <Th>Budget</Th>
            <Th>Spend</Th>
            <Th>Model access</Th>
            <Th className="text-right">Actions</Th>
          </tr>
        </THead>
        <tbody>
          {loading ? (
            <LoadingRow colSpan={6} />
          ) : rows.length === 0 ? (
            <TableMessage colSpan={6}>No users yet. Create one, or create an API key to auto-create one.</TableMessage>
          ) : (
            rows.map((u) => {
              const budget = u.budget_id ? budgetById.get(u.budget_id) : undefined;
              return (
                  <Tr
                    key={u.user_id}
                    selected={editing === u.user_id}
                    onClick={() => {
                      setAddOpen(false);
                      setEditing(u.user_id);
                    }}
                  >
                    <Td className="font-medium text-[var(--otari-ink)]">
                      <div className="flex flex-col gap-0.5">
                        <span className="inline-flex items-center gap-1.5">
                          <code className="text-xs">{u.user_id}</code>
                          {isVirtualUser(u.user_id) ? (
                            <Chip size="sm" color="default">
                              virtual
                            </Chip>
                          ) : null}
                        </span>
                        {u.alias ? <span className="text-xs text-[var(--otari-muted)]">{u.alias}</span> : null}
                      </div>
                    </Td>
                    <Td>
                      <StatusChip user={u} />
                    </Td>
                    <Td className="text-[var(--otari-muted)]">
                      {u.budget_id ? (
                        <span title={u.budget_id}>{budget ? budgetLabel(budget) : shortId(u.budget_id)}</span>
                      ) : (
                        "—"
                      )}
                    </Td>
                    <Td className="text-[var(--otari-muted)]">
                      {formatUSD(u.spend)}
                      {u.reserved > 0 ? (
                        <span className="text-[var(--otari-muted)]"> (+{formatUSD(u.reserved)} held)</span>
                      ) : null}
                    </Td>
                    <Td>
                      <AccessChip allowed={u.allowed_models} />
                    </Td>
                    <Td>
                      {/* Row click opens Edit; action buttons stop propagation so
                          they fire their own handler instead of also opening Edit. */}
                      <div className="flex items-center justify-end gap-1.5" onClick={(e) => e.stopPropagation()}>
                        {u.blocked ? (
                          <Button
                            size="sm"
                            variant="outline"
                            isDisabled={updateUser.isPending}
                            onPress={() => setBlocked(u, false)}
                          >
                            Unblock
                          </Button>
                        ) : (
                          <Button
                            size="sm"
                            variant="outline"
                            isDisabled={updateUser.isPending}
                            onPress={() => setBlocked(u, true)}
                          >
                            Block
                          </Button>
                        )}
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
                              Delete <strong>{u.user_id}</strong>? This deactivates its API keys and hides the user;
                              usage history is preserved.
                            </>
                          }
                          onConfirm={() => deleteUser.mutate(u.user_id)}
                        />
                      </div>
                    </Td>
                  </Tr>
              );
            })
          )}
        </tbody>
      </Table>
    </div>
  );
}
