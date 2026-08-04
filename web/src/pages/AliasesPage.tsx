import { Button, Card, Chip } from "@heroui/react";
import { useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";

import type { AliasResponse } from "@/api/types";
import { useAliases, useCreateAlias, useDeleteAlias, useUsers } from "@/api/hooks";
import { BulkActionBar } from "@/components/BulkActionBar";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { DataTable, type DataTableColumn } from "@/components/DataTable";
import { Field } from "@/components/Field";
import { ModelComboBox } from "@/components/ModelComboBox";
import { UserComboBox } from "@/components/UserComboBox";
import { ConfirmButton, CopyableValue, ErrorBanner, PageHeader } from "@/components/ui";
import { resolveSelectedIds, useTableSelection } from "@/lib/tableSelection";

// Stable row-key getter so DataTable's per-row cache holds across re-renders.
// Scope is part of the key: the same name can exist globally and per user, and
// keying on the name alone would collapse those rows into one. JSON encodes the
// pair rather than joining on a separator, because neither half is
// delimiter-free (a user id has no format restriction, and an alias name only
// bans ":" and "/"), and because DataTable puts this key in a `data-key` DOM
// attribute, which is no place for a control character. The key stays opaque:
// nothing splits it apart, and code needing a row's scope looks the row up.
const getAliasRowKey = (a: AliasResponse): string => JSON.stringify([a.user_id, a.name]);

// Who an alias applies to. Global is the default and the pre-existing behavior;
// scoping to a user lets one display name mean a different model per person, and
// overrides a global alias of the same name for that user only.
function ScopePicker({
  userId,
  onChange,
}: {
  userId: string | null;
  onChange: (userId: string | null) => void;
}) {
  const users = useUsers();
  const scoped = userId !== null;

  const modeButton = (value: boolean, label: string) => (
    <button
      type="button"
      aria-pressed={scoped === value}
      onClick={() => onChange(value ? "" : null)}
      className={
        scoped === value
          ? "rounded-md bg-white px-3 py-1.5 text-sm font-medium text-[var(--otari-ink)] shadow-sm"
          : "rounded-md px-3 py-1.5 text-sm text-[var(--otari-muted)] hover:text-[var(--otari-ink)]"
      }
    >
      {label}
    </button>
  );

  return (
    <div className="flex flex-col gap-3">
      <div>
        <span className="text-sm font-medium text-[var(--otari-ink)]">Applies to</span>
        <p className="text-xs text-[var(--otari-muted)]">
          A global alias resolves for every caller. A user-scoped one resolves only for that user, and takes
          precedence over a global alias of the same name.
        </p>
      </div>
      <div className="flex w-fit items-center gap-1 rounded-lg bg-[var(--otari-bg)] p-1">
        {modeButton(false, "Every caller")}
        {modeButton(true, "One user")}
      </div>
      {scoped ? (
        <UserComboBox
          label="User"
          value={userId ?? ""}
          onChange={onChange}
          users={users.data ?? []}
          placeholder="Pick a user…"
          description="Only this user resolves the alias."
          unknownHint={<span className="text-red-700">No such user. Pick an existing one.</span>}
        />
      ) : null}
    </div>
  );
}

// Edit an existing stored alias's target. Name and scope together are the lookup
// key and are shown read-only; the backend POST /v1/aliases upserts on that pair,
// so the same hook serves both create and edit.
function EditAliasForm({ alias, onClose }: { alias: AliasResponse; onClose: () => void }) {
  const updateAlias = useCreateAlias();
  const [target, setTarget] = useState(alias.target);

  const targetChanged = target.trim() !== "" && target.trim() !== alias.target;

  const submit = () => {
    if (!targetChanged) return;
    updateAlias.mutate({ name: alias.name, target: target.trim(), user_id: alias.user_id }, { onSuccess: onClose });
  };

  return (
    <Card>
      <Card.Content className="flex flex-col gap-4 p-5">
        <div className="text-sm font-semibold text-[var(--otari-ink)]">
          Edit alias <code>{alias.name}</code>
          {alias.user_id ? <> for user <code>{alias.user_id}</code></> : null}
        </div>
        <ErrorBanner error={updateAlias.error} />
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="flex flex-col gap-1">
            <span className="text-sm font-medium text-[var(--otari-ink)]">Alias name</span>
            <code className="text-sm text-[var(--otari-muted)]">{alias.name}</code>
            <span className="text-xs text-[var(--otari-muted)]">
              The alias name and who it applies to ({alias.user_id ? <code>{alias.user_id}</code> : "every caller"})
              are the key and cannot be changed here. Delete and recreate to change either.
            </span>
          </div>
          <ModelComboBox
            label="Target"
            value={target}
            onChange={setTarget}
            isRequired
            description="The real model this resolves to. Callers never see it."
          />
        </div>
        <div className="flex gap-2">
          <Button variant="primary" isDisabled={!targetChanged || updateAlias.isPending} onPress={submit}>
            {updateAlias.isPending ? "Saving…" : "Save changes"}
          </Button>
          <Button variant="ghost" onPress={onClose}>
            Cancel
          </Button>
        </div>
      </Card.Content>
    </Card>
  );
}

// Create a stored alias. Pricing, budgets, and usage all key on the target, so
// an alias is never priced here (see the Models page for the target's price).
function NewAliasForm({ onClose, initialTarget = "" }: { onClose: () => void; initialTarget?: string }) {
  const createAlias = useCreateAlias();
  const [name, setName] = useState("");
  const [target, setTarget] = useState(initialTarget);
  // null = global. "" is "One user" chosen but not yet picked, which is an
  // incomplete form rather than a global alias.
  const [userId, setUserId] = useState<string | null>(null);

  const nameHasDelimiter = /[:/]/.test(name);
  const scopeReady = userId === null || userId.trim() !== "";
  const canSubmit = name.trim() !== "" && target.trim() !== "" && !nameHasDelimiter && scopeReady;

  const submit = () => {
    if (!canSubmit) return;
    createAlias.mutate(
      { name: name.trim(), target: target.trim(), user_id: userId === null ? null : userId.trim() },
      { onSuccess: onClose },
    );
  };

  return (
    <Card>
      <Card.Content className="flex flex-col gap-4 p-5">
        <div className="text-sm font-semibold text-[var(--otari-ink)]">New alias</div>
        <ErrorBanner error={createAlias.error} />
        <div className="grid gap-4 sm:grid-cols-2">
          <Field
            label="Alias name"
            value={name}
            onChange={setName}
            placeholder="fast-model"
            isRequired
            autoFocus
            description={
              nameHasDelimiter ? (
                <span className="text-red-700">An alias name cannot contain “:” or “/”.</span>
              ) : (
                "What callers send as `model`."
              )
            }
          />
          <ModelComboBox
            label="Target"
            value={target}
            onChange={setTarget}
            isRequired
            description="The real model this resolves to. Callers never see it."
          />
        </div>
        <ScopePicker userId={userId} onChange={setUserId} />
        <div className="flex gap-2">
          <Button variant="primary" isDisabled={!canSubmit || createAlias.isPending} onPress={submit}>
            {createAlias.isPending ? "Creating…" : "Create alias"}
          </Button>
          <Button variant="ghost" onPress={onClose}>
            Cancel
          </Button>
        </div>
      </Card.Content>
    </Card>
  );
}

export function AliasesPage() {
  const aliases = useAliases();
  const deleteAlias = useDeleteAlias();
  // A "Make an alias" link from the Models page arrives as ?target=provider:model,
  // opening the form with the target prefilled.
  const [searchParams] = useSearchParams();
  const initialTarget = searchParams.get("target") ?? "";
  const [adding, setAdding] = useState(initialTarget !== "");
  const [editing, setEditing] = useState<AliasResponse | null>(null);
  const selection = useTableSelection();
  const [bulkDeleteOpen, setBulkDeleteOpen] = useState(false);
  const [bulkError, setBulkError] = useState<unknown>(undefined);
  const [bulkPending, setBulkPending] = useState(false);

  const rows = [...(aliases.data ?? [])].sort(
    (a, b) => a.name.localeCompare(b.name) || (a.user_id ?? "").localeCompare(b.user_id ?? ""),
  );
  // Only stored aliases can be deleted; config.yml aliases are read-only.
  const selectableKeys = rows.filter((a) => a.source === "stored").map(getAliasRowKey);
  const disabledKeys = rows.filter((a) => a.source !== "stored").map(getAliasRowKey);
  const selectedRowKeys = resolveSelectedIds(selection.selectedKeys, selectableKeys);
  // Selection hands back row keys; the row they came from carries the scope a
  // delete needs. Looked up rather than unpacked from the key, so the key format
  // stays an implementation detail of the table rather than a wire contract.
  const byRowKey = new Map(rows.map((a) => [getAliasRowKey(a), a]));

  const onBulkDelete = async () => {
    setBulkPending(true);
    setBulkError(undefined);
    try {
      // No bulk endpoint for aliases; delete sequentially so one failure surfaces
      // without firing the rest in parallel against the same small list.
      for (const key of selectedRowKeys) {
        const alias = byRowKey.get(key);
        if (alias === undefined) continue;
        await deleteAlias.mutateAsync({ name: alias.name, userId: alias.user_id });
      }
      selection.clear();
      setBulkDeleteOpen(false);
    } catch (error) {
      setBulkError(error);
    } finally {
      setBulkPending(false);
    }
  };

  // Memoized on the values the cells actually read so DataTable's per-row
  // cache holds across selection clicks; see the DataTable docstring.
  const columns = useMemo<DataTableColumn<AliasResponse>[]>(() => [
    {
      id: "alias",
      header: "Alias",
      isRowHeader: true,
      cell: (a) => (
        <CopyableValue value={a.name} label="alias name" className="font-medium break-all text-[var(--otari-ink)]" />
      ),
    },
    {
      id: "target",
      header: "Target",
      cell: (a) => (
        <CopyableValue value={a.target} label="target model id" className="break-all text-[var(--otari-muted)]" />
      ),
    },
    {
      id: "scope",
      header: "Applies to",
      cell: (a) =>
        a.user_id ? (
          <CopyableValue value={a.user_id} label="user id">
            <code className="break-all text-xs text-[var(--otari-ink)]">{a.user_id}</code>
          </CopyableValue>
        ) : (
          <span className="text-xs text-[var(--otari-muted)]">Every caller</span>
        ),
    },
    {
      id: "source",
      header: "Source",
      cell: (a) => (
        <Chip size="sm" color={a.source === "stored" ? "accent" : "default"}>
          {a.source}
        </Chip>
      ),
    },
    {
      id: "actions",
      header: "Actions",
      align: "end",
      cell: (a) =>
        a.source === "stored" ? (
          <span className="inline-flex items-center gap-2 whitespace-nowrap">
            <Button
              size="sm"
              variant="ghost"
              onPress={() => {
                setAdding(false);
                setEditing(a);
              }}
            >
              Edit
            </Button>
            <ConfirmButton
              confirmLabel="Delete"
              isPending={deleteAlias.isPending}
              onConfirm={() => deleteAlias.mutate({ name: a.name, userId: a.user_id })}
            >
              Delete
            </ConfirmButton>
          </span>
        ) : (
          <span className="text-xs text-[var(--otari-muted)]">set in config.yml</span>
        ),
    },
  ], [deleteAlias.isPending, deleteAlias.mutate]);

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Aliases"
        description="Friendly names that map to a real provider:model. Callers send the alias as the model; pricing, budgets, and usage key on the target."
        action={
          adding || editing ? null : (
            <Button
              variant="primary"
              onPress={() => {
                setEditing(null);
                setAdding(true);
              }}
            >
              New alias
            </Button>
          )
        }
      />

      <ErrorBanner error={aliases.error} />

      {adding ? <NewAliasForm initialTarget={initialTarget} onClose={() => setAdding(false)} /> : null}
      {editing ? <EditAliasForm alias={editing} onClose={() => setEditing(null)} /> : null}

      {deleteAlias.error ? <ErrorBanner error={deleteAlias.error} /> : null}

      {selectedRowKeys.length > 0 ? (
        <BulkActionBar
          selectedCount={selectedRowKeys.length}
          allMatching={false}
          matchingTotal={null}
          canSelectAllMatching={false}
          onSelectAllMatching={() => {}}
          onClear={selection.clear}
        >
          <Button size="sm" variant="danger" onPress={() => setBulkDeleteOpen(true)}>
            Delete
          </Button>
        </BulkActionBar>
      ) : null}

      <DataTable
        ariaLabel="Aliases"
        columns={columns}
        rows={rows}
        getRowKey={getAliasRowKey}
        isLoading={aliases.isLoading}
        emptyContent="No aliases yet. Create one to give a model a friendly name."
        selectionMode="multiple"
        selectedKeys={selection.selectedKeys}
        onSelectionChange={selection.onSelectionChange}
        disabledKeys={disabledKeys}
      />

      <ConfirmDialog
        isOpen={bulkDeleteOpen}
        onOpenChange={setBulkDeleteOpen}
        heading="Delete aliases"
        body={`Delete ${selectedRowKeys.length} stored ${selectedRowKeys.length === 1 ? "alias" : "aliases"}? Callers using ${
          selectedRowKeys.length === 1 ? "it" : "them"
        } will get a model-not-found error.`}
        confirmLabel="Delete"
        isPending={bulkPending}
        error={bulkError}
        onConfirm={onBulkDelete}
      />
    </div>
  );
}
