import { Button, Card, Chip } from "@heroui/react";
import { useState } from "react";
import { useSearchParams } from "react-router-dom";

import type { AliasResponse } from "@/api/types";
import { useAliases, useCreateAlias, useDeleteAlias } from "@/api/hooks";
import { BulkActionBar } from "@/components/BulkActionBar";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { DataTable, type DataTableColumn } from "@/components/DataTable";
import { Field } from "@/components/Field";
import { ModelComboBox } from "@/components/ModelComboBox";
import { ConfirmButton, ErrorBanner, PageHeader } from "@/components/ui";
import { resolveSelectedIds, useTableSelection } from "@/lib/tableSelection";

// Edit an existing stored alias's target. The name is the lookup key and is shown
// read-only; the backend POST /v1/aliases upserts by name, so the same hook serves both.
function EditAliasForm({ alias, onClose }: { alias: AliasResponse; onClose: () => void }) {
  const updateAlias = useCreateAlias();
  const [target, setTarget] = useState(alias.target);

  const targetChanged = target.trim() !== "" && target.trim() !== alias.target;

  const submit = () => {
    if (!targetChanged) return;
    updateAlias.mutate({ name: alias.name, target: target.trim() }, { onSuccess: onClose });
  };

  return (
    <Card>
      <Card.Content className="flex flex-col gap-4 p-5">
        <div className="text-sm font-semibold text-[var(--otari-ink)]">
          Edit alias <code>{alias.name}</code>
        </div>
        <ErrorBanner error={updateAlias.error} />
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="flex flex-col gap-1">
            <span className="text-sm font-medium text-[var(--otari-ink)]">Alias name</span>
            <code className="text-sm text-[var(--otari-muted)]">{alias.name}</code>
            <span className="text-xs text-[var(--otari-muted)]">
              The alias name is the key and cannot be changed here. Delete and recreate to rename.
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

  const nameHasDelimiter = /[:/]/.test(name);
  const canSubmit = name.trim() !== "" && target.trim() !== "" && !nameHasDelimiter;

  const submit = () => {
    if (!canSubmit) return;
    createAlias.mutate({ name: name.trim(), target: target.trim() }, { onSuccess: onClose });
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

  const rows = [...(aliases.data ?? [])].sort((a, b) => a.name.localeCompare(b.name));
  // Only stored aliases can be deleted; config.yml aliases are read-only.
  const selectableKeys = rows.filter((a) => a.source === "stored").map((a) => a.name);
  const disabledKeys = rows.filter((a) => a.source !== "stored").map((a) => a.name);
  const selectedNames = resolveSelectedIds(selection.selectedKeys, selectableKeys);

  const onBulkDelete = async () => {
    setBulkPending(true);
    setBulkError(undefined);
    try {
      // No bulk endpoint for aliases; delete sequentially so one failure surfaces
      // without firing the rest in parallel against the same small list.
      for (const name of selectedNames) {
        await deleteAlias.mutateAsync(name);
      }
      selection.clear();
      setBulkDeleteOpen(false);
    } catch (error) {
      setBulkError(error);
    } finally {
      setBulkPending(false);
    }
  };

  const columns: DataTableColumn<AliasResponse>[] = [
    {
      id: "alias",
      header: "Alias",
      isRowHeader: true,
      cell: (a) => <span className="font-medium break-all text-[var(--otari-ink)]">{a.name}</span>,
    },
    { id: "target", header: "Target", cell: (a) => <span className="break-all text-[var(--otari-muted)]">{a.target}</span> },
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
              onConfirm={() => deleteAlias.mutate(a.name)}
            >
              Delete
            </ConfirmButton>
          </span>
        ) : (
          <span className="text-xs text-[var(--otari-muted)]">set in config.yml</span>
        ),
    },
  ];

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

      {selectedNames.length > 0 ? (
        <BulkActionBar
          selectedCount={selectedNames.length}
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
        getRowKey={(a) => a.name}
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
        body={`Delete ${selectedNames.length} stored ${selectedNames.length === 1 ? "alias" : "aliases"}? Callers using ${
          selectedNames.length === 1 ? "it" : "them"
        } will get a model-not-found error.`}
        confirmLabel="Delete"
        isPending={bulkPending}
        error={bulkError}
        onConfirm={onBulkDelete}
      />
    </div>
  );
}
