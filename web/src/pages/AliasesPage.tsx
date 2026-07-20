import { Button, Card, Chip } from "@heroui/react";
import { useState } from "react";
import { useSearchParams } from "react-router-dom";

import { useAliases, useCreateAlias, useDeleteAlias } from "@/api/hooks";
import { Field } from "@/components/Field";
import { ModelComboBox } from "@/components/ModelComboBox";
import { LoadingRow, Table, TableMessage, Td, Th, THead, Tr } from "@/components/Table";
import { ConfirmButton, ErrorBanner, errorMessage, PageHeader } from "@/components/ui";

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

  const rows = [...(aliases.data ?? [])].sort((a, b) => a.name.localeCompare(b.name));

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Aliases"
        description="Friendly names that map to a real provider:model. Callers send the alias as the model; pricing, budgets, and usage key on the target."
        action={
          adding ? null : (
            <Button variant="primary" onPress={() => setAdding(true)}>
              New alias
            </Button>
          )
        }
      />

      <ErrorBanner error={aliases.error} />

      {adding ? <NewAliasForm initialTarget={initialTarget} onClose={() => setAdding(false)} /> : null}

      <Table>
        <THead>
          <Tr>
            <Th>Alias</Th>
            <Th>Target</Th>
            <Th>Source</Th>
            <Th className="text-right">Actions</Th>
          </Tr>
        </THead>
        <tbody>
          {aliases.isLoading ? (
            <LoadingRow colSpan={4} />
          ) : rows.length === 0 ? (
            <TableMessage colSpan={4}>No aliases yet. Create one to give a model a friendly name.</TableMessage>
          ) : (
            rows.map((alias) => (
              <Tr key={alias.name}>
                <Td className="font-medium break-all text-[var(--otari-ink)]">{alias.name}</Td>
                <Td className="break-all text-[var(--otari-muted)]">{alias.target}</Td>
                <Td>
                  <Chip size="sm" color={alias.source === "stored" ? "accent" : "default"}>
                    {alias.source}
                  </Chip>
                </Td>
                <Td className="text-right whitespace-nowrap">
                  {alias.source === "stored" ? (
                    <span className="inline-flex items-center gap-2">
                      <ConfirmButton
                        confirmLabel="Delete"
                        isPending={deleteAlias.isPending}
                        onConfirm={() => deleteAlias.mutate(alias.name)}
                      >
                        Delete
                      </ConfirmButton>
                      {deleteAlias.error ? (
                        <span className="text-xs text-red-700">{errorMessage(deleteAlias.error)}</span>
                      ) : null}
                    </span>
                  ) : (
                    <span className="text-xs text-[var(--otari-muted)]">set in config.yml</span>
                  )}
                </Td>
              </Tr>
            ))
          )}
        </tbody>
      </Table>
    </div>
  );
}
