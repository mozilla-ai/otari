import { Button, Card, Chip } from "@heroui/react";
import { useState } from "react";

import { useCreateKey, useDeleteKey, useKeys, useSetKeyActive } from "@/api/hooks";
import type { CreateKeyResponse } from "@/api/types";
import { Field } from "@/components/Field";
import { LoadingRow, Table, TableMessage, Td, Th, THead, Tr } from "@/components/Table";
import { ConfirmButton, ErrorBanner, PageHeader } from "@/components/ui";
import { formatDateTime, formatRelative } from "@/lib/format";

function CreatedKeyCallout({ created, onDismiss }: { created: CreateKeyResponse; onDismiss: () => void }) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(created.key);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  };

  return (
    <div className="rounded-lg border border-[var(--otari-brand)] bg-[var(--otari-brand-tint)] px-4 py-3">
      <p className="text-sm font-medium text-[var(--otari-brand-dark)]">
        Key created. Copy it now: it is shown only once.
      </p>
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <code className="rounded bg-white px-2 py-1 text-sm break-all">{created.key}</code>
        <Button size="sm" variant="primary" onPress={copy}>
          {copied ? "Copied" : "Copy"}
        </Button>
        <Button size="sm" variant="ghost" onPress={onDismiss}>
          Dismiss
        </Button>
      </div>
    </div>
  );
}

function CreateKeyForm({ onClose }: { onClose: () => void }) {
  const createKey = useCreateKey();
  const [name, setName] = useState("");
  const [userId, setUserId] = useState("");
  const [created, setCreated] = useState<CreateKeyResponse | null>(null);

  const submit = () => {
    createKey.mutate(
      {
        key_name: name.trim() || null,
        user_id: userId.trim() || null,
      },
      {
        onSuccess: (data) => {
          setCreated(data);
          setName("");
          setUserId("");
        },
      },
    );
  };

  return (
    <Card>
      <Card.Content className="flex flex-col gap-4 p-5">
        <div className="flex items-center justify-between">
          <h2 className="text-base font-semibold text-[var(--otari-ink)]">Create API key</h2>
          <Button size="sm" variant="ghost" onPress={onClose}>
            Close
          </Button>
        </div>
        {created ? <CreatedKeyCallout created={created} onDismiss={() => setCreated(null)} /> : null}
        <ErrorBanner error={createKey.error} />
        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="Key name" value={name} onChange={setName} placeholder="e.g. ci-pipeline" autoFocus />
          <Field
            label="User ID"
            value={userId}
            onChange={setUserId}
            placeholder="(optional) existing or new user"
            description="Leave blank to auto-create a virtual user for this key."
          />
        </div>
        <div>
          <Button variant="primary" isDisabled={createKey.isPending} onPress={submit}>
            {createKey.isPending ? "Creating…" : "Create key"}
          </Button>
        </div>
      </Card.Content>
    </Card>
  );
}

export function KeysPage() {
  const keys = useKeys();
  const setActive = useSetKeyActive();
  const deleteKey = useDeleteKey();
  const [showForm, setShowForm] = useState(false);

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="API keys"
        description="Create virtual keys, revoke or reactivate them, and see when each was last used."
        action={
          <Button variant="primary" onPress={() => setShowForm((value) => !value)}>
            {showForm ? "Hide form" : "Create key"}
          </Button>
        }
      />

      {showForm ? <CreateKeyForm onClose={() => setShowForm(false)} /> : null}

      <ErrorBanner error={keys.error ?? setActive.error ?? deleteKey.error} />

      <Table>
        <THead>
          <Tr>
            <Th>Name</Th>
            <Th>User</Th>
            <Th>Status</Th>
            <Th>Created</Th>
            <Th>Last used</Th>
            <Th>Expires</Th>
            <Th className="text-right">Actions</Th>
          </Tr>
        </THead>
        <tbody>
          {keys.isLoading ? (
            <LoadingRow colSpan={7} />
          ) : keys.data && keys.data.length > 0 ? (
            keys.data.map((key) => (
              <Tr key={key.id}>
                <Td className="font-medium">{key.key_name || <span className="text-[var(--otari-muted)]">unnamed</span>}</Td>
                <Td className="font-mono text-xs">{key.user_id ?? "—"}</Td>
                <Td>
                  <Chip size="sm" color={key.is_active ? "success" : "default"}>
                    {key.is_active ? "active" : "revoked"}
                  </Chip>
                </Td>
                <Td className="text-[var(--otari-muted)]">{formatDateTime(key.created_at)}</Td>
                <Td className="text-[var(--otari-muted)]">{formatRelative(key.last_used_at)}</Td>
                <Td className="text-[var(--otari-muted)]">{key.expires_at ? formatDateTime(key.expires_at) : "never"}</Td>
                <Td className="text-right whitespace-nowrap">
                  <span className="inline-flex items-center gap-2">
                    <Button
                      size="sm"
                      variant="outline"
                      isDisabled={setActive.isPending}
                      onPress={() => setActive.mutate({ id: key.id, isActive: !key.is_active })}
                    >
                      {key.is_active ? "Revoke" : "Activate"}
                    </Button>
                    <ConfirmButton
                      confirmLabel="Delete"
                      isPending={deleteKey.isPending}
                      onConfirm={() => deleteKey.mutate(key.id)}
                    >
                      Delete
                    </ConfirmButton>
                  </span>
                </Td>
              </Tr>
            ))
          ) : (
            <TableMessage colSpan={7}>No API keys yet. Create one to get started.</TableMessage>
          )}
        </tbody>
      </Table>
    </div>
  );
}
