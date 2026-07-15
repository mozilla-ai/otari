import { Button, Card, Chip } from "@heroui/react";
import { useState } from "react";

import { useCreateUser, useDeleteUser, useUsers } from "@/api/hooks";
import { Field } from "@/components/Field";
import { LoadingRow, Table, TableMessage, Td, Th, THead, Tr } from "@/components/Table";
import { ConfirmButton, ErrorBanner, PageHeader } from "@/components/ui";
import { formatCost, formatDateTime } from "@/lib/format";

function CreateUserForm({ onClose }: { onClose: () => void }) {
  const createUser = useCreateUser();
  const [userId, setUserId] = useState("");
  const [alias, setAlias] = useState("");

  const submit = () => {
    if (!userId.trim()) {
      return;
    }
    createUser.mutate(
      { user_id: userId.trim(), alias: alias.trim() || null },
      {
        onSuccess: () => {
          setUserId("");
          setAlias("");
          onClose();
        },
      },
    );
  };

  return (
    <Card>
      <Card.Content className="flex flex-col gap-4 p-5">
        <div className="flex items-center justify-between">
          <h2 className="text-base font-semibold text-[var(--otari-ink)]">Create user</h2>
          <Button size="sm" variant="ghost" onPress={onClose}>
            Close
          </Button>
        </div>
        <ErrorBanner error={createUser.error} />
        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="User ID" value={userId} onChange={setUserId} placeholder="e.g. alice" isRequired autoFocus />
          <Field label="Alias" value={alias} onChange={setAlias} placeholder="(optional) display name" />
        </div>
        <div>
          <Button variant="primary" isDisabled={createUser.isPending || !userId.trim()} onPress={submit}>
            {createUser.isPending ? "Creating…" : "Create user"}
          </Button>
        </div>
      </Card.Content>
    </Card>
  );
}

export function UsersPage() {
  const users = useUsers();
  const deleteUser = useDeleteUser();
  const [showForm, setShowForm] = useState(false);

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Users"
        description="Users own keys and accrue spend. Deleting a user also revokes its keys."
        action={
          <Button variant="primary" onPress={() => setShowForm((value) => !value)}>
            {showForm ? "Hide form" : "Create user"}
          </Button>
        }
      />

      {showForm ? <CreateUserForm onClose={() => setShowForm(false)} /> : null}

      <ErrorBanner error={users.error ?? deleteUser.error} />

      <Table>
        <THead>
          <Tr>
            <Th>User ID</Th>
            <Th>Alias</Th>
            <Th>Status</Th>
            <Th className="text-right">Spend</Th>
            <Th className="text-right">Reserved</Th>
            <Th>Created</Th>
            <Th className="text-right">Actions</Th>
          </Tr>
        </THead>
        <tbody>
          {users.isLoading ? (
            <LoadingRow colSpan={7} />
          ) : users.data && users.data.length > 0 ? (
            users.data.map((user) => (
              <Tr key={user.user_id}>
                <Td className="font-mono text-xs">{user.user_id}</Td>
                <Td>{user.alias || <span className="text-[var(--otari-muted)]">—</span>}</Td>
                <Td>
                  <Chip size="sm" color={user.blocked ? "danger" : "success"}>
                    {user.blocked ? "blocked" : "active"}
                  </Chip>
                </Td>
                <Td className="text-right">{formatCost(user.spend)}</Td>
                <Td className="text-right text-[var(--otari-muted)]">{formatCost(user.reserved)}</Td>
                <Td className="text-[var(--otari-muted)]">{formatDateTime(user.created_at)}</Td>
                <Td className="text-right">
                  <ConfirmButton
                    confirmLabel="Delete"
                    isPending={deleteUser.isPending}
                    onConfirm={() => deleteUser.mutate(user.user_id)}
                  >
                    Delete
                  </ConfirmButton>
                </Td>
              </Tr>
            ))
          ) : (
            <TableMessage colSpan={7}>No users yet.</TableMessage>
          )}
        </tbody>
      </Table>
    </div>
  );
}
