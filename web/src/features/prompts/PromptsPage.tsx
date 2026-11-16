import { useState } from "react"
import { FiEdit2, FiTrash2 } from "react-icons/fi"
import type { StoredPrompt } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { RowAction, RowActionRow } from "@/design-system/actions/RowAction"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { EmptyState } from "@/design-system/feedback/EmptyState"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { PageLoading } from "@/design-system/feedback/PageLoading"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import { useDeletePrompt, usePrompts } from "@/shared/api/prompts"
import { formatDateTimeMinutes } from "@/shared/helpers/format"

import { PromptDialog } from "./PromptDialog"

export function PromptsPage() {
  const prompts = usePrompts()
  const remove = useDeletePrompt()

  const [dialogOpen, setDialogOpen] = useState(false)
  const [editing, setEditing] = useState<StoredPrompt | null>(null)
  const [pendingRemoveId, setPendingRemoveId] = useState<string | null>(null)

  function openAdd() {
    setEditing(null)
    setDialogOpen(true)
  }

  function openEdit(prompt: StoredPrompt) {
    setEditing(prompt)
    setDialogOpen(true)
  }

  const rows = prompts.data ?? []
  const pendingRemove = rows.find((row) => row.id === pendingRemoveId)

  if (prompts.isLoading) {
    return <PageLoading label="Loading prompts…" />
  }

  const isEmpty = !prompts.error && !prompts.isPending && rows.length === 0

  const columns: DataTableColumn<StoredPrompt>[] = [
    {
      id: "name",
      header: "Name",
      isRowHeader: true,
      cell: (row) => <span className="font-mono text-caption">{row.name}</span>,
    },
    {
      id: "content",
      header: "Content",
      cell: (row) => (
        <span className="line-clamp-2 text-caption text-subtle">
          {row.content}
        </span>
      ),
    },
    {
      id: "updated_at",
      header: "Last updated",
      cell: (row) => (
        <span className="text-caption text-subtle">
          {formatDateTimeMinutes(row.updated_at)}
        </span>
      ),
    },
    {
      id: "actions",
      header: "Actions",
      align: "end",
      cell: (row) => (
        <RowActionRow>
          <RowAction
            icon={FiEdit2}
            label="Edit"
            ariaLabel={`Edit ${row.name}`}
            onPress={() => openEdit(row)}
          />
          <RowAction
            icon={FiTrash2}
            label="Delete"
            ariaLabel={`Delete ${row.name}`}
            onPress={() => setPendingRemoveId(row.id)}
          />
        </RowActionRow>
      ),
    },
  ]

  return (
    <div className="flex flex-col gap-6">
      <PageIntro
        title="Prompts"
        action={
          <Button variant="primary" onPress={openAdd}>
            New prompt
          </Button>
        }
      >
        Store named prompts and reference them by name instead of embedding text
        inline. Update a prompt here to change every caller that references it,
        without redeploying.
      </PageIntro>
      <ErrorBanner error={prompts.error ?? remove.error} />
      <PromptDialog
        isOpen={dialogOpen}
        onOpenChange={setDialogOpen}
        editing={editing}
      />
      {isEmpty ? (
        <EmptyState
          title="No prompts stored"
          description="Add a prompt to reference it by name from your API calls."
          actionLabel="New prompt"
          onAction={openAdd}
        />
      ) : (
        <TableScrollFrame className="otari-prompts-table">
          <DataTable
            ariaLabel="Stored prompts"
            mobileLayout="cards"
            columns={columns}
            rows={rows}
            getRowKey={(row) => row.id}
            isLoading={prompts.isLoading}
          />
        </TableScrollFrame>
      )}
      <ConfirmDialog
        isOpen={pendingRemove !== undefined}
        onOpenChange={(isOpen) => {
          if (!isOpen) setPendingRemoveId(null)
        }}
        heading={`Delete "${pendingRemove?.name}"?`}
        body="This cannot be undone. Any caller referencing this prompt by name will stop finding it."
        confirmLabel="Delete"
        onConfirm={() => {
          if (pendingRemoveId) {
            remove.mutate(pendingRemoveId, {
              onSuccess: () => setPendingRemoveId(null),
            })
          }
        }}
        isPending={remove.isPending}
      />
    </div>
  )
}
