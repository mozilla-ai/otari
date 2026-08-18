import { Button, Card } from "@heroui/react"
import { useCallback, useMemo, useState } from "react"

import type { Workspace } from "@/client"
import { canManage } from "@/features/organization/roles"
import { WorkspaceMembersPanel } from "@/features/workspaces/WorkspaceMembersPanel"
import {
  useCreateWorkspace,
  useDeleteWorkspace,
  useOrganizationContext,
  useOrganizationMembers,
  useUpdateWorkspace,
  useWorkspaces,
} from "@/shared/api/hooks"
import { ConfirmDialog } from "@/shared/components/ConfirmDialog"
import { DataTable, type DataTableColumn } from "@/shared/components/DataTable"
import { Field } from "@/shared/components/Field"
import {
  EmptyState,
  ErrorBanner,
  InfoBanner,
  PageHeader,
} from "@/shared/components/ui"

// Workspaces are the unit inside an organization that work is scoped to. This
// page lists the ones the caller can see, and opening a row reveals its roster:
// a workspace's members are only ever a subset of the organization's, which is
// why adding one is a picker over the roster rather than an invitation.

// The workspace vocabulary is the organization one: four fixed roles, the same
// spellings, published on both requests. `asMembershipRole` narrows a picker's
// string back to it.
const getWorkspaceRowKey = (workspace: Workspace): string => workspace.id

function formatDate(value: string): string {
  return new Date(value).toLocaleDateString()
}

function CreateWorkspaceForm({ onClose }: { onClose: () => void }) {
  const create = useCreateWorkspace()
  const [name, setName] = useState("")
  const [description, setDescription] = useState("")
  const trimmed = name.trim()
  return (
    <Card>
      <Card.Content className="flex flex-col gap-4 p-5">
        <div className="text-sm font-semibold text-foreground">
          Create workspace
        </div>
        <ErrorBanner error={create.error} />
        <Field
          label="Name"
          value={name}
          onChange={setName}
          placeholder="Production"
          isRequired
          autoFocus
          description="Unique within this organization. You become its owner."
        />
        <Field
          label="Description (optional)"
          value={description}
          onChange={setDescription}
        />
        <div className="flex gap-2">
          <Button
            variant="primary"
            isDisabled={trimmed === ""}
            isPending={create.isPending}
            onPress={() =>
              create.mutate(
                { name: trimmed, description: description.trim() || null },
                { onSuccess: onClose },
              )
            }
          >
            Create workspace
          </Button>
          <Button variant="ghost" onPress={onClose}>
            Cancel
          </Button>
        </div>
      </Card.Content>
    </Card>
  )
}

function EditWorkspaceForm({
  workspace,
  onClose,
}: {
  workspace: Workspace
  onClose: () => void
}) {
  const update = useUpdateWorkspace()
  const [name, setName] = useState(workspace.name)
  const [description, setDescription] = useState(workspace.description ?? "")
  const trimmed = name.trim()
  return (
    <Card>
      <Card.Content className="flex flex-col gap-4 p-5">
        <div className="text-sm font-semibold text-foreground">
          Edit <code>{workspace.name}</code>
        </div>
        <ErrorBanner error={update.error} />
        <Field label="Name" value={name} onChange={setName} isRequired />
        <Field
          label="Description"
          value={description}
          onChange={setDescription}
        />
        <div className="flex gap-2">
          <Button
            variant="primary"
            isDisabled={trimmed === ""}
            isPending={update.isPending}
            onPress={() =>
              update.mutate(
                {
                  id: workspace.id,
                  body: {
                    name: trimmed,
                    description: description.trim() || null,
                  },
                },
                { onSuccess: onClose },
              )
            }
          >
            Save changes
          </Button>
          <Button variant="ghost" onPress={onClose}>
            Cancel
          </Button>
        </div>
      </Card.Content>
    </Card>
  )
}

export function WorkspacesPage() {
  const context = useOrganizationContext()
  const workspaces = useWorkspaces()
  const orgMembers = useOrganizationMembers()
  const remove = useDeleteWorkspace()

  const [creating, setCreating] = useState(false)
  const [editing, setEditing] = useState<string | null>(null)
  const [expanded, setExpanded] = useState<string | null>(null)
  const [deleting, setDeleting] = useState<Workspace | null>(null)

  const rows = workspaces.data ?? []
  const manages = canManage(context.data)
  const editingWorkspace = rows.find((row) => row.id === editing) ?? null
  const showOnboarding = !workspaces.isLoading && rows.length === 0 && !creating

  const columns = useMemo<DataTableColumn<Workspace>[]>(
    () => [
      {
        id: "name",
        header: "Workspace",
        isRowHeader: true,
        cell: (workspace) => (
          <div className="flex flex-col gap-0.5">
            <span className="text-sm font-medium text-foreground">
              {workspace.name}
            </span>
            {workspace.description ? (
              <span className="text-xs text-muted">
                {workspace.description}
              </span>
            ) : null}
          </div>
        ),
      },
      {
        id: "created",
        header: "Created",
        cell: (workspace) => (
          <span className="text-muted">{formatDate(workspace.created_at)}</span>
        ),
      },
      {
        id: "actions",
        header: "Actions",
        align: "end",
        cell: (workspace) => (
          <div className="flex items-center justify-end gap-1.5">
            <Button
              size="sm"
              variant="outline"
              onPress={() =>
                setExpanded((current) =>
                  current === workspace.id ? null : workspace.id,
                )
              }
            >
              {expanded === workspace.id ? "Hide members" : "Members"}
            </Button>
            <Button
              size="sm"
              variant="ghost"
              isDisabled={!manages}
              onPress={() => {
                setCreating(false)
                setEditing(workspace.id)
              }}
            >
              Edit
            </Button>
            <Button
              size="sm"
              variant="danger-soft"
              isDisabled={!manages}
              onPress={() => setDeleting(workspace)}
            >
              Delete
            </Button>
          </div>
        ),
      },
    ],
    [expanded, manages],
  )

  const renderDetail = useCallback(
    (workspace: Workspace) => (
      <WorkspaceMembersPanel
        workspaceId={workspace.id}
        workspaceName={workspace.name}
        orgMembers={orgMembers.data ?? []}
        rosterResolved={orgMembers.isSuccess}
        canManageWorkspace={manages}
      />
    ),
    [orgMembers.data, orgMembers.isSuccess, manages],
  )

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Workspaces"
        description="The scopes inside this organization that work is grouped into. Each one carries its own members and roles."
        action={
          creating || !manages ? null : (
            <Button
              variant="primary"
              onPress={() => {
                setEditing(null)
                setCreating(true)
              }}
            >
              Create workspace
            </Button>
          )
        }
      />

      <ErrorBanner
        error={
          context.error ?? workspaces.error ?? orgMembers.error ?? remove.error
        }
      />

      {/* Withheld until the context answers, so an owner is not told for one
          paint that they may not manage their own workspaces. */}
      {context.isLoading || manages ? null : (
        <InfoBanner>
          Only organization owners and admins can create, edit, or delete
          workspaces.
        </InfoBanner>
      )}

      {creating ? (
        <CreateWorkspaceForm onClose={() => setCreating(false)} />
      ) : null}

      {/* Keyed on the workspace so switching which one is edited remounts the
          form: its fields seed from `workspace` on mount only. */}
      {editingWorkspace ? (
        <EditWorkspaceForm
          key={editingWorkspace.id}
          workspace={editingWorkspace}
          onClose={() => setEditing(null)}
        />
      ) : null}

      {showOnboarding ? (
        <EmptyState
          title="No workspaces yet"
          description="A workspace groups the work inside this organization and carries its own members and roles. Every organization is created with one, so an empty list usually means the default was deleted."
          actionLabel={manages ? "Create a workspace" : undefined}
          onAction={manages ? () => setCreating(true) : undefined}
        />
      ) : (
        <DataTable
          ariaLabel="Workspaces"
          columns={columns}
          rows={rows}
          getRowKey={getWorkspaceRowKey}
          isLoading={workspaces.isLoading}
          emptyContent="No workspaces yet."
          detailKey={expanded}
          renderDetail={renderDetail}
        />
      )}

      <ConfirmDialog
        isOpen={deleting !== null}
        onOpenChange={(open) => {
          if (!open) setDeleting(null)
        }}
        heading="Delete workspace"
        body={
          <>
            Delete <strong>{deleting?.name}</strong> and its memberships? Usage
            the gateway already recorded is not deleted with it.
          </>
        }
        confirmLabel="Delete workspace"
        isPending={remove.isPending}
        error={remove.error}
        onConfirm={() => {
          if (deleting) {
            remove.mutate(deleting.id, {
              onSuccess: () => {
                if (expanded === deleting.id) setExpanded(null)
                if (editing === deleting.id) setEditing(null)
                setDeleting(null)
              },
            })
          }
        }}
      />
    </div>
  )
}
