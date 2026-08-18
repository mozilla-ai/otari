import { Button, Card, Chip } from "@heroui/react"
import { useCallback, useMemo, useState } from "react"

import type {
  OrganizationMember,
  Workspace,
  WorkspaceMember,
  WorkspaceMemberRole,
} from "@/client"
import {
  asMembershipRole,
  canManage,
  MEMBERSHIP_ROLES,
  memberLabel,
  membershipLabel,
} from "@/features/organization/roles"
import {
  useAddWorkspaceMember,
  useCreateWorkspace,
  useDeleteWorkspace,
  useOrganizationContext,
  useOrganizationMembers,
  useRemoveWorkspaceMember,
  useUpdateWorkspace,
  useUpdateWorkspaceMemberRole,
  useWorkspaceMembers,
  useWorkspaces,
} from "@/shared/api/hooks"
import { ConfirmDialog } from "@/shared/components/ConfirmDialog"
import { DataTable, type DataTableColumn } from "@/shared/components/DataTable"
import { Field } from "@/shared/components/Field"
import {
  EmptyState,
  ErrorBanner,
  FilterSelect,
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
const ROLE_OPTIONS = MEMBERSHIP_ROLES.map((role) => ({
  value: role,
  label: membershipLabel(role),
}))

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

function AddWorkspaceMember({
  workspaceId,
  candidates,
  rosterResolved,
}: {
  workspaceId: string
  candidates: OrganizationMember[]
  /**
   * Whether the organization roster actually answered. An empty candidate list
   * means "everyone is already here" only once it has: while it is loading, or
   * after it failed, the list is empty for a reason the operator should not be
   * told is a full workspace.
   */
  rosterResolved: boolean
}) {
  const add = useAddWorkspaceMember()
  const [userId, setUserId] = useState("")
  const [role, setRole] = useState<WorkspaceMemberRole>("member")

  if (!rosterResolved) {
    return null
  }

  if (candidates.length === 0) {
    return (
      <InfoBanner>
        Every active member of this organization is already in this workspace. A
        workspace's members are always a subset of the organization's, so add
        someone there first, on the Members page.
      </InfoBanner>
    )
  }

  return (
    <div className="flex flex-wrap items-end gap-2">
      <FilterSelect
        label="Organization member"
        value={userId}
        onChange={setUserId}
        options={[
          { value: "", label: "Select a member…" },
          ...candidates.map((member) => ({
            value: member.user_id ?? "",
            label: memberLabel(member),
          })),
        ]}
      />
      <FilterSelect
        label="Role"
        value={role}
        onChange={(value) => setRole(asMembershipRole(value) ?? "member")}
        options={ROLE_OPTIONS}
      />
      <Button
        variant="primary"
        isDisabled={userId === ""}
        isPending={add.isPending}
        onPress={() =>
          add.mutate(
            { workspaceId, userId, role },
            { onSuccess: () => setUserId("") },
          )
        }
      >
        Add member
      </Button>
      <ErrorBanner error={add.error} />
    </div>
  )
}

function WorkspaceMembersPanel({
  workspace,
  orgMembers,
  rosterResolved,
  canManageWorkspace,
}: {
  workspace: Workspace
  orgMembers: OrganizationMember[]
  rosterResolved: boolean
  canManageWorkspace: boolean
}) {
  const members = useWorkspaceMembers(workspace.id)
  const updateRole = useUpdateWorkspaceMemberRole()
  const removeMember = useRemoveWorkspaceMember()
  const [removing, setRemoving] = useState<WorkspaceMember | null>(null)

  const rows = members.data ?? []
  const nameByUserId = useMemo(
    () =>
      new Map(
        orgMembers
          .filter((member) => member.user_id)
          .map((member) => [member.user_id as string, memberLabel(member)]),
      ),
    [orgMembers],
  )
  const present = new Set(rows.map((member) => member.user_id))
  const candidates = orgMembers.filter(
    (member) =>
      member.user_id &&
      member.status === "active" &&
      !present.has(member.user_id),
  )

  return (
    <div className="flex flex-col gap-4 p-4">
      <div className="text-sm font-semibold text-foreground">
        Members of {workspace.name}
      </div>
      <ErrorBanner
        error={members.error ?? updateRole.error ?? removeMember.error}
      />

      {members.isLoading ? (
        <p className="text-sm text-muted">Loading members…</p>
      ) : rows.length === 0 ? (
        <p className="text-sm text-muted">This workspace has no members yet.</p>
      ) : (
        <ul className="flex flex-col gap-2">
          {rows.map((member) => (
            <li
              key={member.id}
              className="flex flex-wrap items-center gap-3 rounded-lg border border-border bg-surface-alt px-3 py-2"
            >
              <span className="text-sm text-foreground">
                {nameByUserId.get(member.user_id) ??
                  `Identity ${member.user_id.slice(0, 8)}`}
              </span>
              <Chip
                size="sm"
                color={member.status === "active" ? "accent" : "default"}
              >
                {member.status}
              </Chip>
              <span className="ml-auto flex items-center gap-2">
                <FilterSelect
                  ariaLabel={`Role for ${nameByUserId.get(member.user_id) ?? member.user_id} in ${workspace.name}`}
                  value={member.role}
                  disabled={!canManageWorkspace || updateRole.isPending}
                  options={ROLE_OPTIONS}
                  onChange={(value) => {
                    const role = asMembershipRole(value)
                    if (role) {
                      updateRole.mutate({
                        workspaceId: workspace.id,
                        userId: member.user_id,
                        role,
                      })
                    }
                  }}
                />
                <Button
                  size="sm"
                  variant="danger-soft"
                  isDisabled={!canManageWorkspace}
                  onPress={() => setRemoving(member)}
                >
                  Remove
                </Button>
              </span>
            </li>
          ))}
        </ul>
      )}

      {canManageWorkspace ? (
        <AddWorkspaceMember
          workspaceId={workspace.id}
          candidates={candidates}
          rosterResolved={rosterResolved}
        />
      ) : null}

      <ConfirmDialog
        isOpen={removing !== null}
        onOpenChange={(open) => {
          if (!open) setRemoving(null)
        }}
        heading="Remove workspace member"
        body={
          <>
            Remove{" "}
            <strong>
              {removing
                ? (nameByUserId.get(removing.user_id) ?? removing.user_id)
                : ""}
            </strong>{" "}
            from {workspace.name}? They keep their organization membership and
            can be added back.
          </>
        }
        confirmLabel="Remove member"
        isPending={removeMember.isPending}
        error={removeMember.error}
        onConfirm={() => {
          if (removing) {
            removeMember.mutate(
              { workspaceId: workspace.id, userId: removing.user_id },
              { onSuccess: () => setRemoving(null) },
            )
          }
        }}
      />
    </div>
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
        workspace={workspace}
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
