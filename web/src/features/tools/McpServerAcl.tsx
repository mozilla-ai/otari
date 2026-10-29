import { useState } from "react"

import type { WorkspaceMcpServer, WorkspaceMcpServerGrant } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { SettingsGroup } from "@/design-system/layout/SettingsGroup"
import { useMemberAttributionLabels } from "@/features/organization/attribution"
import { McpAccessTest } from "@/features/tools/McpAccessTest"
import { McpGrantDialog } from "@/features/tools/McpGrantDialog"
import { McpGrantsTable } from "@/features/tools/McpGrantsTable"
import {
  grantSubject,
  grantUsers,
  keyName,
  userLabel,
} from "@/features/tools/mcpGrantSubjects"
import { useKeys } from "@/shared/api/apiKeys"
import { ApiError } from "@/shared/api/client"
import { useOrganizationMembers } from "@/shared/api/organizations"
import {
  useAddWorkspaceMcpServerGrants,
  useDeleteWorkspaceMcpServerGrant,
  useUpdateWorkspaceMcpServer,
  useUpdateWorkspaceMcpServerGrant,
  useWorkspaceMcpServerAccess,
} from "@/shared/api/tools"
import { useWorkspaceMembers } from "@/shared/api/workspaces"

const MATCH_ORDER =
  "Rules are matched from the most specific: a rule for the key, then a rule for the user who owns it, then the default policy. Tools the server does not expose are never reachable, whatever a rule says."

/**
 * A server's ACL: its rules for keys and users, its default policy, and a
 * trace of one key's call to one tool through them.
 */
export function McpServerAcl({
  workspaceId,
  server,
  canManage,
}: {
  workspaceId: string
  server: WorkspaceMcpServer
  canManage: boolean
}) {
  const access = useWorkspaceMcpServerAccess(workspaceId, server.id, canManage)
  const updateServer = useUpdateWorkspaceMcpServer()
  const addGrants = useAddWorkspaceMcpServerGrants()
  const updateGrant = useUpdateWorkspaceMcpServerGrant()
  const removeGrant = useDeleteWorkspaceMcpServerGrant()
  const keys = useKeys(workspaceId, canManage)
  const workspaceMembers = useWorkspaceMembers(canManage ? workspaceId : null)
  const organizationMembers = useOrganizationMembers(canManage)
  const labels = useMemberAttributionLabels()

  const [isAdding, setAdding] = useState(false)
  // Bumped on each open and used as the dialog's key, so a draft never
  // survives into the next one.
  const [openCount, setOpenCount] = useState(0)
  const [editing, setEditing] = useState<WorkspaceMcpServerGrant>()
  const [isEditingDefault, setEditingDefault] = useState(false)
  const [removing, setRemoving] = useState<WorkspaceMcpServerGrant>()

  // A viewer is refused the list, and so is anyone whose rights changed since
  // the page decided what to offer; both read as "not yours to see".
  const isRefused =
    !canManage ||
    (access.error instanceof ApiError && access.error.status === 403)
  // What a rule can usefully name: the tools the server exposes, which is its
  // allow-list where it has one and what its last check listed otherwise.
  const tools = server.allowed_tools ?? server.health?.tools ?? []
  const grants = access.data?.grants ?? []

  // The workspace's members join the key owners as users who can be granted,
  // through the roster's join from an identity to the id its keys bill to.
  const memberIds = new Set(
    (workspaceMembers.data ?? []).map((member) => member.user_id),
  )
  const userIdsInWorkspace = (organizationMembers.data ?? []).flatMap(
    (member) =>
      member.user_id &&
      memberIds.has(member.user_id) &&
      member.attribution_user_id
        ? [member.attribution_user_id]
        : [],
  )
  const users = grantUsers(keys.data ?? [], userIdsInWorkspace, labels)

  // Every key the page knows of, for the test: the listed ones, and the keys
  // the exceptions name, which a manager who is not an operator cannot list.
  const testKeys = new Map<string, string>()
  for (const user of users) {
    for (const key of user.keys) {
      testKeys.set(key.id, `${key.name} (${user.label})`)
    }
  }
  for (const grant of grants) {
    if (grant.api_key_id && !testKeys.has(grant.api_key_id)) {
      const owner = grant.user_id
        ? ` (${userLabel(grant.user_id, labels)})`
        : ""
      testKeys.set(
        grant.api_key_id,
        `${keyName(grant.key_name, grant.api_key_id)}${owner}`,
      )
    }
  }

  const openAdd = () => {
    addGrants.reset()
    setOpenCount((count) => count + 1)
    setAdding(true)
  }
  const openEdit = (grant: WorkspaceMcpServerGrant) => {
    updateGrant.reset()
    setOpenCount((count) => count + 1)
    setEditing(grant)
  }
  const openEditDefault = () => {
    updateServer.reset()
    setOpenCount((count) => count + 1)
    setEditingDefault(true)
  }
  const ids = { workspaceId, serverId: server.id }

  return (
    <div className="flex flex-col gap-8">
      <SettingsGroup
        title="ACL"
        description={MATCH_ORDER}
        isBounded
        action={
          canManage && !isRefused ? (
            <Button variant="primary" onPress={openAdd}>
              Add rule
            </Button>
          ) : null
        }
      >
        <div className="flex flex-col gap-2">
          {isRefused ? (
            <InfoBanner>
              Only an owner or admin of the workspace or organization can see
              and change the rules for keys and users.
            </InfoBanner>
          ) : (
            <ErrorBanner error={access.error} />
          )}
          <McpGrantsTable
            grants={grants}
            defaultTools={server.everyone_tools}
            labels={labels}
            canManage={canManage}
            showRules={!isRefused}
            onEdit={openEdit}
            onEditDefault={openEditDefault}
            onRemove={(grant) => {
              removeGrant.reset()
              setRemoving(grant)
            }}
          />
        </div>
      </SettingsGroup>

      {canManage && !isRefused ? (
        <SettingsGroup
          title="Trace a call"
          description="Which rule matches when a key calls a tool."
          isBounded
        >
          <McpAccessTest
            workspaceId={workspaceId}
            serverId={server.id}
            serverEnabled={server.enabled}
            keys={[...testKeys].map(([id, label]) => ({ id, label }))}
            tools={tools}
            ruleOwner={(grantId) => {
              const grant = grants.find((entry) => entry.id === grantId)
              return grant ? grantSubject(grant, labels).name : "a key or user"
            }}
          />
        </SettingsGroup>
      ) : null}

      <McpGrantDialog
        key={`add-${openCount}`}
        isOpen={isAdding}
        onOpenChange={setAdding}
        users={users}
        tools={tools}
        isPending={addGrants.isPending}
        error={addGrants.error}
        onSubmit={(draft) =>
          addGrants.mutate(
            { ...ids, body: draft },
            { onSuccess: () => setAdding(false) },
          )
        }
      />
      {editing ? (
        <McpGrantDialog
          key={`edit-${openCount}`}
          isOpen
          onOpenChange={(open) => {
            if (!open) setEditing(undefined)
          }}
          users={[]}
          editing={{
            name: grantSubject(editing, labels).name,
            tools: editing.tools,
          }}
          tools={tools}
          isPending={updateGrant.isPending}
          error={updateGrant.error}
          onSubmit={(draft) =>
            updateGrant.mutate(
              { ...ids, grantId: editing.id, tools: draft.tools },
              { onSuccess: () => setEditing(undefined) },
            )
          }
        />
      ) : null}
      {isEditingDefault ? (
        <McpGrantDialog
          key={`default-${openCount}`}
          isOpen
          onOpenChange={(open) => {
            if (!open) setEditingDefault(false)
          }}
          users={[]}
          editing={{ name: "the default policy", tools: server.everyone_tools }}
          tools={tools}
          isPending={updateServer.isPending}
          error={updateServer.error}
          onSubmit={(draft) =>
            updateServer.mutate(
              { ...ids, body: { everyone_tools: draft.tools } },
              { onSuccess: () => setEditingDefault(false) },
            )
          }
        />
      ) : null}
      <ConfirmDialog
        isOpen={removing !== undefined}
        onOpenChange={(open) => {
          if (!open) setRemoving(undefined)
        }}
        heading="Delete rule"
        body={
          removing
            ? removing.subject === "key"
              ? `${grantSubject(removing, labels).name} then matches the rule for its owner if there is one, otherwise the default policy.`
              : `${grantSubject(removing, labels).name}'s keys then match their own rules where they have one, otherwise the default policy.`
            : null
        }
        confirmLabel="Delete rule"
        isPending={removeGrant.isPending}
        error={removeGrant.error}
        onConfirm={() => {
          if (!removing) return
          removeGrant.mutate(
            { ...ids, grantId: removing.id },
            { onSuccess: () => setRemoving(undefined) },
          )
        }}
      />
    </div>
  )
}
