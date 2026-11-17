import { useEffect, useEffectEvent, useRef, useState } from "react"

import type { WorkspaceMcpServer, WorkspaceMcpServerGrant } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import {
  SettingsBlock,
  SettingsGroup,
} from "@/design-system/layout/SettingsGroup"
import {
  attributionId,
  useMemberAttributionLabels,
} from "@/features/organization/attribution"
import { McpAccessTest } from "@/features/tools/McpAccessTest"
import { McpGrantDialog } from "@/features/tools/McpGrantDialog"
import { McpGrantsTable } from "@/features/tools/McpGrantsTable"
import { isLiteralToolName } from "@/features/tools/mcpAccess"
import {
  grantSubject,
  grantUsers,
  keyName,
  userLabel,
} from "@/features/tools/mcpGrantSubjects"
import { checkedTools } from "@/features/tools/mcpServerHealth"
import { useMcpServerCheck } from "@/features/tools/useMcpServerCheck"
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
import { useUrlValue } from "@/shared/helpers/urlState"

const MATCH_ORDER =
  "Who can use this server. The most specific rule wins: a key's own, then its user's, then the default."
const WHO_A_RULE_NAMES =
  "A rule can name any member or API key, including people who have not connected yet."
const CONNECTED_USER_SHORTCUT =
  "Clicking a connected user above opens that person's rule."

/** A request, from elsewhere on the page, to open one user's rule. */
export interface UserRuleRequest {
  userId: string
  /** Bumped on every request, so asking for the same user twice opens it twice. */
  nonce: number
}

/**
 * A server's access rules: its rules for keys and users, its default policy,
 * and a trace of one key's call to one tool through them.
 */
export function McpServerAcl({
  workspaceId,
  server,
  canManage,
  userRuleRequest,
}: {
  workspaceId: string
  server: WorkspaceMcpServer
  canManage: boolean
  /** Opens the rule for this user: their existing one to edit, else a new one with them ticked. */
  userRuleRequest?: UserRuleRequest
}) {
  const access = useWorkspaceMcpServerAccess(workspaceId, server.id, canManage)
  // A session's tool call links here with its key and tool, to show the rule that decided it.
  const linkedKey = useUrlValue("key")
  const linkedTool = useUrlValue("tool")
  const linkedCall =
    linkedKey && linkedTool ? { keyId: linkedKey, tool: linkedTool } : undefined
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
  const [presetUserId, setPresetUserId] = useState("")
  const section = useRef<HTMLDivElement>(null)

  // A viewer is refused the list, and so is anyone whose rights changed since
  // the page decided what to offer; both read as "not yours to see".
  const isRefused =
    !canManage ||
    (access.error instanceof ApiError && access.error.status === 403)
  // What a rule can usefully name: the tools the server exposes, which is its
  // allow-list where it has one and what its last check listed otherwise.
  const picker = checkedTools(server, { withinAllowList: true })
  const tools = picker.isKnown
    ? picker.tools
    : (server.allowed_tools ?? []).filter(isLiteralToolName)
  const check = useMcpServerCheck({
    workspaceId,
    returnTo: `/tools/mcp-servers/${server.id}`,
  })
  const pickerProps = {
    tools,
    descriptions: picker.descriptions,
    isKnown: picker.isKnown,
    onCheck: () => check.run(server),
    isChecking: check.isRunning(server.id),
    checkError: check.error,
    onDismissCheckError: check.dismissError,
  }
  const grants = access.data?.grants ?? []

  // Everyone the Playground lets into the workspace joins the key owners as a
  // user who can be granted: its members, and the organization's owners and
  // admins, who need no workspace row. The roster joins each identity to the id
  // its keys and sessions bill to.
  const memberIds = new Set(
    (workspaceMembers.data ?? []).map((member) => member.user_id),
  )
  const userIdsInWorkspace = (organizationMembers.data ?? []).flatMap(
    (member) => {
      const id = attributionId(member)
      const entersWorkspace =
        (member.user_id && memberIds.has(member.user_id)) ||
        (member.status === "active" &&
          (member.role === "owner" || member.role === "admin"))
      return entersWorkspace && id ? [id] : []
    },
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
    setPresetUserId("")
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

  // An effect event, so the effect below fires on a new request alone rather
  // than on every refetch of the rules it consults.
  const openUserRule = useEffectEvent((userId: string) => {
    const existing = grants.find(
      (grant) => grant.subject === "user" && grant.user_id === userId,
    )
    section.current?.scrollIntoView?.({ block: "start", behavior: "smooth" })
    if (existing) {
      openEdit(existing)
    } else {
      openAdd()
      setPresetUserId(userId)
    }
  })
  const nonce = userRuleRequest?.nonce
  const requestedUserId = userRuleRequest?.userId ?? ""
  useEffect(() => {
    if (nonce !== undefined && requestedUserId !== "") {
      openUserRule(requestedUserId)
    }
  }, [nonce, requestedUserId])

  return (
    <div ref={section} className="flex scroll-mt-4 flex-col gap-8">
      <SettingsGroup
        title="Access rules"
        description={
          server.credential_mode === "per_user"
            ? `${MATCH_ORDER} ${WHO_A_RULE_NAMES} ${CONNECTED_USER_SHORTCUT}`
            : `${MATCH_ORDER} ${WHO_A_RULE_NAMES}`
        }
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
            <InfoBanner>Only an owner or admin can see these rules.</InfoBanner>
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
          description="Check whether a key can call a tool, and which rule decides."
          isBounded
        >
          <SettingsBlock>
            <McpAccessTest
              workspaceId={workspaceId}
              serverId={server.id}
              serverEnabled={server.enabled}
              keys={[...testKeys].map(([id, label]) => ({ id, label }))}
              tools={tools}
              initial={linkedCall}
              ruleOwner={(grantId) => {
                const grant = grants.find((entry) => entry.id === grantId)
                return grant
                  ? grantSubject(grant, labels).name
                  : "a key or user"
              }}
            />
          </SettingsBlock>
        </SettingsGroup>
      ) : null}

      <McpGrantDialog
        key={`add-${openCount}`}
        isOpen={isAdding}
        onOpenChange={setAdding}
        users={users}
        initialUserIds={presetUserId ? [presetUserId] : []}
        initialQuery={presetUserId ? userLabel(presetUserId, labels) : ""}
        {...pickerProps}
        readTools={server.read_tools}
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
          {...pickerProps}
          readTools={server.read_tools}
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
          {...pickerProps}
          readTools={server.read_tools}
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
