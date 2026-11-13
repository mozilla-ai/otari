import { Link, useNavigate } from "@tanstack/react-router"
import { useState } from "react"
import { FiSettings } from "react-icons/fi"
import type { WorkspaceMcpServer } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { CopyableValue } from "@/design-system/actions/CopyField"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { EmptyState } from "@/design-system/feedback/EmptyState"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { PageLoading } from "@/design-system/feedback/PageLoading"
import { Select } from "@/design-system/forms/Select"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { canManageWorkspace } from "@/features/organization/roles"
import { McpConnectionNoticeBanner } from "@/features/tools/McpConnectionNoticeBanner"
import { McpExposedTools } from "@/features/tools/McpExposedTools"
import { McpInfoHint } from "@/features/tools/McpInfoHint"
import { McpMemberTools } from "@/features/tools/McpMemberTools"
import { McpModeLabel } from "@/features/tools/McpModeLabel"
import { McpNotice } from "@/features/tools/McpNotice"
import {
  McpOwnAccountControl,
  McpOwnAccountNotes,
  useOwnMcpAccount,
} from "@/features/tools/McpOwnAccount"
import { McpReadTools } from "@/features/tools/McpReadTools"
import {
  McpServerAcl,
  type UserRuleRequest,
} from "@/features/tools/McpServerAcl"
import { McpServerConnection } from "@/features/tools/McpServerConnection"
import { McpServerLogo } from "@/features/tools/McpServerLogo"
import { McpServerLogoButton } from "@/features/tools/McpServerLogoDialog"
import {
  McpServerSettingsDialog,
  type McpSettingsFocus,
} from "@/features/tools/McpServerSettingsDialog"
import { McpServerSignIn } from "@/features/tools/McpServerSignIn"
import { McpSharedAccount } from "@/features/tools/McpSharedAccount"
import {
  AVAILABILITY,
  AVAILABILITY_OPTIONS,
  AVAILABILITY_ORDER,
  availabilityFields,
  availabilityOf,
  type McpAvailability,
} from "@/features/tools/mcpAvailability"
import { healthDetail } from "@/features/tools/mcpServerHealth"
import { useOrganizationContext } from "@/shared/api/organizations"
import {
  useDeleteWorkspaceMcpServer,
  useUpdateWorkspaceMcpServer,
  useWorkspaceMcpServer,
  useWorkspaceMcpServerLogos,
} from "@/shared/api/tools"
import { useBreadcrumbLeaf } from "@/shared/hooks/BreadcrumbLeaf"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"

/**
 * One stored MCP server. Everyone sees how it signs in (Shared or Personal)
 * and the account section that explains it; a manager also sees, in the order
 * a call meets them, the tools it exposes, which of them are read, its access
 * rules, then how Otari reaches it. Everything about the server itself is
 * changed in one place, the heading's Settings, which the account section and
 * a refusal that asks for an OAuth client open on the field they are about.
 * Deleting the server is at the foot of Settings.
 *
 * The server is read from `workspaceId` when a link names one (a trace names
 * the workspace its call ran in), else from the selected workspace.
 *
 * Takes the ids as props rather than reading the router, so a test can hand
 * them over without a route tree.
 */
export function McpServerDetailPage({
  serverId,
  workspaceId: linkedWorkspaceId,
  onLeave,
  openWindow,
}: {
  serverId: string
  workspaceId?: string
  /** Sends the whole page to an authorization page; a test observes it here. */
  onLeave?: (url: string) => void
  /** Opens the sign-in popup; a test stands in for the browser's window here. */
  openWindow?: () => Window | undefined
}) {
  const navigate = useNavigate()
  const { selected, isLoading: workspaceLoading } = useSelectedWorkspace()
  const context = useOrganizationContext()
  const workspaceId =
    linkedWorkspaceId ?? (selected ? selected.workspace_id : null)
  const membership = linkedWorkspaceId
    ? context.data?.workspace_memberships?.find(
        (entry) => entry.workspace_id === linkedWorkspaceId,
      )
    : selected
  const canManage = canManageWorkspace(context.data, membership?.role)
  const query = useWorkspaceMcpServer(workspaceId, serverId)
  const update = useUpdateWorkspaceMcpServer()
  const remove = useDeleteWorkspaceMcpServer()
  const [isDeleting, setDeleting] = useState(false)
  const [userRuleRequest, setUserRuleRequest] = useState<UserRuleRequest>()
  // Settings is remounted on each opening, so `count` is its key.
  const [settings, setSettings] = useState<{
    isOpen: boolean
    count: number
    focus: McpSettingsFocus
  }>({ isOpen: false, count: 0, focus: "none" })
  const openSettings = (focus: McpSettingsFocus = "none") =>
    setSettings((previous) => ({
      isOpen: true,
      count: previous.count + 1,
      focus,
    }))
  const closeSettings = () =>
    setSettings((previous) => ({ ...previous, isOpen: false }))
  // What the last save's connection check found, when it found something.
  const [unchecked, setUnchecked] = useState<WorkspaceMcpServer>()
  // The viewer's own account on a Personal server, for the heading's control,
  // the notes under it and their row in Connected users.
  const own = useOwnMcpAccount({
    workspaceId:
      query.data?.credential_mode === "per_user" ? workspaceId : null,
    server: { id: serverId, name: query.data?.name ?? "this server" },
    onLeave,
    openWindow,
  })
  useBreadcrumbLeaf(query.data?.name ?? "")
  const logo = useWorkspaceMcpServerLogos(workspaceId).get(serverId)

  const back = (
    <nav aria-label="Breadcrumb" className="text-caption">
      <Link to="/tools/mcp-servers" className="text-link hover:text-link-hover">
        ← MCP servers
      </Link>
    </nav>
  )

  const server = query.data
  if (!server) {
    if (workspaceLoading || (query.isPending && workspaceId !== null)) {
      return <PageLoading label="Loading MCP server…" />
    }
    return (
      <div className="flex flex-col gap-6">
        {back}
        <ErrorBanner error={query.error} />
        {query.error ? null : (
          <EmptyState
            title="MCP server not found"
            description="It may have been deleted, or it belongs to another workspace."
          />
        )}
      </div>
    )
  }

  const ids = { workspaceId: server.workspace_id, serverId: server.id }
  const openDelete = () => {
    remove.reset()
    setDeleting(true)
  }

  const isPersonal = server.credential_mode === "per_user"
  const addClient = canManage ? () => openSettings("client_id") : undefined
  const availability = availabilityOf(server)

  return (
    <div className="flex min-w-0 flex-col gap-4">
      <McpConnectionNoticeBanner onAddClient={addClient} />
      <PageIntro
        title={server.name}
        leading={
          canManage ? (
            <McpServerLogoButton
              workspaceId={server.workspace_id}
              serverId={server.id}
              name={server.name}
              logo={logo}
            />
          ) : (
            <McpServerLogo
              name={server.name}
              logoUrl={logo?.logo_url}
              tone={logo?.logo_tone}
              size="lg"
            />
          )
        }
        action={
          // The viewer's own account sits on a line of its own above the
          // server's settings, so the heading is no wider for having it.
          <div className="flex flex-col items-start gap-3 sm:items-end">
            {canManage && isPersonal ? (
              <McpOwnAccountControl own={own} />
            ) : null}
            {canManage ? (
              // Bottom edges together: the select carries a label above it,
              // the button does not, and both are one field tall. `mb-1` is
              // the gap the select keeps under its trigger for a message.
              <div className="flex flex-wrap items-end gap-3">
                <Select
                  label="Availability"
                  value={availability}
                  onChange={(next) =>
                    update.mutate({
                      ...ids,
                      body: availabilityFields(next as McpAvailability),
                    })
                  }
                  options={AVAILABILITY_OPTIONS}
                  labelAction={
                    <McpInfoHint label="Explain the three choices">
                      {AVAILABILITY_ORDER.map((value) => (
                        <span key={value}>
                          <span className="font-medium">
                            {AVAILABILITY[value].label}:
                          </span>{" "}
                          {AVAILABILITY[value].help}
                        </span>
                      ))}
                    </McpInfoHint>
                  }
                  isDisabled={update.isPending}
                  // No message line under it, so its edge is the row's edge.
                  shouldReserveMessage={false}
                  className="w-44"
                />
                <Button
                  onPress={() => openSettings()}
                  className="mb-1 h-(--field-height) min-h-(--field-height) shrink-0"
                >
                  <FiSettings aria-hidden className="size-4 shrink-0" />
                  Settings
                </Button>
              </div>
            ) : null}
          </div>
        }
      >
        {/* A label, not a control: the mode is changed in Settings. */}
        <span className="mb-1 block">
          <McpModeLabel mode={server.credential_mode} isCaps />
        </span>
        {/* Separate lines: the hint is the admin's own words and has no
            trailing space or period to lean on. */}
        {server.purpose_hint ? (
          <span className="block break-words">{server.purpose_hint}</span>
        ) : null}
        <span className="mt-1 block text-caption">
          {canManage ? null : (
            <span className="font-medium">
              {AVAILABILITY[availability].label}:{" "}
            </span>
          )}
          {AVAILABILITY[availability].help}
        </span>
        {/* One line at every width: the id is cut with an ellipsis before it
            wraps, and the copy control stays beside it. `w-0 min-w-full` takes
            the column's width without asking for the id's own, so a line that
            cannot wrap does not push the heading's controls off the page. */}
        <span className="mt-2 flex w-0 min-w-full items-center gap-x-2 text-caption whitespace-nowrap">
          <span className="shrink-0">Server ID</span>
          <CopyableValue
            value={server.id}
            label={`id for ${server.name}`}
            isTruncated
          >
            <code className="font-mono text-sm">{server.id}</code>
          </CopyableValue>
        </span>
        {availability === "on_request" ? (
          <span className="block text-caption">
            Add this id to <code className="font-mono">mcp_server_ids</code> in
            your request.
          </span>
        ) : null}
      </PageIntro>
      <ErrorBanner error={update.error} />
      {unchecked ? (
        <McpNotice
          message={
            unchecked.health === null
              ? "Saved, but not checked yet. Use Check under Exposed tools."
              : `Saved, but the connection check found a problem. ${healthDetail(unchecked.health)}`
          }
          onDismiss={() => setUnchecked(undefined)}
        />
      ) : null}
      {isPersonal ? (
        <McpOwnAccountNotes own={own} onAddClient={addClient} />
      ) : null}

      <div className="flex flex-col gap-8">
        {isPersonal ? (
          <McpServerSignIn
            workspaceId={server.workspace_id}
            server={server}
            canManage={canManage}
            own={own}
            onOpenUserRule={(userId) =>
              setUserRuleRequest((previous) => ({
                userId,
                nonce: (previous?.nonce ?? 0) + 1,
              }))
            }
          />
        ) : (
          <McpSharedAccount
            workspaceId={server.workspace_id}
            server={server}
            canManage={canManage}
            onOpenSettings={openSettings}
          />
        )}
        {canManage ? null : (
          <McpMemberTools workspaceId={server.workspace_id} server={server} />
        )}
        {canManage ? (
          <>
            <McpExposedTools
              key={JSON.stringify(server.allowed_tools)}
              workspaceId={server.workspace_id}
              server={server}
              canManage={canManage}
              onAddClient={addClient}
            />
            <McpReadTools
              key={`${server.read_tools_from_server}:${server.read_tools.join()}`}
              workspaceId={server.workspace_id}
              server={server}
              canManage={canManage}
            />
            <McpServerAcl
              workspaceId={server.workspace_id}
              server={server}
              canManage={canManage}
              userRuleRequest={userRuleRequest}
            />
            <McpServerConnection
              server={server}
              onOpenSettings={() => openSettings()}
            />
          </>
        ) : null}
      </div>

      {canManage ? (
        <McpServerSettingsDialog
          key={settings.count}
          isOpen={settings.isOpen}
          onOpenChange={(isOpen) => {
            if (!isOpen) closeSettings()
          }}
          workspaceId={server.workspace_id}
          server={server}
          focus={settings.focus}
          onSaved={(saved) => {
            setUnchecked(saved)
            closeSettings()
          }}
          onDelete={openDelete}
        />
      ) : null}

      <ConfirmDialog
        isOpen={isDeleting}
        onOpenChange={setDeleting}
        heading="Delete MCP server"
        body={`${server.name} is removed, with its token, its access rules and everyone's connection to it. Requests that still ask for it will fail.`}
        confirmLabel="Delete server"
        isPending={remove.isPending}
        error={remove.error}
        onConfirm={() =>
          remove.mutate(ids, {
            onSuccess: () => {
              setDeleting(false)
              void navigate({ to: "/tools/mcp-servers" })
            },
          })
        }
      />
    </div>
  )
}
