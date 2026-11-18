import { Link, useNavigate } from "@tanstack/react-router"
import { useState } from "react"
import { FiSettings } from "react-icons/fi"

import type { WorkspaceMcpServer } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { CopyableValue } from "@/design-system/actions/CopyField"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { EmptyState } from "@/design-system/feedback/EmptyState"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { PageLoading } from "@/design-system/feedback/PageLoading"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { canManage } from "@/features/organization/roles"
import { McpAvailabilitySelect } from "@/features/tools/McpAvailabilitySelect"
import { McpConnectionNoticeBanner } from "@/features/tools/McpConnectionNoticeBanner"
import { McpExposedTools } from "@/features/tools/McpExposedTools"
import { McpModeLabel } from "@/features/tools/McpModeLabel"
import { McpNotice } from "@/features/tools/McpNotice"
import { McpServerConnection } from "@/features/tools/McpServerConnection"
import { McpServerLogo } from "@/features/tools/McpServerLogo"
import {
  McpServerSettingsDialog,
  type McpSettingsFocus,
} from "@/features/tools/McpServerSettingsDialog"
import { McpServerSignIn } from "@/features/tools/McpServerSignIn"
import { McpSharedAccount } from "@/features/tools/McpSharedAccount"
import {
  AVAILABILITY,
  availabilityFields,
  availabilityOf,
} from "@/features/tools/mcpAvailability"
import { healthDetail } from "@/features/tools/mcpServerHealth"
import { ORGANIZATION_MCP_SCOPE } from "@/shared/api/mcpServerScope"
import {
  useDeleteOrganizationMcpServer,
  useOrganizationMcpServer,
} from "@/shared/api/organizationMcpServers"
import { useOrganizationContext } from "@/shared/api/organizations"
import { useUpdateMcpServer } from "@/shared/api/tools"
import { useBreadcrumbLeaf } from "@/shared/hooks/BreadcrumbLeaf"

import { OrganizationMcpServerWorkspaces } from "./OrganizationMcpServerWorkspaces"

const SCOPE = ORGANIZATION_MCP_SCOPE

/**
 * One of the organization's own MCP servers, for its owners and admins: how
 * it signs in and the account behind that, the tools it exposes, how Otari
 * reaches it, and how far each workspace may go with it. Everything about the
 * server itself is changed in Settings, as on a workspace's server page.
 *
 * What a workspace's server page has and this one does not: access rules and
 * the admin's own account, which belong to each workspace that reaches the
 * server; a logo; and members' requests.
 *
 * Takes the id as a prop rather than reading the router, so a test can hand
 * it over without a route tree.
 */
export function OrganizationMcpServerDetailPage({
  serverId,
  onLeave,
  openWindow,
}: {
  serverId: string
  /** Sends the whole page to an authorization page; a test observes it here. */
  onLeave?: (url: string) => void
  /** Opens the sign-in popup; a test stands in for the browser's window here. */
  openWindow?: () => Window | undefined
}) {
  const navigate = useNavigate()
  const context = useOrganizationContext()
  const canEdit = canManage(context.data)
  const query = useOrganizationMcpServer(serverId, canEdit)
  const update = useUpdateMcpServer()
  const remove = useDeleteOrganizationMcpServer()
  const [isDeleting, setDeleting] = useState(false)
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
  // A disabled query still hands back what it cached, so a viewer who loses
  // the role while the page is open would otherwise keep seeing the server.
  const server = canEdit ? query.data : undefined
  useBreadcrumbLeaf(server?.name ?? "")

  const back = (
    <nav aria-label="Breadcrumb" className="text-caption">
      <Link
        to="/organization/mcp-servers"
        className="text-link hover:text-link-hover"
      >
        ← MCP servers
      </Link>
    </nav>
  )

  if (!server) {
    if (context.isPending || (canEdit && query.isPending)) {
      return <PageLoading label="Loading MCP server…" />
    }
    return (
      <div className="flex flex-col gap-6">
        {back}
        <ErrorBanner error={context.error ?? query.error} />
        {!canEdit ? (
          <InfoBanner>
            Only organization owners and admins can see this
            organization&rsquo;s MCP servers.
          </InfoBanner>
        ) : query.error ? null : (
          <EmptyState
            title="MCP server not found"
            description="It may have been deleted, or it belongs to a workspace rather than the organization."
          />
        )}
      </div>
    )
  }

  const openDelete = () => {
    remove.reset()
    setDeleting(true)
  }
  const addClient = () => openSettings("client_id")
  const availability = availabilityOf(server)

  return (
    <div className="flex min-w-0 flex-col gap-4">
      <McpConnectionNoticeBanner onAddClient={addClient} />
      <PageIntro
        title={server.name}
        leading={<McpServerLogo name={server.name} logoUrl={null} size="lg" />}
        action={
          // Bottom edges together, as on a workspace's server page.
          <div className="flex flex-wrap items-end gap-3">
            <McpAvailabilitySelect
              value={availability}
              onChange={(next) =>
                update.mutate({
                  scope: SCOPE,
                  serverId: server.id,
                  body: availabilityFields(next),
                })
              }
              isDisabled={update.isPending}
            />
            <Button
              onPress={() => openSettings()}
              className="mb-1 h-(--field-height) min-h-(--field-height) shrink-0"
            >
              <FiSettings aria-hidden className="size-4 shrink-0" />
              Settings
            </Button>
          </div>
        }
        detail={
          <>
            <span className="mb-1 block">
              <McpModeLabel mode={server.credential_mode} isCaps />
            </span>
            {server.purpose_hint ? (
              <span className="block break-words">{server.purpose_hint}</span>
            ) : null}
            <span className="mt-1 block text-caption">
              {availability === "off"
                ? AVAILABILITY.off.help
                : `${AVAILABILITY[availability].help} Every workspace in your organization can use it, within the limits below.`}
            </span>
            {/* One line at every width, as on a workspace's server page. */}
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
                Add this id to <code className="font-mono">mcp_server_ids</code>{" "}
                in your request.
              </span>
            ) : null}
          </>
        }
      />
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

      <div className="flex flex-col gap-8">
        {server.credential_mode === "per_user" ? (
          <McpServerSignIn scope={SCOPE} server={server} canManage />
        ) : (
          <McpSharedAccount
            scope={SCOPE}
            server={server}
            canManage
            onOpenSettings={openSettings}
            onLeave={onLeave}
            openWindow={openWindow}
          />
        )}
        <McpExposedTools
          key={JSON.stringify(server.allowed_tools)}
          scope={SCOPE}
          server={server}
          canManage
          onAddClient={addClient}
        />
        <OrganizationMcpServerWorkspaces server={server} />
        <McpServerConnection
          server={server}
          onOpenSettings={() => openSettings()}
        />
      </div>

      <McpServerSettingsDialog
        key={settings.count}
        isOpen={settings.isOpen}
        onOpenChange={(isOpen) => {
          if (!isOpen) closeSettings()
        }}
        scope={SCOPE}
        server={server}
        focus={settings.focus}
        onSaved={(saved) => {
          setUnchecked(saved)
          closeSettings()
        }}
        onDelete={openDelete}
      />

      <ConfirmDialog
        isOpen={isDeleting}
        onOpenChange={setDeleting}
        heading="Delete MCP server"
        body={`${server.name} is removed from every workspace, with its token and everyone's connection to it. Requests that still ask for it will fail.`}
        confirmLabel="Delete server"
        isPending={remove.isPending}
        error={remove.error}
        onConfirm={() =>
          remove.mutate(server.id, {
            onSuccess: () => {
              setDeleting(false)
              void navigate({ to: "/organization/mcp-servers" })
            },
          })
        }
      />
    </div>
  )
}
