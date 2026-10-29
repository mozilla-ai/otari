import { Link, useNavigate } from "@tanstack/react-router"
import { useState } from "react"
import { FiTrash2 } from "react-icons/fi"

import { Button } from "@/design-system/actions/Button"
import { RowAction } from "@/design-system/actions/RowAction"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { EmptyState } from "@/design-system/feedback/EmptyState"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { PageLoading } from "@/design-system/feedback/PageLoading"
import { Dot } from "@/design-system/indicators/Dot"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { canManageWorkspace } from "@/features/organization/roles"
import { McpExposedTools } from "@/features/tools/McpExposedTools"
import { McpServerAcl } from "@/features/tools/McpServerAcl"
import { McpServerConnection } from "@/features/tools/McpServerConnection"
import { useOrganizationContext } from "@/shared/api/organizations"
import {
  useDeleteWorkspaceMcpServer,
  useUpdateWorkspaceMcpServer,
  useWorkspaceMcpServer,
} from "@/shared/api/tools"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"

/**
 * One stored MCP server of the selected workspace, on one page in the order a
 * call meets it: the tools it exposes, its ACL, then how otari reaches it.
 *
 * Takes the id as a prop rather than reading the router, so a test can hand it
 * one without a route tree.
 */
export function McpServerDetailPage({ serverId }: { serverId: string }) {
  const navigate = useNavigate()
  const { selected, isLoading: workspaceLoading } = useSelectedWorkspace()
  const context = useOrganizationContext()
  const canManage = canManageWorkspace(context.data, selected?.role)
  const workspaceId = selected ? selected.workspace_id : null
  const query = useWorkspaceMcpServer(workspaceId, serverId)
  const update = useUpdateWorkspaceMcpServer()
  const remove = useDeleteWorkspaceMcpServer()
  const [isDeleting, setDeleting] = useState(false)

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

  return (
    <div className="flex min-w-0 flex-col gap-4">
      {back}
      <PageIntro
        title={server.name}
        action={
          <div className="flex flex-wrap items-center gap-3">
            <span className="flex items-center gap-2 text-mono-caption text-muted">
              <Dot
                className={server.enabled ? "bg-success" : "bg-text-subtle"}
              />
              {server.enabled ? "ENABLED" : "DISABLED"}
            </span>
            {canManage ? (
              <>
                <Button
                  onPress={() =>
                    update.mutate({
                      ...ids,
                      body: { enabled: !server.enabled },
                    })
                  }
                  isPending={update.isPending}
                  className="min-h-11 md:min-h-0"
                >
                  {server.enabled ? "Disable" : "Enable"}
                </Button>
                <RowAction
                  icon={FiTrash2}
                  label="Delete server"
                  ariaLabel={`Delete ${server.name}`}
                  onPress={openDelete}
                />
              </>
            ) : null}
          </div>
        }
      >
        {server.purpose_hint ? (
          <span className="break-words">{server.purpose_hint}</span>
        ) : null}
      </PageIntro>
      <ErrorBanner error={update.error} />
      {canManage ? null : (
        <InfoBanner>
          Only an owner or admin of the workspace or organization can change
          this server.
        </InfoBanner>
      )}

      <div className="flex flex-col gap-8">
        <McpExposedTools
          key={JSON.stringify(server.allowed_tools)}
          workspaceId={server.workspace_id}
          server={server}
          canManage={canManage}
        />
        <McpServerAcl
          workspaceId={server.workspace_id}
          server={server}
          canManage={canManage}
        />
        <McpServerConnection
          key={server.id}
          workspaceId={server.workspace_id}
          server={server}
          canManage={canManage}
        />
      </div>

      <ConfirmDialog
        isOpen={isDeleting}
        onOpenChange={setDeleting}
        heading="Delete MCP server"
        body={`${server.name}, the token stored with it and its access list are removed. A request that still names its id is refused, so update the callers that use it.`}
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
