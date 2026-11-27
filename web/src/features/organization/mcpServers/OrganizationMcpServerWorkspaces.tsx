import { useState } from "react"
import { FiRotateCcw, FiSliders } from "react-icons/fi"

import type {
  OrganizationMcpServerWorkspace,
  WorkspaceMcpServer,
} from "@/client"
import { RowAction, RowActionRow } from "@/design-system/actions/RowAction"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { Dot } from "@/design-system/indicators/Dot"
import { SettingsGroup } from "@/design-system/layout/SettingsGroup"
import { AccessChooser } from "@/features/tools/AccessChooser"
import {
  type AccessMode,
  accessDraft,
  isDraftComplete,
  isLiteralToolName,
  ruleTools,
  toolsFor,
} from "@/features/tools/mcpAccess"
import { checkedTools } from "@/features/tools/mcpServerHealth"
import {
  useLiftOrganizationMcpServerLimit,
  useOrganizationMcpServerWorkspaces,
  useSetOrganizationMcpServerLimit,
} from "@/shared/api/organizationMcpServers"

/** The organization's limit in a workspace, in words. */
function limitLabel(tools: readonly string[] | null): string {
  if (tools === null) return "No limit"
  if (tools.length === 0) return "Turned off"
  return ruleTools(tools) === "Any" ? "Any tool" : ruleTools(tools)
}

/** What the workspace's own admins did with the server, in words. */
function workspaceSideLabel(row: OrganizationMcpServerWorkspace): string {
  if (!row.workspace_enabled) return "Turned off by its admins"
  if (row.narrowed) return "Customized by its admins"
  return "As set here"
}

/**
 * Sets the organization's limit for one workspace: the most its callers reach
 * on the server, which its own rules can narrow further but never widen. Deny
 * turns the server off there. The caller remounts this per opening.
 */
function LimitDialog({
  server,
  workspace,
  onClose,
}: {
  server: WorkspaceMcpServer
  workspace: OrganizationMcpServerWorkspace
  onClose: () => void
}) {
  const set = useSetOrganizationMcpServerLimit()
  const seed = accessDraft(workspace.organization_tools ?? ["*"])
  const [mode, setMode] = useState<AccessMode>(seed.mode)
  const [picked, setPicked] = useState<string[]>(seed.picked)
  // What a limit can usefully name: the tools the server exposes.
  const picker = checkedTools(server, { withinAllowList: true })
  const tools = picker.isKnown
    ? picker.tools
    : (server.allowed_tools ?? []).filter(isLiteralToolName)
  const next = toolsFor(mode, picked)
  // With no limit stored, the editor opens on Any tool: saving it is a
  // change, closing it untouched is not.
  const isChanged =
    JSON.stringify(next) !== JSON.stringify(workspace.organization_tools)
  const isEdited =
    JSON.stringify(next) !== JSON.stringify(toolsFor(seed.mode, seed.picked))

  return (
    <FormDialog
      isOpen
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
      title={`Limit for ${workspace.name}`}
      description={`The most ${workspace.name}'s keys and members reach on ${server.name}. Its admins can narrow this, never widen it. Deny turns the server off there.`}
      submitLabel="Save limit"
      onSubmit={() => {
        if (!isDraftComplete(mode, picked)) return
        set.mutate(
          {
            serverId: server.id,
            workspaceId: workspace.workspace_id,
            tools: next,
          },
          { onSuccess: onClose },
        )
      }}
      isPending={set.isPending}
      isSubmitDisabled={!isDraftComplete(mode, picked) || !isChanged}
      isDirty={isEdited}
      error={set.error}
    >
      <AccessChooser
        label="Tools"
        mode={mode}
        picked={picked}
        tools={tools}
        descriptions={picker.descriptions}
        isKnown={picker.isKnown}
        readTools={server.read_tools}
        isDisabled={set.isPending}
        onChange={(nextMode, nextPicked) => {
          setMode(nextMode)
          setPicked(nextPicked)
        }}
      />
    </FormDialog>
  )
}

/**
 * Every workspace of the organization as the server reaches it: whether its
 * requests reach the server, the organization's limit for it, and whether its
 * own admins narrowed it. The limit is set and lifted per row.
 */
export function OrganizationMcpServerWorkspaces({
  server,
}: {
  server: WorkspaceMcpServer
}) {
  const query = useOrganizationMcpServerWorkspaces(server.id)
  const lift = useLiftOrganizationMcpServerLimit()
  const [editing, setEditing] = useState<OrganizationMcpServerWorkspace>()
  // Bumped on each open and used as the dialog's key, so a draft never
  // survives into the next one.
  const [openCount, setOpenCount] = useState(0)
  const rows = query.data?.data ?? []

  const openLimit = (row: OrganizationMcpServerWorkspace) => {
    // Resetting a pending lift would forget it is in flight, and release the
    // hold on its row while its DELETE could still land after a new limit.
    if (!lift.isPending) lift.reset()
    setOpenCount((count) => count + 1)
    setEditing(row)
  }

  const columns: DataTableColumn<OrganizationMcpServerWorkspace>[] = [
    {
      id: "name",
      header: "Workspace",
      isRowHeader: true,
      cell: (row) => <span className="font-medium">{row.name}</span>,
    },
    {
      id: "reached",
      header: "Reaches it",
      cell: (row) => (
        <span className="flex items-center gap-2 whitespace-nowrap">
          <Dot className={row.reached ? "bg-success" : "bg-text-subtle"} />
          {row.reached ? "Yes" : "No"}
        </span>
      ),
    },
    {
      id: "limit",
      header: "Organization limit",
      cell: (row) => (
        <span
          className={`break-words ${row.organization_tools === null ? "text-muted" : ""}`}
        >
          {limitLabel(row.organization_tools)}
        </span>
      ),
    },
    {
      id: "workspace",
      header: "Workspace's own",
      cell: (row) => (
        <span className="whitespace-nowrap text-muted">
          {workspaceSideLabel(row)}
        </span>
      ),
    },
    {
      id: "actions",
      header: "",
      cell: (row) => {
        // A limit drafted while its lift is in flight would start from the
        // limit being lifted, and could land before or after it.
        const isLifting =
          lift.isPending && lift.variables?.workspaceId === row.workspace_id
        return (
          <RowActionRow>
            <RowAction
              icon={FiSliders}
              label="Set limit…"
              ariaLabel={`Set the limit for ${row.name}${isLifting ? " (its limit is being lifted)" : ""}`}
              isDisabled={isLifting}
              onPress={() => openLimit(row)}
            />
            {row.organization_tools === null ? null : (
              <RowAction
                icon={FiRotateCcw}
                label="Lift limit"
                ariaLabel={`Lift the limit for ${row.name}`}
                isDisabled={lift.isPending}
                onPress={() =>
                  lift.mutate({
                    serverId: server.id,
                    workspaceId: row.workspace_id,
                  })
                }
              />
            )}
          </RowActionRow>
        )
      },
    },
  ]

  return (
    <SettingsGroup
      title="Workspaces"
      count={query.data ? rows.length : undefined}
      description="Each workspace reaches this server unless a limit here or its own admins say otherwise. A limit is the most its keys and members reach; its admins can narrow it, never widen it."
    >
      <ErrorBanner error={query.error ?? lift.error} />
      {/* A table that was never read is not an empty one; a failed reread keeps the rows it had. */}
      {query.isError && !query.data ? null : (
        <DataTable
          ariaLabel={`Workspaces reaching ${server.name}`}
          mobileLayout="cards"
          columns={columns}
          rows={rows}
          getRowKey={(row) => row.workspace_id}
          isLoading={query.isPending && !query.data}
          emptyContent="This organization has no workspaces yet."
        />
      )}
      {editing ? (
        <LimitDialog
          key={openCount}
          server={server}
          workspace={editing}
          onClose={() => setEditing(undefined)}
        />
      ) : null}
    </SettingsGroup>
  )
}
