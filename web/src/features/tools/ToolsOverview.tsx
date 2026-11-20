import { Link } from "@tanstack/react-router"
import { FiChevronRight } from "react-icons/fi"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { Dot } from "@/design-system/indicators/Dot"
import { useIsOrganizationManager } from "@/features/organization/access"
import type { ToolsView } from "@/features/tools/ToolsFrame"
import { useWorkspaceToolState } from "@/features/tools/useWorkspaceToolState"
import { useDeploymentOperator } from "@/shared/api/organizations"
import {
  useTools,
  useWorkspaceCodeExecutionPolicy,
  useWorkspaceMcpServers,
  useWorkspaceWebSearchConfig,
} from "@/shared/api/tools"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"

interface ToolRow {
  id: string
  name: string
  /** "On", "Off", "Unavailable", or a count. */
  state: string
  tone: "on" | "off" | "unknown"
  /** What it lets an agent do, or why it cannot. */
  line: string
  to: ToolsView
  search?: { service: string }
}

/** "GitHub", "GitHub and Linear", "GitHub, Linear and 2 more". */
function listed(names: string[]): string {
  if (names.length <= 2) return names.join(" and ")
  return `${names.slice(0, 2).join(", ")} and ${names.length - 2} more`
}

/**
 * Everything an agent in this workspace can reach beyond the model, one row a
 * tool: whether it is on, what it does, and the way into its settings. The
 * question the page answers is "what can my agents use right now", which the
 * settings pages each answer for one tool only.
 */
export function ToolsOverview() {
  const { selected, isSettled, isUnknown } = useSelectedWorkspace()
  const { isOperator } = useDeploymentOperator()
  const workspaceId = selected?.workspace_id ?? null
  const workspace = selected?.name ?? "this workspace"
  const tools = useTools()
  // Whether a tool is switched on for the workspace is the owners' and
  // admins' to read. Anyone else is refused, so they are told what the
  // deployment offers and not asked a question they cannot have answered.
  const isManager = useIsOrganizationManager()
  const web = useWorkspaceWebSearchConfig(isManager ? workspaceId : null)
  const code = useWorkspaceCodeExecutionPolicy(isManager ? workspaceId : null)
  const mcp = useWorkspaceMcpServers(workspaceId)
  const inWorkspace = useWorkspaceToolState()

  const toolBy = new Map((tools.data?.data ?? []).map((one) => [one.id, one]))

  /**
   * What a row says in place of a fact its read has not given: "…" while it
   * comes, "Could not read" once it failed, a dash with no workspace to ask
   * about. Null once the read has answered.
   */
  const unread = (
    query: { data?: unknown; isError: boolean },
    perWorkspace: boolean,
  ): string | null => {
    if (query.data !== undefined) return null
    if (query.isError) return "Could not read"
    if (perWorkspace && workspaceId === null && isSettled) {
      return isUnknown ? "Could not read" : "—"
    }
    return "…"
  }

  /** A deployment tool this workspace may or may not have switched on. */
  const managed = (
    toolId: string,
    name: string,
    here: { data?: unknown; isError: boolean },
    isOnHere: boolean | undefined,
    does: string,
    to: ToolsView,
  ): ToolRow => {
    const tool = toolBy.get(toolId)
    const unknown = (state: string): ToolRow => ({
      id: toolId,
      name,
      state,
      tone: "unknown",
      line: does,
      to,
    })
    const offered = unread(tools, false)
    if (offered) return unknown(offered)
    if (!tool?.available) {
      return {
        id: toolId,
        name,
        state: "Unavailable",
        tone: "off",
        line: `Not set up on this deployment. ${does}`,
        to,
      }
    }
    if (!isManager) {
      // The workspace's own answer, which any member may read.
      const here = inWorkspace(toolId)
      if (here?.enabled === null) {
        return unknown(here.isError ? "Could not read" : "…")
      }
      return here?.enabled === false
        ? {
            id: toolId,
            name,
            state: "Off",
            tone: "off",
            line: `${here.reason ?? `Off in ${workspace}.`} ${does}`,
            to,
          }
        : { id: toolId, name, state: "Available", tone: "on", line: does, to }
    }
    const switched = unread(here, true)
    if (switched) return unknown(switched)
    return isOnHere
      ? { id: toolId, name, state: "On", tone: "on", line: does, to }
      : {
          id: toolId,
          name,
          state: "Off",
          tone: "off",
          line: `No agent in ${workspace} can use it yet. ${does}`,
          to,
        }
  }

  const mcpUnread = unread(mcp, true)
  const servers = (mcp.data?.data ?? []).filter((server) => server.enabled)
  const serverTools = servers.reduce(
    (count, server) => count + (server.allowed_tools?.length ?? 0),
    0,
  )
  const rows: ToolRow[] = [
    managed(
      "otari_web_search",
      "Web search",
      web,
      web.data?.enabled,
      "Lets the model search the web for current information.",
      "/tools/web-search",
    ),
    managed(
      "otari_web_fetch",
      "Web fetch",
      web,
      web.data?.enabled,
      "Lets the model read a web page by its URL.",
      "/tools/web-search",
    ),
    managed(
      "otari_code_execution",
      "Code execution",
      code,
      code.data?.enabled,
      "Lets the model run code in a sandbox and read the result.",
      "/tools/code-execution",
    ),
    {
      id: "mcp",
      name: "MCP servers",
      state: mcpUnread ?? String(servers.length),
      tone: mcpUnread ? "unknown" : servers.length > 0 ? "on" : "off",
      // Unread, it says what the tool does, as the rows above do.
      line: mcpUnread
        ? "Lets an agent call the tools a connected server offers."
        : servers.length > 0
          ? `${listed(servers.map((server) => server.name))} ${servers.length === 1 ? "is" : "are"} connected${serverTools > 0 ? `: ${serverTools} ${serverTools === 1 ? "tool" : "tools"} between them` : ""}.`
          : `No server is connected to ${workspace}. Connect one to give agents its tools.`,
      to: "/tools/mcp-servers",
    },
    ...(isOperator
      ? [
          {
            id: "guardrails",
            name: "Guardrail services",
            state: "Operator",
            tone: "unknown" as const,
            line: "The endpoints this deployment calls to check a request. Which checks run is set under Routing, Guardrails.",
            to: "/tools" as const,
            search: { service: "guardrails" },
          },
        ]
      : []),
  ]

  return (
    <div className="flex flex-col gap-5">
      <ErrorBanner
        error={tools.error ?? web.error ?? code.error ?? mcp.error}
      />
      <ul className="flex flex-col border-t border-border">
        {rows.map((row) => (
          <li key={row.id} className="border-b border-border-subtle">
            <Link
              to={row.to}
              search={row.search}
              className="flex min-h-16 items-center gap-x-6 gap-y-1 py-3 transition-colors hover:bg-surface-alt motion-reduce:transition-none max-md:flex-wrap"
            >
              <span className="w-44 shrink-0 text-title">{row.name}</span>
              <span className="flex w-32 shrink-0 items-center gap-2.5 text-mono-caption">
                <Dot
                  className={
                    row.tone === "on"
                      ? "bg-success"
                      : row.tone === "off"
                        ? "bg-muted"
                        : "bg-border"
                  }
                />
                {row.state}
              </span>
              <span className="min-w-0 flex-1 basis-64 text-sm text-muted">
                {row.line}
              </span>
              <FiChevronRight aria-hidden className="size-4 shrink-0" />
            </Link>
          </li>
        ))}
      </ul>
    </div>
  )
}
