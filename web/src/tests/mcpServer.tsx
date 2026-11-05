import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render } from "@testing-library/react"
import { vi } from "vitest"

import type {
  ApiKey,
  OrganizationContext,
  OrganizationMember,
  WorkspaceMcpServer,
  WorkspaceMcpServerAccess,
  WorkspaceMcpServerAccessCheck,
} from "@/client"
import { McpServerDetailPage } from "@/features/tools/McpServerDetailPage"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import {
  organizationContext,
  workspaceMcpServer,
  workspaceMember,
} from "@/tests/fixtures"
import { withRouter } from "@/tests/router"

// The harness for a stored MCP server's own page: a stub of the transport that
// answers the reads the page makes and records every call, and a render that
// mounts the page in a real router and query client.

export const WORKSPACE = "44444444-4444-4444-4444-444444444444"

export interface McpCall {
  url: string
  method: string
  body: Record<string, unknown> | undefined
}

export function mockMcpServerApi({
  server = workspaceMcpServer(),
  access = { everyone_tools: ["*"], grants: [] },
  written,
  check,
  keys = [],
  members = [],
  workspaceMemberIds,
  role = "owner",
  workspaceRole = "admin",
  accessStatus,
  deploymentOperator = true,
  otherWorkspaces = [],
}: {
  server?: WorkspaceMcpServer
  access?: WorkspaceMcpServerAccess
  /** What a grant write answers with. */
  written?: WorkspaceMcpServerAccess
  check?: WorkspaceMcpServerAccessCheck
  keys?: ApiKey[]
  /** The organization roster, which names users and joins them to their keys. */
  members?: OrganizationMember[]
  /** Which of `members` belong to the workspace; all of them when left out. */
  workspaceMemberIds?: readonly string[]
  role?: OrganizationContext["role"]
  workspaceRole?: string
  /** A refusal of the access list, as a viewer earns. */
  accessStatus?: number
  deploymentOperator?: boolean
  /** Workspaces the caller also belongs to, after the selected one. */
  otherWorkspaces?: { workspace_id: string; name: string; role: string }[]
} = {}): McpCall[] {
  const calls: McpCall[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
    const body =
      typeof init?.body === "string"
        ? (JSON.parse(init.body) as Record<string, unknown>)
        : undefined
    calls.push({ url, method, body })
    if (url.includes("/access")) {
      if (accessStatus !== undefined) {
        return Response.json({ detail: "refused" }, { status: accessStatus })
      }
      if (url.includes("/access/check")) {
        return Response.json(
          check ?? {
            allowed: true,
            decided_by: "everyone",
            grant_id: null,
            tools: ["*"],
          },
        )
      }
      return Response.json(method === "GET" ? access : (written ?? access))
    }
    if (url.includes("/mcp-servers/")) {
      // Read from any other workspace, the server is not there.
      if (!url.includes(`/workspaces/${server.workspace_id}/`)) {
        return Response.json(
          { detail: `MCP server ${server.id} not found` },
          { status: 404 },
        )
      }
      if (method === "DELETE") return new Response(null, { status: 204 })
      return Response.json(method === "PATCH" ? { ...server, ...body } : server)
    }
    if (url.includes("/members")) {
      if (url.includes("/organizations/me/members")) {
        return Response.json({ data: members, count: members.length })
      }
      const rows = members
        .filter(
          (member) =>
            !workspaceMemberIds ||
            workspaceMemberIds.includes(member.user_id ?? ""),
        )
        .map((member) =>
          workspaceMember({
            user_id: member.user_id ?? "",
            workspace_id: WORKSPACE,
          }),
        )
      return Response.json({ data: rows, count: rows.length })
    }
    if (url.includes("/keys")) return Response.json(keys)
    return Response.json(
      organizationContext({
        role,
        deployment_operator: deploymentOperator,
        workspace_memberships: [
          { workspace_id: WORKSPACE, name: "Alpha", role: workspaceRole },
          ...otherWorkspaces,
        ],
      }),
    )
  })
  return calls
}

/** The calls that wrote something. */
export function mcpWrites(calls: McpCall[]): McpCall[] {
  return calls.filter((call) => call.method !== "GET")
}

export function renderMcpServerPage(
  serverId: string,
  search = "",
  workspaceId?: string,
) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  const url = `/tools/mcp-servers/${serverId}${search}`
  return render(
    <QueryClientProvider client={client}>
      <SelectedWorkspaceProvider>
        <McpServerDetailPage serverId={serverId} workspaceId={workspaceId} />
      </SelectedWorkspaceProvider>
    </QueryClientProvider>,
    {
      wrapper: withRouter({
        url,
        routes: [{ path: "/tools/mcp-servers", element: <p>Server list</p> }],
      }),
    },
  )
}
