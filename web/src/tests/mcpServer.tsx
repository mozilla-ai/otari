import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { useParams } from "@tanstack/react-router"
import { render } from "@testing-library/react"
import { vi } from "vitest"

import type {
  ApiKey,
  McpServerConnection,
  McpServerFoldCandidate,
  McpServerWorkspaceSettings,
  OrganizationContext,
  OrganizationMember,
  WorkspaceMcpServer,
  WorkspaceMcpServerAccess,
  WorkspaceMcpServerAccessCheck,
  WorkspaceMcpServerMyTools,
} from "@/client"
import { McpServerDetailPage } from "@/features/tools/McpServerDetailPage"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  bootstrap,
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
  row = server,
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
  connections = [],
  myTools,
  copies = [],
}: {
  server?: WorkspaceMcpServer
  /**
   * What a read of the server answers with; `server` when left out. Apart from
   * it so a test can answer with a row that names no workspace, while the
   * server is still found only in `server`'s.
   */
  row?: WorkspaceMcpServer
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
  /** Who is signed in to the server, as a manager reads it. */
  connections?: McpServerConnection[]
  /** What the caller may use on the server; unknown, under everyone's rule, when left out. */
  myTools?: WorkspaceMcpServerMyTools
  /** Other workspaces' own copies of the server, as promoting it reads them. */
  copies?: McpServerFoldCandidate[]
} = {}): McpCall[] {
  const calls: McpCall[] = []
  // A read after a write of the workspace's own settings answers with them.
  let saved = row
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
    const body =
      typeof init?.body === "string"
        ? (JSON.parse(init.body) as Record<string, unknown>)
        : undefined
    calls.push({ url, method, body })
    if (url.includes("/mcp-servers/logos")) {
      return Response.json({ data: [] })
    }
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
    if (url.endsWith("/my-tools")) {
      return Response.json(
        myTools ?? {
          server_id: server.id,
          decided_by: "everyone",
          access: server.everyone_tools.length > 0 ? "some" : "none",
          known: false,
          checked_at: null,
          tools: [],
        },
      )
    }
    // Who is signed in to the server.
    if (url.includes("/connections?")) {
      return Response.json({ data: connections, count: connections.length })
    }
    // The caller's own sign-ins: none.
    if (url.includes("/mcp-connections")) {
      return Response.json({ data: [], count: 0 })
    }
    if (
      method === "POST" &&
      url === "/api/v1/organizations/me/mcp-servers/same-url"
    ) {
      return Response.json({ data: copies })
    }
    if (url.includes("/mcp-servers/")) {
      // Read from any other workspace, the server is not there.
      if (!url.includes(`/workspaces/${server.workspace_id}/`)) {
        return Response.json(
          { detail: `MCP server ${server.id} not found` },
          { status: 404 },
        )
      }
      // A workspace's own settings on its organization's server, as the
      // gateway keeps them: a PATCH changes only the fields it sends, and a
      // DELETE takes them all away. The organization's limit is never touched.
      if (url.endsWith("/override") && saved.in_workspace) {
        const own: McpServerWorkspaceSettings =
          method === "PATCH"
            ? ({ ...saved.in_workspace, ...body } as McpServerWorkspaceSettings)
            : {
                ...saved.in_workspace,
                enabled: true,
                everyone_tools: null,
                attach_by_default: null,
              }
        const limit = own.organization_tools
        saved = {
          ...saved,
          in_workspace: {
            ...own,
            reached:
              saved.enabled &&
              own.enabled &&
              (limit === null || limit.length > 0),
          },
        }
        return Response.json(saved)
      }
      if (method === "DELETE") return new Response(null, { status: 204 })
      return Response.json(method === "PATCH" ? { ...row, ...body } : saved)
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

/** Names the server it was opened for, so a test can tell which one. */
function OrganizationServerPageProbe() {
  const { serverId } = useParams({ strict: false })
  return <p>Organization server page {serverId}</p>
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
    <DeploymentProvider value={bootstrap()}>
      <QueryClientProvider client={client}>
        <SelectedWorkspaceProvider>
          <McpServerDetailPage serverId={serverId} workspaceId={workspaceId} />
        </SelectedWorkspaceProvider>
      </QueryClientProvider>
    </DeploymentProvider>,
    {
      wrapper: withRouter({
        url,
        routes: [
          { path: "/tools/mcp-servers", element: <p>Server list</p> },
          {
            path: "/organization/mcp-servers/$serverId",
            element: <OrganizationServerPageProbe />,
          },
        ],
      }),
    },
  )
}
