import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { useParams } from "@tanstack/react-router"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type {
  McpServerRequest,
  McpServerRequestGroup,
  OrganizationContext,
  WorkspaceMcpServer,
  WorkspaceMcpServerHealth,
} from "@/client"
import {
  connectionsLabel,
  WorkspaceMcpServersCard,
} from "@/features/tools/WorkspaceMcpServersCard"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  bootstrap,
  workspaceMcpServerHealth as health,
  mcpCatalogServer,
  organizationContext,
  workspace,
  workspaceMcpServer,
} from "@/tests/fixtures"
import { withRouter } from "@/tests/router"

const ALPHA = "11111111-1111-1111-1111-111111111111"

interface Call {
  url: string
  method: string
  body: Record<string, unknown> | undefined
}

function mockApi({
  memberships = [{ workspace_id: ALPHA, name: "Alpha", role: "admin" }],
  servers = [] as WorkspaceMcpServer[],
  written,
  tested,
  writeStatus,
  writeDetail,
  role = "owner",
  workspaceStatus,
  connectedIds = [],
  requests = [],
  mine = [],
}: {
  memberships?: { workspace_id: string; name: string; role: string }[]
  servers?: WorkspaceMcpServer[]
  // What a successful write answers with, when it is not the first listed row.
  written?: WorkspaceMcpServer
  // What Test connection answers with.
  tested?: WorkspaceMcpServerHealth
  writeStatus?: number
  writeDetail?: string
  // The caller's organization role; a member-view test lowers it.
  role?: OrganizationContext["role"]
  // Refuses the read of the workspace's own row with this status.
  workspaceStatus?: number
  // The servers the caller has connected their own account to.
  connectedIds?: string[]
  // What members are waiting for, as a manager reads it.
  requests?: McpServerRequestGroup[]
  // The caller's own requests, as a member reads them.
  mine?: McpServerRequest[]
} = {}): Call[] {
  const calls: Call[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
    if (url.includes("/mcp-server-requests")) {
      calls.push({
        url,
        method,
        body:
          typeof init?.body === "string"
            ? (JSON.parse(init.body) as Record<string, unknown>)
            : undefined,
      })
      if (url.endsWith("/mine")) return Response.json({ data: mine })
      if (url.endsWith("/acknowledge")) {
        return new Response(null, { status: 204 })
      }
      if (url.endsWith("/decline")) return Response.json({ declined: 1 })
      return Response.json({
        data: requests,
        count: requests.reduce(
          (total, group) => total + group.requesters.length,
          0,
        ),
      })
    }
    if (url.includes("/mcp-catalog/")) {
      return Response.json({ data: [], count: 0 })
    }
    if (url.includes("/mcp-connections")) {
      calls.push({ url, method, body: undefined })
      if (method === "POST") {
        return Response.json({
          authorization_url: "https://auth.example.com/authorize",
        })
      }
      if (method === "DELETE") return new Response(null, { status: 204 })
      return Response.json({
        data: connectedIds.map((server_id) => ({
          server_id,
          connected_at: "2026-11-01T00:00:00Z",
          updated_at: "2026-11-01T00:00:00Z",
          expires_at: null,
          renewable: true,
        })),
      })
    }
    if (url.endsWith("/check/sign-in")) {
      calls.push({ url, method, body: undefined })
      return Response.json({
        authorization_url:
          "https://auth.example.com/authorize?state=check-state",
      })
    }
    if (url.includes("/mcp-servers")) {
      calls.push({
        url,
        method,
        body:
          typeof init?.body === "string"
            ? (JSON.parse(init.body) as Record<string, unknown>)
            : undefined,
      })
      if (method !== "GET" && writeStatus !== undefined) {
        return Response.json(
          { detail: writeDetail ?? "refused" },
          { status: writeStatus },
        )
      }
      if (method === "GET") {
        return Response.json({ data: servers, count: servers.length })
      }
      if (url.endsWith("/mcp-servers/check")) {
        return Response.json(tested ?? health())
      }
      return Response.json(written ?? servers[0] ?? workspaceMcpServer())
    }
    if (url.includes(`/workspaces/${ALPHA}`) || url.includes("/workspaces?")) {
      calls.push({
        url,
        method,
        body:
          typeof init?.body === "string"
            ? (JSON.parse(init.body) as Record<string, unknown>)
            : undefined,
      })
      const row = workspace({ id: ALPHA, name: "Alpha" })
      if (method !== "GET") {
        return Response.json({ ...row, request_mcp_servers_allowed: false })
      }
      if (workspaceStatus !== undefined) {
        return Response.json(
          { detail: "Workspace unavailable" },
          { status: workspaceStatus },
        )
      }
      return url.includes("/workspaces?")
        ? Response.json({ data: [row], count: 1 })
        : Response.json(row)
    }
    return Response.json(
      organizationContext({ role, workspace_memberships: memberships }),
    )
  })
  return calls
}

function ServerPageProbe() {
  const { serverId } = useParams({ strict: false })
  return <p>Server page {serverId}</p>
}

function renderCard(variant?: "card" | "page", surfaces?: string[]) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <DeploymentProvider value={bootstrap(surfaces ? { surfaces } : {})}>
      <QueryClientProvider client={client}>
        <SelectedWorkspaceProvider>
          <WorkspaceMcpServersCard variant={variant} />
        </SelectedWorkspaceProvider>
      </QueryClientProvider>
    </DeploymentProvider>,
    {
      // A real router, because a row and a new server open the server's own
      // page; the probe stands in for that page.
      wrapper: withRouter({
        url: "/tools/mcp-servers",
        routes: [
          {
            path: "/tools/mcp-servers/$serverId",
            element: <ServerPageProbe />,
          },
        ],
      }),
    },
  )
}

// The Add button is rendered before the list resolves, so a test that clicks it
// immediately would submit against a table that had not loaded. The rendered
// list is the signal that the workspace and its servers are both in hand.
async function renderLoaded(rows: WorkspaceMcpServer[] = []) {
  renderCard()
  if (rows.length === 0) {
    await screen.findByText("No MCP servers yet.")
    return
  }
  await screen.findByText(rows[0].name)
}

function writes(calls: Call[]): Call[] {
  return calls.filter((call) => call.method !== "GET")
}

describe("WorkspaceMcpServersCard", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
    window.sessionStorage.clear()
  })

  it("opens the page it is the whole of, heading and register control", async () => {
    // `variant="page"` is what renders /tools/mcp-servers: its h1, the
    // description under it, and the register control in that heading row. The
    // page component is a one-line wrapper with no test of its own, so without
    // this the page's only h1 could be dropped silently.
    mockApi()
    renderCard("page")

    expect(
      await screen.findByRole("heading", { level: 1, name: "MCP servers" }),
    ).toBeInTheDocument()
    expect(screen.getByText(/tools of remote MCP servers/)).toBeInTheDocument()
    expect(
      await screen.findByRole("button", { name: "Add MCP server" }),
    ).toBeInTheDocument()
  })

  it("keeps its own heading where it is one card among several", async () => {
    mockApi()
    renderCard()

    expect(
      await screen.findByRole("heading", { level: 2, name: "MCP servers" }),
    ).toBeInTheDocument()
    expect(screen.queryByRole("heading", { level: 1 })).toBeNull()
  })

  it("lists the workspace's servers with what the API says about each", async () => {
    const servers = [
      workspaceMcpServer({
        name: "github",
        url: "https://mcp.example.com/github",
        has_token: true,
        allowed_tools: ["list_issues", "get_issue"],
      }),
      workspaceMcpServer({
        id: "66666666-6666-6666-6666-666666666666",
        name: "wiki",
        url: "https://mcp.example.com/wiki",
        enabled: false,
      }),
    ]
    mockApi({ servers })
    await renderLoaded(servers)

    expect(screen.getByText("https://mcp.example.com/github")).toBeVisible()
    // A count, never the names.
    expect(screen.getByText("2 tools")).toBeVisible()
    expect(screen.queryByText("list_issues")).not.toBeInTheDocument()
    expect(screen.getByText("All")).toBeVisible()
    expect(screen.queryByText(/^Server ID|^ID$/)).not.toBeInTheDocument()
    // One availability indicator per row, not two badges.
    expect(screen.getByText("ON REQUEST")).toBeVisible()
    expect(screen.getByText("OFF")).toBeVisible()
  })

  it("says per server how it signs in, and how many members connected a personal one", async () => {
    const servers = [
      workspaceMcpServer({ name: "github" }),
      workspaceMcpServer({
        id: "66666666-6666-6666-6666-666666666666",
        name: "drive",
        credential_mode: "per_user",
        connected_users: 3,
      }),
      workspaceMcpServer({
        id: "77777777-7777-7777-7777-777777777777",
        name: "notion",
        credential_mode: "per_user",
        connected_users: 0,
      }),
    ]
    mockApi({ servers, connectedIds: [servers[1].id] })
    await renderLoaded(servers)

    const rowOf = (name: string) =>
      screen.getByText(name).closest("tr") as HTMLElement
    // The two columns read without a legend: how a server signs in, then what
    // stands behind that.
    expect(screen.getByRole("columnheader", { name: "Sign-in" })).toBeVisible()
    expect(
      screen.getByRole("columnheader", { name: "Connections" }),
    ).toBeVisible()
    expect(
      screen.queryByRole("columnheader", { name: "You" }),
    ).not.toBeInTheDocument()
    expect(within(rowOf("github")).getByText("Shared account")).toBeVisible()
    expect(within(rowOf("drive")).getByText("Personal accounts")).toBeVisible()
    // The admin's own connection is a suffix on the count, not a column.
    expect(
      await within(rowOf("drive")).findByText("3 members, including you"),
    ).toBeVisible()
    expect(within(rowOf("notion")).getByText("No one yet")).toBeVisible()
  })

  it("counts the members connected to a Personal server in plain words", () => {
    expect(connectionsLabel(0, false)).toBe("No one yet")
    expect(connectionsLabel(1, false)).toBe("1 member")
    expect(connectionsLabel(4, false)).toBe("4 members")
    expect(connectionsLabel(1, true)).toBe("You")
    expect(connectionsLabel(2, true)).toBe("2 members, including you")
    // A sign-in that just finished is known before the count catches up.
    expect(connectionsLabel(0, true)).toBe("You")
  })

  it("says per Shared server whether its one account is set up", async () => {
    const servers = [
      workspaceMcpServer({ name: "github", has_token: true }),
      workspaceMcpServer({
        id: "66666666-6666-6666-6666-666666666666",
        name: "linear",
        needs_setup: true,
      }),
      workspaceMcpServer({
        id: "77777777-7777-7777-7777-777777777777",
        name: "docs",
        has_token: false,
      }),
    ]
    mockApi({ servers })
    await renderLoaded(servers)

    const rowOf = (name: string) =>
      screen.getByText(name).closest("tr") as HTMLElement
    expect(within(rowOf("github")).getByText("Set up")).toBeVisible()
    expect(within(rowOf("linear")).getByText("Not set up")).toBeVisible()
    // A server that answers without an account has none to set up.
    expect(within(rowOf("docs")).getByText("No account needed")).toBeVisible()
  })

  it("reads an empty allow-list as admitting no tool, the way the gateway does", async () => {
    // The tool loop treats `[]` as a deny-all, distinct from null. "All" would
    // tell an operator the opposite. This form never sends `[]`, but the API
    // accepts one, so a row can arrive here holding it.
    const servers = [workspaceMcpServer({ allowed_tools: [] })]
    mockApi({ servers })
    await renderLoaded(servers)

    expect(screen.getByText("0 tools")).toBeVisible()
    expect(screen.queryByText("All")).not.toBeInTheDocument()
  })

  it("registers a server from the add form", async () => {
    const calls = mockApi()
    const user = userEvent.setup()
    await renderLoaded()

    await user.click(screen.getByRole("button", { name: "Add MCP server" }))
    await user.type(screen.getByLabelText("Name"), "github")
    await user.type(
      screen.getByLabelText("URL"),
      "https://mcp.example.com/github",
    )
    await user.type(
      screen.getByLabelText("Allowed tools"),
      "list_issues, get_issue",
    )
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", {
        name: "Add MCP server",
      }),
    )

    const post = writes(calls).find((call) => call.method === "POST")
    expect(post?.body).toEqual({
      name: "github",
      url: "https://mcp.example.com/github",
      purpose_hint: null,
      allowed_tools: ["list_issues", "get_issue"],
      enabled: true,
      attach_by_default: false,
      everyone_tools: ["*"],
      credential_mode: "shared",
    })
    // No `authorization_token` key at all rather than an explicit null: the
    // operator typed no token, so the request says nothing about one.
    expect(post?.body).not.toHaveProperty("authorization_token")
    expect(post?.body).not.toHaveProperty("oauth_client_id")
  })

  it("shows which servers are attached to every request", async () => {
    const servers = [workspaceMcpServer({ attach_by_default: true })]
    mockApi({ servers })
    await renderLoaded(servers)

    expect(screen.getByText("EVERY REQUEST")).toBeVisible()
  })

  it("registers a server each user signs in to, which takes no shared token", async () => {
    const calls = mockApi()
    const user = userEvent.setup()
    await renderLoaded()

    await user.click(screen.getByRole("button", { name: "Add MCP server" }))
    const dialog = within(screen.getByRole("dialog"))
    expect(screen.getByRole("radio", { name: /^Shared/ })).toBeChecked()
    await user.type(screen.getByLabelText("Name"), "drive")
    await user.type(
      screen.getByLabelText("URL"),
      "http://mcp.example.com/drive",
    )
    await user.click(screen.getByRole("radio", { name: /^Personal/ }))

    // The token box and the pre-save test go: there is no shared token to send.
    expect(
      screen.queryByLabelText("Authorization token"),
    ).not.toBeInTheDocument()
    expect(
      dialog.queryByRole("button", { name: "Test connection" }),
    ).not.toBeInTheDocument()
    // A person's own token travels to it, so it needs https as a token does.
    expect(
      dialog.getByRole("button", { name: "Add MCP server" }),
    ).toBeDisabled()
    await user.clear(screen.getByLabelText("URL"))
    await user.type(
      screen.getByLabelText("URL"),
      "https://mcp.example.com/drive",
    )
    // The secret is on screen beside the ID, and takes a value only once
    // there is a client for it to belong to.
    expect(screen.getByLabelText("Client secret")).toBeDisabled()
    await user.type(screen.getByLabelText("OAuth client ID"), "otari-client")
    await user.type(screen.getByLabelText("Client secret"), "s3cret")
    await user.click(dialog.getByRole("button", { name: "Add MCP server" }))

    const post = writes(calls).find((call) => call.method === "POST")
    expect(post?.body).toMatchObject({
      name: "drive",
      credential_mode: "per_user",
      oauth_client_id: "otari-client",
      oauth_client_secret: "s3cret",
    })
    expect(post?.body).not.toHaveProperty("authorization_token")
  })

  it("surfaces the 409 a duplicate name earns rather than hiding it", async () => {
    mockApi({
      writeStatus: 409,
      writeDetail: "Workspace Alpha already has an MCP server named 'github'",
    })
    const user = userEvent.setup()
    await renderLoaded()

    await user.click(screen.getByRole("button", { name: "Add MCP server" }))
    await user.type(screen.getByLabelText("Name"), "github")
    await user.type(
      screen.getByLabelText("URL"),
      "https://mcp.example.com/github",
    )
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", {
        name: "Add MCP server",
      }),
    )

    expect(
      await screen.findByText(/already has an MCP server named 'github'/),
    ).toBeVisible()
  })

  it("does not show a refused write's banner over the next blank form", async () => {
    mockApi({ writeStatus: 409, writeDetail: "already taken" })
    const user = userEvent.setup()
    await renderLoaded()

    await user.click(screen.getByRole("button", { name: "Add MCP server" }))
    await user.type(screen.getByLabelText("Name"), "github")
    await user.type(screen.getByLabelText("URL"), "https://mcp.example.com")
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", {
        name: "Add MCP server",
      }),
    )
    expect(await screen.findByText("already taken")).toBeVisible()

    // The typed draft is dirty, so leaving goes through the guard.
    await user.click(screen.getByRole("button", { name: "Cancel" }))
    await user.click(screen.getByRole("button", { name: "Discard" }))
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull())
    await user.click(screen.getByRole("button", { name: "Add MCP server" }))

    expect(screen.queryByText("already taken")).not.toBeInTheDocument()
  })

  it("refuses an http URL carrying a token before asking the server", async () => {
    const calls = mockApi()
    const user = userEvent.setup()
    await renderLoaded()

    await user.click(screen.getByRole("button", { name: "Add MCP server" }))
    await user.type(screen.getByLabelText("Name"), "github")
    await user.type(screen.getByLabelText("URL"), "http://mcp.example.com")
    await user.type(screen.getByLabelText("Authorization token"), "ghp_secret")

    // Asserted through the input's own accessible description rather than by
    // finding the text somewhere on screen: the reason for putting it in the
    // field's error slot is that a screen reader on the URL box is told why.
    await waitFor(() =>
      expect(screen.getByLabelText("URL")).toHaveAccessibleDescription(
        /needs an https URL/,
      ),
    )
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", {
        name: "Add MCP server",
      }),
    )
    expect(writes(calls)).toHaveLength(0)
  })

  it("deletes a server once the confirmation is accepted", async () => {
    const servers = [workspaceMcpServer()]
    const calls = mockApi({ servers })
    const user = userEvent.setup()
    await renderLoaded(servers)

    await user.click(screen.getByRole("button", { name: "Delete" }))
    const dialog = await screen.findByRole("alertdialog")
    expect(
      within(dialog).getByText(
        /is removed, with its token and its access rules/,
      ),
    ).toBeVisible()
    await user.click(
      within(dialog).getByRole("button", { name: "Delete server" }),
    )

    await waitFor(() =>
      expect(
        writes(calls).some(
          (call) =>
            call.method === "DELETE" && call.url.endsWith(servers[0].id),
        ),
      ).toBe(true),
    )
  })

  it("gives a member a card for each server they use, with no admin controls", async () => {
    // The list is a member read (otari-ai#1942); only what changes the rows
    // takes the management role.
    const servers = [
      workspaceMcpServer({
        name: "github",
        purpose_hint: "Issues and pull requests",
        caller_access: "read",
      }),
      workspaceMcpServer({
        id: "66666666-6666-6666-6666-666666666666",
        name: "drive",
        credential_mode: "per_user",
        caller_access: "all",
        attach_by_default: true,
      }),
      workspaceMcpServer({
        id: "77777777-7777-7777-7777-777777777777",
        name: "notion",
        credential_mode: "per_user",
      }),
      workspaceMcpServer({
        id: "88888888-8888-8888-8888-888888888888",
        name: "retired",
        enabled: false,
      }),
      workspaceMcpServer({
        id: "99999999-9999-9999-9999-999999999999",
        name: "waiting",
        needs_setup: true,
      }),
    ]
    const calls = mockApi({
      memberships: [{ workspace_id: ALPHA, name: "Alpha", role: "member" }],
      servers,
      role: "member",
      connectedIds: [servers[1].id],
    })
    renderCard()

    const list = await screen.findByRole("list", { name: "Your MCP servers" })
    // Shared servers and the Personal ones connected, together.
    const drive = (await within(list).findByText("drive")).closest("li")
    const github = within(list).getByText("github").closest("li")
    expect(drive).not.toBeNull()
    expect(github).not.toBeNull()
    // Each card says how the server signs in and what the rules give the member.
    expect(within(github as HTMLElement).getByText("Shared")).toBeVisible()
    expect(
      within(drive as HTMLElement).queryByText("Shared"),
    ).not.toBeInTheDocument()
    expect(within(github as HTMLElement).getByText("Read-only")).toBeVisible()
    expect(
      within(github as HTMLElement).getByText("Issues and pull requests"),
    ).toBeVisible()
    expect(
      within(drive as HTMLElement).getByText(/^Connected since /),
    ).toBeVisible()
    expect(within(drive as HTMLElement).getByText("All tools")).toBeVisible()
    // Only a server a request has to ask for says so.
    expect(within(github as HTMLElement).getByText("On request")).toBeVisible()
    expect(
      within(drive as HTMLElement).queryByText("On request"),
    ).not.toBeInTheDocument()
    // The card opens the server's page.
    expect(within(list).getByRole("link", { name: "drive" })).toHaveAttribute(
      "href",
      `/tools/mcp-servers/${servers[1].id}`,
    )
    expect(
      within(list).getByRole("button", { name: "Disconnect drive" }),
    ).toBeVisible()
    expect(
      within(list).queryByRole("button", { name: /github/ }),
    ).not.toBeInTheDocument()
    // Not connected yet, so not a server they use.
    expect(within(list).queryByText("notion")).not.toBeInTheDocument()
    // A disabled server is nothing a member can use, and neither is a Shared
    // one whose account an admin has not set up.
    expect(screen.queryByText("retired")).not.toBeInTheDocument()
    expect(screen.queryByText("waiting")).not.toBeInTheDocument()
    for (const name of ["Add MCP server", "Delete", "Check github"]) {
      expect(screen.queryByRole("button", { name })).not.toBeInTheDocument()
    }
    expect(
      writes(calls).filter((call) => call.url.includes("/mcp-servers")),
    ).toEqual([])
  })

  it("offers a member the servers left to connect as a searchable grid", async () => {
    const servers = [
      workspaceMcpServer({ name: "github" }),
      workspaceMcpServer({
        id: "66666666-6666-6666-6666-666666666666",
        name: "drive",
        credential_mode: "per_user",
      }),
      workspaceMcpServer({
        id: "77777777-7777-7777-7777-777777777777",
        name: "notion",
        credential_mode: "per_user",
      }),
    ]
    mockApi({
      memberships: [{ workspace_id: ALPHA, name: "Alpha", role: "member" }],
      servers,
      role: "member",
      connectedIds: [servers[1].id],
    })
    const user = userEvent.setup()
    renderCard()

    const open = await screen.findByRole("button", {
      name: "Connect a server",
    })
    await waitFor(() => expect(open).toBeEnabled())
    await user.click(open)

    const dialog = await screen.findByRole("dialog", {
      name: "Connect a server",
    })
    const grid = within(dialog).getByRole("list", {
      name: "Servers you can connect",
    })
    expect(
      within(grid).getByRole("button", { name: "Connect notion" }),
    ).toBeVisible()
    // Connected already, or nothing to connect.
    expect(within(grid).queryByText("drive")).not.toBeInTheDocument()
    expect(within(grid).queryByText("github")).not.toBeInTheDocument()
  })

  it("signs a member in from a tile, in a popup, and falls back to the whole page when it is blocked", async () => {
    const servers = [
      workspaceMcpServer({
        name: "notion",
        credential_mode: "per_user",
      }),
    ]
    const calls = mockApi({
      memberships: [{ workspace_id: ALPHA, name: "Alpha", role: "member" }],
      servers,
      role: "member",
    })
    const assign = vi.fn()
    vi.spyOn(window, "location", "get").mockReturnValue({
      ...window.location,
      assign,
    })
    // The browser blocks the popup.
    vi.spyOn(window, "open").mockReturnValue(null)
    const user = userEvent.setup()
    renderCard()

    // Nothing to show yet, so the page is one message and the way to add the first.
    expect(
      await screen.findByRole("heading", { name: "No servers yet" }),
    ).toBeVisible()
    expect(
      screen.queryByRole("list", { name: "Your MCP servers" }),
    ).not.toBeInTheDocument()
    const open = screen.getByRole("button", { name: "Connect a server" })
    await waitFor(() => expect(open).toBeEnabled())
    await user.click(open)
    await user.click(
      await screen.findByRole("button", { name: "Connect notion" }),
    )

    await waitFor(() =>
      expect(assign).toHaveBeenCalledWith("https://auth.example.com/authorize"),
    )
    const started = calls.find(
      (call) => call.method === "POST" && call.url.includes("/mcp-connections"),
    )
    expect(started?.url).toContain(`/mcp-connections/${servers[0].id}`)
    expect(window.sessionStorage.getItem("otari.mcp-sign-in.origin")).toBe(
      "/tools/mcp-servers",
    )
  })

  it("shows each server's last check as a dot beside its name", async () => {
    const servers = [
      workspaceMcpServer({
        name: "github",
        health: health({
          checked_at: new Date().toISOString(),
          tool_count: 3,
          tools: ["list_issues", "get_issue", "close_issue"],
        }),
      }),
      workspaceMcpServer({
        id: "66666666-6666-6666-6666-666666666666",
        name: "wiki",
        health: health({
          status: "failing",
          error: "unauthorized",
          http_status: 401,
          tool_count: null,
        }),
      }),
      workspaceMcpServer({
        id: "77777777-7777-7777-7777-777777777777",
        name: "jira",
      }),
      workspaceMcpServer({
        id: "99999999-9999-9999-9999-999999999999",
        name: "old",
        health: health({ checked_at: "2020-01-01T00:00:00+00:00" }),
      }),
    ]
    mockApi({ servers })
    await renderLoaded(servers)

    expect(screen.getByRole("img", { name: /^github: Healthy/ })).toBeVisible()
    expect(
      screen.getByRole("img", {
        name: /^wiki: Failing\. .*authorization token/,
      }),
    ).toBeVisible()
    expect(
      screen.getByRole("img", {
        name: /^jira: Not checked\. Not checked yet/,
      }),
    ).toBeVisible()
    expect(
      screen.getByRole("img", {
        name: /^old: Needs attention\. No successful contact in the last 7 days/,
      }),
    ).toBeVisible()
    expect(screen.getByText("3 tools")).toBeVisible()
    // The dot replaces the Health column.
    expect(
      screen.queryByRole("columnheader", { name: "Health" }),
    ).not.toBeInTheDocument()
  })

  it("says when a server last answered, for a screen reader as for a pointer", async () => {
    const servers = [
      workspaceMcpServer({
        health: health({
          status: "failing",
          error: "unauthorized",
          http_status: 401,
          tool_count: null,
        }),
      }),
    ]
    mockApi({ servers })
    await renderLoaded(servers)

    expect(
      screen.getByRole("img", { name: /Last contact .*\.$/ }),
    ).toBeInTheDocument()
  })

  it("tests the typed connection without saving, and picks tools from what it lists", async () => {
    const calls = mockApi({
      tested: health({
        tools: ["list_issues", "get_issue", "close_issue"],
        tool_descriptions: { get_issue: "Read one issue." },
      }),
    })
    const user = userEvent.setup()
    await renderLoaded()

    await user.click(screen.getByRole("button", { name: "Add MCP server" }))
    await user.type(
      screen.getByLabelText("URL"),
      "https://mcp.example.com/github",
    )
    await user.click(screen.getByRole("button", { name: "Test connection" }))

    const list = await screen.findByRole("list", { name: "Allowed tools" })
    expect(within(list).getByText("Read one issue.")).toBeVisible()
    const post = writes(calls).find((call) => call.url.endsWith("/check"))
    expect(post?.body).toEqual({
      url: "https://mcp.example.com/github",
      allowed_tools: null,
    })
    expect(
      writes(calls).filter((call) => !call.url.endsWith("/check")),
    ).toEqual([])

    await user.click(within(list).getByRole("checkbox", { name: /get_issue/ }))
    await user.click(
      within(list).getByRole("checkbox", { name: /close_issue/ }),
    )
    expect(screen.getByText("2 of 3 tools selected")).toBeVisible()
    await user.type(screen.getByLabelText("Name"), "github")
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", {
        name: "Add MCP server",
      }),
    )
    const created = writes(calls).find(
      (call) => call.method === "POST" && !call.url.endsWith("/check"),
    )
    expect(created?.body).toMatchObject({
      allowed_tools: ["get_issue", "close_issue"],
      enabled: true,
      attach_by_default: false,
    })
  })

  it("says why a test failed", async () => {
    mockApi({
      tested: health({
        status: "failing",
        error: "not_found",
        http_status: 404,
        tool_count: null,
        tools: [],
      }),
    })
    const user = userEvent.setup()
    await renderLoaded()

    await user.click(screen.getByRole("button", { name: "Add MCP server" }))
    await user.type(
      screen.getByLabelText("URL"),
      "https://mcp.example.com/github",
    )
    await user.click(screen.getByRole("button", { name: "Test connection" }))

    expect(await screen.findByText(/The test failed\./)).toBeVisible()
    expect(screen.getByText(/HTTP 404/)).toBeVisible()
  })

  it("opens the new server's page once it is added, whatever its check found", async () => {
    // The page is where the check's result, the access list and every later
    // change live, so a failing check no longer holds the form open.
    const failing = workspaceMcpServer({
      id: "99999999-9999-9999-9999-999999999999",
      health: health({
        status: "failing",
        error: "not_found",
        http_status: 404,
        tool_count: null,
      }),
    })
    mockApi({ written: failing })
    const user = userEvent.setup()
    await renderLoaded()

    await user.click(screen.getByRole("button", { name: "Add MCP server" }))
    await user.type(screen.getByLabelText("Name"), "github")
    await user.type(
      screen.getByLabelText("URL"),
      "https://mcp.example.com/github",
    )
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", {
        name: "Add MCP server",
      }),
    )

    expect(
      await screen.findByText(`Server page ${failing.id}`),
    ).toBeInTheDocument()
  })

  it("rechecks a server from its row", async () => {
    const servers = [workspaceMcpServer()]
    const calls = mockApi({ servers })
    const user = userEvent.setup()
    await renderLoaded(servers)

    await user.click(screen.getByRole("button", { name: "Check github" }))

    await waitFor(() =>
      expect(
        writes(calls).some(
          (call) =>
            call.method === "POST" &&
            call.url.endsWith(`/mcp-servers/${servers[0].id}/check`),
        ),
      ).toBe(true),
    )
  })

  it("says when a check could not run, in a line that can be dismissed, and says nothing when it ran", async () => {
    const servers = [workspaceMcpServer()]
    mockApi({ servers, writeStatus: 409, writeDetail: "busy" })
    const user = userEvent.setup()
    await renderLoaded(servers)

    // Nothing is announced before a check, and a check that works adds no line either.
    expect(screen.queryByRole("status")).not.toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Check github" }))

    const notice = await screen.findByRole("status")
    await user.click(within(notice).getByRole("button", { name: "Dismiss" }))
    expect(screen.queryByRole("status")).not.toBeInTheDocument()
  })

  it("checks a Personal server through a sign-in in a popup, never with a connection", async () => {
    const servers = [
      workspaceMcpServer({ name: "notion", credential_mode: "per_user" }),
    ]
    const calls = mockApi({ servers })
    const popup = {
      closed: false,
      close: vi.fn(),
      opener: {},
      location: { href: "" },
      document: { title: "", body: { textContent: "" } },
    }
    vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window)
    const user = userEvent.setup()
    await renderLoaded(servers)

    expect(
      screen.getByRole("img", { name: /^notion: Not checked/ }),
    ).toBeVisible()
    await user.click(screen.getByRole("button", { name: "Check notion" }))

    await waitFor(() =>
      expect(popup.location.href).toBe(
        "https://auth.example.com/authorize?state=check-state",
      ),
    )
    expect(popup.opener).toBeNull()
    expect(
      writes(calls).map((call) => call.url.replace(/^.*\/mcp-servers/, "")),
    ).toEqual([`/${servers[0].id}/check/sign-in`])
    // The popup's callback page knows this sign-in as one of its own.
    expect(window.localStorage.getItem("otari.mcp-sign-in.windows")).toContain(
      "check-state",
    )
  })

  it("keeps each row's status to one line, leaving the default policy to the server's page", async () => {
    const servers = [workspaceMcpServer({ name: "github", everyone_tools: [] })]
    mockApi({ servers })
    await renderLoaded(servers)

    expect(screen.getByText("ON REQUEST")).toBeVisible()
    expect(screen.queryByText(/^DEFAULT:/)).not.toBeInTheDocument()
  })

  it("registers a server nobody can use yet when asked to", async () => {
    const calls = mockApi()
    const user = userEvent.setup()
    await renderLoaded()

    await user.click(screen.getByRole("button", { name: "Add MCP server" }))
    expect(screen.getByRole("radio", { name: "Allow" })).toBeChecked()
    await user.type(screen.getByLabelText("Name"), "payroll")
    await user.type(
      screen.getByLabelText("URL"),
      "https://mcp.example.com/payroll",
    )
    await user.click(screen.getByRole("radio", { name: "Deny" }))
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", {
        name: "Add MCP server",
      }),
    )

    const post = writes(calls).find((call) => call.method === "POST")
    expect(post?.body).toMatchObject({ everyone_tools: [] })
  })

  it("opens a server's own page from its row", async () => {
    const servers = [workspaceMcpServer()]
    mockApi({ servers })
    const user = userEvent.setup()
    await renderLoaded(servers)

    await user.click(screen.getByRole("button", { name: "Manage github" }))

    expect(
      await screen.findByText(`Server page ${servers[0].id}`),
    ).toBeInTheDocument()
  })

  it("switches whether requests may name their own servers", async () => {
    const calls = mockApi({ servers: [workspaceMcpServer()] })
    const user = userEvent.setup()
    await renderLoaded([workspaceMcpServer()])

    const toggle = await screen.findByRole("switch", {
      name: "Allow servers outside this list",
    })
    expect(toggle).toBeChecked()
    // One line: the label is on the page, the explanation behind the icon.
    expect(screen.getByText("Allow servers outside this list")).toBeVisible()
    expect(screen.queryByText(/^On: an API request/)).not.toBeInTheDocument()
    await user.click(
      screen.getByRole("button", { name: "Explain this switch" }),
    )
    expect(
      await screen.findByText(
        /^On: an API request may carry the URL and token of any MCP server/,
      ),
    ).toBeVisible()
    // Not a dashboard setting: what members connect is unchanged.
    expect(
      screen.getByText(
        /^Off: requests can only use the servers in this list\. Members connect/,
      ),
    ).toBeVisible()
    await user.click(toggle)

    await waitFor(() =>
      expect(
        calls.find(
          (call) =>
            call.method === "PATCH" &&
            call.url.endsWith(`/workspaces/${ALPHA}`),
        )?.body,
      ).toEqual({ request_mcp_servers_allowed: false }),
    )
    // The written row replaces the one read, so the switch shows what was saved.
    await waitFor(() => expect(toggle).not.toBeChecked())
  })

  it("reads the selected workspace's own row, not the whole list", async () => {
    const calls = mockApi({ servers: [workspaceMcpServer()] })
    await renderLoaded([workspaceMcpServer()])

    expect(
      await screen.findByRole("switch", {
        name: "Allow servers outside this list",
      }),
    ).toBeInTheDocument()
    expect(
      calls.some(
        (call) =>
          call.method === "GET" && call.url.endsWith(`/workspaces/${ALPHA}`),
      ),
    ).toBe(true)
    expect(calls.some((call) => call.url.includes("/workspaces?"))).toBe(false)
  })

  it("says why the workspace's request switch could not be read", async () => {
    mockApi({ servers: [workspaceMcpServer()], workspaceStatus: 500 })
    await renderLoaded([workspaceMcpServer()])

    expect(await screen.findByText("Workspace unavailable")).toBeVisible()
    expect(
      screen.queryByRole("switch", {
        name: "Allow servers outside this list",
      }),
    ).not.toBeInTheDocument()
  })

  it("offers the request switch only to those who manage the workspace", async () => {
    mockApi({
      servers: [workspaceMcpServer()],
      role: "member",
      memberships: [{ workspace_id: ALPHA, name: "Alpha", role: "member" }],
    })
    await renderLoaded([workspaceMcpServer()])

    expect(
      screen.queryByRole("switch", {
        name: "Allow servers outside this list",
      }),
    ).not.toBeInTheDocument()
  })

  it("offers Browse catalog beside Add to a manager, on a deployment that serves the catalog", async () => {
    mockApi()
    await renderLoaded()

    expect(
      screen.getByRole("button", { name: "Browse catalog" }),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Add MCP server" }),
    ).toBeInTheDocument()
  })

  it("leaves Browse catalog out where the deployment switched the catalog off", async () => {
    mockApi()
    renderCard("card", ["tools", "workspaces"])
    await screen.findByText("No MCP servers yet.")

    expect(
      screen.queryByRole("button", { name: "Browse catalog" }),
    ).not.toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Add MCP server" }),
    ).toBeInTheDocument()
  })

  it("does not offer Browse catalog to a member who cannot manage the workspace", async () => {
    mockApi({
      servers: [workspaceMcpServer()],
      role: "member",
      memberships: [{ workspace_id: ALPHA, name: "Alpha", role: "member" }],
    })
    await renderLoaded([workspaceMcpServer()])

    expect(
      screen.queryByRole("button", { name: "Browse catalog" }),
    ).not.toBeInTheDocument()
  })

  it("shows a manager what members asked for, and opens the confirm step for one from Add", async () => {
    const sentry = mcpCatalogServer()
    mockApi({
      requests: [
        {
          catalog_name: sentry.name,
          title: sentry.title,
          url: sentry.url,
          catalog_server: sentry,
          requesters: [
            {
              request_id: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
              user_id: "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
              full_name: "Ada Member",
              email: "ada@example.com",
              note: "Triage production errors",
              requested_at: "2026-11-10T09:00:00+00:00",
            },
          ],
          first_requested_at: "2026-11-10T09:00:00+00:00",
        },
      ],
    })
    const user = userEvent.setup()
    await renderLoaded()

    const row = await screen.findByRole("region", {
      name: "Requests for Sentry",
    })
    expect(within(row).getByText(/^Asked by Ada Member on /)).toBeVisible()
    expect(within(row).getByText("Triage production errors")).toBeVisible()
    await user.click(within(row).getByRole("button", { name: "Add Sentry" }))

    // Straight to the catalog's confirm step for that server: Shared or
    // Personal, and Availability, with no browse in between.
    const dialog = await screen.findByRole("dialog", {
      name: "Add Sentry to Alpha",
    })
    expect(
      within(dialog).getByRole("radiogroup", { name: "Sign-in mode" }),
    ).toBeVisible()
    expect(
      within(dialog).getByRole("radiogroup", { name: "Availability" }),
    ).toBeVisible()
    expect(
      within(dialog).getByRole("button", { name: "Confirm" }),
    ).toBeVisible()
    expect(
      within(dialog).queryByRole("button", { name: "Back" }),
    ).not.toBeInTheDocument()
  })

  it("shows no Requests block while nobody is waiting, and asks for none without the catalog", async () => {
    const calls = mockApi()
    renderCard("card", ["tools", "workspaces"])
    await screen.findByText("No MCP servers yet.")

    expect(
      screen.queryByRole("heading", { name: /Requests/ }),
    ).not.toBeInTheDocument()
    expect(
      calls.some((call) => call.url.includes("/mcp-server-requests")),
    ).toBe(false)
  })

  it("offers a member with no servers a way to ask an admin for one", async () => {
    mockApi({
      role: "member",
      memberships: [{ workspace_id: ALPHA, name: "Alpha", role: "member" }],
    })
    const user = userEvent.setup()
    renderCard()

    expect(
      await screen.findByText(/Ask an admin to add one from the catalog/),
    ).toBeVisible()
    await user.click(screen.getByRole("button", { name: "Request a server" }))

    expect(
      await screen.findByRole("dialog", { name: "Request a server" }),
    ).toBeVisible()
  })

  it("tells a member the answer to their request above their servers, and connects from it", async () => {
    const notion = workspaceMcpServer({
      id: "66666666-6666-6666-6666-666666666666",
      name: "notion",
      credential_mode: "per_user",
    })
    const calls = mockApi({
      memberships: [{ workspace_id: ALPHA, name: "Alpha", role: "member" }],
      servers: [notion],
      role: "member",
      mine: [
        {
          id: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
          workspace_id: ALPHA,
          catalog_name: "so.notion/mcp",
          title: "Notion",
          url: notion.url,
          note: null,
          status: "approved",
          decision_reason: null,
          decided_at: "2026-11-11T09:00:00+00:00",
          server_id: notion.id,
          catalog_server: null,
          created_at: "2026-11-10T09:00:00+00:00",
        },
      ],
    })
    const user = userEvent.setup()
    renderCard()

    // Above the empty page: the member has connected nothing yet.
    const answer = await screen.findByText(
      "notion was added. Connect your account to use it.",
    )
    const empty = screen.getByRole("heading", { name: "No servers yet" })
    expect(
      answer.compareDocumentPosition(empty) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy()
    await user.click(screen.getByRole("button", { name: "Connect notion" }))

    await waitFor(() =>
      expect(
        calls.some(
          (call) =>
            call.method === "POST" && call.url.includes("/mcp-connections"),
        ),
      ).toBe(true),
    )
  })

  it("leads a member from Connect a server to the catalog", async () => {
    const servers = [
      workspaceMcpServer({ name: "github" }),
      workspaceMcpServer({
        id: "66666666-6666-6666-6666-666666666666",
        name: "notion",
        credential_mode: "per_user",
      }),
    ]
    mockApi({
      memberships: [{ workspace_id: ALPHA, name: "Alpha", role: "member" }],
      servers,
      role: "member",
    })
    const user = userEvent.setup()
    renderCard()

    const open = await screen.findByRole("button", { name: "Connect a server" })
    await waitFor(() => expect(open).toBeEnabled())
    await user.click(open)
    const connect = await screen.findByRole("dialog", {
      name: "Connect a server",
    })
    await user.click(
      within(connect).getByRole("button", { name: "Request a server" }),
    )

    // One dialog at a time: the catalog takes the place of the connect grid.
    expect(
      await screen.findByRole("dialog", { name: "Request a server" }),
    ).toBeVisible()
    await waitFor(() =>
      expect(
        screen.queryByRole("dialog", { name: "Connect a server" }),
      ).not.toBeInTheDocument(),
    )
  })

  it("offers a member no request where the deployment switched the catalog off", async () => {
    mockApi({
      role: "member",
      memberships: [{ workspace_id: ALPHA, name: "Alpha", role: "member" }],
    })
    renderCard("card", ["tools", "workspaces"])

    expect(
      await screen.findByText("This workspace has no servers for you yet."),
    ).toBeVisible()
    expect(
      screen.queryByRole("button", { name: "Request a server" }),
    ).not.toBeInTheDocument()
  })
})
