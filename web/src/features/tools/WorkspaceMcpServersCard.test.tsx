import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type {
  OrganizationContext,
  WorkspaceMcpServer,
  WorkspaceMcpServerHealth,
} from "@/client"
import { WorkspaceMcpServersCard } from "@/features/tools/WorkspaceMcpServersCard"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import {
  workspaceMcpServerHealth as health,
  organizationContext,
  workspaceMcpServer,
} from "@/tests/fixtures"

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
} = {}): Call[] {
  const calls: Call[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
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
    return Response.json(
      organizationContext({ role, workspace_memberships: memberships }),
    )
  })
  return calls
}

function renderCard(variant?: "card" | "page") {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <SelectedWorkspaceProvider>
        <WorkspaceMcpServersCard variant={variant} />
      </SelectedWorkspaceProvider>
    </QueryClientProvider>,
  )
}

// The Add button is rendered before the list resolves, so a test that clicks it
// immediately would submit against a table that had not loaded. The rendered
// list is the signal that the workspace and its servers are both in hand.
async function renderLoaded(rows: WorkspaceMcpServer[] = []) {
  renderCard()
  if (rows.length === 0) {
    await screen.findByText(/No MCP server registered/i)
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
    expect(
      screen.getByText(/checked for SSRF safety and tested when saved/),
    ).toBeInTheDocument()
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
    // The token is never returned, so "Stored" is the whole of what a row can
    // say about it.
    expect(screen.getByText("STORED")).toBeVisible()
    expect(screen.getByText("2 allowed")).toBeVisible()
    expect(screen.getByText("All")).toBeVisible()
    expect(screen.getByText("ENABLED")).toBeVisible()
    expect(screen.getByText("DISABLED")).toBeVisible()
  })

  it("shows each server's id truncated, with a copy of the whole id", async () => {
    const servers = [
      workspaceMcpServer({
        id: "66666666-6666-6666-6666-666666666666",
        name: "wiki",
      }),
    ]
    mockApi({ servers })
    const user = userEvent.setup()
    await renderLoaded(servers)

    // The id is what a request names in `mcp_server_ids`, so the full value
    // stays on the tooltip and the copy even though the cell shows a prefix.
    const id = screen.getByText("66666666…")
    expect(id).toBeVisible()
    expect(id).toHaveAttribute("title", "66666666-6666-6666-6666-666666666666")
    await user.click(screen.getByRole("button", { name: /Copy id for wiki/ }))
    expect(await navigator.clipboard.readText()).toBe(
      "66666666-6666-6666-6666-666666666666",
    )
  })

  it("reads an empty allow-list as admitting no tool, the way the gateway does", async () => {
    // The tool loop treats `[]` as a deny-all, distinct from null. "All" would
    // tell an operator the opposite. This form never sends `[]`, but the API
    // accepts one, so a row can arrive here holding it.
    const servers = [workspaceMcpServer({ allowed_tools: [] })]
    mockApi({ servers })
    await renderLoaded(servers)

    expect(screen.getByText("0 allowed")).toBeVisible()
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
    })
    // No `authorization_token` key at all rather than an explicit null: the
    // operator typed no token, so the request says nothing about one.
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

  it("leaves a stored token alone when the edit form never touches it", async () => {
    const servers = [workspaceMcpServer({ has_token: true })]
    const calls = mockApi({ servers })
    const user = userEvent.setup()
    await renderLoaded(servers)

    await user.click(screen.getByRole("button", { name: "Edit" }))
    await user.clear(screen.getByLabelText("Name"))
    await user.type(screen.getByLabelText("Name"), "github-prod")
    await user.click(screen.getByRole("button", { name: "Save server" }))

    const patch = writes(calls).find((call) => call.method === "PATCH")
    expect(patch?.body).toMatchObject({ name: "github-prod" })
    // The whole point of the three-state token: a form that serialized an empty
    // box back as null or "" would destroy a credential it was never shown.
    expect(patch?.body).not.toHaveProperty("authorization_token")
  })

  it("clears a stored token only when the operator asks for it", async () => {
    const servers = [workspaceMcpServer({ has_token: true })]
    const calls = mockApi({ servers })
    const user = userEvent.setup()
    await renderLoaded(servers)

    await user.click(screen.getByRole("button", { name: "Edit" }))
    await user.click(
      screen.getByRole("checkbox", { name: "Remove the stored token" }),
    )
    await user.click(screen.getByRole("button", { name: "Save server" }))

    const patch = writes(calls).find((call) => call.method === "PATCH")
    expect(patch?.body).toMatchObject({ authorization_token: "" })
  })

  it("rotates the token when the operator types a replacement", async () => {
    const servers = [workspaceMcpServer({ has_token: true })]
    const calls = mockApi({ servers })
    const user = userEvent.setup()
    await renderLoaded(servers)

    await user.click(screen.getByRole("button", { name: "Edit" }))
    await user.type(screen.getByLabelText("Authorization token"), "ghp_rotated")
    await user.click(screen.getByRole("button", { name: "Save server" }))

    // The third of the three states, and the only one that writes a credential.
    const patch = writes(calls).find((call) => call.method === "PATCH")
    expect(patch?.body).toMatchObject({ authorization_token: "ghp_rotated" })
  })

  it("takes the tick off Remove when a replacement token is typed", async () => {
    const servers = [workspaceMcpServer({ has_token: true })]
    const calls = mockApi({ servers })
    const user = userEvent.setup()
    await renderLoaded(servers)

    await user.click(screen.getByRole("button", { name: "Edit" }))
    const remove = screen.getByRole("checkbox", {
      name: "Remove the stored token",
    })
    await user.click(remove)
    expect(remove).toBeChecked()

    // Clearing and rotating say opposite things, so the later instruction wins
    // rather than both being held and one silently losing at submit.
    await user.type(screen.getByLabelText("Authorization token"), "ghp_new")
    expect(remove).not.toBeChecked()

    await user.click(screen.getByRole("button", { name: "Save server" }))
    const patch = writes(calls).find((call) => call.method === "PATCH")
    expect(patch?.body).toMatchObject({ authorization_token: "ghp_new" })
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
      within(dialog).getByText(/the token stored with it are removed/),
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

  it("lists the servers read-only for a member who cannot manage the workspace", async () => {
    // The list is a member read now (otari-ai#1942): the rows shape every
    // request the member sends through the workspace. Only the affordances
    // that change them take the management role.
    const servers = [
      workspaceMcpServer({
        name: "github",
        url: "https://mcp.example.com/github",
        has_token: true,
      }),
    ]
    const calls = mockApi({
      memberships: [{ workspace_id: ALPHA, name: "Alpha", role: "member" }],
      servers,
      role: "member",
    })
    const user = userEvent.setup()
    await renderLoaded(servers)

    // The URL as the server chose to render it for this caller: the service
    // masks any credential in it for a reader who cannot manage the workspace,
    // so the card renders what it is given and adds no masking of its own.
    expect(screen.getByText("https://mcp.example.com/github")).toBeVisible()
    // The token stays write-only whoever is reading. Uppercase in the source
    // rather than through a text-transform, so the accessible name is what is
    // asserted here and in the operator case above.
    expect(screen.getByText("STORED")).toBeVisible()
    expect(
      screen.getByText(/Only an owner or admin of the workspace/),
    ).toBeVisible()
    expect(
      screen.queryByRole("button", { name: "Add MCP server" }),
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Edit" }),
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Delete" }),
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Recheck" }),
    ).not.toBeInTheDocument()
    // Nothing on the page can write, so nothing did.
    await user.click(screen.getByText("github"))
    expect(writes(calls)).toHaveLength(0)
  })

  it("shows what each server's last connection check found", async () => {
    const servers = [
      workspaceMcpServer({
        name: "github",
        health: health({
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
    ]
    mockApi({ servers })
    await renderLoaded(servers)

    expect(screen.getByText("CONNECTED")).toBeVisible()
    expect(screen.getByText("3 tools")).toBeVisible()
    expect(screen.getByText("TOKEN REJECTED")).toBeVisible()
    expect(screen.getByText("Not checked")).toBeVisible()
  })

  it("opens what to fix from the health label, not only on hover", async () => {
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
    const user = userEvent.setup()
    await renderLoaded(servers)

    await user.click(screen.getByRole("button", { name: "TOKEN REJECTED" }))

    const detail = await screen.findByRole("dialog", {
      name: "Last check of github",
    })
    expect(
      within(detail).getByText(/Check the authorization token/),
    ).toBeVisible()
    expect(within(detail).getByText(/^Checked /)).toBeVisible()
    expect(within(detail).getByText("list_issues, get_issue")).toBeVisible()
  })

  it("tests the typed connection without saving, and picks tools from what it lists", async () => {
    const calls = mockApi({
      tested: health({ tools: ["list_issues", "get_issue", "close_issue"] }),
    })
    const user = userEvent.setup()
    await renderLoaded()

    await user.click(screen.getByRole("button", { name: "Add MCP server" }))
    await user.type(
      screen.getByLabelText("URL"),
      "https://mcp.example.com/github",
    )
    await user.click(screen.getByRole("button", { name: "Test connection" }))

    expect(await screen.findByText(/3 tools available/)).toBeVisible()
    const post = writes(calls).find((call) => call.url.endsWith("/check"))
    expect(post?.body).toEqual({
      url: "https://mcp.example.com/github",
      allowed_tools: null,
    })
    expect(
      writes(calls).filter((call) => !call.url.endsWith("/check")),
    ).toEqual([])

    await user.click(screen.getByRole("button", { name: "get_issue" }))
    await user.click(screen.getByRole("button", { name: "close_issue" }))
    expect(screen.getByLabelText("Allowed tools")).toHaveValue(
      "get_issue, close_issue",
    )
    expect(screen.getByRole("button", { name: "get_issue" })).toHaveAttribute(
      "aria-pressed",
      "true",
    )
    await user.click(screen.getByRole("button", { name: "get_issue" }))
    expect(screen.getByLabelText("Allowed tools")).toHaveValue("close_issue")
    // Taking the last name off would leave the list blank, which allows every
    // tool, so the picker refuses; clearing the field is the deliberate way.
    expect(screen.getByRole("button", { name: "close_issue" })).toBeDisabled()
    await user.click(screen.getByRole("button", { name: "close_issue" }))
    expect(screen.getByLabelText("Allowed tools")).toHaveValue("close_issue")
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

  it("hides the stored catalog once a test of the same connection fails", async () => {
    const servers = [workspaceMcpServer({ health: health() })]
    mockApi({
      servers,
      tested: health({
        status: "failing",
        error: "timeout",
        tool_count: null,
        tools: [],
      }),
    })
    const user = userEvent.setup()
    await renderLoaded(servers)

    await user.click(screen.getByRole("button", { name: "Edit" }))
    expect(screen.getByText(/2 tools available/)).toBeVisible()
    await user.click(screen.getByRole("button", { name: "Test connection" }))

    expect(await screen.findByText(/The test failed\./)).toBeVisible()
    expect(screen.queryByText(/tools available/)).not.toBeInTheDocument()
  })

  it("tests an edit with the stored token unless a new one is typed", async () => {
    const servers = [workspaceMcpServer({ has_token: true })]
    const calls = mockApi({ servers })
    const user = userEvent.setup()
    await renderLoaded(servers)

    await user.click(screen.getByRole("button", { name: "Edit" }))
    await user.click(screen.getByRole("button", { name: "Test connection" }))

    await waitFor(() =>
      expect(
        writes(calls).find((call) => call.url.endsWith("/check"))?.body,
      ).toEqual({
        url: servers[0].url,
        allowed_tools: null,
        server_id: servers[0].id,
      }),
    )
  })

  it("keeps the add form open on the stored server when its check fails", async () => {
    // Save and warn: the server is stored either way, so the form turns into
    // an edit of it rather than inviting a second add of the same name.
    const failing = workspaceMcpServer({
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

    const dialog = await screen.findByRole("dialog", {
      name: "Edit MCP server",
    })
    expect(
      within(dialog).getByText(/Saved, but the connection failed/),
    ).toBeVisible()
    expect(within(dialog).getByText(/HTTP 404/)).toBeVisible()
    expect(
      within(dialog).getByRole("button", { name: "Save server" }),
    ).toBeVisible()
  })

  it("closes the add form when the check finds nothing to fix", async () => {
    mockApi({ written: workspaceMcpServer({ health: health() }) })
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

    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull())
  })

  it("says so when an edit's check could not run, even on a server never checked", async () => {
    // The gateway had no free check slot, so the row comes back with no health
    // at all: the same as before the edit, which is why "did a check run" is
    // read off what the edit changed rather than off the health it returned.
    const servers = [workspaceMcpServer()]
    mockApi({ servers })
    const user = userEvent.setup()
    await renderLoaded(servers)

    await user.click(screen.getByRole("button", { name: "Edit" }))
    await user.clear(screen.getByLabelText("URL"))
    await user.type(screen.getByLabelText("URL"), "https://mcp.example.com/v2")
    await user.click(screen.getByRole("button", { name: "Save server" }))

    expect(await screen.findByText(/the gateway was busy/)).toBeVisible()
    expect(screen.getByRole("dialog")).toBeVisible()
  })

  it("closes on an edit that leaves the connection alone", async () => {
    const servers = [workspaceMcpServer()]
    mockApi({ servers })
    const user = userEvent.setup()
    await renderLoaded(servers)

    await user.click(screen.getByRole("button", { name: "Edit" }))
    await user.clear(screen.getByLabelText("Name"))
    await user.type(screen.getByLabelText("Name"), "github-prod")
    await user.click(screen.getByRole("button", { name: "Save server" }))

    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull())
  })

  it("rechecks a server from its row", async () => {
    const servers = [workspaceMcpServer()]
    const calls = mockApi({ servers })
    const user = userEvent.setup()
    await renderLoaded(servers)

    await user.click(screen.getByRole("button", { name: "Recheck" }))

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
})
