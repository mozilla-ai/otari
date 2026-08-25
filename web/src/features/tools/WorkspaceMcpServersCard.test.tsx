import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { WorkspaceMcpServer } from "@/client"
import { WorkspaceMcpServersCard } from "@/features/tools/WorkspaceMcpServersCard"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { organizationContext, workspaceMcpServer } from "@/tests/fixtures"

const ALPHA = "11111111-1111-1111-1111-111111111111"

interface Call {
  url: string
  method: string
  body: Record<string, unknown> | undefined
}

function mockApi({
  memberships = [{ workspace_id: ALPHA, name: "Alpha", role: "admin" }],
  servers = [] as WorkspaceMcpServer[],
  writeStatus,
  writeDetail,
}: {
  memberships?: { workspace_id: string; name: string; role: string }[]
  servers?: WorkspaceMcpServer[]
  writeStatus?: number
  writeDetail?: string
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
      return Response.json(servers[0] ?? workspaceMcpServer())
    }
    return Response.json(
      organizationContext({ workspace_memberships: memberships }),
    )
  })
  return calls
}

function renderCard() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <SelectedWorkspaceProvider>
        <WorkspaceMcpServersCard />
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
    expect(screen.getByText("Stored")).toBeVisible()
    expect(screen.getByText("2 allowed")).toBeVisible()
    expect(screen.getByText("All")).toBeVisible()
    expect(screen.getByText("Enabled")).toBeVisible()
    expect(screen.getByText("Disabled")).toBeVisible()
  })

  it("reads an empty allow-list as every tool, the way the gateway does", async () => {
    // `mcp_client` takes a falsy `allowed_tools` as no allow-list at all, so a
    // row holding `[]` exposes every tool exactly as null does. "0 allowed"
    // would tell an operator the opposite. This form never sends `[]`, but the
    // API accepts one, so a row can arrive here holding it.
    const servers = [workspaceMcpServer({ allowed_tools: [] })]
    mockApi({ servers })
    await renderLoaded(servers)

    expect(screen.getByText("All")).toBeVisible()
    expect(screen.queryByText("0 allowed")).not.toBeInTheDocument()
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
    await user.click(screen.getByRole("button", { name: "Add server" }))

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
    await user.click(screen.getByRole("button", { name: "Add server" }))

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
    await user.click(screen.getByRole("button", { name: "Add server" }))
    expect(await screen.findByText("already taken")).toBeVisible()

    await user.click(screen.getByRole("button", { name: "Cancel" }))
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
    await user.click(screen.getByRole("button", { name: "Add server" }))
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

  it("does not list the servers at all for a member who cannot manage the workspace", async () => {
    // The read takes the management role server-side, so asking would earn a
    // 403 over a table that could never fill.
    const requests: string[] = []
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input)
      if (url.includes("/mcp-servers")) {
        requests.push(url)
        return new Response("forbidden", { status: 403 })
      }
      return Response.json(
        organizationContext({
          role: "member",
          workspace_memberships: [
            { workspace_id: ALPHA, name: "Alpha", role: "member" },
          ],
        }),
      )
    })
    renderCard()

    expect(
      await screen.findByText(/managed by an owner or admin/i),
    ).toBeVisible()
    expect(requests).toHaveLength(0)
    expect(
      screen.queryByRole("button", { name: "Add server" }),
    ).not.toBeInTheDocument()
  })
})
