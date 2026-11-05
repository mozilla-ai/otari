import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import {
  workspaceMcpServerHealth as health,
  workspaceMcpServer,
} from "@/tests/fixtures"
import {
  type McpCall,
  mcpWrites,
  mockMcpServerApi,
  renderMcpServerPage,
  WORKSPACE,
} from "@/tests/mcpServer"

const SERVER = workspaceMcpServer({ workspace_id: WORKSPACE, has_token: true })
const path = `/workspaces/${WORKSPACE}/mcp-servers/${SERVER.id}`

async function openPage() {
  renderMcpServerPage(SERVER.id)
  await screen.findByRole("heading", { level: 1, name: "github" })
}

async function editConnection(user: ReturnType<typeof userEvent.setup>) {
  await openPage()
  await user.click(screen.getByRole("button", { name: "Edit connection" }))
}

async function patchBody(calls: McpCall[]) {
  let body: Record<string, unknown> | undefined
  await waitFor(() => {
    body = mcpWrites(calls).find(
      (call) => call.method === "PATCH" && call.url.endsWith(path),
    )?.body
    expect(body).toBeDefined()
  })
  return body
}

describe("McpServerDetailPage, exposed tools and connection", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("leaves a stored token alone when the form never touches it", async () => {
    const calls = mockMcpServerApi({ server: SERVER })
    const user = userEvent.setup()
    await editConnection(user)

    await user.clear(screen.getByLabelText("Name"))
    await user.type(screen.getByLabelText("Name"), "github-prod")
    await user.click(screen.getByRole("button", { name: "Save server" }))

    const body = await patchBody(calls)
    expect(body).toMatchObject({ name: "github-prod" })
    // A form that sent an empty box back as null or "" would destroy a
    // credential it was never shown.
    expect(body).not.toHaveProperty("authorization_token")
  })

  it("clears a stored token only when asked to", async () => {
    const calls = mockMcpServerApi({ server: SERVER })
    const user = userEvent.setup()
    await editConnection(user)

    await user.click(
      screen.getByRole("checkbox", { name: "Remove the stored token" }),
    )
    await user.click(screen.getByRole("button", { name: "Save server" }))

    expect(await patchBody(calls)).toMatchObject({ authorization_token: "" })
  })

  it("rotates the token when a replacement is typed, taking the tick off Remove", async () => {
    const calls = mockMcpServerApi({ server: SERVER })
    const user = userEvent.setup()
    await editConnection(user)

    const remove = screen.getByRole("checkbox", {
      name: "Remove the stored token",
    })
    await user.click(remove)
    await user.type(screen.getByLabelText("Authorization token"), "ghp_new")
    expect(remove).not.toBeChecked()
    await user.click(screen.getByRole("button", { name: "Save server" }))

    expect(await patchBody(calls)).toMatchObject({
      authorization_token: "ghp_new",
    })
  })

  it("drops a typed replacement when Remove is ticked after it", async () => {
    const calls = mockMcpServerApi({ server: SERVER })
    const user = userEvent.setup()
    await editConnection(user)

    await user.type(screen.getByLabelText("Authorization token"), "ghp_new")
    await user.click(
      screen.getByRole("checkbox", { name: "Remove the stored token" }),
    )
    expect(screen.getByLabelText("Authorization token")).toHaveValue("")
    await user.click(screen.getByRole("button", { name: "Save server" }))

    expect(await patchBody(calls)).toMatchObject({ authorization_token: "" })
  })

  it("says so when a save's check could not run", async () => {
    // The gateway had no free check slot, so the row comes back with no health.
    mockMcpServerApi({
      server: workspaceMcpServer({ workspace_id: WORKSPACE }),
    })
    const user = userEvent.setup()
    await editConnection(user)

    await user.clear(screen.getByLabelText("URL"))
    await user.type(screen.getByLabelText("URL"), "https://mcp.example.com/v2")
    await user.click(screen.getByRole("button", { name: "Save server" }))

    expect(await screen.findByText(/the gateway was busy/)).toBeVisible()
  })

  it("refuses an http URL while a token is stored", async () => {
    const calls = mockMcpServerApi({ server: SERVER })
    const user = userEvent.setup()
    await editConnection(user)

    await user.clear(screen.getByLabelText("URL"))
    await user.type(screen.getByLabelText("URL"), "http://mcp.example.com")

    await waitFor(() =>
      expect(screen.getByLabelText("URL")).toHaveAccessibleDescription(
        /needs an https URL/,
      ),
    )
    expect(screen.getByRole("button", { name: "Save server" })).toBeDisabled()
    expect(mcpWrites(calls)).toHaveLength(0)
  })

  it("disables the server from its heading", async () => {
    const calls = mockMcpServerApi({ server: SERVER })
    const user = userEvent.setup()
    await openPage()

    await user.click(screen.getByRole("button", { name: "Disable" }))

    expect(await patchBody(calls)).toEqual({ enabled: false })
  })

  it("deletes the server and goes back to the list", async () => {
    const calls = mockMcpServerApi({ server: SERVER })
    const user = userEvent.setup()
    await openPage()

    await user.click(screen.getByRole("button", { name: "Delete github" }))
    const confirm = await screen.findByRole("alertdialog")
    await user.click(
      within(confirm).getByRole("button", { name: "Delete server" }),
    )

    await waitFor(() =>
      expect(
        mcpWrites(calls).some(
          (call) => call.method === "DELETE" && call.url.endsWith(path),
        ),
      ).toBe(true),
    )
    expect(await screen.findByText("Server list")).toBeInTheDocument()
  })

  it("narrows the tools the server exposes to the ones ticked", async () => {
    const calls = mockMcpServerApi({
      server: workspaceMcpServer({
        workspace_id: WORKSPACE,
        health: health({
          tool_count: 3,
          tools: ["list_issues", "get_issue", "close_issue"],
        }),
      }),
    })
    const user = userEvent.setup()
    await openPage()

    expect(screen.getByText(/The server offers 3 tools/)).toBeVisible()
    await user.click(screen.getByRole("button", { name: "Edit exposed tools" }))
    await user.click(
      screen.getByRole("radio", { name: "Only the tools ticked" }),
    )
    await user.click(screen.getByRole("checkbox", { name: "get_issue" }))
    await user.click(screen.getByRole("button", { name: "Save exposed tools" }))

    expect(await patchBody(calls)).toEqual({ allowed_tools: ["get_issue"] })
  })

  it("labels read tools, naming the server's own hints without ticking them", async () => {
    const calls = mockMcpServerApi({
      server: workspaceMcpServer({
        workspace_id: WORKSPACE,
        health: health({
          tools: ["list_issues", "get_issue", "delete_issue"],
          read_only_hinted_tools: ["list_issues", "delete_issue"],
        }),
      }),
    })
    const user = userEvent.setup()
    await openPage()

    expect(screen.getByText(/None labeled/)).toBeVisible()
    await user.click(screen.getByRole("button", { name: "Edit read tools" }))
    // The server calls delete_issue read-only; the manager is who decides, one tick at a time.
    expect(
      screen.getByText(
        /The server marks list_issues, delete_issue as read-only/,
      ),
    ).toBeVisible()
    const list = screen.getByRole("list", { name: "Read tools" })
    for (const name of ["list_issues", "get_issue", "delete_issue"]) {
      expect(within(list).getByRole("checkbox", { name })).not.toBeChecked()
    }
    expect(
      screen.getByRole("button", { name: "Save read tools" }),
    ).toBeDisabled()
    await user.click(
      within(list).getByRole("checkbox", { name: "list_issues" }),
    )
    await user.click(within(list).getByRole("checkbox", { name: "get_issue" }))
    await user.click(screen.getByRole("button", { name: "Save read tools" }))

    expect(await patchBody(calls)).toEqual({
      read_tools: ["list_issues", "get_issue"],
    })
  })

  it("checks the connection beside the exposed tools", async () => {
    const calls = mockMcpServerApi({ server: SERVER })
    const user = userEvent.setup()
    await openPage()

    await user.click(screen.getByRole("button", { name: "Check connection" }))

    await waitFor(() =>
      expect(
        mcpWrites(calls).some(
          (call) =>
            call.method === "POST" && call.url.endsWith(`${path}/check`),
        ),
      ).toBe(true),
    )
  })
})
