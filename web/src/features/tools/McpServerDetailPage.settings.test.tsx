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
import { pickOption } from "@/tests/select"

const SERVER = workspaceMcpServer({ workspace_id: WORKSPACE, has_token: true })
const path = `/workspaces/${WORKSPACE}/mcp-servers/${SERVER.id}`

async function openPage() {
  renderMcpServerPage(SERVER.id)
  await screen.findByRole("heading", { level: 1, name: "github" })
}

async function openSettings(user: ReturnType<typeof userEvent.setup>) {
  await openPage()
  await user.click(screen.getByRole("button", { name: "Settings" }))
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

describe("McpServerDetailPage, exposed tools and settings", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("leaves a stored token alone when the form never touches it", async () => {
    const calls = mockMcpServerApi({ server: SERVER })
    const user = userEvent.setup()
    await openSettings(user)

    await user.clear(screen.getByLabelText("Name"))
    await user.type(screen.getByLabelText("Name"), "github-prod")
    await user.click(screen.getByRole("button", { name: "Save" }))

    const body = await patchBody(calls)
    expect(body).toMatchObject({ name: "github-prod" })
    // A form that sent an empty box back as null or "" would destroy a
    // credential it was never shown.
    expect(body).not.toHaveProperty("authorization_token")
  })

  it("clears a stored token only when asked to", async () => {
    const calls = mockMcpServerApi({ server: SERVER })
    const user = userEvent.setup()
    await openSettings(user)

    await user.click(
      screen.getByRole("checkbox", { name: "Remove the stored token" }),
    )
    await user.click(screen.getByRole("button", { name: "Save" }))

    expect(await patchBody(calls)).toMatchObject({ authorization_token: "" })
  })

  it("rotates the token when a replacement is typed, taking the tick off Remove", async () => {
    const calls = mockMcpServerApi({ server: SERVER })
    const user = userEvent.setup()
    await openSettings(user)

    const remove = screen.getByRole("checkbox", {
      name: "Remove the stored token",
    })
    await user.click(remove)
    await user.type(screen.getByLabelText("Authorization token"), "ghp_new")
    expect(remove).not.toBeChecked()
    await user.click(screen.getByRole("button", { name: "Save" }))

    expect(await patchBody(calls)).toMatchObject({
      authorization_token: "ghp_new",
    })
  })

  it("drops a typed replacement when Remove is ticked after it", async () => {
    const calls = mockMcpServerApi({ server: SERVER })
    const user = userEvent.setup()
    await openSettings(user)

    await user.type(screen.getByLabelText("Authorization token"), "ghp_new")
    await user.click(
      screen.getByRole("checkbox", { name: "Remove the stored token" }),
    )
    expect(screen.getByLabelText("Authorization token")).toHaveValue("")
    await user.click(screen.getByRole("button", { name: "Save" }))

    expect(await patchBody(calls)).toMatchObject({ authorization_token: "" })
  })

  it("says so when a save's check could not run", async () => {
    // The gateway had no free check slot, so the row comes back with no health.
    mockMcpServerApi({
      server: workspaceMcpServer({ workspace_id: WORKSPACE }),
    })
    const user = userEvent.setup()
    await openSettings(user)

    await user.clear(screen.getByLabelText("URL"))
    await user.type(screen.getByLabelText("URL"), "https://mcp.example.com/v2")
    await user.click(screen.getByRole("button", { name: "Save" }))

    expect(await screen.findByText(/Saved, but not checked yet/)).toBeVisible()
  })

  it("refuses an http URL while a token is stored", async () => {
    const calls = mockMcpServerApi({ server: SERVER })
    const user = userEvent.setup()
    await openSettings(user)

    await user.clear(screen.getByLabelText("URL"))
    await user.type(screen.getByLabelText("URL"), "http://mcp.example.com")

    await waitFor(() =>
      expect(screen.getByLabelText("URL")).toHaveAccessibleDescription(
        /needs an https URL/,
      ),
    )
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled()
    expect(mcpWrites(calls)).toHaveLength(0)
  })

  it("turns the server off from its one Availability choice", async () => {
    const calls = mockMcpServerApi({ server: SERVER })
    const user = userEvent.setup()
    await openPage()

    await pickOption(
      user,
      screen.getByRole("button", { name: /Availability/ }),
      /^Off/,
    )

    expect(await patchBody(calls)).toEqual({
      enabled: false,
      attach_by_default: false,
    })
  })

  it("adds the server to every request from the same choice", async () => {
    const calls = mockMcpServerApi({ server: SERVER })
    const user = userEvent.setup()
    await openPage()

    expect(
      screen.getByText(
        "Available to people building things, invisible to coding agents.",
      ),
    ).toBeInTheDocument()
    await pickOption(
      user,
      screen.getByRole("button", { name: /Availability/ }),
      /^Every request/,
    )

    expect(await patchBody(calls)).toEqual({
      enabled: true,
      attach_by_default: true,
    })
  })

  it("deletes the server and goes back to the list", async () => {
    const calls = mockMcpServerApi({ server: SERVER })
    const user = userEvent.setup()
    await openSettings(user)

    // Deleting is at the foot of Settings, said before it is pressed.
    const settings = screen.getByRole("dialog", { name: "Settings for github" })
    expect(
      within(settings).getByText(
        /Removes the server for everyone in the workspace and ends every connection to it/,
      ),
    ).toBeVisible()
    await user.click(
      within(settings).getByRole("button", { name: "Delete github" }),
    )
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
      screen.getByRole("radio", { name: "Only the tools I pick" }),
    )
    await user.click(screen.getByRole("checkbox", { name: "get_issue" }))
    await user.click(screen.getByRole("button", { name: "Save exposed tools" }))

    expect(await patchBody(calls)).toEqual({ allowed_tools: ["get_issue"] })
  })

  it("starts read tools from the ones the server marks read-only, until a manager edits them", async () => {
    const calls = mockMcpServerApi({
      server: workspaceMcpServer({
        workspace_id: WORKSPACE,
        // What the gateway answers while nobody has edited the labels.
        read_tools: ["list_issues", "delete_issue"],
        read_tools_from_server: true,
        health: health({
          tools: ["list_issues", "get_issue", "delete_issue"],
          read_only_hinted_tools: ["list_issues", "delete_issue"],
        }),
      }),
    })
    const user = userEvent.setup()
    await openPage()

    expect(
      screen.getByText(
        "These are the tools the server marks read-only. Edit them to set your own.",
      ),
    ).toBeVisible()
    await user.click(screen.getByRole("button", { name: "Edit read tools" }))
    const list = screen.getByRole("list", { name: "Read tools" })
    expect(
      within(list).getByRole("checkbox", { name: "list_issues" }),
    ).toBeChecked()
    expect(
      within(list).getByRole("checkbox", { name: "delete_issue" }),
    ).toBeChecked()
    expect(
      within(list).getByRole("checkbox", { name: "get_issue" }),
    ).not.toBeChecked()
    // The server calls delete_issue read-only; the manager disagrees.
    await user.click(
      within(list).getByRole("checkbox", { name: "delete_issue" }),
    )
    await user.click(within(list).getByRole("checkbox", { name: "get_issue" }))
    await user.click(screen.getByRole("button", { name: "Save read tools" }))

    expect(await patchBody(calls)).toEqual({
      read_tools: ["list_issues", "get_issue"],
    })
  })

  it("says nothing about the server once the labels are a manager's own", async () => {
    mockMcpServerApi({
      server: workspaceMcpServer({
        workspace_id: WORKSPACE,
        read_tools: ["list_issues"],
        health: health({ tools: ["list_issues", "get_issue"] }),
      }),
    })
    await openPage()

    expect(screen.getByText("list_issues")).toBeVisible()
    expect(
      screen.queryByText(/the server marks read-only/),
    ).not.toBeInTheDocument()
    await userEvent
      .setup()
      .click(screen.getByRole("button", { name: "Edit read tools" }))
    // Nothing changed yet, and the labels are already the manager's.
    expect(
      screen.getByRole("button", { name: "Save read tools" }),
    ).toBeDisabled()
  })

  it("checks the connection beside the exposed tools", async () => {
    const calls = mockMcpServerApi({ server: SERVER })
    const user = userEvent.setup()
    await openPage()

    await user.click(screen.getByRole("button", { name: "Check" }))

    await waitFor(() =>
      expect(
        mcpWrites(calls).some(
          (call) =>
            call.method === "POST" && call.url.endsWith(`${path}/check`),
        ),
      ).toBe(true),
    )
  })

  it("moves a server to Personal, which drops the shared token box", async () => {
    const calls = mockMcpServerApi({ server: SERVER })
    const user = userEvent.setup()
    await openSettings(user)

    expect(screen.getByRole("radio", { name: /^Shared/ })).toBeChecked()
    await user.click(screen.getByRole("radio", { name: /^Personal/ }))
    expect(
      screen.queryByLabelText("Authorization token"),
    ).not.toBeInTheDocument()
    await user.type(screen.getByLabelText("OAuth client ID"), "otari-client")
    await user.click(screen.getByRole("button", { name: "Save" }))

    const body = await patchBody(calls)
    expect(body).toMatchObject({
      credential_mode: "per_user",
      oauth_client_id: "otari-client",
    })
    expect(body).not.toHaveProperty("authorization_token")
  })

  it("warns that moving a server to another host clears the client a manager entered", async () => {
    const perUser = workspaceMcpServer({
      workspace_id: WORKSPACE,
      credential_mode: "per_user",
      oauth_client_id: "otari-client",
      has_oauth_client_secret: true,
    })
    mockMcpServerApi({ server: perUser })
    const user = userEvent.setup()
    await openSettings(user)

    const url = screen.getByLabelText("URL")
    await user.clear(url)
    await user.type(url, "https://mcp.example.com/other")
    expect(screen.queryByText(/another host/)).not.toBeInTheDocument()
    await user.clear(url)
    await user.type(url, "https://elsewhere.example.com/mcp")
    expect(screen.getByText(/another host/)).toBeVisible()
  })

  it("shows the mode as a label and opens Settings on the field a press is about", async () => {
    mockMcpServerApi({
      server: workspaceMcpServer({
        workspace_id: WORKSPACE,
        needs_setup: true,
      }),
    })
    const user = userEvent.setup()
    await openPage()

    expect(screen.getByText("Not set up yet")).toBeVisible()
    // The mode is read in the heading, and changed nowhere but Settings.
    const heading = screen.getByRole("heading", { level: 1 })
    const intro = heading.closest("header") as HTMLElement
    expect(within(intro).getByText("SHARED")).toBeVisible()
    expect(
      screen.queryByRole("button", { name: /Account mode/ }),
    ).not.toBeInTheDocument()
    // Deleting is not in the heading either.
    expect(
      within(intro).queryByRole("button", { name: /Delete/ }),
    ).not.toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Settings" }))
    const settings = screen.getByRole("dialog", {
      name: "Settings for github",
    })
    for (const group of ["Server", "Credentials", "OAuth client"]) {
      expect(
        within(settings).getByRole("heading", { level: 3, name: group }),
      ).toBeVisible()
    }
    expect(
      within(settings).getByRole("radio", { name: "Shared account" }),
    ).toBeChecked()
    expect(
      within(settings).getByText(/Switching ends existing connections/),
    ).toBeVisible()
    await user.click(within(settings).getByRole("button", { name: "Cancel" }))
    await waitFor(() =>
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument(),
    )

    // The Connection block opens the same dialog, from its Edit and its link.
    for (const name of ["Edit connection", "Open Settings"]) {
      await user.click(screen.getByRole("button", { name }))
      expect(
        screen.getByRole("dialog", { name: "Settings for github" }),
      ).toBeVisible()
      await user.keyboard("{Escape}")
      await waitFor(() =>
        expect(screen.queryByRole("dialog")).not.toBeInTheDocument(),
      )
    }

    // A press about the token lands in the token field.
    await user.click(screen.getByRole("button", { name: "Paste a token" }))
    const token = await screen.findByLabelText("Authorization token")
    await waitFor(() => expect(token).toHaveFocus())
    await user.keyboard("{Escape}")
    await waitFor(() =>
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument(),
    )

    // And the sentence about members' own accounts lands on the mode.
    await user.click(
      screen.getByRole("button", { name: "change the sign-in mode" }),
    )
    await waitFor(() =>
      expect(
        screen.getByRole("radio", { name: "Shared account" }),
      ).toHaveFocus(),
    )
  })

  it("asks before closing Settings on unsaved changes", async () => {
    const calls = mockMcpServerApi({ server: SERVER })
    const user = userEvent.setup()
    await openSettings(user)

    await user.type(screen.getByLabelText("Purpose hint"), "Issues")
    await user.keyboard("{Escape}")
    expect(screen.getByText("Unsaved changes")).toBeVisible()
    await user.click(screen.getByRole("button", { name: "Discard" }))
    await waitFor(() =>
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument(),
    )
    expect(mcpWrites(calls)).toHaveLength(0)
  })

  it("offers Settings to nobody who cannot manage the workspace", async () => {
    mockMcpServerApi({
      server: SERVER,
      role: "member",
      workspaceRole: "member",
    })
    await openPage()

    expect(
      screen.queryByRole("button", { name: "Settings" }),
    ).not.toBeInTheDocument()
    expect(screen.getByText("SHARED")).toBeVisible()
  })

  it("shows the client secret beside the client ID, and removes a stored one only when asked", async () => {
    const perUser = workspaceMcpServer({
      workspace_id: WORKSPACE,
      credential_mode: "per_user",
      oauth_client_id: "otari-client",
      has_oauth_client_secret: true,
    })
    const calls = mockMcpServerApi({ server: perUser })
    const user = userEvent.setup()
    await openPage()

    // The summary says a secret is stored, and never what it is.
    expect(screen.getByText(/with a stored secret/)).toBeVisible()
    expect(screen.getByText(/Your client ID/)).toBeVisible()
    await user.click(screen.getByRole("button", { name: "Settings" }))
    const secret = screen.getByLabelText("Client secret")
    expect(secret).toBeEnabled()
    expect(secret).toHaveValue("")
    await user.click(
      screen.getByRole("checkbox", { name: "Remove the stored secret" }),
    )
    await user.click(screen.getByRole("button", { name: "Save" }))

    const body = await patchBody(calls)
    expect(body).toMatchObject({ oauth_client_secret: "" })
    // The client itself is the same one, so nobody is signed out.
    expect(body).not.toHaveProperty("oauth_client_id")
  })

  it("sends a typed client secret with the client it belongs to, on a shared server too", async () => {
    const calls = mockMcpServerApi({ server: SERVER })
    const user = userEvent.setup()
    await openSettings(user)

    // On screen before there is a client, but it takes nothing until there is one.
    expect(screen.getByLabelText("Client secret")).toBeDisabled()
    await user.type(screen.getByLabelText("OAuth client ID"), "Iv1.github")
    await user.type(screen.getByLabelText("Client secret"), " gh-secret ")
    await user.click(screen.getByRole("button", { name: "Save" }))

    expect(await patchBody(calls)).toMatchObject({
      oauth_client_id: "Iv1.github",
      oauth_client_secret: "gh-secret",
    })
  })

  it("keeps the server id on one line beside its copy control", async () => {
    mockMcpServerApi({ server: SERVER })
    await openPage()

    const id = screen.getByText(SERVER.id)
    expect(id.tagName).toBe("CODE")
    // Cut with an ellipsis before it wraps: no break-anywhere, and a line that cannot wrap.
    expect(id).not.toHaveClass("break-all")
    expect(id.parentElement).toHaveClass("truncate")
    const line = id.closest(".whitespace-nowrap") as HTMLElement
    expect(line).not.toHaveClass("flex-wrap")
    expect(
      within(line).getByRole("button", { name: /id for github/i }),
    ).toBeVisible()
  })

  it("warns before a save that signs the shared account out", async () => {
    mockMcpServerApi({
      server: workspaceMcpServer({
        workspace_id: WORKSPACE,
        shared_account: {
          signed_in_by: "Ada Admin",
          signed_in_at: "2026-11-01T10:00:00+00:00",
          renewable: true,
        },
      }),
    })
    const user = userEvent.setup()
    await openSettings(user)

    expect(
      screen.queryByText(/signs the shared account out/),
    ).not.toBeInTheDocument()
    await user.type(screen.getByLabelText("Authorization token"), "tok")
    expect(screen.getByText(/signs the shared account out/)).toBeVisible()
  })

  it("shows a per-user server's sign-in and warns before a change that ends everyone's", async () => {
    const perUser = workspaceMcpServer({
      workspace_id: WORKSPACE,
      credential_mode: "per_user",
      oauth_client_id: "otari-client",
      has_oauth_client_secret: true,
    })
    const calls = mockMcpServerApi({ server: perUser })
    const user = userEvent.setup()
    await openPage()

    // A manager's own account is the heading's control, not a section of its
    // own and not a button on Connected users.
    expect(
      screen.queryByRole("heading", { level: 2, name: "Your account" }),
    ).not.toBeInTheDocument()
    const connect = await screen.findByRole("button", {
      name: "Connect your account",
    })
    expect(connect.closest("header")).not.toBeNull()
    await user.click(screen.getByRole("button", { name: "Settings" }))
    // A rename keeps every connection, so it says nothing and sends no mode.
    expect(screen.queryByText(/disconnects everyone/)).not.toBeInTheDocument()
    await user.clear(screen.getByLabelText("OAuth client ID"))
    expect(screen.getByText(/disconnects everyone/)).toBeVisible()
    await user.click(screen.getByRole("button", { name: "Save" }))

    const body = await patchBody(calls)
    // An emptied box goes back to the client Otari registers itself.
    expect(body).toMatchObject({ oauth_client_id: "" })
    expect(body).not.toHaveProperty("credential_mode")
    expect(body).not.toHaveProperty("oauth_client_secret")
  })
})
