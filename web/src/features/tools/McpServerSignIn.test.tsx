import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { McpConnection, McpServerConnection } from "@/client"
import {
  McpOwnAccountControl,
  McpOwnAccountNotes,
  useOwnMcpAccount,
} from "@/features/tools/McpOwnAccount"
import { McpServerSignIn } from "@/features/tools/McpServerSignIn"
import {
  forgetMcpConnectionNotice,
  readMcpConnectionNotice,
} from "@/features/tools/mcpConnectionNotice"
import { publishSignInResult } from "@/features/tools/mcpSignInWindow"
import { organizationContext, workspaceMcpServer } from "@/tests/fixtures"

const SERVER = workspaceMcpServer({
  name: "drive",
  credential_mode: "per_user",
})
const WORKSPACE = SERVER.workspace_id
const MINE: McpConnection = {
  server_id: SERVER.id,
  connected_at: "2026-11-01T10:00:00+00:00",
  updated_at: "2026-11-01T10:00:00+00:00",
  expires_at: null,
  renewable: true,
}
const ALICE: McpServerConnection = {
  user_id: "alice-id",
  email: "alice@example.com",
  full_name: "Alice",
  connected_at: "2026-11-02T10:00:00+00:00",
  updated_at: "2026-11-02T10:00:00+00:00",
}

interface Call {
  url: string
  method: string
  body: unknown
}

function mockApi({
  mine = [],
  others = [],
}: {
  mine?: McpConnection[]
  others?: McpServerConnection[]
} = {}): Call[] {
  const calls: Call[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
    calls.push({
      url,
      method,
      body: typeof init?.body === "string" ? JSON.parse(init.body) : undefined,
    })
    if (method === "DELETE") return new Response(null, { status: 204 })
    if (url.endsWith(`/mcp-connections/${SERVER.id}`)) {
      return Response.json({
        authorization_url: "https://auth.example.com/authorize?state=s",
      })
    }
    if (url.includes("/connections?")) {
      return Response.json({ data: others, count: others.length })
    }
    if (url.endsWith("/organizations/me")) {
      return Response.json(organizationContext())
    }
    return Response.json({ data: mine })
  })
  return calls
}

function renderSignIn(
  canManage = false,
  openWindow: () => Window | undefined = () => undefined,
) {
  const onLeave = vi.fn()
  render(
    <QueryClientProvider
      client={
        new QueryClient({ defaultOptions: { queries: { retry: false } } })
      }
    >
      <Page canManage={canManage} onLeave={onLeave} openWindow={openWindow} />
    </QueryClientProvider>,
  )
  return onLeave
}

/** The three places the server's page puts a person's own account, as it wires them. */
function Page({
  canManage,
  onLeave,
  openWindow,
}: {
  canManage: boolean
  onLeave: (url: string) => void
  openWindow: () => Window | undefined
}) {
  const own = useOwnMcpAccount({
    workspaceId: WORKSPACE,
    server: SERVER,
    onLeave,
    openWindow,
  })
  return (
    <>
      {canManage ? (
        <section aria-label="Page heading">
          <McpOwnAccountControl own={own} />
        </section>
      ) : null}
      <McpOwnAccountNotes own={own} />
      <McpServerSignIn
        workspaceId={WORKSPACE}
        server={SERVER}
        canManage={canManage}
        own={own}
      />
    </>
  )
}

describe("McpServerSignIn", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
    forgetMcpConnectionNotice()
  })

  it("connects in a popup and updates the page in place when it reports back", async () => {
    const calls = mockApi()
    const popup = {
      closed: false,
      close: vi.fn(),
      opener: {},
      location: { href: "" },
    }
    const user = userEvent.setup()
    const onLeave = renderSignIn(false, () => popup as unknown as Window)

    const connect = await screen.findByRole("button", {
      name: "Connect your account",
    })
    await waitFor(() => expect(connect).toBeEnabled())
    await user.click(connect)

    await waitFor(() =>
      expect(popup.location.href).toBe(
        "https://auth.example.com/authorize?state=s",
      ),
    )
    expect(onLeave).not.toHaveBeenCalled()
    expect(
      screen.getByText("Finish signing in in the window that opened."),
    ).toBeVisible()
    const readsBefore = calls.filter((call) => call.method === "GET").length

    publishSignInResult({
      state: "s",
      outcome: "connected",
      serverId: SERVER.id,
      serverName: "drive",
    })

    // The connection is read again, so the page shows it without a reload.
    await waitFor(() =>
      expect(
        calls.filter((call) => call.method === "GET").length,
      ).toBeGreaterThan(readsBefore),
    )
    // Nothing is announced: the row turning to Connected is the result.
    expect(readMcpConnectionNotice()).toBeUndefined()
    expect(
      screen.queryByText("Finish signing in in the window that opened."),
    ).not.toBeInTheDocument()
  })

  it("stops, and leaves the page where it is, when the popup is closed before the sign-in starts", async () => {
    const calls = mockApi()
    // Closed by the person while the start request was on its way.
    const popup = {
      closed: true,
      close: vi.fn(),
      opener: {},
      location: { href: "" },
    }
    const user = userEvent.setup()
    const onLeave = renderSignIn(false, () => popup as unknown as Window)

    const connect = await screen.findByRole("button", {
      name: "Connect your account",
    })
    await waitFor(() => expect(connect).toBeEnabled())
    await user.click(connect)

    await waitFor(() =>
      expect(calls.some((call) => call.method === "POST")).toBe(true),
    )
    await waitFor(() =>
      expect(
        screen.queryByText("Finish signing in in the window that opened."),
      ).not.toBeInTheDocument(),
    )
    expect(onLeave).not.toHaveBeenCalled()
    expect(popup.location.href).toBe("")
  })

  it("sends a person who has not connected to the server's authorization page", async () => {
    const calls = mockApi()
    const onLeave = renderSignIn()

    expect(await screen.findByText("Not connected")).toBeVisible()
    await userEvent.click(
      await screen.findByRole("button", { name: "Connect your account" }),
    )

    await waitFor(() =>
      expect(onLeave).toHaveBeenCalledWith(
        "https://auth.example.com/authorize?state=s",
      ),
    )
    const started = calls.find((call) => call.method === "POST")
    expect(started?.url).toContain(
      `/workspaces/${WORKSPACE}/mcp-connections/${SERVER.id}`,
    )
    // Where the sign-in lands afterwards: this server's own page.
    expect(started?.body).toEqual({
      return_to: `/tools/mcp-servers/${SERVER.id}`,
    })
    // A member is not shown, and does not ask for, who else is connected.
    expect(screen.queryByText("Connected users")).not.toBeInTheDocument()
    expect(calls.some((call) => call.url.includes("/connections?"))).toBe(false)
  })

  it("claims nothing while the status loads, and says so when it cannot be read", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async () =>
      Response.json({ detail: "boom" }, { status: 500 }),
    )
    renderSignIn()

    expect(screen.getByText("Checking…")).toBeVisible()
    expect(screen.queryByText("Not connected")).not.toBeInTheDocument()
    expect(await screen.findByText("Status unavailable")).toBeVisible()
    expect(
      screen.getByRole("button", { name: "Connect your account" }),
    ).toBeDisabled()
  })

  it("shows a connection and ends it after a confirmation", async () => {
    const calls = mockApi({ mine: [MINE] })
    renderSignIn()

    expect(await screen.findByText(/^Connected since /)).toBeVisible()
    expect(
      screen.queryByRole("button", { name: "Connect your account" }),
    ).not.toBeInTheDocument()
    await userEvent.click(screen.getByRole("button", { name: "Disconnect" }))
    expect(calls.some((call) => call.method === "DELETE")).toBe(false)
    const dialog = await screen.findByRole("alertdialog")
    await userEvent.click(
      within(dialog).getByRole("button", { name: "Disconnect" }),
    )

    await waitFor(() =>
      expect(calls.find((call) => call.method === "DELETE")?.url).toContain(
        `/workspaces/${WORKSPACE}/mcp-connections/${SERVER.id}`,
      ),
    )
  })

  it("says when a connection cannot renew itself", async () => {
    mockApi({ mine: [{ ...MINE, renewable: false }] })
    renderSignIn()

    expect(
      await screen.findByText(/connect again when this sign-in expires/),
    ).toBeVisible()
  })

  it("lets a manager see who is connected and remove one of them", async () => {
    const calls = mockApi({ others: [ALICE] })
    renderSignIn(true)

    expect(await screen.findByText("Alice")).toBeVisible()
    expect(screen.getByText("alice@example.com")).toBeVisible()
    await userEvent.click(
      screen.getByRole("button", { name: "Remove the connection of Alice" }),
    )
    const dialog = await screen.findByRole("alertdialog")
    await userEvent.click(
      within(dialog).getByRole("button", { name: "Remove connection" }),
    )

    await waitFor(() =>
      expect(calls.find((call) => call.method === "DELETE")?.url).toContain(
        `/mcp-servers/${SERVER.id}/connections/alice-id`,
      ),
    )
  })

  it("gives a manager Connect your account in the page heading, not in Connected users", async () => {
    const calls = mockApi({ others: [ALICE] })
    const onLeave = renderSignIn(true)

    expect(
      screen.queryByRole("heading", { name: "Your account" }),
    ).not.toBeInTheDocument()
    expect(
      screen.getByText(/each member signs in with their own account/),
    ).toBeVisible()
    // The one Connect on the page is in its heading, and names whose account it is.
    const connect = await screen.findByRole("button", {
      name: "Connect your account",
    })
    expect(
      within(screen.getByRole("region", { name: "Page heading" })).getByRole(
        "button",
        { name: "Connect your account" },
      ),
    ).toBe(connect)
    expect(screen.getAllByRole("button", { name: /^(Re)?connect/i })).toEqual([
      connect,
    ])
    await userEvent.click(connect)
    await waitFor(() =>
      expect(onLeave).toHaveBeenCalledWith(
        "https://auth.example.com/authorize?state=s",
      ),
    )
    expect(
      calls.some(
        (call) =>
          call.method === "POST" &&
          call.url.endsWith(`/mcp-connections/${SERVER.id}`),
      ),
    ).toBe(true)
  })

  it("leaves an empty list of connected users with no control of its own", async () => {
    mockApi()
    renderSignIn(true)

    expect(await screen.findByText("Nobody has connected yet.")).toBeVisible()
    const group = screen.getByRole("heading", { name: "Connected users" })
      .parentElement?.parentElement?.parentElement as HTMLElement
    expect(within(group).queryByRole("button")).not.toBeInTheDocument()
  })

  it("lets a connected manager reconnect from the heading's menu", async () => {
    mockApi({ mine: [MINE], others: [ALICE] })
    const user = userEvent.setup()
    const onLeave = renderSignIn(true)

    const control = await screen.findByRole("button", {
      name: "Your account: connected. Reconnect or disconnect",
    })
    expect(control).toHaveTextContent("Your account: connected")
    expect(
      screen.queryByRole("button", { name: "Connect your account" }),
    ).not.toBeInTheDocument()
    await user.click(control)
    await user.click(await screen.findByRole("menuitem", { name: "Reconnect" }))

    await waitFor(() => expect(onLeave).toHaveBeenCalled())
  })

  it("lets a connected manager disconnect from the heading's menu, after a confirmation", async () => {
    const calls = mockApi({ mine: [MINE], others: [ALICE] })
    const user = userEvent.setup()
    renderSignIn(true)

    await user.click(
      await screen.findByRole("button", {
        name: "Your account: connected. Reconnect or disconnect",
      }),
    )
    await user.click(
      await screen.findByRole("menuitem", { name: "Disconnect" }),
    )
    expect(calls.some((call) => call.method === "DELETE")).toBe(false)
    const dialog = await screen.findByRole("alertdialog")
    await user.click(within(dialog).getByRole("button", { name: "Disconnect" }))

    await waitFor(() =>
      expect(calls.find((call) => call.method === "DELETE")?.url).toMatch(
        new RegExp(`/workspaces/${WORKSPACE}/mcp-connections/${SERVER.id}$`),
      ),
    )
  })

  it("gives the manager's own row the one remove action every row has, as Disconnect", async () => {
    // Not in the list the gateway returned, as right after a sign-in: the row is theirs all the same.
    const calls = mockApi({ mine: [MINE], others: [ALICE] })
    renderSignIn(true)

    const table = await screen.findByRole("grid", {
      name: "Users connected to drive",
    })
    await waitFor(() =>
      expect(within(table).getAllByRole("rowheader")[0]).toHaveTextContent(
        "(you)",
      ),
    )
    // No Reconnect in the list: that is the heading's.
    expect(
      within(table).queryByRole("button", { name: /Reconnect/ }),
    ).not.toBeInTheDocument()
    expect(within(table).getAllByRole("button")).toHaveLength(2)

    // Removing their own row is Disconnect: the same confirmation, the same call.
    await userEvent.click(
      within(table).getByRole("button", { name: "Disconnect your account" }),
    )
    const dialog = await screen.findByRole("alertdialog")
    expect(
      within(dialog).getByText(
        /Your keys and the Playground stop reaching drive/,
      ),
    ).toBeVisible()
    await userEvent.click(
      within(dialog).getByRole("button", { name: "Disconnect" }),
    )
    await waitFor(() =>
      expect(calls.find((call) => call.method === "DELETE")?.url).toMatch(
        new RegExp(`/workspaces/${WORKSPACE}/mcp-connections/${SERVER.id}$`),
      ),
    )
  })

  it("puts the manager's own connection first among the connected users, marked", async () => {
    const own: McpServerConnection = {
      // The fixture caller's id: a connection is held by the dashboard account's own.
      user_id: organizationContext().caller?.user_id ?? "",
      email: null,
      full_name: "Operator",
      connected_at: "2026-11-05T10:00:00+00:00",
      updated_at: "2026-11-05T10:00:00+00:00",
    }
    // Oldest first, as the gateway lists them, so the manager's is last.
    mockApi({ others: [ALICE, own] })
    renderSignIn(true)

    const table = await screen.findByRole("grid", {
      name: "Users connected to drive",
    })
    await waitFor(() =>
      expect(
        within(table)
          .getAllByRole("rowheader")
          .map((cell) => cell.textContent),
      ).toEqual(["Operator (you)", "Alicealice@example.com"]),
    )
  })
})
