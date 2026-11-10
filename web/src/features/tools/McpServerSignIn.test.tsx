import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { McpConnection, McpServerConnection } from "@/client"
import { McpServerSignIn } from "@/features/tools/McpServerSignIn"
import { workspaceMcpServer } from "@/tests/fixtures"

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
    return Response.json({ data: mine })
  })
  return calls
}

function renderSignIn(canManage = false) {
  const onLeave = vi.fn()
  render(
    <QueryClientProvider
      client={
        new QueryClient({ defaultOptions: { queries: { retry: false } } })
      }
    >
      <McpServerSignIn
        workspaceId={WORKSPACE}
        server={SERVER}
        canManage={canManage}
        onLeave={onLeave}
      />
    </QueryClientProvider>,
  )
  return onLeave
}

describe("McpServerSignIn", () => {
  afterEach(() => {
    vi.restoreAllMocks()
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

    expect(await screen.findByText(/issues no refresh token/)).toBeVisible()
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
})
