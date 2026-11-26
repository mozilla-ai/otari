import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { WorkspaceMcpServer } from "@/client"
import {
  McpSharedAccount,
  sharedAccountState,
} from "@/features/tools/McpSharedAccount"
import { forgetMcpConnectionNotice } from "@/features/tools/mcpConnectionNotice"
import { workspaceMcpServer, workspaceMcpServerHealth } from "@/tests/fixtures"

const NOT_SET_UP = workspaceMcpServer({
  name: "github",
  needs_setup: true,
  health: workspaceMcpServerHealth({
    status: "failing",
    error: "unauthorized",
  }),
})
const OPEN = workspaceMcpServer({
  name: "docs",
  health: workspaceMcpServerHealth({ status: "healthy" }),
})
const UNCHECKED = workspaceMcpServer({ name: "docs", health: null })
const SIGNED_IN = workspaceMcpServer({
  name: "github",
  shared_account: {
    signed_in_by: "Ada Admin",
    signed_in_at: "2026-11-01T10:00:00+00:00",
    renewable: true,
  },
})

interface Call {
  url: string
  method: string
}

function mockApi(): Call[] {
  const calls: Call[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
    calls.push({ url, method })
    if (method === "DELETE") return Response.json(NOT_SET_UP)
    return Response.json({
      authorization_url: "https://auth.example.com/authorize?state=s",
    })
  })
  return calls
}

function renderAccount(server: WorkspaceMcpServer, canManage = true) {
  const onLeave = vi.fn()
  const onOpenSettings = vi.fn()
  render(
    <QueryClientProvider
      client={
        new QueryClient({ defaultOptions: { queries: { retry: false } } })
      }
    >
      <McpSharedAccount
        workspaceId={server.workspace_id}
        server={server}
        canManage={canManage}
        onOpenSettings={onOpenSettings}
        onLeave={onLeave}
        openWindow={() => undefined}
      />
    </QueryClientProvider>,
  )
  return { onLeave, onOpenSettings }
}

describe("McpSharedAccount", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
    forgetMcpConnectionNotice()
  })

  it("reads a server's one credential in five states", () => {
    expect(sharedAccountState(SIGNED_IN)).toBe("signed_in")
    expect(sharedAccountState(workspaceMcpServer({ has_token: true }))).toBe(
      "token",
    )
    expect(sharedAccountState(NOT_SET_UP)).toBe("not_set_up")
    expect(sharedAccountState(OPEN)).toBe("open")
    // No check has answered, or the last one failed for another reason: an
    // answer without an account was never seen.
    expect(sharedAccountState(UNCHECKED)).toBe("unchecked")
    expect(
      sharedAccountState(
        workspaceMcpServer({
          health: workspaceMcpServerHealth({
            status: "failing",
            error: "unreachable",
          }),
        }),
      ),
    ).toBe("unchecked")
  })

  it("claims no answer from a server that was never checked", () => {
    mockApi()
    renderAccount(UNCHECKED)

    expect(screen.getByText("No account set up")).toBeVisible()
    expect(
      screen.getByText(/^Requests reach the server without an account\./),
    ).toBeVisible()
    expect(screen.queryByText(/answered without/)).not.toBeInTheDocument()
  })

  it("tells a member no account is set up on a server that was never checked", () => {
    mockApi()
    renderAccount(UNCHECKED, false)

    expect(
      screen.getByText(/No account is set up, so there is nothing for you/),
    ).toBeVisible()
    expect(screen.queryByText(/answered without/)).not.toBeInTheDocument()
  })

  it("offers a manager both ways to set up a server that is not set up yet", async () => {
    const calls = mockApi()
    const { onLeave, onOpenSettings } = renderAccount(NOT_SET_UP)

    expect(screen.getByText("Not set up yet")).toBeVisible()
    // The consequence is said before the press, not after it.
    expect(
      screen.getByText(/Everyone in the workspace acts as the account/),
    ).toBeVisible()
    expect(screen.getByText(/use a service account/)).toBeVisible()

    await userEvent.click(screen.getByRole("button", { name: "Paste a token" }))
    expect(onOpenSettings).toHaveBeenLastCalledWith("token")
    await userEvent.click(
      screen.getByRole("button", { name: "change the sign-in mode" }),
    )
    expect(onOpenSettings).toHaveBeenLastCalledWith("mode")

    await userEvent.click(
      screen.getByRole("button", { name: "Sign in for everyone" }),
    )
    await waitFor(() =>
      expect(onLeave).toHaveBeenCalledWith(
        "https://auth.example.com/authorize?state=s",
      ),
    )
    expect(calls[0]).toEqual({
      method: "POST",
      url: expect.stringMatching(
        new RegExp(`/mcp-servers/${NOT_SET_UP.id}/shared-account$`),
      ),
    })
  })

  it("links a refusal that asks for an OAuth client to Settings", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      Response.json(
        {
          detail:
            "This server's sign-in offers no way to register Otari as a client. Add an OAuth client ID: on the server's page, open Settings.",
        },
        { status: 409 },
      ),
    )
    const { onOpenSettings } = renderAccount(NOT_SET_UP)

    await userEvent.click(
      screen.getByRole("button", { name: "Sign in for everyone" }),
    )
    expect(await screen.findByText(/open Settings\.$/)).toBeVisible()
    await userEvent.click(screen.getByRole("button", { name: "Open Settings" }))
    expect(onOpenSettings).toHaveBeenLastCalledWith("client_id")
  })

  it("says who signed in and when, and disconnects after a confirmation", async () => {
    const calls = mockApi()
    renderAccount(SIGNED_IN)

    expect(screen.getByText(/^Signed in by Ada Admin on /)).toBeVisible()
    expect(screen.getByRole("button", { name: "Reconnect" })).toBeVisible()
    await userEvent.click(screen.getByRole("button", { name: "Disconnect" }))
    expect(calls).toEqual([])
    const dialog = await screen.findByRole("alertdialog")
    expect(within(dialog).getByText(/Everyone's keys/)).toBeVisible()
    await userEvent.click(
      within(dialog).getByRole("button", { name: "Disconnect" }),
    )
    await waitFor(() =>
      expect(calls.find((call) => call.method === "DELETE")?.url).toMatch(
        new RegExp(`/mcp-servers/${SIGNED_IN.id}/shared-account$`),
      ),
    )
  })

  it("tells a member what Shared means and offers them nothing to press", () => {
    mockApi()
    renderAccount(SIGNED_IN, false)

    expect(
      screen.getByText(/everyone in this workspace uses one account/),
    ).toBeVisible()
    expect(screen.getByText(/nothing for you to connect/)).toBeVisible()
    expect(screen.queryByRole("button")).not.toBeInTheDocument()
  })

  it("tells a member when no account is set up yet", () => {
    mockApi()
    renderAccount(NOT_SET_UP, false)

    expect(
      screen.getByText(/An admin has not set up an account yet/),
    ).toBeVisible()
    expect(screen.queryByText(/An admin set it up/)).not.toBeInTheDocument()
  })
})
