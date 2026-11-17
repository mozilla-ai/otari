import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import {
  McpConnectionCallbackPage,
  type McpConnectionCallbackSearch,
} from "@/features/tools/McpConnectionCallbackPage"
import {
  forgetMcpConnectionNotice,
  readMcpConnectionNotice,
} from "@/features/tools/mcpConnectionNotice"
import {
  type McpSignInResult,
  markSignInWindow,
  subscribeToSignInResult,
} from "@/features/tools/mcpSignInWindow"
import { CANCELLED_NOTICE } from "@/features/tools/useMcpSignIn"
import { withRouter } from "@/tests/router"

const SERVER_ID = "55555555-5555-5555-5555-555555555555"

function mockComplete(answer: () => Response) {
  const calls: { url: string; body: unknown }[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    calls.push({
      url: String(input),
      body: typeof init?.body === "string" ? JSON.parse(init.body) : undefined,
    })
    return answer()
  })
  return calls
}

function renderCallback(search: McpConnectionCallbackSearch) {
  return render(
    <QueryClientProvider client={new QueryClient()}>
      <McpConnectionCallbackPage search={search} />
    </QueryClientProvider>,
    {
      wrapper: withRouter({
        url: "/mcp-connections/callback",
        routes: [
          { path: "/tools/mcp-servers/$serverId", element: <p>Server page</p> },
          { path: "/tools/mcp-servers", element: <p>Server list</p> },
          { path: "/playground", element: <p>Playground</p> },
        ],
      }),
    },
  )
}

describe("McpConnectionCallbackPage", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.sessionStorage.clear()
    window.localStorage.clear()
    forgetMcpConnectionNotice()
  })

  it("finishes a popup's sign-in, reports it to the page that opened it, and closes", async () => {
    const calls = mockComplete(() =>
      Response.json({
        server_id: SERVER_ID,
        workspace_id: "w",
        server_name: "notion",
        return_to: "/tools/mcp-servers",
        checked: true,
      }),
    )
    const close = vi.spyOn(window, "close").mockImplementation(() => {})
    markSignInWindow("popup-state")
    const heard = new Promise<McpSignInResult>((resolve) =>
      subscribeToSignInResult("popup-state", resolve),
    )

    renderCallback({ state: "popup-state", code: "the-code" })

    expect(await heard).toEqual({
      state: "popup-state",
      outcome: "checked",
      serverId: SERVER_ID,
      serverName: "notion",
    })
    expect(calls).toHaveLength(1)
    await waitFor(() => expect(close).toHaveBeenCalled())
    // It stays where it is rather than opening the dashboard inside the popup.
    expect(screen.queryByText("Server list")).not.toBeInTheDocument()
    expect(window.localStorage.getItem("otari.mcp-sign-in.windows")).toBe("{}")
  })

  it("reports a sign-in cancelled in a popup without spending anything", async () => {
    const calls = mockComplete(() => Response.json({}))
    vi.spyOn(window, "close").mockImplementation(() => {})
    markSignInWindow("popup-state")
    const heard = new Promise<McpSignInResult>((resolve) =>
      subscribeToSignInResult("popup-state", resolve),
    )

    renderCallback({ state: "popup-state", error: "access_denied" })

    expect(await heard).toEqual({ state: "popup-state", outcome: "cancelled" })
    expect(calls).toEqual([])
  })

  it("treats a sign-in no popup noted as a full page, whatever its state", async () => {
    mockComplete(() =>
      Response.json({
        server_id: SERVER_ID,
        workspace_id: "w",
        server_name: "drive",
        return_to: "/tools/mcp-servers",
      }),
    )
    markSignInWindow("another-state")

    renderCallback({ state: "the-state", code: "the-code" })

    expect(await screen.findByText("Server list")).toBeVisible()
  })

  it("spends the code once and lands where the sign-in was started from", async () => {
    const calls = mockComplete(() =>
      Response.json({
        server_id: SERVER_ID,
        workspace_id: "w",
        server_name: "drive",
        return_to: "/playground",
      }),
    )
    const { rerender } = renderCallback({
      state: "the-state",
      code: "the-code",
    })
    rerender(
      <QueryClientProvider client={new QueryClient()}>
        <McpConnectionCallbackPage
          search={{ state: "the-state", code: "the-code" }}
        />
      </QueryClientProvider>,
    )

    expect(await screen.findByText("Playground")).toBeVisible()
    // A sign-in that worked says nothing: the page it lands on shows it.
    expect(readMcpConnectionNotice()).toBeUndefined()
    expect(calls).toHaveLength(1)
    expect(calls[0].url).toContain("/mcp-connections/complete")
    expect(calls[0].body).toEqual({ state: "the-state", code: "the-code" })
  })

  it("sends the authorization server's iss along with the code", async () => {
    const calls = mockComplete(() =>
      Response.json({
        server_id: SERVER_ID,
        workspace_id: "w",
        server_name: "drive",
        return_to: null,
      }),
    )
    renderCallback({ state: "s", code: "c", iss: "https://auth.example.com" })

    expect(await screen.findByText("Server page")).toBeVisible()
    expect(calls[0].body).toEqual({
      state: "s",
      code: "c",
      iss: "https://auth.example.com",
    })
  })

  it("lands on the server's page when the sign-in named nowhere, or somewhere outside the dashboard", async () => {
    mockComplete(() =>
      Response.json({
        server_id: SERVER_ID,
        workspace_id: "w",
        server_name: "drive",
        return_to: "//evil.example/",
      }),
    )
    renderCallback({ state: "s", code: "c" })

    expect(await screen.findByText("Server page")).toBeVisible()
  })

  it("shows a refusal and offers the way back, without retrying", async () => {
    const calls = mockComplete(() =>
      Response.json(
        { detail: "This sign-in was started by another person." },
        { status: 400 },
      ),
    )
    renderCallback({ state: "s", code: "c" })

    expect(await screen.findByText(/started by another person/)).toBeVisible()
    expect(screen.getByRole("link", { name: /MCP servers/ })).toBeVisible()
    expect(calls).toHaveLength(1)
  })

  it("goes back to the list with a notice when the person cancels at the provider", async () => {
    const calls = mockComplete(() => Response.json({}))
    renderCallback({ error: "access_denied", state: "s" })

    expect(await screen.findByText("Server list")).toBeVisible()
    expect(readMcpConnectionNotice()).toEqual({
      tone: "info",
      message: CANCELLED_NOTICE,
    })
    expect(calls).toHaveLength(0)
  })

  it("goes back to the page the sign-in started from when it is cancelled", async () => {
    window.sessionStorage.setItem(
      "otari.mcp-sign-in.origin",
      `/tools/mcp-servers/${SERVER_ID}`,
    )
    mockComplete(() => Response.json({}))
    renderCallback({ error: "access_denied", state: "s" })

    expect(await screen.findByText("Server page")).toBeVisible()
    expect(window.sessionStorage.getItem("otari.mcp-sign-in.origin")).toBeNull()
  })

  it("ignores a noted origin that would leave the dashboard", async () => {
    window.sessionStorage.setItem("otari.mcp-sign-in.origin", "//evil.example/")
    mockComplete(() => Response.json({}))
    renderCallback({ error: "access_denied", state: "s" })

    expect(await screen.findByText("Server list")).toBeVisible()
  })

  it("keeps explaining a refusal that is not a cancel", async () => {
    mockComplete(() => Response.json({}))
    renderCallback({ error: "unknown_state", state: "s" })

    expect(await screen.findByText(/unknown, already finished/)).toBeVisible()
  })

  it.each(["constructor", "__proto__", "toString"])(
    "shows the generic refusal for an error named %s rather than crashing",
    async (error) => {
      mockComplete(() => Response.json({}))
      renderCallback({ error, state: "s" })

      expect(
        await screen.findByText(/did not finish, so nothing was connected/),
      ).toBeVisible()
    },
  )

  it("goes straight on for a sign-in the gateway already finished", async () => {
    const calls = mockComplete(() => Response.json({}))
    renderCallback({ connected: SERVER_ID })

    await waitFor(() => expect(screen.getByText("Server page")).toBeVisible())
    expect(calls).toHaveLength(0)
  })
})
