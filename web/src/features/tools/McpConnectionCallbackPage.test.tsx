import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import {
  McpConnectionCallbackPage,
  type McpConnectionCallbackSearch,
} from "@/features/tools/McpConnectionCallbackPage"
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

  it("sends nothing when the authorization server refused", async () => {
    const calls = mockComplete(() => Response.json({}))
    renderCallback({ error: "access_denied", state: "s" })

    expect(await screen.findByText(/declined/)).toBeVisible()
    expect(calls).toHaveLength(0)
  })

  it("goes straight on for a sign-in the gateway already finished", async () => {
    const calls = mockComplete(() => Response.json({}))
    renderCallback({ connected: SERVER_ID })

    await waitFor(() => expect(screen.getByText("Server page")).toBeVisible())
    expect(calls).toHaveLength(0)
  })
})
