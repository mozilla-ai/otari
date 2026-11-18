import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import {
  createMemoryHistory,
  createRootRoute,
  createRoute,
  createRouter,
  RouterProvider,
} from "@tanstack/react-router"
import { act, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { Route } from "@/routes/tools.mcp-servers.$serverId"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap, workspaceMcpServer } from "@/tests/fixtures"
import { mockMcpServerApi, WORKSPACE } from "@/tests/mcpServer"

const SERVER = workspaceMcpServer({ workspace_id: WORKSPACE })
const OTHER = workspaceMcpServer({
  id: "66666666-6666-6666-6666-666666666666",
  workspace_id: WORKSPACE,
  name: "linear",
})

// The route's own component, mounted at its own path, so what is tested is
// how the route renders the page rather than the page alone. Its component is
// split into a chunk of its own, so the first load waits for that chunk too.
async function renderRoute(serverId: string) {
  const root = createRootRoute()
  const page = createRoute({
    getParentRoute: () => root,
    path: "/tools/mcp-servers/$serverId",
    validateSearch: Route.options.validateSearch,
    component: Route.options.component,
  })
  const router = createRouter({
    routeTree: root.addChildren([page]),
    history: createMemoryHistory({
      initialEntries: [`/tools/mcp-servers/${serverId}`],
    }),
  })
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  render(
    <DeploymentProvider value={bootstrap()}>
      <QueryClientProvider client={client}>
        <SelectedWorkspaceProvider>
          <RouterProvider router={router} />
        </SelectedWorkspaceProvider>
      </QueryClientProvider>
    </DeploymentProvider>,
  )
  await act(() => router.load())
  return router
}

describe("a workspace's MCP server route", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("starts afresh on another server, so a dialog opened on one never saves to the next", async () => {
    mockMcpServerApi({ server: SERVER })
    // The stock mock answers any id in the workspace with `SERVER`, so the
    // second server's own read is answered here with a row of its own.
    const answer = vi.mocked(globalThis.fetch).getMockImplementation()
    vi.mocked(globalThis.fetch).mockImplementation(async (input, init) =>
      String(input).endsWith(`/mcp-servers/${OTHER.id}`)
        ? Response.json(OTHER)
        : (answer as typeof fetch)(input, init),
    )
    const user = userEvent.setup()
    const router = await renderRoute(SERVER.id)
    await screen.findByRole("heading", { level: 1, name: SERVER.name })

    await user.click(screen.getByRole("button", { name: "Settings" }))
    await screen.findByRole("dialog", { name: `Settings for ${SERVER.name}` })

    // As Back or Forward through history moves to another server.
    await act(() =>
      router.navigate({
        to: "/tools/mcp-servers/$serverId",
        params: { serverId: OTHER.id },
        search: {},
      }),
    )

    // Settled on the new id: its read answered and the heading is back. A
    // dialog still open would hide the heading from this query.
    await waitFor(() =>
      expect(router.state.location.pathname).toBe(
        `/tools/mcp-servers/${OTHER.id}`,
      ),
    )
    await screen.findByRole("heading", { level: 1, name: OTHER.name })
    expect(screen.queryByRole("dialog")).toBeNull()
  })
})
