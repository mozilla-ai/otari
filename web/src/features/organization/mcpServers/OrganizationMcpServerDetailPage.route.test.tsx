import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import {
  createMemoryHistory,
  createRootRoute,
  createRoute,
  createRouter,
  RouterProvider,
} from "@tanstack/react-router"
import { act, render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { Route } from "@/routes/organization.mcp-servers.$serverId"
import * as apiClient from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  bootstrap,
  organizationContext,
  workspaceMcpServer,
} from "@/tests/fixtures"

const FIRST = workspaceMcpServer({
  scope: "organization",
  workspace_id: null,
  name: "linear",
})
const SECOND = workspaceMcpServer({
  scope: "organization",
  workspace_id: null,
  id: "66666666-6666-6666-6666-666666666666",
  name: "notion",
})

function mockApi() {
  vi.spyOn(apiClient, "apiFetch").mockImplementation(async (path) => {
    if (path === "/organizations/me") {
      return organizationContext({ role: "admin" }) as never
    }
    if (path.endsWith("/workspaces")) return { data: [], count: 0 } as never
    for (const server of [FIRST, SECOND]) {
      if (path === `/organizations/me/mcp-servers/${server.id}`) {
        return server as never
      }
    }
    throw new apiClient.ApiError(404, `Not mocked: ${path}`)
  })
}

// The route's own component, mounted at its own path, so what is tested is
// how the route renders the page rather than the page alone. Its component is
// split into a chunk of its own, so the first load waits for that chunk too.
async function renderRoute(serverId: string) {
  const root = createRootRoute()
  const page = createRoute({
    getParentRoute: () => root,
    path: "/organization/mcp-servers/$serverId",
    component: Route.options.component,
  })
  const router = createRouter({
    routeTree: root.addChildren([page]),
    history: createMemoryHistory({
      initialEntries: [`/organization/mcp-servers/${serverId}`],
    }),
  })
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  render(
    <DeploymentProvider value={bootstrap()}>
      <QueryClientProvider client={client}>
        <RouterProvider router={router} />
      </QueryClientProvider>
    </DeploymentProvider>,
  )
  await act(() => router.load())
  return router
}

describe("the organization's MCP server route", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("starts afresh on another server, so a dialog opened on one never saves to the next", async () => {
    mockApi()
    const user = userEvent.setup()
    const router = await renderRoute(FIRST.id)
    await screen.findByRole("heading", { level: 1, name: FIRST.name })

    await user.click(screen.getByRole("button", { name: "Settings" }))
    await screen.findByRole("dialog", { name: `Settings for ${FIRST.name}` })

    // As Back or Forward through history moves to another server.
    await act(() =>
      router.navigate({
        to: "/organization/mcp-servers/$serverId",
        params: { serverId: SECOND.id },
      }),
    )

    await screen.findByRole("heading", { level: 1, name: SECOND.name })
    expect(screen.queryByRole("dialog")).toBeNull()
  })
})
