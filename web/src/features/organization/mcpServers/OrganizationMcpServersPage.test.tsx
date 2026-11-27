import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { OrganizationContext, WorkspaceMcpServer } from "@/client"
import { OrganizationMcpServersPage } from "@/features/organization/mcpServers/OrganizationMcpServersPage"
import * as apiClient from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  bootstrap,
  organizationContext,
  workspaceMcpServer,
  workspaceMcpServerHealth,
} from "@/tests/fixtures"
import { withRouter } from "@/tests/router"

const SERVERS = "/organizations/me/mcp-servers"

const GITHUB = workspaceMcpServer({
  scope: "organization",
  workspace_id: null,
  health: workspaceMcpServerHealth({ tools: ["list_issues", "get_issue"] }),
})

interface Call {
  path: string
  method: string
  body: Record<string, unknown> | undefined
}

function mockApi({
  servers = [GITHUB],
  role = "owner",
}: {
  servers?: WorkspaceMcpServer[]
  role?: OrganizationContext["role"]
} = {}): Call[] {
  const calls: Call[] = []
  vi.spyOn(apiClient, "apiFetch").mockImplementation(async (path, init) => {
    const method = (init?.method ?? "GET").toUpperCase()
    const body =
      typeof init?.body === "string"
        ? (JSON.parse(init.body) as Record<string, unknown>)
        : undefined
    calls.push({ path, method, body })
    if (path.startsWith(SERVERS)) {
      if (method === "POST") {
        return {
          ...GITHUB,
          id: "66666666-6666-6666-6666-666666666666",
          ...body,
        } as never
      }
      return { data: servers, count: servers.length } as never
    }
    if (path === "/organizations/me") {
      return organizationContext({ role }) as never
    }
    throw new apiClient.ApiError(404, `Not mocked: ${path}`)
  })
  return calls
}

function renderPage() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <DeploymentProvider value={bootstrap()}>
      <QueryClientProvider client={client}>
        <OrganizationMcpServersPage />
      </QueryClientProvider>
    </DeploymentProvider>,
    { wrapper: withRouter({ url: "/organization/mcp-servers" }) },
  )
}

describe("OrganizationMcpServersPage", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("lists the organization's servers with their health", async () => {
    mockApi()
    renderPage()

    const table = await screen.findByRole("grid", {
      name: "Organization MCP servers",
    })
    expect(await within(table).findByText("github")).toBeVisible()
    expect(within(table).getByText(GITHUB.url)).toBeVisible()
    expect(
      screen.getByText(
        "Servers every workspace in your organization can use. Each workspace's admins can narrow access or turn one off, never widen it.",
      ),
    ).toBeVisible()
  })

  it("adds a server through the organization's endpoint", async () => {
    const calls = mockApi({ servers: [] })
    const user = userEvent.setup()
    renderPage()

    await user.click(await screen.findByRole("button", { name: "Add server" }))
    const dialog = await screen.findByRole("dialog", { name: "New MCP server" })
    expect(
      within(dialog).getByText(
        "Connect a remote MCP server every workspace in your organization can use.",
      ),
    ).toBeVisible()
    await user.type(within(dialog).getByLabelText(/^Name/), "linear")
    await user.type(
      within(dialog).getByLabelText(/^URL/),
      "https://mcp.example.com/linear",
    )
    await user.click(
      within(dialog).getByRole("button", { name: "Add MCP server" }),
    )

    await waitFor(() =>
      expect(calls.filter((call) => call.method === "POST")).toEqual([
        expect.objectContaining({
          path: SERVERS,
          body: expect.objectContaining({
            name: "linear",
            url: "https://mcp.example.com/linear",
          }),
        }),
      ]),
    )
    // No workspace route is asked anything.
    expect(calls.some((call) => call.path.startsWith("/workspaces"))).toBe(
      false,
    )
  })

  it("tests a draft's connection through the organization's endpoint", async () => {
    const calls = mockApi({ servers: [] })
    const user = userEvent.setup()
    renderPage()

    await user.click(await screen.findByRole("button", { name: "Add server" }))
    const dialog = await screen.findByRole("dialog", { name: "New MCP server" })
    await user.type(
      within(dialog).getByLabelText(/^URL/),
      "https://mcp.example.com/linear",
    )
    await user.click(
      within(dialog).getByRole("button", { name: "Test connection" }),
    )

    await waitFor(() =>
      expect(calls.map((call) => [call.method, call.path])).toContainEqual([
        "POST",
        `${SERVERS}/check`,
      ]),
    )
  })

  it("asks a member nothing and says the page is for owners and admins", async () => {
    const calls = mockApi({ role: "member" })
    renderPage()

    expect(
      await screen.findByText(/Only organization owners and admins can see/),
    ).toBeVisible()
    expect(calls.some((call) => call.path.startsWith(SERVERS))).toBe(false)
    expect(screen.queryByRole("button", { name: "Add server" })).toBeNull()
  })
})
