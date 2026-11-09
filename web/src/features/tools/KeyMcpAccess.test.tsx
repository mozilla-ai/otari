import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { OrganizationContext, WorkspaceMcpServerKeyAccess } from "@/client"
import { KeyMcpAccess } from "@/features/tools/KeyMcpAccess"
import { organizationContext } from "@/tests/fixtures"
import { withRouter } from "@/tests/router"

const WORKSPACE = "44444444-4444-4444-4444-444444444444"

function serverAccess(
  overrides: Partial<WorkspaceMcpServerKeyAccess> = {},
): WorkspaceMcpServerKeyAccess {
  return {
    server_id: "55555555-5555-5555-5555-555555555555",
    name: "github",
    enabled: true,
    decided_by: "everyone",
    grant_id: null,
    tools: ["*"],
    allowed_tools: null,
    ...overrides,
  }
}

function mockApi({
  rows = [],
  status = 200,
  workspaceRole = "admin",
  role = "owner",
}: {
  rows?: WorkspaceMcpServerKeyAccess[]
  status?: number
  workspaceRole?: string
  role?: OrganizationContext["role"]
} = {}) {
  const calls: string[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input)
    calls.push(url)
    if (url.includes("/mcp-servers/keys/")) {
      return status === 200
        ? Response.json(rows)
        : Response.json({ detail: "refused" }, { status })
    }
    return Response.json(
      organizationContext({
        role,
        workspace_memberships: [
          { workspace_id: WORKSPACE, name: "Alpha", role: workspaceRole },
        ],
      }),
    )
  })
  return calls
}

function renderSection() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <KeyMcpAccess workspaceId={WORKSPACE} apiKeyId="key-1" />
    </QueryClientProvider>,
    { wrapper: withRouter({ url: "/keys" }) },
  )
}

describe("KeyMcpAccess", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("says what the key reaches on each server and which rule decided", async () => {
    mockApi({
      rows: [
        serverAccess(),
        serverAccess({
          server_id: "66666666-6666-6666-6666-666666666666",
          name: "jira",
          decided_by: "key",
          grant_id: "g-1",
          tools: ["get_issue", "list_*"],
          allowed_tools: ["get_issue", "list_issues", "create_issue"],
        }),
        serverAccess({
          server_id: "77777777-7777-7777-7777-777777777777",
          name: "payroll",
          decided_by: "user",
          grant_id: "g-2",
          tools: [],
        }),
      ],
    })
    renderSection()

    expect(
      await screen.findByRole("region", { name: "MCP access" }),
    ).toBeVisible()
    expect(screen.getByRole("link", { name: "github" })).toHaveAttribute(
      "href",
      "/tools/mcp-servers/55555555-5555-5555-5555-555555555555",
    )
    expect(screen.getByText("All tools")).toBeVisible()
    expect(screen.getByText("matched the default policy")).toBeVisible()
    // A pattern is read through the server's own ceiling.
    expect(screen.getByText("get_issue, list_issues")).toBeVisible()
    expect(screen.getByText("matched the rule for this key")).toBeVisible()
    expect(screen.getByText("No access")).toBeVisible()
    expect(screen.getByText("matched the rule for its owner")).toBeVisible()
  })

  it("stays out of the form when the workspace has no servers", async () => {
    const calls = mockApi({ rows: [] })
    renderSection()

    await waitFor(() =>
      expect(calls.some((url) => url.includes("/mcp-servers/keys/"))).toBe(
        true,
      ),
    )
    expect(screen.queryByText("MCP access")).not.toBeInTheDocument()
  })

  it("stays out of the form when the read is refused", async () => {
    const calls = mockApi({ status: 403 })
    renderSection()

    await waitFor(() =>
      expect(calls.some((url) => url.includes("/mcp-servers/keys/"))).toBe(
        true,
      ),
    )
    expect(screen.queryByText("MCP access")).not.toBeInTheDocument()
  })

  it("is not asked for by someone who does not manage the workspace", async () => {
    const calls = mockApi({ workspaceRole: "member", role: "member" })
    renderSection()

    await waitFor(() =>
      expect(calls.some((url) => url.includes("/organizations/me"))).toBe(true),
    )
    expect(calls.some((url) => url.includes("/mcp-servers/keys/"))).toBe(false)
    expect(screen.queryByText("MCP access")).not.toBeInTheDocument()
  })
})
