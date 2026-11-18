import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { OrganizationMcpServerWorkspace } from "@/client"
import { OrganizationMcpServerDetailPage } from "@/features/organization/mcpServers/OrganizationMcpServerDetailPage"
import * as apiClient from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  bootstrap,
  organizationContext,
  workspaceMcpServer,
  workspaceMcpServerHealth,
} from "@/tests/fixtures"
import { withRouter } from "@/tests/router"
import { pressRowAction, rowMenuItemNames } from "@/tests/rowMenu"

const SERVER = workspaceMcpServer({
  scope: "organization",
  workspace_id: null,
  read_tools: ["list_issues"],
  health: workspaceMcpServerHealth({ tools: ["list_issues", "get_issue"] }),
})
const PATH = `/organizations/me/mcp-servers/${SERVER.id}`

const ALPHA: OrganizationMcpServerWorkspace = {
  workspace_id: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
  name: "Alpha",
  reached: true,
  organization_tools: null,
  workspace_enabled: true,
  narrowed: false,
}
const BETA: OrganizationMcpServerWorkspace = {
  workspace_id: "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
  name: "Beta",
  reached: false,
  organization_tools: [],
  workspace_enabled: true,
  narrowed: true,
}

interface Call {
  path: string
  method: string
  body: Record<string, unknown> | undefined
}

function mockApi(): Call[] {
  const calls: Call[] = []
  vi.spyOn(apiClient, "apiFetch").mockImplementation(async (path, init) => {
    const method = (init?.method ?? "GET").toUpperCase()
    const body =
      typeof init?.body === "string"
        ? (JSON.parse(init.body) as Record<string, unknown>)
        : undefined
    calls.push({ path, method, body })
    if (path.startsWith(`${PATH}/workspaces/`)) {
      const row = path.endsWith(ALPHA.workspace_id) ? ALPHA : BETA
      return {
        ...row,
        organization_tools: method === "PUT" ? body?.tools : null,
      } as never
    }
    if (path === `${PATH}/workspaces`) {
      return { data: [ALPHA, BETA], count: 2 } as never
    }
    if (path === PATH) return SERVER as never
    if (path === "/organizations/me") {
      return organizationContext({ role: "admin" }) as never
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
        <OrganizationMcpServerDetailPage serverId={SERVER.id} />
      </QueryClientProvider>
    </DeploymentProvider>,
    {
      wrapper: withRouter({ url: `/organization/mcp-servers/${SERVER.id}` }),
    },
  )
}

async function workspacesTable() {
  const table = await screen.findByRole("grid", {
    name: `Workspaces reaching ${SERVER.name}`,
  })
  await within(table).findByText("Alpha")
  return table
}

const reads = (calls: Call[], path: string) =>
  calls.filter((call) => call.method === "GET" && call.path === path).length

describe("OrganizationMcpServerDetailPage", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("shows the server's account, tools, workspaces and connection, and nothing per workspace", async () => {
    mockApi()
    renderPage()

    await screen.findByRole("heading", { level: 1, name: SERVER.name })
    expect(
      screen
        .getAllByRole("heading", { level: 2 })
        .map((heading) => heading.textContent),
    ).toEqual(
      expect.arrayContaining([
        "Shared account",
        "Exposed tools",
        expect.stringMatching(/^Workspaces/),
        "Connection",
      ]),
    )
    // Access rules belong to each workspace that reaches the server.
    expect(screen.queryByRole("heading", { name: "Access rules" })).toBeNull()
    expect(
      screen.getByText(
        "Shared: everyone in every workspace of your organization uses one account on this server.",
        { exact: false },
      ),
    ).toBeVisible()
  })

  it("lists each workspace with whether it reaches the server and the limit set for it", async () => {
    mockApi()
    renderPage()

    const table = await workspacesTable()
    const alpha = within(table).getByRole("row", { name: /Alpha/ })
    expect(alpha).toHaveTextContent("No limit")
    expect(alpha).toHaveTextContent("As set here")
    const beta = within(table).getByRole("row", { name: /Beta/ })
    expect(beta).toHaveTextContent("Turned off")
    expect(beta).toHaveTextContent("Narrowed by its admins")
  })

  it("sets a workspace's limit through PUT and reads the table again", async () => {
    const calls = mockApi()
    const user = userEvent.setup()
    renderPage()
    const table = await workspacesTable()
    expect(reads(calls, `${PATH}/workspaces`)).toBe(1)

    await pressRowAction(user, "Set the limit for Alpha", table)
    const dialog = await screen.findByRole("dialog", {
      name: "Limit for Alpha",
    })
    await user.click(
      within(dialog).getByRole("radio", { name: /Read tools only/ }),
    )
    await user.click(within(dialog).getByRole("button", { name: "Save limit" }))

    await waitFor(() =>
      expect(calls.filter((call) => call.method === "PUT")).toEqual([
        {
          path: `${PATH}/workspaces/${ALPHA.workspace_id}`,
          method: "PUT",
          body: { tools: ["@read"] },
        },
      ]),
    )
    await waitFor(() => expect(reads(calls, `${PATH}/workspaces`)).toBe(2))
    await waitFor(() =>
      expect(
        screen.queryByRole("dialog", { name: "Limit for Alpha" }),
      ).toBeNull(),
    )
  })

  it("turns the server off in a workspace with an empty limit", async () => {
    const calls = mockApi()
    const user = userEvent.setup()
    renderPage()
    const table = await workspacesTable()

    await pressRowAction(user, "Set the limit for Alpha", table)
    const dialog = await screen.findByRole("dialog", {
      name: "Limit for Alpha",
    })
    await user.click(within(dialog).getByRole("radio", { name: "Deny" }))
    await user.click(within(dialog).getByRole("button", { name: "Save limit" }))

    await waitFor(() =>
      expect(calls.find((call) => call.method === "PUT")?.body).toEqual({
        tools: [],
      }),
    )
  })

  it("lifts a workspace's limit through DELETE and reads the table again", async () => {
    const calls = mockApi()
    const user = userEvent.setup()
    renderPage()
    const table = await workspacesTable()

    await pressRowAction(user, "Lift the limit for Beta", table)

    await waitFor(() =>
      expect(calls.filter((call) => call.method === "DELETE")).toEqual([
        {
          path: `${PATH}/workspaces/${BETA.workspace_id}`,
          method: "DELETE",
          body: undefined,
        },
      ]),
    )
    await waitFor(() => expect(reads(calls, `${PATH}/workspaces`)).toBe(2))
  })

  it("offers no lift where no limit is set", async () => {
    mockApi()
    const user = userEvent.setup()
    renderPage()
    const table = await workspacesTable()

    expect(
      await rowMenuItemNames(
        user,
        within(table).getByRole("row", { name: /Alpha/ }),
      ),
    ).toEqual(["Set the limit for Alpha"])
  })

  it("saves its settings through the organization's endpoint", async () => {
    const calls = mockApi()
    const user = userEvent.setup()
    renderPage()
    await screen.findByRole("heading", { level: 1, name: SERVER.name })

    await user.click(screen.getAllByRole("button", { name: /Settings/ })[0])
    const dialog = await screen.findByRole("dialog", {
      name: `Settings for ${SERVER.name}`,
    })
    const hint = within(dialog).getByLabelText(/^Purpose hint/)
    await user.type(hint, "Issues")
    await user.click(within(dialog).getByRole("button", { name: "Save" }))

    await waitFor(() =>
      expect(calls.filter((call) => call.method === "PATCH")).toEqual([
        expect.objectContaining({
          path: PATH,
          body: expect.objectContaining({ purpose_hint: "Issues" }),
        }),
      ]),
    )
  })
})
