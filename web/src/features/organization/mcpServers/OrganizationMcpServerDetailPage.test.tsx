import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type {
  OrganizationMcpServerWorkspace,
  WorkspaceMcpServer,
} from "@/client"
import { OrganizationMcpServerDetailPage } from "@/features/organization/mcpServers/OrganizationMcpServerDetailPage"
import * as apiClient from "@/shared/api/client"
import {
  mcpServerKey,
  ORGANIZATION_MCP_SCOPE,
} from "@/shared/api/mcpServerScope"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  bootstrap,
  organizationContext,
  workspaceMcpServer,
  workspaceMcpServerHealth,
} from "@/tests/fixtures"
import { withRouter } from "@/tests/router"
import {
  getRowMenuItem,
  pressRowAction,
  rowMenuItemNames,
} from "@/tests/rowMenu"
import { pickOption, selectTrigger } from "@/tests/select"

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

const reads = (calls: Call[], path: string) =>
  calls.filter((call) => call.method === "GET" && call.path === path).length

function mockApi({
  server = SERVER,
  lift,
  limit,
  reread,
  workspaces,
  role = "admin",
}: {
  server?: WorkspaceMcpServer
  /** Runs before a lift is answered, to hold it or refuse it. */
  lift?: () => Promise<unknown>
  /** Holds a limit's PUT until it settles. */
  limit?: Promise<void>
  /** Holds every read of the table after the first. */
  reread?: Promise<void>
  /** Runs before every read of the table, to refuse it. */
  workspaces?: () => Promise<unknown>
  role?: "owner" | "admin" | "member"
} = {}): Call[] {
  const calls: Call[] = []
  // Each workspace's row as the server holds it, so a reread answers with what was saved.
  const rows = [{ ...ALPHA }, { ...BETA }]
  vi.spyOn(apiClient, "apiFetch").mockImplementation(async (path, init) => {
    const method = (init?.method ?? "GET").toUpperCase()
    const body =
      typeof init?.body === "string"
        ? (JSON.parse(init.body) as Record<string, unknown>)
        : undefined
    calls.push({ path, method, body })
    if (method === "DELETE" && lift) await lift()
    if (method === "PUT" && limit) await limit
    if (path.startsWith(`${PATH}/workspaces/`)) {
      const row = rows.find((each) => path.endsWith(each.workspace_id))
      if (!row) throw new apiClient.ApiError(404, `Not mocked: ${path}`)
      const tools = method === "PUT" ? (body?.tools as string[]) : null
      row.organization_tools = tools
      row.reached = row.workspace_enabled && tools?.length !== 0
      return { ...row } as never
    }
    if (path === `${PATH}/workspaces`) {
      if (workspaces) await workspaces()
      if (reread && reads(calls, path) > 1) await reread
      return {
        data: rows.map((row) => ({ ...row })),
        count: rows.length,
      } as never
    }
    if (path === PATH) return server as never
    if (path === "/organizations/me") {
      return organizationContext({ role }) as never
    }
    throw new apiClient.ApiError(404, `Not mocked: ${path}`)
  })
  return calls
}

function renderPage(
  client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  }),
) {
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
    expect(beta).toHaveTextContent("Customized by its admins")
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
    // The dialog closes once the table was read again, which kept the saved limit.
    expect(within(table).getByRole("row", { name: /Alpha/ })).toHaveTextContent(
      "Read tools",
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

  it("holds a row's Set limit while its lift is in flight, until the table is read again", async () => {
    let settleReread = () => {}
    const reread = new Promise<void>((resolve) => {
      settleReread = resolve
    })
    const calls = mockApi({ reread })
    const user = userEvent.setup()
    renderPage()
    const table = await workspacesTable()

    await pressRowAction(user, "Lift the limit for Beta", table)
    await waitFor(() => expect(reads(calls, `${PATH}/workspaces`)).toBe(2))

    // The lift answered, but the table has not been read again yet.
    const beta = within(table).getByRole("row", { name: /Beta/ })
    expect(
      await getRowMenuItem(user, /its limit is being lifted/, beta),
    ).toHaveAttribute("aria-disabled", "true")
    await user.keyboard("{Escape}")
    // Only the row being lifted is held.
    const alpha = within(table).getByRole("row", { name: /Alpha/ })
    expect(
      await getRowMenuItem(user, "Set the limit for Alpha", alpha),
    ).not.toHaveAttribute("aria-disabled")
    await user.keyboard("{Escape}")

    settleReread()
    await waitFor(() =>
      expect(within(beta).getByText("No limit")).toBeVisible(),
    )
    expect(
      await getRowMenuItem(user, "Set the limit for Beta", beta),
    ).not.toHaveAttribute("aria-disabled")
  })

  it("shows nothing of the server to a viewer without the role, even from what is cached", async () => {
    mockApi({ role: "member" })
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    })
    client.setQueryData(mcpServerKey(ORGANIZATION_MCP_SCOPE, SERVER.id), SERVER)
    renderPage(client)

    expect(
      await screen.findByText(/Only organization owners and admins/),
    ).toBeVisible()
    expect(screen.queryByRole("heading", { name: SERVER.name })).toBeNull()
  })

  it("closes an untouched limit dialog without asking, though Any tool is not the stored value", async () => {
    mockApi()
    const user = userEvent.setup()
    renderPage()
    const table = await workspacesTable()

    await pressRowAction(user, "Set the limit for Alpha", table)
    const dialog = await screen.findByRole("dialog", {
      name: "Limit for Alpha",
    })
    // Alpha has no limit, so the editor opens on Any tool, which saving would set.
    expect(
      within(dialog).getByRole("button", { name: "Save limit" }),
    ).toBeEnabled()
    await user.keyboard("{Escape}")

    await waitFor(() =>
      expect(
        screen.queryByRole("dialog", { name: "Limit for Alpha" }),
      ).toBeNull(),
    )
    expect(screen.queryByRole("button", { name: "Discard" })).toBeNull()
  })

  it("holds the tool choice while a limit saves", async () => {
    let settle = () => {}
    const limit = new Promise<void>((resolve) => {
      settle = resolve
    })
    mockApi({ limit })
    const user = userEvent.setup()
    renderPage()
    const table = await workspacesTable()

    await pressRowAction(user, "Set the limit for Alpha", table)
    const dialog = await screen.findByRole("dialog", {
      name: "Limit for Alpha",
    })
    await user.click(
      within(dialog).getByRole("radio", { name: /Read tools only/ }),
    )
    await user.click(within(dialog).getByRole("button", { name: "Save limit" }))

    await waitFor(() =>
      expect(
        within(dialog).getByRole("radio", { name: "Deny" }),
      ).toBeDisabled(),
    )
    settle()
    await waitFor(() =>
      expect(
        screen.queryByRole("dialog", { name: "Limit for Alpha" }),
      ).toBeNull(),
    )
  })

  it("shows the saved limit when the table cannot be read again", async () => {
    const refused = Promise.reject(new apiClient.ApiError(500, "Read refused"))
    // Handled here so the rejection is not reported before the reread awaits it.
    refused.catch(() => {})
    mockApi({ reread: refused })
    const user = userEvent.setup()
    renderPage()
    const table = await workspacesTable()

    await pressRowAction(user, "Set the limit for Alpha", table)
    const dialog = await screen.findByRole("dialog", {
      name: "Limit for Alpha",
    })
    await user.click(
      within(dialog).getByRole("radio", { name: /Read tools only/ }),
    )
    await user.click(within(dialog).getByRole("button", { name: "Save limit" }))

    expect(await screen.findByText("Read refused")).toBeVisible()
    expect(within(table).getByRole("row", { name: /Alpha/ })).toHaveTextContent(
      "Read tools",
    )
  })

  it("shows the error and no empty table when the workspaces cannot be read", async () => {
    mockApi({
      workspaces: () =>
        Promise.reject(new apiClient.ApiError(500, "Read refused")),
    })
    renderPage()

    expect(await screen.findByText("Read refused")).toBeVisible()
    expect(
      screen.queryByText("This organization has no workspaces yet."),
    ).toBeNull()
  })

  it("keeps holding a row whose lift is in flight when another row's limit is opened and closed", async () => {
    let settleLift = () => {}
    const lifted = new Promise<void>((resolve) => {
      settleLift = resolve
    })
    mockApi({
      lift: () => lifted,
      reread: Promise.resolve(),
    })
    const user = userEvent.setup()
    renderPage()
    const table = await workspacesTable()

    await pressRowAction(user, "Lift the limit for Beta", table)
    await pressRowAction(user, "Set the limit for Alpha", table)
    await screen.findByRole("dialog", { name: "Limit for Alpha" })
    await user.keyboard("{Escape}")
    await waitFor(() =>
      expect(
        screen.queryByRole("dialog", { name: "Limit for Alpha" }),
      ).toBeNull(),
    )

    const beta = within(table).getByRole("row", { name: /Beta/ })
    expect(
      await getRowMenuItem(user, /its limit is being lifted/, beta),
    ).toHaveAttribute("aria-disabled", "true")
    await user.keyboard("{Escape}")

    settleLift()
    await waitFor(() =>
      expect(within(beta).getByText("No limit")).toBeVisible(),
    )
    expect(
      await getRowMenuItem(user, "Set the limit for Beta", beta),
    ).not.toHaveAttribute("aria-disabled")
  })

  it("keeps a row's Set limit after its lift fails", async () => {
    mockApi({
      lift: () => Promise.reject(new apiClient.ApiError(500, "Lift refused")),
    })
    const user = userEvent.setup()
    renderPage()
    const table = await workspacesTable()

    await pressRowAction(user, "Lift the limit for Beta", table)

    expect(await screen.findByText("Lift refused")).toBeVisible()
    const beta = within(table).getByRole("row", { name: /Beta/ })
    expect(
      await getRowMenuItem(user, "Set the limit for Beta", beta),
    ).not.toHaveAttribute("aria-disabled")
  })

  it("turns a server created as Off on from its Availability choice", async () => {
    const calls = mockApi({
      server: { ...SERVER, enabled: false, attach_by_default: false },
    })
    const user = userEvent.setup()
    renderPage()
    await screen.findByRole("heading", { level: 1, name: SERVER.name })

    expect(selectTrigger("Availability")).toHaveTextContent("Off")
    expect(screen.getByText("Nobody can use it.")).toBeVisible()
    await pickOption(user, selectTrigger("Availability"), /^On request/)

    await waitFor(() =>
      expect(calls.filter((call) => call.method === "PATCH")).toEqual([
        {
          path: PATH,
          method: "PATCH",
          body: { enabled: true, attach_by_default: false },
        },
      ]),
    )
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
