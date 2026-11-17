import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import {
  apiKey,
  workspaceMcpServerGrant as grant,
  workspaceMcpServerHealth as health,
  organizationMember,
  workspaceMcpServer,
} from "@/tests/fixtures"
import {
  mcpWrites,
  mockMcpServerApi,
  renderMcpServerPage,
  WORKSPACE,
} from "@/tests/mcpServer"
import { pickOption } from "@/tests/select"

const SERVER = workspaceMcpServer({
  workspace_id: WORKSPACE,
  purpose_hint: "Issues for our repos",
  health: health({ tools: ["list_issues", "get_issue", "create_issue"] }),
})
const BO = "bo-0000"
const ALICE = "alice-000"
const MEMBERS = [
  organizationMember({
    user_id: "identity-bo",
    attribution_user_id: BO,
    full_name: "Bo",
  }),
  organizationMember({
    organization_member_id: "m-alice",
    user_id: "identity-alice",
    attribution_user_id: ALICE,
    full_name: "Alice",
  }),
]
const BO_LAPTOP = apiKey({
  id: "key-bo",
  key_name: "Bo's laptop",
  user_id: BO,
  workspace_id: WORKSPACE,
})

const path = `/workspaces/${WORKSPACE}/mcp-servers/${SERVER.id}`

async function openPage() {
  renderMcpServerPage(SERVER.id)
  await screen.findByRole("heading", { level: 1, name: "github" })
}

describe("McpServerDetailPage, Access", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("opens on the server's name, what it is for and where it is", async () => {
    mockMcpServerApi({ server: SERVER })
    await openPage()

    expect(screen.getByText(/Issues for our repos/)).toBeVisible()
    expect(screen.getByText(SERVER.url)).toBeVisible()
    expect(
      screen.getByRole("button", { name: /Availability/ }),
    ).toHaveTextContent("On request")
    // Its id is at the top, with how a request asks for it.
    expect(screen.getByText(SERVER.id)).toBeVisible()
    expect(
      screen.getByRole("button", { name: `Copy id for ${SERVER.name}` }),
    ).toBeVisible()
    expect(screen.getByText("mcp_server_ids").parentElement).toHaveTextContent(
      "Add this id to mcp_server_ids in your request.",
    )
    // One page, in the order a call meets the server.
    expect(
      screen
        .getAllByRole("heading", { level: 2 })
        .map((heading) => heading.textContent),
    ).toEqual(
      expect.arrayContaining([
        "Exposed tools",
        "Access rules",
        "Trace a call",
        "Connection",
      ]),
    )
    // The mode is said at the top, beside the server's state.
    expect(screen.getByText("SHARED")).toBeVisible()
    // What the rules are for, without a connected-users list to point at on a Shared server.
    expect(
      screen.getByText(
        /A rule can name any member or API key, including people who have not connected yet\.$/,
      ),
    ).toBeVisible()
  })

  it("explains each Availability choice, in the list and behind the icon", async () => {
    mockMcpServerApi({ server: SERVER })
    const user = userEvent.setup()
    await openPage()

    await user.click(
      screen.getByRole("button", { name: "Explain the three choices" }),
    )
    const tip = await screen.findByRole("tooltip")
    for (const line of [
      "Off: Nobody can use it.",
      "On request: Available to people building things, invisible to coding agents.",
      "Every request: Available everywhere.",
    ]) {
      expect(tip).toHaveTextContent(line)
    }

    await user.click(screen.getByRole("button", { name: /Availability/ }))
    expect(
      await screen.findByRole("option", { name: "On request" }),
    ).toHaveAccessibleDescription(
      "Available to people building things, invisible to coding agents",
    )
    expect(
      screen.getByRole("option", { name: "Off" }),
    ).not.toHaveAccessibleDescription()
  })

  it("edits the default policy to allow specific tools", async () => {
    const calls = mockMcpServerApi({ server: SERVER })
    const user = userEvent.setup()
    await openPage()

    await user.click(
      await screen.findByRole("button", { name: "Edit the default policy" }),
    )
    const dialog = await screen.findByRole("dialog", {
      name: "Edit the default policy",
    })
    const group = within(dialog).getByRole("radiogroup", { name: "Action" })
    expect(
      within(group).getByRole("radio", { name: "Allow any tool" }),
    ).toBeChecked()
    await user.click(
      within(group).getByRole("radio", { name: "Allow specific tools…" }),
    )
    const save = within(dialog).getByRole("button", { name: "Save rule" })
    // Specific tools with nothing ticked is not a rule anybody meant.
    expect(save).toBeDisabled()
    await user.click(
      within(dialog).getByRole("checkbox", { name: "get_issue" }),
    )
    await user.click(
      within(dialog).getByRole("checkbox", { name: "list_issues" }),
    )
    await user.click(save)

    await waitFor(() =>
      expect(
        mcpWrites(calls).find(
          (call) => call.method === "PATCH" && call.url.endsWith(path),
        )?.body,
      ).toEqual({ everyone_tools: ["get_issue", "list_issues"] }),
    )
  })

  it("sets the default policy to the tools labeled read", async () => {
    const calls = mockMcpServerApi({ server: SERVER })
    const user = userEvent.setup()
    await openPage()

    await user.click(
      await screen.findByRole("button", { name: "Edit the default policy" }),
    )
    const dialog = await screen.findByRole("dialog", {
      name: "Edit the default policy",
    })
    // Nothing is labeled on this server, which the choice says before it is made.
    expect(
      within(dialog).getByText(
        "Allows nothing yet: no tool is labeled read on this server.",
      ),
    ).toBeVisible()
    await user.click(
      within(dialog).getByRole("radio", { name: /Read tools only/ }),
    )
    await user.click(within(dialog).getByRole("button", { name: "Save rule" }))

    await waitFor(() =>
      expect(
        mcpWrites(calls).find((call) => call.method === "PATCH")?.body,
      ).toEqual({ everyone_tools: ["@read"] }),
    )
  })

  it("sets a default deny policy with an empty list", async () => {
    const calls = mockMcpServerApi({ server: SERVER })
    const user = userEvent.setup()
    await openPage()

    await user.click(
      await screen.findByRole("button", { name: "Edit the default policy" }),
    )
    const dialog = await screen.findByRole("dialog", {
      name: "Edit the default policy",
    })
    await user.click(within(dialog).getByRole("radio", { name: "Deny" }))
    await user.click(within(dialog).getByRole("button", { name: "Save rule" }))

    await waitFor(() =>
      expect(
        mcpWrites(calls).find((call) => call.method === "PATCH")?.body,
      ).toEqual({ everyone_tools: [] }),
    )
  })

  it("lists the access rules in match order: key rules, user rules, the default policy", async () => {
    mockMcpServerApi({
      server: SERVER,
      members: MEMBERS,
      access: {
        everyone_tools: [],
        grants: [
          grant({
            id: "g-key",
            api_key_id: "key-bo",
            key_name: "Bo's laptop",
            key_active: false,
            user_id: BO,
            tools: ["get_issue", "list_issues"],
          }),
          grant({
            id: "g-user",
            subject: "user",
            api_key_id: null,
            key_name: null,
            key_active: null,
            user_id: ALICE,
            tools: [],
          }),
        ],
      },
    })
    await openPage()

    const table = await screen.findByRole("grid", { name: "Access rules" })
    expect(await within(table).findByText("Bo's laptop")).toBeVisible()
    expect(within(table).getByText("key of Bo")).toBeVisible()
    expect(within(table).getByText("inactive")).toBeVisible()
    expect(within(table).getByText("get_issue, list_issues")).toBeVisible()
    expect(within(table).getByText("Alice")).toBeVisible()
    expect(within(table).getByText("user, all their keys")).toBeVisible()
    expect(within(table).getByText("Default policy")).toBeVisible()
    const rows = within(table).getAllByRole("row").slice(1)
    expect(rows.map((row) => row.textContent)).toEqual([
      expect.stringMatching(/^Bo's laptop.*ALLOW/),
      expect.stringMatching(/^Alice.*DENY/),
      expect.stringMatching(/^Default policy.*ALLOW/),
    ])
    expect(screen.getByText(/The most specific rule wins/)).toBeVisible()
  })

  it("shows only the default policy when there are no rules", async () => {
    mockMcpServerApi({ server: SERVER })
    await openPage()

    const table = await screen.findByRole("grid", { name: "Access rules" })
    expect(within(table).getByText("Default policy")).toBeVisible()
    expect(within(table).getAllByRole("row")).toHaveLength(2)
    expect(
      within(table).queryByRole("button", { name: /^Delete/ }),
    ).not.toBeInTheDocument()
  })

  it("adds a key and a user with the tools picked", async () => {
    const calls = mockMcpServerApi({
      server: SERVER,
      members: MEMBERS,
      keys: [BO_LAPTOP],
      written: {
        everyone_tools: ["*"],
        grants: [grant({ id: "g-new", api_key_id: "key-bo", user_id: BO })],
      },
    })
    const user = userEvent.setup()
    await openPage()

    await user.click(await screen.findByRole("button", { name: "Add rule" }))
    const dialog = await screen.findByRole("dialog", { name: "Add rule" })
    await user.click(
      await within(dialog).findByRole("checkbox", {
        name: "Key Bo's laptop of Bo",
      }),
    )
    await user.click(
      within(dialog).getByRole("checkbox", {
        name: "Alice, all their keys",
      }),
    )
    await user.click(
      within(dialog).getByRole("radio", { name: "Allow specific tools…" }),
    )
    await user.click(
      within(dialog).getByRole("checkbox", { name: "get_issue" }),
    )
    await user.click(within(dialog).getByRole("button", { name: "Add rule" }))

    await waitFor(() =>
      expect(
        mcpWrites(calls).find((call) =>
          call.url.endsWith(`${path}/access/grants`),
        )?.body,
      ).toEqual({
        api_key_ids: ["key-bo"],
        user_ids: [ALICE],
        tools: ["get_issue"],
      }),
    )
    // The answer is the whole list, which the table shows without a refetch.
    expect(
      await within(
        await screen.findByRole("grid", { name: "Access rules" }),
      ).findByText("ci-bot"),
    ).toBeVisible()
  })

  it("finds a user or key by name in the picker", async () => {
    mockMcpServerApi({ server: SERVER, members: MEMBERS, keys: [BO_LAPTOP] })
    const user = userEvent.setup()
    await openPage()

    await user.click(await screen.findByRole("button", { name: "Add rule" }))
    const dialog = await screen.findByRole("dialog", { name: "Add rule" })
    await within(dialog).findByRole("checkbox", {
      name: "Key Bo's laptop of Bo",
    })
    await user.type(
      within(dialog).getByRole("searchbox", { name: "Source" }),
      "alice",
    )

    await waitFor(() =>
      expect(
        within(dialog).queryByRole("checkbox", {
          name: "Key Bo's laptop of Bo",
        }),
      ).not.toBeInTheDocument(),
    )
    expect(
      within(dialog).getByRole("checkbox", { name: "Alice, all their keys" }),
    ).toBeVisible()
  })

  it("offers an organization admin outside the workspace, whom the Playground lets in", async () => {
    mockMcpServerApi({
      server: SERVER,
      members: [
        ...MEMBERS,
        organizationMember({
          organization_member_id: "m-dana",
          user_id: "identity-dana",
          attribution_user_id: null,
          full_name: "Dana",
          role: "admin",
        }),
        organizationMember({
          organization_member_id: "m-eve",
          user_id: "identity-eve",
          attribution_user_id: null,
          full_name: "Eve",
          role: "member",
        }),
      ],
      workspaceMemberIds: ["identity-bo", "identity-alice"],
      keys: [BO_LAPTOP],
    })
    const user = userEvent.setup()
    await openPage()

    await user.click(await screen.findByRole("button", { name: "Add rule" }))
    const dialog = await screen.findByRole("dialog", { name: "Add rule" })

    expect(
      await within(dialog).findByRole("checkbox", { name: /^Dana/ }),
    ).toBeVisible()
    expect(
      within(dialog).queryByRole("checkbox", { name: /^Eve/ }),
    ).not.toBeInTheDocument()
  })

  it("edits one rule", async () => {
    const calls = mockMcpServerApi({
      server: SERVER,
      members: MEMBERS,
      access: { everyone_tools: ["*"], grants: [grant({ id: "g-1" })] },
    })
    const user = userEvent.setup()
    await openPage()

    await user.click(
      await screen.findByRole("button", { name: "Edit the rule for ci-bot" }),
    )
    const dialog = await screen.findByRole("dialog", { name: "Edit ci-bot" })
    await user.click(within(dialog).getByRole("radio", { name: "Deny" }))
    await user.click(within(dialog).getByRole("button", { name: "Save rule" }))

    await waitFor(() =>
      expect(
        mcpWrites(calls).find(
          (call) =>
            call.method === "PATCH" &&
            call.url.endsWith(`${path}/access/grants/g-1`),
        )?.body,
      ).toEqual({ tools: [] }),
    )
  })

  it("deletes a rule once confirmed", async () => {
    const calls = mockMcpServerApi({
      server: SERVER,
      access: { everyone_tools: ["*"], grants: [grant({ id: "g-1" })] },
      written: { everyone_tools: ["*"], grants: [] },
    })
    const user = userEvent.setup()
    await openPage()

    await user.click(
      await screen.findByRole("button", {
        name: "Delete the rule for ci-bot",
      }),
    )
    const confirm = await screen.findByRole("alertdialog")
    expect(
      within(confirm).getByText(/matches the rule for its owner/),
    ).toBeVisible()
    await user.click(
      within(confirm).getByRole("button", { name: "Delete rule" }),
    )

    await waitFor(() =>
      expect(
        mcpWrites(calls).some(
          (call) =>
            call.method === "DELETE" &&
            call.url.endsWith(`${path}/access/grants/g-1`),
        ),
      ).toBe(true),
    )
    await waitFor(() =>
      expect(
        within(screen.getByRole("grid", { name: "Access rules" })).queryByText(
          "ci-bot",
        ),
      ).not.toBeInTheDocument(),
    )
  })

  it("traces one key's call to one tool and says which rule matched", async () => {
    const calls = mockMcpServerApi({
      server: SERVER,
      members: MEMBERS,
      keys: [BO_LAPTOP],
      access: { everyone_tools: [], grants: [] },
      check: {
        allowed: false,
        decided_by: "everyone",
        grant_id: null,
        tools: [],
      },
    })
    const user = userEvent.setup()
    await openPage()

    await pickOption(user, "Source key", "Bo's laptop (Bo)")
    await user.type(screen.getByRole("combobox", { name: "Tool" }), "create")
    await user.click(
      await screen.findByRole("option", { name: "create_issue" }),
    )
    await user.click(screen.getByRole("button", { name: "Trace" }))

    expect(
      await screen.findByText(/DENY, matched the default policy \(deny\)/),
    ).toBeVisible()
    expect(screen.getByRole("status")).toHaveTextContent(
      "Bo's laptop (Bo) · create_issue → DENY, matched the default policy (deny)",
    )
    expect(
      calls.some((call) =>
        call.url.endsWith(
          `${path}/access/check?api_key_id=key-bo&tool=create_issue`,
        ),
      ),
    ).toBe(true)
  })

  it("traces the call a session links to, on arrival", async () => {
    const calls = mockMcpServerApi({
      server: SERVER,
      members: MEMBERS,
      keys: [BO_LAPTOP],
      access: { everyone_tools: [], grants: [] },
      check: {
        allowed: false,
        decided_by: "everyone",
        grant_id: null,
        tools: [],
      },
    })
    renderMcpServerPage(SERVER.id, "?key=key-bo&tool=create_issue")

    await screen.findByText(/DENY, matched the default policy \(deny\)/)
    expect(screen.getByRole("status")).toHaveTextContent(
      "Bo's laptop (Bo) · create_issue → DENY, matched the default policy (deny)",
    )
    expect(
      calls.some((call) =>
        call.url.endsWith(
          `${path}/access/check?api_key_id=key-bo&tool=create_issue`,
        ),
      ),
    ).toBe(true)
  })

  it("names the rule that allowed a key", async () => {
    mockMcpServerApi({
      server: SERVER,
      members: MEMBERS,
      keys: [BO_LAPTOP],
      access: {
        everyone_tools: [],
        grants: [
          grant({
            id: "g-bo",
            api_key_id: "key-bo",
            key_name: "Bo's laptop",
            user_id: BO,
          }),
        ],
      },
      check: {
        allowed: true,
        decided_by: "key",
        grant_id: "g-bo",
        tools: ["*"],
      },
    })
    const user = userEvent.setup()
    await openPage()

    await pickOption(user, "Source key", "Bo's laptop (Bo)")
    await user.type(screen.getByRole("combobox", { name: "Tool" }), "get_is")
    await user.click(await screen.findByRole("option", { name: "get_issue" }))
    await user.click(screen.getByRole("button", { name: "Trace" }))

    expect(
      await screen.findByText(/ALLOW, matched the rule for Bo's laptop/),
    ).toBeVisible()
  })

  it("reads the server from the workspace a trace links it in, not the selected one", async () => {
    const other = "66666666-6666-6666-6666-666666666666"
    const calls = mockMcpServerApi({
      server: workspaceMcpServer({ workspace_id: other }),
      role: "member",
      workspaceRole: "member",
      otherWorkspaces: [{ workspace_id: other, name: "Beta", role: "admin" }],
    })
    renderMcpServerPage(SERVER.id, "", other)

    await screen.findByRole("heading", { level: 1, name: "github" })
    expect(
      calls.some((call) =>
        call.url.endsWith(`/workspaces/${other}/mcp-servers/${SERVER.id}`),
      ),
    ).toBe(true)
    // Managing follows the role in the linked workspace, not the selected one.
    expect(screen.getByRole("button", { name: /Availability/ })).toBeVisible()
  })

  it("does not find a server of another workspace without a link naming it", async () => {
    mockMcpServerApi({
      server: workspaceMcpServer({
        workspace_id: "66666666-6666-6666-6666-666666666666",
      }),
    })
    renderMcpServerPage(SERVER.id)

    expect(
      await screen.findByText(`MCP server ${SERVER.id} not found`),
    ).toBeVisible()
  })

  it("shows a member none of the admin sections, and asks nothing of them", async () => {
    const calls = mockMcpServerApi({
      server: workspaceMcpServer({
        workspace_id: WORKSPACE,
        everyone_tools: [],
      }),
      role: "member",
      workspaceRole: "member",
      deploymentOperator: false,
    })
    await openPage()

    expect(
      screen
        .getAllByRole("heading", { level: 2 })
        .map((heading) => heading.textContent),
    ).toEqual(["Shared account", "Your tools"])
    expect(screen.getByText(/nothing for you to connect/)).toBeVisible()
    expect(
      await screen.findByText(/don't let you use any tool on this server/),
    ).toBeVisible()
    for (const name of ["Check", "Add rule", "Edit the default policy"]) {
      expect(screen.queryByRole("button", { name })).not.toBeInTheDocument()
    }
    // The access list is a manager's read, so it is not asked for at all.
    expect(calls.some((call) => call.url.includes("/access"))).toBe(false)
  })

  it("shows a member the tools their own rule lets them use", async () => {
    mockMcpServerApi({
      server: workspaceMcpServer({ workspace_id: WORKSPACE }),
      role: "member",
      workspaceRole: "member",
      deploymentOperator: false,
      myTools: {
        server_id: SERVER.id,
        decided_by: "user",
        access: "some",
        known: true,
        checked_at: "2026-11-10T10:00:00+00:00",
        tools: [
          {
            name: "list_issues",
            description: "List the open issues.",
            read_only: true,
          },
          { name: "create_issue", description: null, read_only: false },
        ],
      },
    })
    const user = userEvent.setup()
    await openPage()

    const list = await screen.findByRole("list", {
      name: `Your tools on ${SERVER.name}`,
    })
    expect(within(list).getByText("list_issues")).toBeVisible()
    expect(within(list).getByText("List the open issues.")).toBeVisible()
    expect(within(list).getByText("create_issue")).toBeVisible()
    // Only the tool labeled read wears the chip.
    expect(within(list).getAllByText("Read-only")).toHaveLength(1)

    // The search reads names and descriptions.
    const search = screen.getByRole("searchbox", { name: "Search your tools" })
    await user.type(search, "open issues")
    expect(within(list).getByText("list_issues")).toBeVisible()
    expect(within(list).queryByText("create_issue")).not.toBeInTheDocument()
    await user.clear(search)
    await user.type(search, "nothing like this")
    expect(
      screen.getByText('No tool matches "nothing like this".'),
    ).toBeVisible()
  })

  it("says why a member's tools are not known yet", async () => {
    mockMcpServerApi({
      server: workspaceMcpServer({
        workspace_id: WORKSPACE,
        credential_mode: "per_user",
      }),
      role: "member",
      workspaceRole: "member",
      deploymentOperator: false,
    })
    await openPage()

    expect(
      await screen.findByText(/Connect your account above to see the tools/),
    ).toBeVisible()
  })

  it("falls back to read-only when the access list is refused", async () => {
    mockMcpServerApi({ server: SERVER, accessStatus: 403 })
    await openPage()

    expect(
      await screen.findByText("Only an owner or admin can see these rules."),
    ).toBeVisible()
    expect(
      screen.queryByRole("button", { name: "Add rule" }),
    ).not.toBeInTheDocument()
  })

  it("opens a connected user's access rule from Connected users", async () => {
    const personal = workspaceMcpServer({
      workspace_id: WORKSPACE,
      credential_mode: "per_user",
      health: health({ tools: ["list_issues"] }),
    })
    mockMcpServerApi({
      server: personal,
      members: MEMBERS,
      connections: [
        {
          user_id: BO,
          full_name: "Bo",
          email: null,
          connected_at: "2026-11-01T00:00:00Z",
          updated_at: "2026-11-01T00:00:00Z",
        },
      ],
    })
    const user = userEvent.setup()
    await openPage()

    const table = await screen.findByRole("grid", {
      name: "Users connected to github",
    })
    await user.click(await within(table).findByText("Bo"))

    const dialog = await screen.findByRole("dialog", { name: "Add rule" })
    expect(
      within(dialog).getByRole("checkbox", { name: "Bo, all their keys" }),
    ).toBeChecked()
  })

  it("edits the rule a connected user already has", async () => {
    const personal = workspaceMcpServer({
      workspace_id: WORKSPACE,
      credential_mode: "per_user",
    })
    mockMcpServerApi({
      server: personal,
      members: MEMBERS,
      access: {
        everyone_tools: ["*"],
        grants: [
          grant({
            id: "g-bo",
            subject: "user",
            user_id: BO,
            api_key_id: null,
            key_name: null,
          }),
        ],
      },
      connections: [
        {
          user_id: BO,
          full_name: "Bo",
          email: null,
          connected_at: "2026-11-01T00:00:00Z",
          updated_at: "2026-11-01T00:00:00Z",
        },
      ],
    })
    const user = userEvent.setup()
    await openPage()

    const table = await screen.findByRole("grid", {
      name: "Users connected to github",
    })
    // Wait for the rules, so the click finds Bo's.
    await screen.findByRole("button", { name: "Edit the rule for Bo" })
    await user.click(within(table).getByText("Bo"))

    expect(await screen.findByRole("dialog", { name: "Edit Bo" })).toBeVisible()
  })
})
