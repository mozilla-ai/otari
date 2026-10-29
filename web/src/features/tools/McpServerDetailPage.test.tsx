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
    expect(screen.getByText("ENABLED")).toBeVisible()
    // One page, in the order a call meets the server.
    expect(
      screen
        .getAllByRole("heading", { level: 2 })
        .map((heading) => heading.textContent),
    ).toEqual(
      expect.arrayContaining([
        "Exposed tools",
        "ACL",
        "Trace a call",
        "Connection",
      ]),
    )
    expect(
      screen.getByRole("link", { name: "← MCP servers" }),
    ).toBeInTheDocument()
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

  it("lists the ACL in match order: key rules, user rules, the default policy", async () => {
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

    const table = await screen.findByRole("grid", { name: "ACL" })
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
    expect(
      screen.getByText(/Rules are matched from the most specific/),
    ).toBeVisible()
  })

  it("shows only the default policy when there are no rules", async () => {
    mockMcpServerApi({ server: SERVER })
    await openPage()

    const table = await screen.findByRole("grid", { name: "ACL" })
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
      await within(await screen.findByRole("grid", { name: "ACL" })).findByText(
        "ci-bot",
      ),
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
        within(screen.getByRole("grid", { name: "ACL" })).queryByText("ci-bot"),
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

  it("is read-only for someone who cannot manage the workspace", async () => {
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

    const table = screen.getByRole("grid", { name: "ACL" })
    expect(within(table).getByText("Default policy")).toBeVisible()
    expect(within(table).getByText("DENY")).toBeVisible()
    expect(
      screen.getByText(/can see and change the rules for keys and users/),
    ).toBeVisible()
    expect(
      screen.queryByRole("button", { name: "Edit the default policy" }),
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Add rule" }),
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Disable" }),
    ).not.toBeInTheDocument()
    // The access list is a manager's read, so it is not asked for at all.
    expect(calls.some((call) => call.url.includes("/access"))).toBe(false)
  })

  it("falls back to read-only when the access list is refused", async () => {
    mockMcpServerApi({ server: SERVER, accessStatus: 403 })
    await openPage()

    expect(
      await screen.findByText(
        /can see and change the rules for keys and users/,
      ),
    ).toBeVisible()
    expect(
      screen.queryByRole("button", { name: "Add rule" }),
    ).not.toBeInTheDocument()
  })
})
