import { act, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { McpServerWorkspaceSettings } from "@/client"
import { mcpServerKey, workspaceMcpScope } from "@/shared/api/mcpServerScope"
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
import { getRowMenuItem, pressRowAction } from "@/tests/rowMenu"
import { pickOption } from "@/tests/select"

// An organization's server read through a workspace: the organization's
// definition, with the workspace's own settings on it in `in_workspace`.

function settings(
  overrides: Partial<McpServerWorkspaceSettings> = {},
): McpServerWorkspaceSettings {
  return {
    reached: true,
    enabled: true,
    everyone_tools: null,
    attach_by_default: null,
    organization_tools: null,
    ...overrides,
  }
}

function organizationServer(own: Partial<McpServerWorkspaceSettings> = {}) {
  return workspaceMcpServer({
    // Found only through the workspace, which the harness reads it from.
    workspace_id: WORKSPACE,
    scope: "organization",
    name: "hub",
    url: "https://mcp.example.com/***",
    health: health({ tools: ["list_issues", "get_issue", "create_issue"] }),
    in_workspace: settings(own),
  })
}

/** The row as the gateway answers: an organization's server names no workspace. */
function asRead(server: ReturnType<typeof organizationServer>) {
  return { ...server, workspace_id: null }
}

const BO = "bo-0000"
const BO_LAPTOP = apiKey({
  id: "key-bo",
  key_name: "Bo's laptop",
  user_id: BO,
  workspace_id: WORKSPACE,
})

/**
 * Holds the first write of the workspace's own settings until the returned
 * function is called with how it answers: the stub's answer, or a refusal.
 */
function holdFirstOverrideWrite(): (outcome: "save" | "refuse") => void {
  const answer = vi.mocked(globalThis.fetch).getMockImplementation()
  let release: (outcome: "save" | "refuse") => void = () => {}
  const held = new Promise<"save" | "refuse">((resolve) => {
    release = resolve
  })
  let isHeld = false
  vi.mocked(globalThis.fetch).mockImplementation(async (input, init) => {
    if (String(input).endsWith("/override") && init?.method === "PATCH") {
      if (!isHeld) {
        isHeld = true
        if ((await held) === "refuse") {
          return Response.json({ detail: "Write refused" }, { status: 500 })
        }
      }
    }
    return answer?.(input, init) as Promise<Response>
  })
  return release
}

async function openPage(serverId: string) {
  const { client } = renderMcpServerPage(serverId)
  await screen.findByRole("heading", { level: 1, name: "hub" })
  return client
}

/** From now on, every read of the server itself fails. */
function failServerReads(serverId: string) {
  const answer = vi.mocked(globalThis.fetch).getMockImplementation()
  vi.mocked(globalThis.fetch).mockImplementation(async (input, init) => {
    const isRead = (init?.method ?? "GET").toUpperCase() === "GET"
    if (isRead && String(input).endsWith(`/mcp-servers/${serverId}`)) {
      return Response.json({ detail: "Unavailable" }, { status: 503 })
    }
    return answer?.(input, init) as Promise<Response>
  })
}

const serverKey = (serverId: string) =>
  mcpServerKey(workspaceMcpScope(WORKSPACE), serverId)

describe("McpServerDetailPage, an organization's server", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("shows the definition read-only, without what the organization keeps", async () => {
    const server = organizationServer({ organization_tools: ["@read"] })
    mockMcpServerApi({ server, row: asRead(server) })
    await openPage(server.id)

    expect(screen.getByText("Organization")).toBeVisible()
    expect(
      screen.getByText(
        "Your organization manages this server. You can change how this workspace uses it.",
      ),
    ).toBeVisible()
    expect(
      screen.getByText("Organization limit:").parentElement,
    ).toHaveTextContent("Organization limit: Read tools.")
    // Its definition is the organization's to change.
    expect(screen.queryByRole("button", { name: "Settings" })).toBeNull()
    expect(screen.queryByRole("button", { name: "Edit connection" })).toBeNull()
    expect(
      screen.queryByRole("button", { name: "Edit exposed tools" }),
    ).toBeNull()
    expect(screen.queryByRole("button", { name: "Check" })).toBeNull()
    expect(
      screen.queryByRole("button", { name: "Change the logo of hub" }),
    ).toBeNull()
    expect(
      screen.queryByRole("button", { name: "Sign in for everyone" }),
    ).toBeNull()
    // It is the organization's already.
    expect(
      screen.queryByRole("button", { name: "Make it an organization server" }),
    ).toBeNull()
    // Still read: how Otari reaches it.
    expect(screen.getByText("https://mcp.example.com/***")).toBeVisible()
  })

  it("keeps the workspace's access rules and their grants", async () => {
    const server = organizationServer()
    mockMcpServerApi({
      server,
      row: asRead(server),
      access: { everyone_tools: ["*"], grants: [grant({ id: "g-1" })] },
    })
    await openPage(server.id)

    const rules = await screen.findByRole("grid", { name: "Access rules" })
    expect(await within(rules).findByText("ci-bot")).toBeVisible()
    expect(
      within(rules).getByText("any other key, as your organization set it"),
    ).toBeVisible()
    expect(screen.getByRole("button", { name: "Add rule" })).toBeVisible()
    // The workspace set nothing of its own, so there is nothing to clear.
    expect(
      screen.getByRole("button", { name: "Use organization defaults" }),
    ).toBeDisabled()
  })

  it("sets the workspace's Availability through its own settings", async () => {
    const server = organizationServer({ everyone_tools: ["@read"] })
    const calls = mockMcpServerApi({ server, row: asRead(server) })
    const user = userEvent.setup()
    await openPage(server.id)

    await pickOption(user, "Availability", "Every request")

    await waitFor(() =>
      expect(
        mcpWrites(calls).find((call) => call.url.endsWith("/override")),
      ).toMatchObject({ method: "PATCH" }),
    )
    // Only what Availability sets: the default policy is left as stored.
    expect(
      mcpWrites(calls).find((call) => call.url.endsWith("/override"))?.body,
    ).toEqual({ enabled: true, attach_by_default: true })
    // Nothing that changes the organization's definition.
    expect(
      mcpWrites(calls).some((call) => !call.url.endsWith("/override")),
    ).toBe(false)
  })

  it("sets the workspace's default policy from the access rules", async () => {
    const server = organizationServer({ enabled: false })
    const calls = mockMcpServerApi({ server, row: asRead(server) })
    const user = userEvent.setup()
    await openPage(server.id)

    await pressRowAction(user, "Edit the default policy")
    const dialog = await screen.findByRole("dialog", {
      name: "Edit the default policy",
    })
    await user.click(within(dialog).getByRole("radio", { name: "Deny" }))
    await user.click(within(dialog).getByRole("button", { name: "Save rule" }))

    await waitFor(() =>
      expect(
        mcpWrites(calls).find((call) => call.url.endsWith("/override")),
      ).toEqual(
        expect.objectContaining({
          method: "PATCH",
          // Only the policy: the workspace's switch is left as stored.
          body: { everyone_tools: [] },
        }),
      ),
    )
  })

  it("saves nothing when the default policy is saved unchanged", async () => {
    const server = organizationServer()
    const calls = mockMcpServerApi({ server, row: asRead(server) })
    const user = userEvent.setup()
    await openPage(server.id)

    await user.click(
      screen.getByRole("button", {
        name: "Edit the default policy for this workspace",
      }),
    )
    const dialog = await screen.findByRole("dialog", {
      name: "Edit the default policy",
    })
    await user.click(within(dialog).getByRole("button", { name: "Save rule" }))

    // Saving the inherited policy would stop it following the organization's.
    await waitFor(() =>
      expect(
        screen.queryByRole("dialog", { name: "Edit the default policy" }),
      ).toBeNull(),
    )
    expect(mcpWrites(calls)).toEqual([])
  })

  it("holds the default policy's Edit while an Availability write is in flight, and keeps its refusal", async () => {
    const server = organizationServer()
    const calls = mockMcpServerApi({ server, row: asRead(server) })
    const release = holdFirstOverrideWrite()
    const user = userEvent.setup()
    await openPage(server.id)

    await pickOption(user, "Availability", "Off")
    const edit = screen.getByRole("button", {
      name: "Edit the default policy for this workspace",
    })
    await waitFor(() => expect(edit).toBeDisabled())

    release("refuse")
    expect(await screen.findByText("Write refused")).toBeVisible()
    // The write may have landed all the same, so the server is read again first.
    const written = calls.findIndex((call) => call.method === "PATCH")
    await waitFor(() =>
      expect(
        calls
          .slice(written + 1)
          .some(
            (call) =>
              call.method === "GET" &&
              call.url.endsWith(`/mcp-servers/${server.id}`),
          ),
      ).toBe(true),
    )
    await waitFor(() => expect(edit).toBeEnabled())
    expect(screen.getByText("Write refused")).toBeVisible()

    // Opening the editor clears the settled refusal.
    await user.click(edit)
    await screen.findByRole("dialog", { name: "Edit the default policy" })
    expect(screen.queryByText("Write refused")).toBeNull()
  })

  it("keeps Off when the default policy is saved after it", async () => {
    const server = organizationServer()
    const calls = mockMcpServerApi({ server, row: asRead(server) })
    const release = holdFirstOverrideWrite()
    const user = userEvent.setup()
    await openPage(server.id)

    await pickOption(user, "Availability", "Off")
    // The access rules write the same settings, so their Edit waits too.
    const held = await getRowMenuItem(user, /^Edit…/)
    expect(held).toHaveAttribute("aria-disabled", "true")
    expect(held).toHaveTextContent("Saving another change first")
    await user.keyboard("{Escape}")

    release("save")
    await waitFor(() =>
      expect(
        screen.getByRole("button", {
          name: "Edit the default policy for this workspace",
        }),
      ).toBeEnabled(),
    )
    await pressRowAction(user, "Edit the default policy")
    const dialog = await screen.findByRole("dialog", {
      name: "Edit the default policy",
    })
    await user.click(within(dialog).getByRole("radio", { name: "Deny" }))
    await user.click(within(dialog).getByRole("button", { name: "Save rule" }))

    await waitFor(() =>
      expect(
        mcpWrites(calls).filter((call) => call.url.endsWith("/override")),
      ).toHaveLength(2),
    )
    const [off, policy] = mcpWrites(calls).filter((call) =>
      call.url.endsWith("/override"),
    )
    expect(off?.body).toEqual({ enabled: false })
    // The policy alone, so it cannot undo Off even from a view that missed it.
    expect(policy?.body).toEqual({ everyone_tools: [] })
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: /Availability/ }),
      ).toHaveTextContent("Off"),
    )
  })

  it("compares the default policy with the one the editor opened on", async () => {
    const server = organizationServer()
    const calls = mockMcpServerApi({ server, row: asRead(server) })
    const user = userEvent.setup()
    const client = await openPage(server.id)
    const moved = asRead({ ...server, everyone_tools: ["@read"] })
    const open = async () => {
      await user.click(
        screen.getByRole("button", {
          name: "Edit the default policy for this workspace",
        }),
      )
      return screen.findByRole("dialog", { name: "Edit the default policy" })
    }

    // The organization's default moves while the editor is open, untouched.
    let dialog = await open()
    act(() => client.setQueryData(serverKey(server.id), moved))
    await user.click(within(dialog).getByRole("button", { name: "Save rule" }))
    await waitFor(() =>
      expect(
        screen.queryByRole("dialog", { name: "Edit the default policy" }),
      ).toBeNull(),
    )
    expect(mcpWrites(calls)).toEqual([])

    // An edit is the admin's own choice, even when it equals the moved default.
    act(() => client.setQueryData(serverKey(server.id), asRead(server)))
    dialog = await open()
    act(() => client.setQueryData(serverKey(server.id), moved))
    await user.click(
      within(dialog).getByRole("radio", { name: "Read tools only" }),
    )
    await user.click(within(dialog).getByRole("button", { name: "Save rule" }))
    await waitFor(() =>
      expect(mcpWrites(calls)).toEqual([
        expect.objectContaining({
          method: "PATCH",
          body: { everyone_tools: ["@read"] },
        }),
      ]),
    )
  })

  it("holds every write of the settings after a failed write whose re-read fails too", async () => {
    const server = organizationServer()
    const calls = mockMcpServerApi({ server, row: asRead(server) })
    const release = holdFirstOverrideWrite()
    const user = userEvent.setup()
    await openPage(server.id)

    // The write fails from a dialog still open, and the read after it fails too.
    await user.click(
      screen.getByRole("button", {
        name: "Edit the default policy for this workspace",
      }),
    )
    const dialog = await screen.findByRole("dialog", {
      name: "Edit the default policy",
    })
    await user.click(within(dialog).getByRole("radio", { name: "Deny" }))
    await user.click(within(dialog).getByRole("button", { name: "Save rule" }))
    failServerReads(server.id)
    release("refuse")
    expect(await within(dialog).findByText("Write refused")).toBeVisible()
    await waitFor(() =>
      expect(
        within(dialog).getByRole("button", { name: "Save rule" }),
      ).toBeDisabled(),
    )

    // The open dialog says so beside the write's refusal, offers the refresh
    // itself, and cannot save again: the refused write is the only one sent.
    expect(within(dialog).getByText(/could not be read again/)).toBeVisible()
    expect(
      within(dialog).getByRole("button", { name: "Refresh" }),
    ).toBeVisible()
    const save = within(dialog).getByRole("button", { name: "Save rule" })
    expect(save).toBeDisabled()
    const sent = vi
      .mocked(globalThis.fetch)
      .mock.calls.filter(
        ([input, init]) =>
          String(input).endsWith("/override") && init?.method === "PATCH",
      )
    expect(sent).toHaveLength(1)
    expect(mcpWrites(calls)).toEqual([])
    await user.click(within(dialog).getByRole("button", { name: "Cancel" }))
    await user.click(await screen.findByRole("button", { name: "Discard" }))
    await waitFor(() =>
      expect(
        screen.queryByRole("dialog", { name: "Edit the default policy" }),
      ).toBeNull(),
    )

    // The settings card and the access rules both wait for a refresh.
    expect(
      screen.getByRole("button", {
        name: "Edit the default policy for this workspace",
      }),
    ).toBeDisabled()
    expect(screen.getByRole("button", { name: "Refresh" })).toBeVisible()
    const held = await getRowMenuItem(user, /^Edit…/)
    expect(held).toHaveAttribute("aria-disabled", "true")
    expect(held).toHaveTextContent("Refresh the settings first")
  })

  it("holds the access rules' default-policy dialog after a failed write whose re-read fails too", async () => {
    const server = organizationServer()
    const calls = mockMcpServerApi({ server, row: asRead(server) })
    const release = holdFirstOverrideWrite()
    const user = userEvent.setup()
    await openPage(server.id)

    // The write fails from the access rules' dialog, still open, and the read
    // after it fails too.
    await pressRowAction(user, "Edit the default policy")
    const dialog = await screen.findByRole("dialog", {
      name: "Edit the default policy",
    })
    await user.click(within(dialog).getByRole("radio", { name: "Deny" }))
    await user.click(within(dialog).getByRole("button", { name: "Save rule" }))
    failServerReads(server.id)
    release("refuse")
    expect(await within(dialog).findByText("Write refused")).toBeVisible()
    expect(
      await within(dialog).findByText(/could not be read again/),
    ).toBeVisible()

    // It offers the refresh itself, cannot save again, and the refused write
    // is the only one sent.
    expect(
      within(dialog).getByRole("button", { name: "Refresh" }),
    ).toBeVisible()
    expect(
      within(dialog).getByRole("button", { name: "Save rule" }),
    ).toBeDisabled()
    const sent = vi
      .mocked(globalThis.fetch)
      .mock.calls.filter(
        ([input, init]) =>
          String(input).endsWith("/override") && init?.method === "PATCH",
      )
    expect(sent).toHaveLength(1)
    expect(mcpWrites(calls)).toEqual([])
  })

  it("clears the workspace's own settings back to the organization's", async () => {
    const server = organizationServer({ attach_by_default: true })
    const calls = mockMcpServerApi({ server, row: asRead(server) })
    const user = userEvent.setup()
    await openPage(server.id)

    const reset = screen.getByRole("button", {
      name: "Use organization defaults",
    })
    expect(reset).toBeEnabled()
    await user.click(reset)

    await waitFor(() =>
      expect(
        mcpWrites(calls).find((call) => call.url.endsWith("/override"))?.method,
      ).toBe("DELETE"),
    )
  })

  it("says the organization turned it off here, when the workspace did not", async () => {
    const server = organizationServer({
      reached: false,
      organization_tools: [],
    })
    mockMcpServerApi({ server, row: asRead(server) })
    await openPage(server.id)

    expect(
      screen.getByText(
        "Your organization turned this server off for this workspace.",
      ),
    ).toBeVisible()
  })

  it("says when a trace was decided by the organization's limit", async () => {
    const server = organizationServer({ organization_tools: ["get_issue"] })
    mockMcpServerApi({
      server,
      row: asRead(server),
      members: [
        organizationMember({
          user_id: "identity-bo",
          attribution_user_id: BO,
          full_name: "Bo",
        }),
      ],
      keys: [BO_LAPTOP],
      check: {
        allowed: false,
        decided_by: "organization",
        grant_id: null,
        tools: ["get_issue"],
      },
    })
    const user = userEvent.setup()
    await openPage(server.id)

    await pickOption(user, "Source key", "Bo's laptop (Bo)")
    await user.type(screen.getByRole("combobox", { name: "Tool" }), "create")
    await user.click(
      await screen.findByRole("option", { name: "create_issue" }),
    )
    await user.click(screen.getByRole("button", { name: "Trace" }))

    expect(await screen.findByRole("status")).toHaveTextContent(
      "DENY, the organization's limit for this workspace decided it",
    )
  })
})
