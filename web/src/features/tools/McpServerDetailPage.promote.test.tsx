import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { mcpServerFoldCandidate, workspaceMcpServer } from "@/tests/fixtures"
import {
  mcpWrites,
  mockMcpServerApi,
  renderMcpServerPage,
  WORKSPACE,
} from "@/tests/mcpServer"

// Making a workspace's own server its organization's, from the server's page.

const PROMOTE = "Make it an organization server"

const KEEPS = mcpServerFoldCandidate({
  server_id: "aaaaaaaa-0000-0000-0000-000000000001",
  workspace_name: "Beta",
  name: "github",
  same_credential: true,
  signed_in: 3,
  keep_sign_in: 3,
})
const SIGNS_AGAIN = mcpServerFoldCandidate({
  server_id: "aaaaaaaa-0000-0000-0000-000000000002",
  workspace_name: "Gamma",
  name: "gh",
  same_credential: false,
  signed_in: 2,
  keep_sign_in: 0,
  sign_in_again: 2,
})

async function openPage(serverId: string) {
  renderMcpServerPage(serverId)
  await screen.findByRole("heading", { level: 1, name: "github" })
}

describe("McpServerDetailPage, making a server the organization's", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("is offered to an organization owner or admin", async () => {
    const server = workspaceMcpServer({ workspace_id: WORKSPACE })
    mockMcpServerApi({ server, role: "admin" })
    await openPage(server.id)

    expect(screen.getByRole("button", { name: PROMOTE })).toBeVisible()
  })

  it("is not offered to a workspace's own admin", async () => {
    const server = workspaceMcpServer({ workspace_id: WORKSPACE })
    mockMcpServerApi({ server, role: "member", workspaceRole: "admin" })
    await openPage(server.id)

    // The workspace's own controls are there, so the page has settled.
    expect(screen.getByRole("button", { name: "Settings" })).toBeVisible()
    expect(screen.queryByRole("button", { name: PROMOTE })).toBeNull()
  })

  it("shows the other workspaces' copies, posts the ticked ones and opens the organization's page", async () => {
    const server = workspaceMcpServer({
      workspace_id: WORKSPACE,
      credential_mode: "per_user",
    })
    const calls = mockMcpServerApi({
      server,
      copies: [KEEPS, SIGNS_AGAIN],
    })
    const user = userEvent.setup()
    await openPage(server.id)

    await user.click(screen.getByRole("button", { name: PROMOTE }))

    const dialog = await screen.findByRole("dialog", {
      name: "Make github an organization server",
    })
    const copies = await within(dialog).findByRole("list", {
      name: "Workspace copies",
    })
    expect(
      within(copies).getByText("3 people will keep their sign-in."),
    ).toBeVisible()
    expect(
      within(copies).getByText("2 people will need to sign in again."),
    ).toBeVisible()
    expect(
      within(copies).getByRole("checkbox", { name: "Beta: github" }),
    ).toBeChecked()
    expect(
      within(copies).getByRole("checkbox", { name: "Gamma: gh" }),
    ).not.toBeChecked()
    const read = calls.find((call) => call.url.includes("same-url"))
    const params = new URL(String(read?.url), "http://localhost").searchParams
    expect(params.get("url")).toBe(server.url)
    expect(params.get("server_id")).toBe(server.id)

    await user.click(
      within(copies).getByRole("checkbox", { name: "Gamma: gh" }),
    )
    await user.click(within(dialog).getByRole("button", { name: PROMOTE }))

    await waitFor(() =>
      expect(
        mcpWrites(calls).find((call) => call.url.endsWith("/promote")),
      ).toMatchObject({
        url: `/api/v1/workspaces/${WORKSPACE}/mcp-servers/${server.id}/promote`,
        method: "POST",
        body: {
          fold_server_ids: [KEEPS.server_id, SIGNS_AGAIN.server_id],
        },
      }),
    )
    expect(
      await screen.findByText("Organization server page"),
    ).toBeInTheDocument()
  })

  it("promotes with nothing to fold when no other workspace has a copy", async () => {
    const server = workspaceMcpServer({ workspace_id: WORKSPACE })
    const calls = mockMcpServerApi({ server })
    const user = userEvent.setup()
    await openPage(server.id)

    await user.click(screen.getByRole("button", { name: PROMOTE }))
    const dialog = await screen.findByRole("dialog", {
      name: "Make github an organization server",
    })
    expect(
      await within(dialog).findByText(
        "No other workspace has its own server at this URL.",
      ),
    ).toBeVisible()
    await user.click(within(dialog).getByRole("button", { name: PROMOTE }))

    await waitFor(() =>
      expect(
        mcpWrites(calls).find((call) => call.url.endsWith("/promote"))?.body,
      ).toEqual({ fold_server_ids: [] }),
    )
  })
})
