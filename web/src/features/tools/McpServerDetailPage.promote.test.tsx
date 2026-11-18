import { focusManager } from "@tanstack/react-query"
import { act, screen, waitFor, within } from "@testing-library/react"
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
  workspace_id: "bbbbbbbb-0000-0000-0000-000000000001",
  workspace_name: "Beta",
  name: "github",
  same_credential: true,
  signed_in: 3,
  keep_sign_in: 3,
})
const SIGNS_AGAIN = mcpServerFoldCandidate({
  server_id: "aaaaaaaa-0000-0000-0000-000000000002",
  workspace_id: "bbbbbbbb-0000-0000-0000-000000000002",
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
    focusManager.setFocused(undefined)
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
    // By the server's id alone: its stored URL, which may carry a credential, is never sent.
    expect(calls.filter((call) => call.url.includes("same-url"))).toEqual([
      {
        url: "/api/v1/organizations/me/mcp-servers/same-url",
        method: "POST",
        body: { server_id: server.id },
      },
    ])

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
      await screen.findByText(`Organization server page ${server.id}`),
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

  it("holds the promote until this opening's read answers, and offers Retry when it fails", async () => {
    const server = workspaceMcpServer({ workspace_id: WORKSPACE })
    const calls = mockMcpServerApi({ server, copies: [KEEPS] })
    const answer = vi.mocked(globalThis.fetch).getMockImplementation()
    let lookup: "answer" | "hold" | "fail" = "answer"
    let release: (response: Response) => void = () => {}
    vi.mocked(globalThis.fetch).mockImplementation(async (input, init) => {
      if (String(input).includes("/same-url")) {
        if (lookup === "hold") {
          return new Promise<Response>((resolve) => {
            release = resolve
          })
        }
        if (lookup === "fail") {
          return Response.json({ detail: "unavailable" }, { status: 503 })
        }
      }
      return answer?.(input, init) as Promise<Response>
    })
    const user = userEvent.setup()
    await openPage(server.id)

    // A first opening reads the copies, which stay cached once it closes.
    await user.click(screen.getByRole("button", { name: PROMOTE }))
    let dialog = await screen.findByRole("dialog", {
      name: "Make github an organization server",
    })
    await within(dialog).findByRole("list", { name: "Workspace copies" })
    await user.click(within(dialog).getByRole("button", { name: "Cancel" }))
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull())

    // While this opening's read is out, the cached copies are neither shown nor
    // sent, whatever time the cache says they were read at.
    lookup = "hold"
    await user.click(screen.getByRole("button", { name: PROMOTE }))
    dialog = await screen.findByRole("dialog", {
      name: "Make github an organization server",
    })
    await within(dialog).findByText(
      "Looking for copies of it in other workspaces.",
    )
    expect(within(dialog).getByRole("button", { name: PROMOTE })).toBeDisabled()
    expect(
      within(dialog).queryByRole("list", { name: "Workspace copies" }),
    ).toBeNull()

    lookup = "fail"
    act(() =>
      release(Response.json({ detail: "unavailable" }, { status: 503 })),
    )
    const retry = await within(dialog).findByRole("button", { name: "Retry" })
    expect(within(dialog).getByRole("button", { name: PROMOTE })).toBeDisabled()
    expect(
      within(dialog).queryByRole("list", { name: "Workspace copies" }),
    ).toBeNull()

    lookup = "answer"
    await user.click(retry)
    await within(dialog).findByRole("list", { name: "Workspace copies" })
    await user.click(within(dialog).getByRole("button", { name: PROMOTE }))
    await waitFor(() =>
      expect(
        mcpWrites(calls).find((call) => call.url.endsWith("/promote"))?.body,
      ).toEqual({ fold_server_ids: [KEEPS.server_id] }),
    )
  })

  it("lists a copy that is on, but does not fold it into a server that is off", async () => {
    const server = workspaceMcpServer({
      workspace_id: WORKSPACE,
      enabled: false,
    })
    const calls = mockMcpServerApi({ server, copies: [KEEPS] })
    const user = userEvent.setup()
    await openPage(server.id)

    await user.click(screen.getByRole("button", { name: PROMOTE }))
    const dialog = await screen.findByRole("dialog", {
      name: "Make github an organization server",
    })
    const copy = await within(dialog).findByRole("checkbox", {
      name: "Beta: github",
    })
    expect(copy).not.toBeChecked()
    expect(copy).toBeDisabled()
    expect(
      within(dialog).getByText(
        "It is on, and an organization server that is off cannot take its place.",
      ),
    ).toBeVisible()

    await user.click(within(dialog).getByRole("button", { name: PROMOTE }))
    await waitFor(() =>
      expect(
        mcpWrites(calls).find((call) => call.url.endsWith("/promote"))?.body,
      ).toEqual({ fold_server_ids: [] }),
    )
  })

  it("drops a ticked copy that is gone after a refresh, and says so", async () => {
    const server = workspaceMcpServer({ workspace_id: WORKSPACE })
    const calls = mockMcpServerApi({ server, copies: [KEEPS, SIGNS_AGAIN] })
    const answer = vi.mocked(globalThis.fetch).getMockImplementation()
    let onlyKeeps = false
    vi.mocked(globalThis.fetch).mockImplementation(async (input, init) => {
      if (onlyKeeps && String(input).includes("/same-url")) {
        return Response.json({ data: [KEEPS] })
      }
      return answer?.(input, init) as Promise<Response>
    })
    const user = userEvent.setup()
    await openPage(server.id)

    await user.click(screen.getByRole("button", { name: PROMOTE }))
    const dialog = await screen.findByRole("dialog", {
      name: "Make github an organization server",
    })
    await user.click(
      await within(dialog).findByRole("checkbox", { name: "Gamma: gh" }),
    )

    onlyKeeps = true
    act(() => {
      focusManager.setFocused(false)
      focusManager.setFocused(true)
    })

    expect(
      await within(dialog).findByText(
        "No longer there, so no longer folded: Gamma: gh.",
      ),
    ).toBeVisible()
    expect(
      within(dialog).queryByRole("checkbox", { name: "Gamma: gh" }),
    ).toBeNull()
    await user.click(within(dialog).getByRole("button", { name: PROMOTE }))
    await waitFor(() =>
      expect(
        mcpWrites(calls).find((call) => call.url.endsWith("/promote"))?.body,
      ).toEqual({ fold_server_ids: [KEEPS.server_id] }),
    )
  })
})
