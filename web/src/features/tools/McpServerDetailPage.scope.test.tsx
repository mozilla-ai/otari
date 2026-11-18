import { screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { workspaceMcpServer } from "@/tests/fixtures"
import {
  mcpWrites,
  mockMcpServerApi,
  renderMcpServerPage,
  WORKSPACE,
} from "@/tests/mcpServer"
import { pickOption } from "@/tests/select"

// A server row's `workspace_id` is null for an organization's server. This
// page reads and writes through the workspace it read the server from, so a
// row without one changes nothing about where its calls go.
const SERVER = workspaceMcpServer({ workspace_id: WORKSPACE, has_token: true })
const ROW = { ...SERVER, workspace_id: null }
const path = `/workspaces/${WORKSPACE}/mcp-servers/${SERVER.id}`

async function openPage() {
  renderMcpServerPage(SERVER.id)
  await screen.findByRole("heading", { level: 1, name: "github" })
}

describe("McpServerDetailPage, a row that names no workspace", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("opens the server's page with its sections", async () => {
    mockMcpServerApi({ server: SERVER, row: ROW })
    await openPage()

    expect(screen.getByRole("heading", { name: "Exposed tools" })).toBeVisible()
    expect(screen.getByRole("heading", { name: "Access rules" })).toBeVisible()
    expect(screen.getByText(SERVER.url)).toBeVisible()
  })

  it("changes its availability through the workspace it was read from", async () => {
    const calls = mockMcpServerApi({ server: SERVER, row: ROW })
    const user = userEvent.setup()
    await openPage()

    await pickOption(user, "Availability", "Off")

    await waitFor(() =>
      expect(
        mcpWrites(calls).some(
          (call) => call.method === "PATCH" && call.url.endsWith(path),
        ),
      ).toBe(true),
    )
    expect(mcpWrites(calls).every((call) => !call.url.includes("null"))).toBe(
      true,
    )
  })

  it("saves its settings through the workspace it was read from", async () => {
    const calls = mockMcpServerApi({ server: SERVER, row: ROW })
    const user = userEvent.setup()
    await openPage()

    await user.click(screen.getByRole("button", { name: "Settings" }))
    await user.clear(screen.getByLabelText("Name"))
    await user.type(screen.getByLabelText("Name"), "github-prod")
    await user.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() =>
      expect(
        mcpWrites(calls).find(
          (call) => call.method === "PATCH" && call.url.endsWith(path),
        )?.body,
      ).toMatchObject({ name: "github-prod" }),
    )
  })
})
