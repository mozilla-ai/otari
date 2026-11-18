import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { McpServerFoldCandidate } from "@/client"
import { McpServerDialog } from "@/features/tools/McpServerDialog"
import {
  type McpServerScope,
  workspaceMcpScope,
} from "@/shared/api/mcpServerScope"
import { mcpServerFoldCandidate } from "@/tests/fixtures"

const WORKSPACE = "11111111-1111-1111-1111-111111111111"

const KEEPS = mcpServerFoldCandidate({
  server_id: "aaaaaaaa-0000-0000-0000-000000000001",
  workspace_name: "Beta",
  name: "github",
  same_credential: true,
  signed_in: 3,
  keep_sign_in: 3,
  sign_in_again: 0,
})
const OWN_TOKEN = mcpServerFoldCandidate({
  server_id: "aaaaaaaa-0000-0000-0000-000000000002",
  workspace_name: "Gamma",
  name: "gh",
  credential_mode: "shared",
  same_credential: false,
  signed_in: 2,
  keep_sign_in: 0,
  sign_in_again: 2,
  replaces_shared_credential: true,
})

/** Answers the read of the copies at a URL, and records the URLs asked. */
function mockCopies(copies: McpServerFoldCandidate[]): string[] {
  const asked: string[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input)
    asked.push(url)
    if (url.includes("/mcp-servers/same-url")) {
      return Response.json({ data: copies })
    }
    return Response.json({})
  })
  return asked
}

function renderDialog({
  scope = workspaceMcpScope(WORKSPACE),
  canAddForOrganization = true,
}: {
  scope?: McpServerScope
  canAddForOrganization?: boolean
} = {}) {
  const onSubmit = vi.fn()
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  render(
    <QueryClientProvider client={client}>
      <McpServerDialog
        isOpen
        onOpenChange={() => {}}
        scope={scope}
        canAddForOrganization={canAddForOrganization}
        workspaceName="Alpha"
        isPending={false}
        error={null}
        onSubmit={onSubmit}
      />
    </QueryClientProvider>,
  )
  return onSubmit
}

async function fillAndSubmit(user: ReturnType<typeof userEvent.setup>) {
  await user.type(screen.getByLabelText("Name"), "github")
  await user.type(screen.getByLabelText("URL"), "https://mcp.example.com/gh")
  await user.click(
    within(screen.getByRole("dialog")).getByRole("button", {
      name: "Add MCP server",
    }),
  )
}

describe("McpServerDialog", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("offers an organization owner or admin the whole organization, preselected", () => {
    mockCopies([])
    renderDialog()

    const choice = screen.getByRole("radiogroup", { name: "Add to" })
    expect(
      within(choice).getByRole("radio", { name: /Whole organization/ }),
    ).toBeChecked()
    expect(
      within(choice).getByRole("radio", { name: /This workspace/ }),
    ).not.toBeChecked()
  })

  it("asks a workspace's own admin nothing, and adds to the workspace", async () => {
    const asked = mockCopies([KEEPS])
    const user = userEvent.setup()
    const onSubmit = renderDialog({ canAddForOrganization: false })

    expect(screen.queryByRole("radiogroup", { name: "Add to" })).toBeNull()
    await fillAndSubmit(user)

    expect(onSubmit).toHaveBeenCalledWith(
      expect.not.objectContaining({ fold_server_ids: expect.anything() }),
      { kind: "workspace", workspaceId: WORKSPACE },
    )
    expect(asked.some((url) => url.includes("same-url"))).toBe(false)
  })

  it("skips the fold step when no workspace has a copy", async () => {
    const asked = mockCopies([])
    const user = userEvent.setup()
    const onSubmit = renderDialog()

    await fillAndSubmit(user)

    await waitFor(() => expect(onSubmit).toHaveBeenCalled())
    expect(onSubmit.mock.calls[0][1]).toEqual({ kind: "organization" })
    expect(onSubmit.mock.calls[0][0]).toMatchObject({
      name: "github",
      url: "https://mcp.example.com/gh",
    })
    const read = asked.find((url) => url.includes("same-url"))
    const params = new URL(String(read), "http://localhost").searchParams
    expect(params.get("url")).toBe("https://mcp.example.com/gh")
    expect(params.get("credential_mode")).toBe("shared")
  })

  it("shows each copy with what folding it does, ticking only those on the same account", async () => {
    mockCopies([KEEPS, OWN_TOKEN])
    const user = userEvent.setup()
    const onSubmit = renderDialog()

    await fillAndSubmit(user)

    const copies = await screen.findByRole("list", {
      name: "Workspace copies",
    })
    expect(
      within(copies).getByText("3 people will keep their sign-in."),
    ).toBeVisible()
    expect(
      within(copies).getByText("2 people will need to sign in again."),
    ).toBeVisible()
    expect(
      within(copies).getByText(
        "Its shared account or token will be replaced by the organization's.",
      ),
    ).toBeVisible()
    expect(
      within(copies).getByRole("checkbox", { name: "Beta: github" }),
    ).toBeChecked()
    expect(
      within(copies).getByRole("checkbox", { name: "Gamma: gh" }),
    ).not.toBeChecked()
    expect(onSubmit).not.toHaveBeenCalled()

    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", {
        name: "Add MCP server",
      }),
    )
    expect(onSubmit).toHaveBeenCalledWith(
      expect.objectContaining({ fold_server_ids: [KEEPS.server_id] }),
      { kind: "organization" },
    )
  })

  it("sends the copies ticked by hand", async () => {
    mockCopies([KEEPS, OWN_TOKEN])
    const user = userEvent.setup()
    const onSubmit = renderDialog()

    await fillAndSubmit(user)
    await user.click(
      await screen.findByRole("checkbox", { name: "Beta: github" }),
    )
    await user.click(screen.getByRole("checkbox", { name: "Gamma: gh" }))
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", {
        name: "Add MCP server",
      }),
    )

    expect(onSubmit).toHaveBeenCalledWith(
      expect.objectContaining({ fold_server_ids: [OWN_TOKEN.server_id] }),
      { kind: "organization" },
    )
  })

  it("adds to the workspace when the admin picks it, with no fold step", async () => {
    const asked = mockCopies([KEEPS])
    const user = userEvent.setup()
    const onSubmit = renderDialog()

    await user.click(screen.getByRole("radio", { name: /This workspace/ }))
    await fillAndSubmit(user)

    expect(onSubmit).toHaveBeenCalledWith(expect.anything(), {
      kind: "workspace",
      workspaceId: WORKSPACE,
    })
    expect(asked.some((url) => url.includes("same-url"))).toBe(false)
  })
})
