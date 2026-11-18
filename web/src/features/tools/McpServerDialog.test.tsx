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

// As the gateway answers for a new shared server with no token: a shared copy
// with no credential of its own is on the same account, and nobody is signed
// in to a shared copy with their own.
const KEEPS = mcpServerFoldCandidate({
  server_id: "aaaaaaaa-0000-0000-0000-000000000001",
  workspace_id: "bbbbbbbb-0000-0000-0000-000000000001",
  workspace_name: "Beta",
  name: "github",
  credential_mode: "shared",
  same_credential: true,
  signed_in: 0,
  keep_sign_in: 0,
  sign_in_again: 0,
})
const OWN_TOKEN = mcpServerFoldCandidate({
  server_id: "aaaaaaaa-0000-0000-0000-000000000002",
  workspace_id: "bbbbbbbb-0000-0000-0000-000000000002",
  workspace_name: "Gamma",
  name: "gh",
  credential_mode: "shared",
  same_credential: false,
  signed_in: 0,
  keep_sign_in: 0,
  sign_in_again: 0,
  replaces_shared_credential: true,
})

const LOOKUP = "/api/v1/organizations/me/mcp-servers/same-url"

interface Asked {
  url: string
  method: string
  body: unknown
}

/**
 * Answers the read of the copies at a URL, and records every request. A
 * function answer stands for one that is still being read, or refused.
 */
function mockCopies(
  copies: McpServerFoldCandidate[] | (() => Promise<Response>),
): Asked[] {
  const asked: Asked[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
    asked.push({
      url,
      method,
      body: typeof init?.body === "string" ? JSON.parse(init.body) : undefined,
    })
    if (method === "POST" && url === LOOKUP) {
      return typeof copies === "function"
        ? copies()
        : Response.json({ data: copies })
    }
    return Response.json({})
  })
  return asked
}

/** The body of the one copy lookup made. */
function lookupBody(asked: Asked[]): unknown {
  const lookups = asked.filter((call) => call.url === LOOKUP)
  expect(lookups).toHaveLength(1)
  expect(lookups[0].method).toBe("POST")
  return lookups[0].body
}

function renderDialog({
  scope = workspaceMcpScope(WORKSPACE),
  canAddForOrganization = true,
  error = null,
}: {
  scope?: McpServerScope
  canAddForOrganization?: boolean
  error?: unknown
} = {}) {
  const onSubmit = vi.fn()
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  const dialog = (canAdd: boolean, shown: unknown) => (
    <QueryClientProvider client={client}>
      <McpServerDialog
        isOpen
        onOpenChange={() => {}}
        scope={scope}
        canAddForOrganization={canAdd}
        workspaceName="Alpha"
        isPending={false}
        error={shown}
        onSubmit={onSubmit}
      />
    </QueryClientProvider>
  )
  const { rerender } = render(dialog(canAddForOrganization, error))
  return Object.assign(onSubmit, {
    rerender: (canAdd: boolean, shown: unknown = error) =>
      rerender(dialog(canAdd, shown)),
  })
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
    expect(asked.some((call) => call.url === LOOKUP)).toBe(false)
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
    expect(lookupBody(asked)).toEqual({
      url: "https://mcp.example.com/gh",
      credential_mode: "shared",
      with_token: false,
    })
  })

  it("shows each copy with what folding it does, ticking only those on the same account", async () => {
    mockCopies([KEEPS, OWN_TOKEN])
    const user = userEvent.setup()
    const onSubmit = renderDialog()

    await fillAndSubmit(user)

    const copies = await screen.findByRole("list", {
      name: "Workspace copies",
    })
    expect(within(copies).getByText("No one's sign-in changes.")).toBeVisible()
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
    expect(asked.some((call) => call.url === LOOKUP)).toBe(false)
  })

  it("weighs a personal server's copies by who must sign in again, and leaves them unticked", async () => {
    // A server not stored yet has no client of its own, so nobody keeps a sign-in.
    const personal = mcpServerFoldCandidate({
      server_id: "aaaaaaaa-0000-0000-0000-000000000003",
      workspace_name: "Delta",
      name: "github",
      credential_mode: "per_user",
      same_credential: true,
      signed_in: 2,
      keep_sign_in: 0,
      sign_in_again: 2,
    })
    const asked = mockCopies([personal])
    const user = userEvent.setup()
    const onSubmit = renderDialog()

    await user.click(screen.getByRole("radio", { name: /Personal accounts/ }))
    await fillAndSubmit(user)

    const copies = await screen.findByRole("list", { name: "Workspace copies" })
    expect(
      within(copies).getByText("2 people will need to sign in again."),
    ).toBeVisible()
    expect(
      within(copies).getByRole("checkbox", { name: "Delta: github" }),
    ).not.toBeChecked()
    expect(lookupBody(asked)).toEqual({
      url: "https://mcp.example.com/gh",
      credential_mode: "per_user",
      with_token: false,
    })
    expect(onSubmit).not.toHaveBeenCalled()
  })

  it("says a token of its own changes a keyless copy's account, and leaves it unticked", async () => {
    // The gateway's answer once told the new server brings a token.
    const keyless = { ...KEEPS, same_credential: false }
    const asked = mockCopies([keyless])
    const user = userEvent.setup()
    renderDialog()

    await user.type(screen.getByLabelText("Authorization token"), "secret")
    await fillAndSubmit(user)

    const copies = await screen.findByRole("list", { name: "Workspace copies" })
    expect(
      within(copies).getByText("The account its calls use will change."),
    ).toBeVisible()
    expect(
      within(copies).getByRole("checkbox", { name: "Beta: github" }),
    ).not.toBeChecked()
    expect(lookupBody(asked)).toEqual({
      url: "https://mcp.example.com/gh",
      credential_mode: "shared",
      with_token: true,
    })
  })

  it("freezes the form while it looks for copies", async () => {
    let answer: (response: Response) => void = () => {}
    mockCopies(
      () =>
        new Promise<Response>((resolve) => {
          answer = resolve
        }),
    )
    const user = userEvent.setup()
    const onSubmit = renderDialog()

    await fillAndSubmit(user)

    await waitFor(() => expect(screen.getByLabelText("URL")).toBeDisabled())
    expect(screen.getByLabelText("Name")).toBeDisabled()
    answer(Response.json({ data: [] }))
    await waitFor(() => expect(onSubmit).toHaveBeenCalled())
    expect(onSubmit.mock.calls[0][0]).toMatchObject({
      url: "https://mcp.example.com/gh",
    })
  })

  it("does not add to the workspace a server reviewed for the organization once that access ends", async () => {
    mockCopies([KEEPS])
    const user = userEvent.setup()
    const onSubmit = renderDialog()

    await fillAndSubmit(user)
    await screen.findByRole("list", { name: "Workspace copies" })
    onSubmit.rerender(false)

    expect(
      screen.getByText(
        /You can no longer add servers for the whole organization/,
      ),
    ).toBeVisible()
    expect(
      within(screen.getByRole("dialog")).getByRole("button", {
        name: "Add MCP server",
      }),
    ).toBeDisabled()
    expect(onSubmit).not.toHaveBeenCalled()
  })

  it("clears a failed look for copies once the admin adds to the workspace instead", async () => {
    mockCopies(async () =>
      Response.json({ detail: "Copies unavailable" }, { status: 503 }),
    )
    const user = userEvent.setup()
    const onSubmit = renderDialog()

    await fillAndSubmit(user)
    expect(await screen.findByText("Copies unavailable")).toBeVisible()

    await user.click(screen.getByRole("radio", { name: /This workspace/ }))
    expect(screen.queryByText("Copies unavailable")).toBeNull()
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", {
        name: "Add MCP server",
      }),
    )
    expect(onSubmit).toHaveBeenCalledWith(
      expect.objectContaining({ name: "github" }),
      workspaceMcpScope(WORKSPACE),
    )
    onSubmit.rerender(true, new Error("Name in use"))
    expect(await screen.findByText("Name in use")).toBeVisible()
  })

  it("shows the fold limit, and holds the add while more copies than it are ticked", async () => {
    const many = Array.from({ length: 51 }, (_, index) =>
      mcpServerFoldCandidate({
        ...KEEPS,
        server_id: `aaaaaaaa-0000-0000-0000-${String(index).padStart(12, "0")}`,
        workspace_id: `bbbbbbbb-0000-0000-0000-${String(index).padStart(12, "0")}`,
        workspace_name: `Workspace ${index}`,
      }),
    )
    mockCopies(many)
    const user = userEvent.setup()
    renderDialog()

    await fillAndSubmit(user)

    expect(
      await screen.findByText(
        "Up to 50 copies can be folded at a time, and 51 are ticked. Untick some to continue.",
      ),
    ).toBeVisible()
    const add = within(screen.getByRole("dialog")).getByRole("button", {
      name: "Add MCP server",
    })
    expect(add).toBeDisabled()
    await user.click(
      screen.getByRole("checkbox", { name: "Workspace 0: github" }),
    )
    expect(add).toBeEnabled()
  })

  it("folds one copy of each workspace, which the admin picks where it has several", async () => {
    const second = {
      ...KEEPS,
      server_id: "aaaaaaaa-0000-0000-0000-000000000009",
      name: "gh-2",
    }
    mockCopies([KEEPS, second])
    const user = userEvent.setup()
    const onSubmit = renderDialog()

    await fillAndSubmit(user)
    const copies = await screen.findByRole("list", { name: "Workspace copies" })
    const first = within(copies).getByRole("checkbox", { name: "Beta: github" })
    const other = within(copies).getByRole("checkbox", { name: "Beta: gh-2" })
    expect(first).not.toBeChecked()
    expect(other).not.toBeChecked()

    await user.click(first)
    await user.click(other)
    expect(first).not.toBeChecked()
    expect(other).toBeChecked()
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", {
        name: "Add MCP server",
      }),
    )
    expect(onSubmit).toHaveBeenCalledWith(
      expect.objectContaining({ fold_server_ids: [second.server_id] }),
      { kind: "organization" },
    )
  })

  it("does not fold a copy that is on into a server added Off, and says why", async () => {
    const off = {
      ...OWN_TOKEN,
      same_credential: true,
      replaces_shared_credential: false,
      enabled: false,
    }
    mockCopies([KEEPS, off])
    const user = userEvent.setup()
    const onSubmit = renderDialog()

    await user.click(screen.getByRole("radio", { name: /Off/ }))
    await fillAndSubmit(user)

    const copies = await screen.findByRole("list", { name: "Workspace copies" })
    const on = within(copies).getByRole("checkbox", { name: "Beta: github" })
    expect(on).not.toBeChecked()
    expect(on).toBeDisabled()
    expect(
      within(copies).getByText(
        "It is on, and an organization server that is off cannot take its place.",
      ),
    ).toBeVisible()
    expect(
      within(copies).getByRole("checkbox", { name: "Gamma: gh" }),
    ).toBeChecked()
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", {
        name: "Add MCP server",
      }),
    )
    expect(onSubmit).toHaveBeenCalledWith(
      expect.objectContaining({
        enabled: false,
        fold_server_ids: [off.server_id],
      }),
      { kind: "organization" },
    )
  })

  it("keeps a draft for the organization there once that access ends, until the admin picks the workspace", async () => {
    const asked = mockCopies([])
    const user = userEvent.setup()
    const onSubmit = renderDialog()

    await user.type(screen.getByLabelText("Name"), "github")
    await user.type(screen.getByLabelText("URL"), "https://mcp.example.com/gh")
    onSubmit.rerender(false)

    expect(
      screen.getByText(
        /You can no longer add servers for the whole organization/,
      ),
    ).toBeVisible()
    const add = within(screen.getByRole("dialog")).getByRole("button", {
      name: "Add MCP server",
    })
    expect(add).toBeDisabled()

    await user.click(screen.getByRole("radio", { name: /This workspace/ }))
    await user.click(add)
    expect(onSubmit).toHaveBeenCalledWith(
      expect.objectContaining({ name: "github" }),
      workspaceMcpScope(WORKSPACE),
    )
    expect(asked.some((call) => call.url === LOOKUP)).toBe(false)
  })
})
