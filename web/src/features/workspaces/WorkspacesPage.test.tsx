import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactElement } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import type {
  OrganizationContext,
  OrganizationMember,
  Workspace,
  WorkspaceMember,
} from "@/client"
import { WorkspacesPage } from "@/features/workspaces/WorkspacesPage"
import {
  organizationContext,
  organizationMember,
  workspace,
  workspaceMember,
} from "@/tests/fixtures"

interface Request {
  url: string
  method: string
  body: unknown
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

function mockApi(
  opts: {
    context?: OrganizationContext
    workspaces?: Workspace[]
    members?: WorkspaceMember[]
    orgMembers?: OrganizationMember[]
  } = {},
) {
  const context = opts.context ?? organizationContext()
  const list = opts.workspaces ?? [workspace()]
  const members = opts.members ?? [workspaceMember()]
  const orgMembers = opts.orgMembers ?? [organizationMember()]
  const requests: Request[] = []

  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
    requests.push({
      url,
      method,
      body: init?.body ? JSON.parse(String(init.body)) : undefined,
    })

    if (url.includes("/members") && url.includes("/v1/workspaces/")) {
      if (method === "GET") {
        return jsonResponse({ data: members, count: members.length })
      }
      return jsonResponse(members[0] ?? workspaceMember())
    }
    if (url.includes("/v1/workspaces")) {
      if (method === "GET") {
        return jsonResponse({ data: list, count: list.length })
      }
      if (method === "DELETE") return jsonResponse({ message: "deleted" })
      return jsonResponse(workspace({ name: "Created" }))
    }
    if (url.includes("/v1/organizations/me/members")) {
      return jsonResponse({ data: orgMembers, count: orgMembers.length })
    }
    return jsonResponse(context)
  })

  return requests
}

function renderPage(ui: ReactElement) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>)
}

afterEach(() => {
  vi.restoreAllMocks()
})

const SECOND = "66666666-6666-6666-6666-666666666666"

describe("WorkspacesPage", () => {
  it("lists the organization's workspaces", async () => {
    mockApi({
      workspaces: [
        workspace({ name: "Production", description: "Live traffic" }),
      ],
    })
    renderPage(<WorkspacesPage />)

    expect(await screen.findByText("Production")).toBeInTheDocument()
    expect(screen.getByText("Live traffic")).toBeInTheDocument()
  })

  it("creates a workspace with a name and a description", async () => {
    const requests = mockApi({})
    const user = userEvent.setup()
    renderPage(<WorkspacesPage />)

    await user.click(
      await screen.findByRole("button", { name: "Create workspace" }),
    )
    await user.type(screen.getByLabelText("Name"), "Research")
    await user.type(
      screen.getByLabelText("Description (optional)"),
      "Experiments",
    )
    // The header action hides itself while the form is open, so the only
    // button left with this name is the form's own submit.
    await user.click(screen.getByRole("button", { name: "Create workspace" }))

    const post = requests.find((request) => request.method === "POST")
    expect(post?.url).toContain("/v1/workspaces")
    expect(post?.body).toEqual({
      name: "Research",
      description: "Experiments",
    })
  })

  it("opens a workspace's roster and names the identity behind each row", async () => {
    mockApi({
      orgMembers: [
        organizationMember({
          user_id: "33333333-3333-3333-3333-333333333333",
          full_name: "Operator",
        }),
      ],
    })
    const user = userEvent.setup()
    renderPage(<WorkspacesPage />)

    await user.click(await screen.findByRole("button", { name: "Members" }))
    expect(
      await screen.findByText("Members of Default Workspace"),
    ).toBeInTheDocument()
    // The roster carries user ids only, so the name comes from the
    // organization's roster rather than from the workspace endpoint.
    expect(screen.getByText("Operator")).toBeInTheDocument()
  })

  it("says why there is nobody left to add once the roster is exhausted", async () => {
    mockApi({})
    const user = userEvent.setup()
    renderPage(<WorkspacesPage />)

    await user.click(await screen.findByRole("button", { name: "Members" }))
    // The one organization member is already in the workspace, which in a
    // standalone deployment is the permanent state until sign-in lands.
    expect(
      await screen.findByText(/already in this workspace/),
    ).toBeInTheDocument()
    expect(screen.queryByLabelText("Organization member")).toBeNull()
  })

  it("reports a roster that failed instead of calling the workspace full", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input)
      if (url.includes("/v1/organizations/me/members")) {
        return jsonResponse({ detail: "Roster unavailable" }, 500)
      }
      if (url.includes("/members")) {
        return jsonResponse({ data: [workspaceMember()], count: 1 })
      }
      if (url.includes("/v1/workspaces")) {
        return jsonResponse({ data: [workspace()], count: 1 })
      }
      return jsonResponse(organizationContext())
    })
    const user = userEvent.setup()
    renderPage(<WorkspacesPage />)

    await user.click(await screen.findByRole("button", { name: "Members" }))
    // An empty candidate list is only "everyone is already here" once the
    // roster has answered; a failed one has to say so instead.
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Roster unavailable",
    )
    expect(screen.queryByText(/already in this workspace/)).toBeNull()
  })

  it("adds an organization member to a workspace with a chosen role", async () => {
    const requests = mockApi({
      orgMembers: [
        organizationMember({
          organization_member_id: "owner-membership",
          user_id: "33333333-3333-3333-3333-333333333333",
          full_name: "Operator",
        }),
        organizationMember({
          organization_member_id: "analyst-membership",
          user_id: "77777777-7777-7777-7777-777777777777",
          full_name: "Analyst",
          role: "member",
        }),
      ],
    })
    const user = userEvent.setup()
    renderPage(<WorkspacesPage />)

    await user.click(await screen.findByRole("button", { name: "Members" }))
    const picker = await screen.findByLabelText("Organization member")
    await user.selectOptions(picker, "77777777-7777-7777-7777-777777777777")
    await user.selectOptions(screen.getByLabelText("Role"), "admin")
    await user.click(screen.getByRole("button", { name: "Add member" }))

    const post = requests.find(
      (request) =>
        request.method === "POST" && request.url.includes("/members"),
    )
    // The role travels as a query parameter, which is the rehomed wire
    // contract; a body would be ignored.
    expect(post?.url).toContain(
      "/v1/workspaces/44444444-4444-4444-4444-444444444444/members/77777777-7777-7777-7777-777777777777?role=admin",
    )
  })

  it("confirms before deleting a workspace", async () => {
    // Two, because the last workspace cannot be deleted and its button says so.
    const requests = mockApi({
      workspaces: [workspace(), workspace({ id: SECOND, name: "Bravo" })],
    })
    const user = userEvent.setup()
    renderPage(<WorkspacesPage />)

    const deletes = await screen.findAllByRole("button", { name: "Delete" })
    await user.click(deletes[0])
    await user.click(screen.getByRole("button", { name: "Delete workspace" }))

    const remove = requests.find((request) => request.method === "DELETE")
    expect(remove?.url).toContain(
      "/v1/workspaces/44444444-4444-4444-4444-444444444444",
    )
  })

  it("refuses to offer the last workspace's deletion, and says why", async () => {
    // The server keeps every organization on at least one workspace, and first
    // boot is that state, so an enabled button here is one that always 400s.
    mockApi({})
    renderPage(<WorkspacesPage />)

    const remove = await screen.findByRole("button", {
      name: /^Delete Default Workspace \(/,
    })
    expect(remove).toBeDisabled()
    expect(remove).toHaveAccessibleName(/keeps at least one workspace/)
  })

  it("offers a non-manager the roster but none of the write controls", async () => {
    // Two workspaces, so the Delete button is about the caller's role rather
    // than about the last-workspace rule.
    mockApi({
      context: organizationContext({ role: "viewer" }),
      workspaces: [workspace(), workspace({ id: SECOND, name: "Bravo" })],
    })
    const user = userEvent.setup()
    renderPage(<WorkspacesPage />)

    await screen.findByText("Default Workspace")
    expect(
      screen.queryByRole("button", { name: "Create workspace" }),
    ).toBeNull()
    expect(screen.getAllByRole("button", { name: "Edit" })[0]).toBeDisabled()
    expect(screen.getAllByRole("button", { name: "Delete" })[0]).toBeDisabled()

    await user.click(screen.getAllByRole("button", { name: "Members" })[0])
    const roster = await screen.findByText("Members of Default Workspace")
    expect(roster).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Remove" })).toBeDisabled()
  })

  it("renames a workspace through the update endpoint", async () => {
    const requests = mockApi({})
    const user = userEvent.setup()
    renderPage(<WorkspacesPage />)

    await user.click(await screen.findByRole("button", { name: "Edit" }))
    const name = screen.getByLabelText("Name")
    await user.clear(name)
    await user.type(name, "Renamed")
    await user.click(screen.getByRole("button", { name: "Save changes" }))

    const patch = requests.find((request) => request.method === "PATCH")
    expect(patch?.url).toContain(
      "/v1/workspaces/44444444-4444-4444-4444-444444444444",
    )
    expect(patch?.body).toEqual({ name: "Renamed", description: null })
  })
})
