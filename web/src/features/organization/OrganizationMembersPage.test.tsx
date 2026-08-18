import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactElement } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import type {
  OrganizationContext,
  OrganizationMember,
  Workspace,
} from "@/client"
import { OrganizationMembersPage } from "@/features/organization/OrganizationMembersPage"
import {
  organizationContext,
  organizationMember,
  workspace,
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

function mockApi(opts: {
  context?: OrganizationContext
  members?: OrganizationMember[]
  workspaces?: Workspace[]
}) {
  const context = opts.context ?? organizationContext()
  const members = opts.members ?? [organizationMember()]
  const workspaces = opts.workspaces ?? []
  const requests: Request[] = []

  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
    requests.push({
      url,
      method,
      body: init?.body ? JSON.parse(String(init.body)) : undefined,
    })

    if (url.includes("/v1/workspaces")) {
      return jsonResponse({ data: workspaces, count: workspaces.length })
    }
    if (url.includes("/v1/organizations/me/members")) {
      if (method === "GET") {
        return jsonResponse({ data: members, count: members.length })
      }
      if (method === "POST") {
        return jsonResponse(
          { status: "active", email: "new@example.com", role: "member" },
          201,
        )
      }
      return jsonResponse(members[0])
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

const OWNER = organizationMember({
  organization_member_id: "owner-membership",
  user_id: "aaaaaaaa-0000-0000-0000-000000000000",
  full_name: "Operator",
  role: "owner",
})
const ANALYST = organizationMember({
  organization_member_id: "analyst-membership",
  user_id: "bbbbbbbb-0000-0000-0000-000000000000",
  full_name: "Analyst",
  email: "analyst@example.com",
  role: "member",
})

function rowFor(name: string) {
  return screen
    .getAllByRole("row")
    .find((row) => within(row).queryByText(name) !== null) as HTMLElement
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe("OrganizationMembersPage", () => {
  it("lists the roster with each member's role and status", async () => {
    mockApi({ members: [OWNER, ANALYST] })
    renderPage(<OrganizationMembersPage />)

    expect(await screen.findByText("Analyst")).toBeInTheDocument()
    expect(screen.getByText("analyst@example.com")).toBeInTheDocument()
    expect(screen.getByLabelText("Role for Analyst")).toHaveValue("member")
  })

  it("changes a member's role through the membership endpoint", async () => {
    const requests = mockApi({ members: [OWNER, ANALYST] })
    const user = userEvent.setup()
    renderPage(<OrganizationMembersPage />)

    const role = await screen.findByLabelText("Role for Analyst")
    await user.selectOptions(role, "admin")

    const patch = requests.find((request) => request.method === "PATCH")
    expect(patch?.url).toContain(
      "/v1/organizations/me/members/analyst-membership",
    )
    expect(patch?.body).toEqual({ role: "admin" })
  })

  it("locks the last active owner's membership rather than letting it be cleared", async () => {
    mockApi({ members: [OWNER, ANALYST] })
    renderPage(<OrganizationMembersPage />)

    await screen.findByText("Operator")
    const owner = rowFor("Operator")
    // Demoting or removing the last active owner leaves nobody able to manage
    // the organization, so the server refuses it and the page says so up front.
    expect(within(owner).getByLabelText("Role for Operator")).toBeDisabled()
    expect(within(owner).getByRole("button", { name: "Remove" })).toBeDisabled()
    expect(within(owner).getByText("Active")).toBeInTheDocument()
  })

  it("suspends a member rather than deleting them, and says so", async () => {
    const requests = mockApi({ members: [OWNER, ANALYST] })
    const user = userEvent.setup()
    renderPage(<OrganizationMembersPage />)

    await screen.findByText("Analyst")
    await user.click(
      within(rowFor("Analyst")).getByRole("button", { name: "Remove" }),
    )
    expect(
      screen.getByText(/suspended rather than\s+deleted/),
    ).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Remove member" }))

    const remove = requests.find((request) => request.method === "DELETE")
    expect(remove?.url).toContain(
      "/v1/organizations/me/members/analyst-membership",
    )
  })

  it("shows a status rather than offering one to set", async () => {
    mockApi({ members: [OWNER, ANALYST] })
    renderPage(<OrganizationMembersPage />)

    await screen.findByText("Analyst")
    // The gateway takes two settable statuses, and suspending is what Remove
    // already does with a confirmation in front of it, so a dropdown here would
    // be an unconfirmed removal. The other direction has no subject: a
    // suspended membership is not listable, so no row exists to reactivate.
    expect(screen.queryByLabelText("Status for Analyst")).toBeNull()
    expect(within(rowFor("Analyst")).getByText("Active")).toBeInTheDocument()
  })

  it("adds a member by address, into the workspaces that were ticked", async () => {
    const requests = mockApi({
      members: [OWNER],
      workspaces: [workspace({ id: "ws-1", name: "Production" })],
    })
    const user = userEvent.setup()
    renderPage(<OrganizationMembersPage />)

    await user.click(await screen.findByRole("button", { name: "Add member" }))
    await user.type(screen.getByLabelText("Email address"), "ada@example.com")
    await user.selectOptions(screen.getByLabelText("Role"), "admin")
    // Ticked by default, since a member in no workspace can reach nothing.
    expect(await screen.findByLabelText("Production")).toBeChecked()
    // The header action hides itself while the form is open, so the remaining
    // button of this name is the form's own submit.
    await user.click(screen.getByRole("button", { name: "Add member" }))

    const post = requests.find((request) => request.method === "POST")
    expect(post?.url).toContain("/v1/organizations/me/members")
    expect(post?.body).toEqual({
      email: "ada@example.com",
      role: "admin",
      workspace_assignments: [{ workspace_id: "ws-1", role: "member" }],
    })
  })

  it("leaves the default alone once the operator has cleared it", async () => {
    mockApi({
      members: [OWNER],
      workspaces: [workspace({ id: "ws-1", name: "Production" })],
    })
    const user = userEvent.setup()
    renderPage(<OrganizationMembersPage />)

    await user.click(await screen.findByRole("button", { name: "Add member" }))
    const production = await screen.findByLabelText("Production")
    await user.click(production)

    // The seed is a starting point, not a value re-imposed on every render:
    // typing after clearing it must not tick the box again.
    await user.type(screen.getByLabelText("Email address"), "ada@example.com")
    expect(production).not.toBeChecked()
  })

  it("sends no assignment list, and says so, when every workspace is unticked", async () => {
    const requests = mockApi({
      members: [OWNER],
      workspaces: [workspace({ id: "ws-1", name: "Production" })],
    })
    const user = userEvent.setup()
    renderPage(<OrganizationMembersPage />)

    await user.click(await screen.findByRole("button", { name: "Add member" }))
    await user.type(screen.getByLabelText("Email address"), "ada@example.com")
    // Deliberate now rather than the default: clearing the seeded workspace is
    // a choice, and the form says what it costs before the request goes.
    await user.click(await screen.findByLabelText("Production"))
    expect(screen.getByText(/no workspace/)).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Add member" }))

    const post = requests.find((request) => request.method === "POST")
    // No assignment is not the same request as an empty list of them.
    expect(post?.body).toEqual({
      email: "ada@example.com",
      role: "member",
      workspace_assignments: null,
    })
  })

  it("offers no membership control to a caller who cannot manage the organization", async () => {
    mockApi({
      context: organizationContext({ role: "viewer" }),
      members: [OWNER, ANALYST],
    })
    renderPage(<OrganizationMembersPage />)

    await screen.findByText("Analyst")
    const analyst = rowFor("Analyst")
    expect(within(analyst).getByLabelText("Role for Analyst")).toBeDisabled()
    expect(
      within(analyst).getByRole("button", { name: "Remove" }),
    ).toBeDisabled()
    expect(
      screen.getByText(/Only organization owners and admins/),
    ).toBeInTheDocument()
  })
})
