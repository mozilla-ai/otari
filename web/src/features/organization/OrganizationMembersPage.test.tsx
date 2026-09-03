import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactElement } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import type {
  Budget,
  DeploymentBootstrap,
  OrganizationContext,
  OrganizationMember,
  ScopedBudget,
  User,
  Workspace,
  WorkspaceMember,
} from "@/client"
import { OrganizationMembersPage } from "@/features/organization/OrganizationMembersPage"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  bootstrap,
  budget,
  organizationContext,
  organizationMember,
  scopedBudget,
  user,
  workspace,
  workspaceMember,
} from "@/tests/fixtures"
import { pickOption, selectTrigger } from "@/tests/select"

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
  inviteResult?: unknown
  // The gateway's spend rows. The roster joins them on `attribution_user_id`
  // to show what a member may call and what they have spent, neither of which
  // is a column on the membership itself.
  users?: User[]
  // Workspace rosters, keyed by workspace id, and the ceilings keyed on those
  // membership rows. Together they answer "which workspaces, and what budget in
  // each", which the editor writes to.
  workspaceMembers?: Record<string, WorkspaceMember[]>
  scopedBudgets?: ScopedBudget[]
  // The editor picks a budget rather than typing an amount, so the list has to
  // be served for the picker to have anything in it.
  budgets?: Budget[]
}) {
  const context = opts.context ?? organizationContext()
  const members = opts.members ?? [organizationMember()]
  const workspaces = opts.workspaces ?? []
  const users = opts.users ?? []
  const workspaceMembers = opts.workspaceMembers ?? {}
  const scopedBudgets = opts.scopedBudgets ?? []
  const budgetList = opts.budgets ?? []
  const requests: Request[] = []

  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
    requests.push({
      url,
      method,
      body: init?.body ? JSON.parse(String(init.body)) : undefined,
    })

    if (url.includes("/v1/providers")) {
      return jsonResponse({ providers: [] })
    }
    if (url.includes("/v1/models/discoverable")) {
      return jsonResponse({ models: [] })
    }
    if (url.includes("/v1/aliases")) {
      return jsonResponse([])
    }
    if (url.includes("/v1/budgets")) {
      return jsonResponse(budgetList)
    }
    if (url.includes("/v1/scoped-budgets")) {
      if (method === "GET") return jsonResponse(scopedBudgets)
      return jsonResponse(
        scopedBudgets[0] ?? {},
        method === "DELETE" ? 204 : 200,
      )
    }
    if (url.includes("/members") && url.includes("/v1/workspaces/")) {
      const id = url.split("/v1/workspaces/")[1]?.split("/")[0] ?? ""
      const roster = workspaceMembers[id] ?? []
      if (method === "GET") {
        return jsonResponse({ data: roster, count: roster.length })
      }
      return jsonResponse(roster[0] ?? {})
    }
    if (url.includes("/v1/workspaces")) {
      return jsonResponse({ data: workspaces, count: workspaces.length })
    }
    if (url.includes("/v1/users")) {
      return jsonResponse(method === "PATCH" ? users[0] : users)
    }
    if (url.includes("/v1/organizations/me/member-invitations")) {
      if (method === "POST") {
        return jsonResponse(
          opts.inviteResult ?? {
            invitation_id: "invitation-1",
            organization_member_id: "invited-membership",
            email: "new@example.com",
            role: "member",
            status: "invited",
            mail_sent: false,
            accept_link: "/#/accept-invitation?token=abc123",
            expires_at: "2026-01-08T00:00:00+00:00",
            created_at: "2026-01-01T00:00:00+00:00",
          },
          201,
        )
      }
      return jsonResponse({ message: "Invitation revoked" })
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

function renderPage(
  ui: ReactElement,
  bootstrapOverrides: Partial<DeploymentBootstrap> = {},
) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <DeploymentProvider value={bootstrap(bootstrapOverrides)}>
      <QueryClientProvider client={client}>{ui}</QueryClientProvider>
    </DeploymentProvider>,
  )
}

const OWNER = organizationMember({
  organization_member_id: "owner-membership",
  user_id: "aaaaaaaa-0000-0000-0000-000000000000",
  full_name: "Operator",
  role: "owner",
})
const SECOND = "66666666-6666-6666-6666-666666666666"

// What the scope control seeds from and writes back unchanged when the operator
// does not touch it, which is what makes the access write assertable here.
const ANALYST_ACCESS = ["openai:gpt-4o"]

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
    expect(selectTrigger("Role for Analyst")).toHaveTextContent("Member")
  })

  it("changes a member's role through the membership endpoint", async () => {
    const requests = mockApi({ members: [OWNER, ANALYST] })
    const user = userEvent.setup()
    renderPage(<OrganizationMembersPage />)

    await screen.findByText("Analyst")
    await pickOption(user, "Role for Analyst", "Admin")

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
    // The reason is in each control's own name, not only in a tooltip: neither
    // takes focus while disabled, so a pointer is the only thing a `title`
    // would reach.
    expect(
      within(owner).getByLabelText(
        /Role for Operator \(This is the last active owner/,
      ),
    ).toBeDisabled()
    expect(
      within(owner).getByRole("button", {
        name: /Remove Operator \(This is the last active owner/,
      }),
    ).toBeDisabled()
    expect(within(owner).getByText("ACTIVE")).toBeInTheDocument()
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
    expect(within(rowFor("Analyst")).getByText("ACTIVE")).toBeInTheDocument()
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
    await pickOption(user, "Role", "Admin")
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
    expect(
      within(analyst).getByLabelText(
        /Role for Analyst \(Only organization owners and admins/,
      ),
    ).toBeDisabled()
    expect(
      within(analyst).getByRole("button", {
        name: /Remove Analyst \(Only organization owners and admins/,
      }),
    ).toBeDisabled()
    expect(
      screen.getByText(/Only organization owners and admins/),
    ).toBeInTheDocument()
  })

  it("invites a member by email and shows the accept link when mail is not configured", async () => {
    const requests = mockApi({ members: [OWNER] })
    const user = userEvent.setup()
    renderPage(<OrganizationMembersPage />, { mail_ready: false })

    await user.click(
      await screen.findByRole("button", { name: "Invite member" }),
    )
    expect(
      screen.getByText(/Invitation email is unavailable/),
    ).toBeInTheDocument()
    await user.type(screen.getByLabelText("Email address"), "ada@example.com")
    await user.click(screen.getByRole("button", { name: "Send invitation" }))

    const post = requests.find(
      (request) =>
        request.method === "POST" && request.url.includes("member-invitations"),
    )
    expect(post?.body).toMatchObject({
      email: "ada@example.com",
      role: "member",
    })

    // mail_sent is false in the mocked response, so the link is offered to
    // share by hand rather than the form just closing.
    expect(await screen.findByText("Invitation sent")).toBeInTheDocument()
    expect(
      screen.getByText("/#/accept-invitation?token=abc123"),
    ).toBeInTheDocument()
  })

  it("says the email will be sent when mail is configured", async () => {
    mockApi({ members: [OWNER] })
    const user = userEvent.setup()
    renderPage(<OrganizationMembersPage />, { mail_ready: true })

    await user.click(
      await screen.findByRole("button", { name: "Invite member" }),
    )
    expect(
      screen.getByText(/An email with an accept link is sent here/),
    ).toBeInTheDocument()
  })

  it("offers Revoke instead of Remove for an invited row, and revokes through the invitation endpoint", async () => {
    const invited = organizationMember({
      organization_member_id: "invited-membership",
      user_id: "cccccccc-0000-0000-0000-000000000000",
      email: "pending@example.com",
      full_name: null,
      role: "member",
      status: "invited",
      invitation_id: "invitation-1",
    })
    const requests = mockApi({ members: [OWNER, invited] })
    const user = userEvent.setup()
    renderPage(<OrganizationMembersPage />)

    await screen.findByText("pending@example.com")
    const row = rowFor("pending@example.com")
    expect(within(row).queryByRole("button", { name: "Remove" })).toBeNull()
    await user.click(within(row).getByRole("button", { name: "Revoke" }))
    await user.click(screen.getByRole("button", { name: "Revoke invitation" }))

    const revoke = requests.find((request) => request.method === "DELETE")
    expect(revoke?.url).toContain(
      "/v1/organizations/me/member-invitations/invitation-1",
    )
  })

  it("shows what a member may call and what they have spent", async () => {
    // Both read off the gateway's `users` row, reached through the membership's
    // `attribution_user_id`. The roster is the only place they are now, so a
    // regression here is a capability that quietly disappeared with the page
    // these columns replaced.
    mockApi({
      members: [OWNER, ANALYST],
      users: [
        user({
          user_id: ANALYST.attribution_user_id as string,
          allowed_models: ["openai:gpt-4o"],
          spend: 12.5,
          reserved: 2.25,
        }),
      ],
    })
    renderPage(<OrganizationMembersPage />)

    await screen.findByText("Analyst")
    // Awaited rather than read synchronously: these two columns are the
    // deployment operator's and render once the organization context says the
    // caller is one, which can be a paint after the roster itself.
    expect(
      await within(rowFor("Analyst")).findByText("Selected models"),
    ).toBeInTheDocument()
    const row = rowFor("Analyst")
    expect(within(row).getByText("$12.50")).toBeInTheDocument()
    expect(within(row).getByText("$2.25 in flight")).toBeInTheDocument()
  })

  it("leaves the spend cells empty for a member with no spend row", async () => {
    // A member added by address before any key was issued has no `users` row, so
    // there is nothing to report. Empty rather than zero: zero would claim they
    // are on the gateway and have spent nothing.
    mockApi({
      members: [
        OWNER,
        organizationMember({
          organization_member_id: "pending-membership",
          user_id: "cccccccc-0000-0000-0000-000000000000",
          attribution_user_id: null,
          full_name: "Pending",
          role: "member",
        }),
      ],
      users: [],
    })
    renderPage(<OrganizationMembersPage />)

    await screen.findByText("Pending")
    const row = rowFor("Pending")
    expect(within(row).queryByText("All models")).not.toBeInTheDocument()
    expect(
      within(row).queryByRole("button", { name: "Block" }),
    ).not.toBeInTheDocument()
  })

  it("blocks a member through their spend row, not their membership", async () => {
    // Blocking stops their keys without touching the membership, which is what
    // makes it a different act from Remove one column over.
    const requests = mockApi({
      members: [OWNER, ANALYST],
      users: [
        user({
          user_id: ANALYST.attribution_user_id as string,
          blocked: false,
        }),
      ],
    })
    const actor = userEvent.setup()
    renderPage(<OrganizationMembersPage />)

    await screen.findByText("Analyst")
    // Block lives in the Actions column but reads the spend row, so like the
    // two operator-only columns it arrives with the organization context.
    const block = await within(rowFor("Analyst")).findByRole("button", {
      name: "Block",
    })
    await actor.click(block)

    const patch = requests.find(
      (r) => r.method === "PATCH" && r.url.includes("/v1/users/"),
    )
    expect(patch?.body).toEqual({ blocked: true })
  })

  it("edits model access, workspace membership and the workspace budget in one save", async () => {
    // The three used to be separate controls on the row. They are three tables
    // underneath, so this asserts all three writes land from a single save, and
    // that the ceiling is written against the membership rather than the person.
    const requests = mockApi({
      members: [OWNER, ANALYST],
      users: [
        user({
          user_id: ANALYST.attribution_user_id as string,
          allowed_models: ANALYST_ACCESS,
        }),
      ],
      workspaces: [workspace(), workspace({ id: SECOND, name: "Bravo" })],
      workspaceMembers: {
        "44444444-4444-4444-4444-444444444444": [
          workspaceMember({
            id: "membership-1",
            user_id: ANALYST.user_id as string,
            role: "member",
          }),
        ],
      },
      budgets: [
        budget({ budget_id: "bud-small", name: "Small", max_budget: 50 }),
        budget({ budget_id: "bud-large", name: "Large", max_budget: 125 }),
      ],
      scopedBudgets: [
        scopedBudget({
          id: "ceiling-1",
          scope_type: "workspace_member",
          scope_id: "membership-1",
          budget_id: "bud-small",
          max_budget: 50,
        }),
      ],
    })
    const actor = userEvent.setup()
    renderPage(<OrganizationMembersPage />)

    await screen.findByText("Analyst")
    const row = rowFor("Analyst")
    await actor.click(within(row).getByRole("button", { name: "Edit" }))
    await screen.findByText("Workspace access")

    // Already in Default Workspace on the Small budget; move to Large and join
    // Bravo. A budget is picked, never an amount: the figure is the budget's.
    await pickOption(actor, "Budget in Default Workspace", "Large")
    await actor.click(screen.getByLabelText("Bravo"))
    await actor.click(screen.getByRole("button", { name: "Save changes" }))

    // All three writes land from the one save. Model access goes to the spend
    // row the membership is joined to, and it is the third table the editor
    // touches: without this the title would be claiming it without proof.
    const access = requests.find(
      (r) => r.method === "PATCH" && r.url.includes("/v1/users/"),
    )
    expect(access?.body).toEqual({ allowed_models: ANALYST_ACCESS })

    const join = requests.find(
      (r) =>
        r.method === "POST" &&
        r.url.includes(`/v1/workspaces/${SECOND}/members/`),
    )
    expect(join).toBeDefined()

    const ceiling = requests.find(
      (r) =>
        r.method === "PATCH" && r.url.includes("/v1/scoped-budgets/ceiling-1"),
    )
    expect(ceiling?.body).toEqual({ budget_id: "bud-large" })

    // The ordering, not just the presence of both: a ceiling names a membership,
    // so a workspace just joined has no id to name until the server answers.
    // Asserting the sequence is the point of testing the two together.
    expect(requests.indexOf(join!)).toBeLessThan(requests.indexOf(ceiling!))
  })
})

describe("OrganizationMembersPage for a tenant who does not operate the deployment", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("asks for none of the deployment-wide reads", async () => {
    // `/v1/users`, `/v1/budgets` and `/v1/scoped-budgets` have refused a tenant
    // since #821. An organization owner is one, so the page must not ask: the
    // refusals rendered as "this endpoint requires deployment operator access"
    // across a page that is theirs (otari#838).
    const requests = mockApi({
      members: [OWNER, ANALYST],
      context: organizationContext({ deployment_operator: false }),
    })
    renderPage(<OrganizationMembersPage />)

    // The roster having painted is what proves the page got as far as fetching,
    // so an empty list below is a decision not to ask rather than a page that
    // asked for nothing at all.
    await screen.findByText("Analyst")
    expect(
      requests.some((r) => r.url.includes("/v1/organizations/me/members")),
    ).toBe(true)
    for (const path of ["/v1/users", "/v1/budgets", "/v1/scoped-budgets"]) {
      expect(
        requests.filter((r) => r.method === "GET" && r.url.includes(path)),
      ).toHaveLength(0)
    }
  })

  it("drops the columns those reads fed, rather than emptying them", async () => {
    // An em dash here would be indistinguishable from the em dash this table
    // already shows for a member with no gateway identity yet, so the columns go.
    mockApi({
      members: [OWNER, ANALYST],
      context: organizationContext({ deployment_operator: false }),
    })
    renderPage(<OrganizationMembersPage />)

    await screen.findByText("Analyst")
    // "Spend" is still a column and is withheld by id. Model access is not a
    // column any more, it reads under the member's name, so what is asserted
    // here is the marker itself rather than a header that no longer exists.
    expect(screen.queryAllByText("All models")).toHaveLength(0)
    expect(screen.queryByText("Spend")).not.toBeInTheDocument()
    // What is theirs stays.
    expect(screen.getByText("Role")).toBeInTheDocument()
    expect(screen.getByText("Workspaces")).toBeInTheDocument()
  })

  it("reports no refusal, which is the symptom this fixes", async () => {
    mockApi({
      members: [OWNER, ANALYST],
      context: organizationContext({ deployment_operator: false }),
    })
    renderPage(<OrganizationMembersPage />)

    await screen.findByText("Analyst")
    expect(
      screen.queryByText(/requires deployment operator access/i),
    ).not.toBeInTheDocument()
  })

  it("saves a workspace placement without asking for scoped budgets", async () => {
    // The gate cannot live on the query alone: `refetch()` runs the query
    // function even when `enabled` is false, so the editor's ceilings pass would
    // still ask `/v1/scoped-budgets`, be refused, and put the operator refusal
    // back on a page this branch just cleared of it (otari#838).
    const requests = mockApi({
      members: [OWNER, ANALYST],
      workspaces: [workspace(), workspace({ id: SECOND, name: "Bravo" })],
      context: organizationContext({ deployment_operator: false }),
    })
    const actor = userEvent.setup()
    renderPage(<OrganizationMembersPage />)

    await screen.findByText("Analyst")
    await actor.click(
      within(rowFor("Analyst")).getByRole("button", { name: "Edit" }),
    )
    await screen.findByText("Workspace access")
    // No Budget column for this caller, so the only thing to save is placement.
    expect(
      screen.queryByLabelText("Budget in Default Workspace"),
    ).not.toBeInTheDocument()
    await actor.click(screen.getByRole("button", { name: "Save changes" }))

    await waitFor(() =>
      expect(requests.some((r) => r.url.includes("/v1/scoped-budgets"))).toBe(
        false,
      ),
    )
  })

  it("still shows the columns to an operator, so the case above is not vacuous", async () => {
    mockApi({
      members: [OWNER, ANALYST],
      users: [user({ user_id: ANALYST.attribution_user_id as string })],
      context: organizationContext({ deployment_operator: true }),
    })
    renderPage(<OrganizationMembersPage />)

    await screen.findByText("Analyst")
    expect((await screen.findAllByText("All models")).length).toBeGreaterThan(0)
    expect(screen.getByText("Spend")).toBeInTheDocument()
  })
})
