import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
import { WorkspaceMembersPage } from "@/features/workspaces/WorkspaceMembersPage"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import {
  organizationContext,
  organizationMember,
  workspaceMember,
} from "@/tests/fixtures"
import { renderWithRouter } from "@/tests/router"
import { pickOption } from "@/tests/select"

const ALPHA = "11111111-1111-1111-1111-111111111111"
const USER = "33333333-3333-3333-3333-333333333333"

interface Request {
  url: string
  method: string
}

function mockApi({
  memberships = [{ workspace_id: ALPHA, name: "Alpha", role: "admin" }],
  members = [workspaceMember({ user_id: USER })],
  orgMembers = [organizationMember({ user_id: USER, full_name: "Alex Avery" })],
  context = organizationContext(),
  rosterFails = false,
}: {
  memberships?: { workspace_id: string; name: string; role: string }[]
  members?: ReturnType<typeof workspaceMember>[]
  orgMembers?: ReturnType<typeof organizationMember>[]
  context?: ReturnType<typeof organizationContext>
  rosterFails?: boolean
} = {}) {
  const requests: Request[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    requests.push({ url, method: (init?.method ?? "GET").toUpperCase() })
    if (url.includes("/members") && url.includes("/v1/workspaces/")) {
      return Response.json({ data: members, count: members.length })
    }
    if (url.includes("/v1/organizations/me/members")) {
      if (rosterFails) {
        return Response.json({ detail: "Roster unavailable" }, { status: 500 })
      }
      return Response.json({ data: orgMembers, count: orgMembers.length })
    }
    return Response.json({
      ...context,
      workspace_memberships: memberships,
    })
  })
  return requests
}

// In a router, because the page links to Organization > Members & roles for a
// caller who manages the organization.
function renderPage() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return renderWithRouter(
    <QueryClientProvider client={client}>
      <SelectedWorkspaceProvider>
        <WorkspaceMembersPage />
      </SelectedWorkspaceProvider>
    </QueryClientProvider>,
    { url: "/members" },
  )
}

describe("WorkspaceMembersPage", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("shows the roster of the selected workspace, named by the organization", async () => {
    mockApi()
    await renderPage()

    // The page is about the selected workspace, and says which one.
    expect(await screen.findByText("Members of Alpha")).toBeInTheDocument()
  })

  it("says so rather than showing an empty roster when there is no workspace", async () => {
    mockApi({ memberships: [] })
    await renderPage()

    // An empty roster would read as "this workspace has nobody in it", which is
    // a different fact from "you are in no workspace".
    expect(await screen.findByText("No workspace selected")).toBeInTheDocument()
  })

  it("enables management controls for a workspace owner even without an organization role", async () => {
    // "member" at the organization level, but "owner" of the selected
    // workspace itself: the server's OR rule
    // (`_require_workspace_management_access`), which the page has to match.
    mockApi({
      memberships: [{ workspace_id: ALPHA, name: "Alpha", role: "owner" }],
    })
    await renderPage()

    await screen.findByText("Members of Alpha")
    expect(await screen.findByRole("button", { name: "Remove" })).toBeEnabled()
  })

  // The three below moved here with the panel. They used to run against the
  // copy embedded in the Workspaces page, which is gone: the same component
  // rendered on two rails was the duplication Fede flagged, and this is the
  // surface that survives.

  it("says why there is nobody left to add once the roster is exhausted", async () => {
    // The one organization member is already in the workspace, which in a
    // standalone deployment is the permanent state until sign-in lands.
    mockApi()
    await renderPage()

    expect(
      await screen.findByText(/already in this workspace/),
    ).toBeInTheDocument()
    expect(screen.queryByLabelText("Organization member")).toBeNull()
  })

  it("reports a roster that failed instead of calling the workspace full", async () => {
    // An empty candidate list is only "everyone is already here" once the
    // roster has answered; a failed one has to say so instead.
    mockApi({ rosterFails: true })
    await renderPage()

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Roster unavailable",
    )
    expect(screen.queryByText(/already in this workspace/)).toBeNull()
  })

  it("adds an organization member to a workspace with a chosen role", async () => {
    const requests = mockApi({
      orgMembers: [
        organizationMember({ user_id: USER, full_name: "Alex Avery" }),
        organizationMember({
          organization_member_id: "analyst-membership",
          user_id: "77777777-7777-7777-7777-777777777777",
          full_name: "Analyst",
          role: "member",
        }),
      ],
    })
    const user = userEvent.setup()
    await renderPage()

    await pickOption(user, "Organization member", "Analyst")
    await pickOption(user, "Role", "Admin")
    await user.click(screen.getByRole("button", { name: "Add member" }))

    const post = requests.find(
      (request) =>
        request.method === "POST" && request.url.includes("/members"),
    )
    // The role travels as a query parameter, which is the rehomed wire
    // contract; a body would be ignored.
    expect(post?.url).toContain(
      `/v1/workspaces/${ALPHA}/members/77777777-7777-7777-7777-777777777777?role=admin`,
    )
  })

  it("offers a non-manager the roster but none of its write controls", async () => {
    // A viewer at the organization level with no role on this workspace: the
    // roster still reads, and nothing on it can be changed.
    mockApi({
      context: organizationContext({ role: "viewer" }),
      memberships: [{ workspace_id: ALPHA, name: "Alpha", role: "viewer" }],
    })
    await renderPage()

    await screen.findByText("Members of Alpha")
    expect(await screen.findByRole("button", { name: "Remove" })).toBeDisabled()
  })

  // otari-ai#1960: the organization's roster is Members & roles, on a rail the
  // shell opens only to a caller who manages the organization, so a member had
  // nowhere to read who else is in their organization.

  it("shows a member the organization roster they have no other page for", async () => {
    mockApi({
      context: organizationContext({
        role: "member",
        deployment_operator: false,
      }),
      memberships: [{ workspace_id: ALPHA, name: "Alpha", role: "member" }],
      orgMembers: [
        organizationMember({
          user_id: USER,
          full_name: "Alex Avery",
          role: "member",
        }),
        organizationMember({
          organization_member_id: "owner-membership",
          user_id: "77777777-7777-7777-7777-777777777777",
          full_name: "Bea Brooks",
          role: "owner",
        }),
      ],
    })
    await renderPage()

    const roster = await screen.findByRole("grid", {
      name: "Organization members",
    })
    // Someone the workspace roster above does not name: the point of the card is
    // the organization beyond this workspace.
    expect(within(roster).getByText("Bea Brooks")).toBeInTheDocument()
    expect(within(roster).getByText("Owner")).toBeInTheDocument()
    // Read-only: the roles and the removals are the management page's.
    expect(within(roster).queryByRole("button")).toBeNull()
  })

  it("shows the organization roster to a member who is in no workspace yet", async () => {
    // The caller with the most use for it: nothing else on this page answers
    // for them, and who to ask about a workspace is on the list.
    mockApi({
      context: organizationContext({
        role: "member",
        deployment_operator: false,
      }),
      memberships: [],
    })
    await renderPage()

    expect(await screen.findByText("No workspace selected")).toBeInTheDocument()
    expect(
      await screen.findByRole("grid", { name: "Organization members" }),
    ).toBeInTheDocument()
  })

  it("reports a failed roster to a member rather than an empty organization", async () => {
    // The card cannot tell an empty organization from an unanswered read, so the
    // page's banner is what says which happened.
    mockApi({
      context: organizationContext({
        role: "member",
        deployment_operator: false,
      }),
      memberships: [{ workspace_id: ALPHA, name: "Alpha", role: "member" }],
      rosterFails: true,
    })
    await renderPage()

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Roster unavailable",
    )
    const roster = await screen.findByRole("grid", {
      name: "Organization members",
    })
    expect(within(roster).getByText("No members to show.")).toBeInTheDocument()
  })

  it("points a manager at the organization page instead of repeating its roster", async () => {
    // An owner or admin has Members & roles, which lists the same people and
    // can change them, so the read-only copy would be the same table twice.
    mockApi()
    await renderPage()

    expect(
      await screen.findByRole("link", { name: /Members & roles/ }),
    ).toHaveAttribute("href", "/organization/members")
    expect(
      screen.queryByRole("grid", { name: "Organization members" }),
    ).toBeNull()
  })
})
