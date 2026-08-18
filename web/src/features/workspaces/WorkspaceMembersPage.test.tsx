import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { WorkspaceMembersPage } from "@/features/workspaces/WorkspaceMembersPage"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import {
  organizationContext,
  organizationMember,
  workspaceMember,
} from "@/tests/fixtures"

const ALPHA = "11111111-1111-1111-1111-111111111111"
const USER = "33333333-3333-3333-3333-333333333333"

function mockApi({
  memberships = [{ workspace_id: ALPHA, name: "Alpha", role: "admin" }],
  members = [workspaceMember({ user_id: USER })],
}: {
  memberships?: { workspace_id: string; name: string; role: string }[]
  members?: ReturnType<typeof workspaceMember>[]
} = {}) {
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input)
    if (url.includes("/members") && url.includes("/v1/workspaces/")) {
      return Response.json({ data: members, count: members.length })
    }
    if (url.includes("/v1/organizations/me/members")) {
      return Response.json({
        data: [organizationMember({ user_id: USER, full_name: "Alex Avery" })],
        count: 1,
      })
    }
    return Response.json(
      organizationContext({ workspace_memberships: memberships }),
    )
  })
}

function renderPage() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <SelectedWorkspaceProvider>
        <WorkspaceMembersPage />
      </SelectedWorkspaceProvider>
    </QueryClientProvider>,
  )
}

describe("WorkspaceMembersPage", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("shows the roster of the selected workspace, named by the organization", async () => {
    mockApi()
    renderPage()

    // The page is about the selected workspace, and says which one.
    expect(await screen.findByText("Members of Alpha")).toBeInTheDocument()
  })

  it("says so rather than showing an empty roster when there is no workspace", async () => {
    mockApi({ memberships: [] })
    renderPage()

    // An empty roster would read as "this workspace has nobody in it", which is
    // a different fact from "you are in no workspace".
    expect(await screen.findByText("No workspace selected")).toBeInTheDocument()
  })
})
