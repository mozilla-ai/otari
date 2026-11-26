import { Outlet } from "@tanstack/react-router"
import { screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
import type { MailReadiness, OrganizationMember } from "@/client"
import { PeoplePage } from "@/features/people/PeoplePage"
import { API_ROOT } from "@/shared/api/client"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  bootstrap,
  organizationContext,
  organizationMember,
  SURFACES,
} from "@/tests/fixtures"
import { AppProviders as Provider } from "@/tests/providers"
import { renderWithRouter } from "@/tests/router"

type Role = NonNullable<
  NonNullable<Parameters<typeof organizationContext>[0]>["role"]
>

function person(name: string, workspaceId: string): OrganizationMember {
  return organizationMember({
    organization_member_id: `om-${name}`,
    user_id: `login-${name}`,
    attribution_user_id: `user-${name}`,
    email: `${name}@example.com`,
    full_name: name,
    role: "member",
    status: "active",
    created_at: "2026-11-01T00:00:00Z",
    workspaces: [
      {
        workspace_id: workspaceId,
        workspace_name: workspaceId === "ws-platform" ? "Platform" : "Mobile",
        workspace_member_id: `wm-${name}`,
        role: "member",
      },
    ],
  })
}

const MEMBERS = [person("Wei", "ws-platform"), person("Hana", "ws-mobile")]

async function renderPeople(
  role: Role,
  selected = "ws-platform",
  { hasTraces = true, mail = "delivers" as MailReadiness } = {},
) {
  window.localStorage.setItem("otari.dashboard.selectedWorkspace", selected)
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const path = String(input).slice(API_ROOT.length).split("?")[0]
    const paged = (data: unknown[]) =>
      Response.json({ data, count: data.length })
    switch (path) {
      case "/organizations/me":
        return Response.json(
          organizationContext({
            role,
            deployment_operator: false,
            workspace_memberships: [
              { workspace_id: "ws-platform", name: "Platform", role: "member" },
              { workspace_id: "ws-mobile", name: "Mobile", role: "member" },
            ],
          }),
        )
      case "/organizations/me/members":
        return paged(MEMBERS)
      case "/organizations/me/traces":
        return Response.json({ items: [], total: 0 })
      case "/organizations/me/keys":
        return Response.json([])
      case "/organizations/me/usage/summary":
        return Response.json({
          totals: { cost: 0, request_count: 0, error_count: 0 },
          series: [],
          groups: [],
        })
      default:
        return paged([])
    }
  })
  return renderWithRouter(<PeoplePage />, {
    url: "/people",
    shell: (
      <Provider>
        <DeploymentProvider
          value={bootstrap({
            mail,
            ...(hasTraces ? { surfaces: [...SURFACES, "traces"] } : {}),
          })}
        >
          <SelectedWorkspaceProvider>
            <Outlet />
          </SelectedWorkspaceProvider>
        </DeploymentProvider>
      </Provider>
    ),
  })
}

describe("People", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("lists only the people of the selected workspace", async () => {
    await renderPeople("owner", "ws-mobile")
    const table = await screen.findByRole("grid", { name: "People" })
    expect(await within(table).findByText("Hana")).toBeVisible()
    expect(within(table).queryByText("Wei")).toBeNull()
  })

  it("follows the workspace when another is selected", async () => {
    await renderPeople("owner", "ws-platform")
    const table = await screen.findByRole("grid", { name: "People" })
    expect(await within(table).findByText("Wei")).toBeVisible()
    expect(within(table).queryByText("Hana")).toBeNull()
  })

  it.each([
    ["delivers", /They get an email with an invitation link/],
    ["sends_only", /reaches an inbox, so you get a link to share/],
    ["none", /reaches an inbox, so you get a link to share/],
  ] as const)(
    "says how an invitee is reached when mail is %s",
    async (mail, wording) => {
      await renderPeople("owner", "ws-platform", { mail })

      await userEvent
        .setup()
        .click(await screen.findByRole("button", { name: "Invite people" }))

      expect(await screen.findByText(wording)).toBeInTheDocument()
    },
  )

  it("links the organization's Members page for an owner", async () => {
    await renderPeople("owner")
    const link = await screen.findByRole("link", { name: "Members & roles" })
    expect(link).toHaveAttribute("href", "/organization/members")
  })

  it("links the organization's Members page for an admin", async () => {
    await renderPeople("admin")
    expect(
      await screen.findByRole("link", { name: "Members & roles" }),
    ).toHaveAttribute("href", "/organization/members")
  })

  it("counts each person's sessions where the deployment records them", async () => {
    await renderPeople("member")
    const table = await screen.findByRole("grid", { name: "People" })
    expect(await within(table).findByText("Wei")).toBeVisible()
    expect(
      within(table).getByRole("columnheader", { name: "Sessions" }),
    ).toBeVisible()
    expect(
      screen.getByText(/You see spend and sessions in your own workspaces/),
    ).toBeVisible()
  })

  it("has no Sessions column where the deployment records none", async () => {
    await renderPeople("member", "ws-platform", { hasTraces: false })
    const table = await screen.findByRole("grid", { name: "People" })
    expect(await within(table).findByText("Wei")).toBeVisible()
    expect(
      within(table).queryByRole("columnheader", { name: "Sessions" }),
    ).toBeNull()
    expect(screen.queryByText(/sessions/i)).toBeNull()
  })

  for (const role of ["member", "viewer"] as const) {
    it(`does not link the organization's Members page for a ${role}`, async () => {
      await renderPeople(role)
      // The workspace roster link renders in the same paragraph, so waiting
      // for it means the page is up before the absence is read.
      expect(
        await screen.findByRole("link", { name: "Workspace members" }),
      ).toHaveAttribute("href", "/members")
      expect(screen.queryByRole("link", { name: "Members & roles" })).toBeNull()
    })
  }
})
