import { Outlet } from "@tanstack/react-router"
import { screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { StrictMode } from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { WorkspaceSwitcher } from "@/app/nav/WorkspaceSwitcher"
import { Provider } from "@/app/provider"
import type { CallerOrganizationMembership } from "@/client"
import { ENTER_HOLD_MS } from "@/features/workspaces/WorkspacesPage"
import * as apiClient from "@/shared/api/client"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import {
  callerOrganizationMembership,
  organization,
  organizationContext,
  workspace,
} from "@/tests/fixtures"
import { renderWithRouter } from "@/tests/router"

const SECOND_ORGANIZATION_ID = "99999999-9999-9999-9999-999999999999"
const CREATED_WORKSPACE_ID = "77777777-7777-7777-7777-777777777777"

/** One membership per organization, the second one being the one to switch to. */
function twoOrganizations(): CallerOrganizationMembership[] {
  return [
    callerOrganizationMembership(),
    callerOrganizationMembership({
      organization_member_id: "88888888-8888-8888-8888-888888888888",
      organization: organization({
        id: SECOND_ORGANIZATION_ID,
        name: "Research",
        slug: "research-1a2b3c4d",
      }),
      role: "member",
      is_active_organization: false,
    }),
  ]
}

interface Recorded {
  url: string
  method: string
  body: unknown
}

/** Spy on the transport, not the hooks, so the real query keys and paths run. */
function mockApi(
  options: {
    memberships?: CallerOrganizationMembership[]
    context?: Parameters<typeof organizationContext>[0]
    switchFails?: boolean
  } = {},
) {
  const requests: Recorded[] = []
  const memberships = options.memberships ?? [callerOrganizationMembership()]
  // Flipped by a successful create, so the context afterwards carries the
  // membership the server would have made the caller an owner through. The
  // switcher reads its workspace list from there, not from the create's answer.
  let createdWorkspace = false
  vi.spyOn(apiClient, "apiFetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = init?.method ?? "GET"
    requests.push({
      url,
      method,
      body: init?.body ? JSON.parse(String(init.body)) : undefined,
    })
    if (url.startsWith("/v1/budgets")) {
      return [] as never
    }
    if (url === "/v1/workspaces" && method === "POST") {
      createdWorkspace = true
      return workspace({ id: CREATED_WORKSPACE_ID, name: "Staging" }) as never
    }
    if (url === "/v1/organizations/me") {
      const base = organizationContext(options.context)
      if (!createdWorkspace) return base as never
      return {
        ...base,
        workspace_memberships: [
          ...(base.workspace_memberships ?? []),
          {
            workspace_id: CREATED_WORKSPACE_ID,
            name: "Staging",
            role: "owner",
          },
        ],
      } as never
    }
    if (url.startsWith("/v1/organizations/me/memberships")) {
      return { data: memberships, count: memberships.length } as never
    }
    if (url === "/v1/organizations/me/switch") {
      if (options.switchFails) {
        throw new apiClient.ApiError(404, "Organization not found")
      }
      return organizationContext() as never
    }
    if (url === "/v1/organizations") {
      return organization({ id: SECOND_ORGANIZATION_ID }) as never
    }
    return organizationContext(options.context) as never
  })
  return requests
}

function renderSwitcher() {
  return renderWithRouter(
    <Provider>
      <SelectedWorkspaceProvider>
        <WorkspaceSwitcher collapsed={false} />
      </SelectedWorkspaceProvider>
    </Provider>,
  )
}

async function openMenu() {
  const user = userEvent.setup()
  await user.click(
    await screen.findByRole("button", { name: /^Switch workspace/ }),
  )
  return { user, menu: await screen.findByRole("dialog") }
}

describe("the organization half of the scope switcher", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.localStorage.clear()
  })

  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("states the organization rather than offering a switch when there is one", async () => {
    mockApi()
    await renderSwitcher()

    const { menu } = await openMenu()

    // The name is there, and not as something to press: a row whose only effect
    // would be to close the menu reads as broken.
    expect(await within(menu).findByText("Default Organization")).toBeVisible()
    expect(
      within(menu).queryByRole("button", { name: /Default Organization/ }),
    ).toBeNull()
  })

  it("switches to another organization the caller belongs to", async () => {
    const requests = mockApi({ memberships: twoOrganizations() })
    await renderSwitcher()

    const { user, menu } = await openMenu()
    await user.click(
      await within(menu).findByRole("button", { name: /Research/ }),
    )

    const posted = requests.find((request) => request.method === "POST")
    expect(posted?.url).toBe("/v1/organizations/me/switch")
    expect(posted?.body).toEqual({ organization_id: SECOND_ORGANIZATION_ID })
  })

  it("marks the organization it is already in, and does not re-switch to it", async () => {
    const requests = mockApi({ memberships: twoOrganizations() })
    await renderSwitcher()

    const { user, menu } = await openMenu()
    const current = await within(menu).findByRole("button", {
      name: /Default Organization/,
    })
    expect(within(current).getByText("Selected")).toBeInTheDocument()
    await user.click(current)

    expect(requests.some((request) => request.method === "POST")).toBe(false)
  })

  it("reports a switch that was refused instead of closing on a scope that did not move", async () => {
    mockApi({ memberships: twoOrganizations(), switchFails: true })
    await renderSwitcher()

    const { user, menu } = await openMenu()
    await user.click(
      await within(menu).findByRole("button", { name: /Research/ }),
    )

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Organization not found",
    )
    // Still open, which is what makes the message readable at all.
    expect(screen.getByRole("dialog")).toBeInTheDocument()
  })

  it("creates an organization and moves into it", async () => {
    const requests = mockApi()
    await renderSwitcher()

    const { user, menu } = await openMenu()
    await user.click(
      within(menu).getByRole("button", { name: /Create organization/ }),
    )
    await user.type(await screen.findByLabelText(/Name/), "Research")
    const form = await screen.findByRole("dialog")
    await user.click(
      within(form).getByRole("button", { name: "Create organization" }),
    )

    const posts = requests.filter((request) => request.method === "POST")
    expect(posts.map((request) => request.url)).toEqual([
      "/v1/organizations",
      "/v1/organizations/me/switch",
    ])
    expect(posts[0]?.body).toEqual({ name: "Research" })
    // The second call is what makes the new organization the one on screen; the
    // server deliberately does not switch as a side effect of creating.
    expect(posts[1]?.body).toEqual({
      organization_id: SECOND_ORGANIZATION_ID,
    })
  })

  it("offers Create organization whatever the caller's role in the one they are in", async () => {
    // No role in an organization gates creating one: it is not an action inside
    // a tenant, which is also why the server checks only the credential. Create
    // *workspace* beside it is the one that is gated.
    mockApi({ context: { role: "viewer" } })
    await renderSwitcher()

    const { menu } = await openMenu()

    expect(
      within(menu).getByRole("button", { name: /Create organization/ }),
    ).toBeInTheDocument()
    expect(
      within(menu).queryByRole("button", { name: "Create workspace" }),
    ).toBeNull()
  })
})

// The workspace half's create flow, which ends inside the workspace it made.
// Rendered in the shell slot rather than as the page, and started away from "/",
// so the navigation has somewhere to land and something to leave: a switcher
// rendered as the page's own content would unmount the moment it navigated.
function renderSwitcherOnAPage(options: { strict?: boolean } = {}) {
  const shell = (
    <Provider>
      <SelectedWorkspaceProvider>
        <WorkspaceSwitcher collapsed={false} />
        <Outlet />
      </SelectedWorkspaceProvider>
    </Provider>
  )
  return renderWithRouter(<div>USAGE PAGE</div>, {
    url: "/usage",
    routes: [{ path: "/", element: <div>OVERVIEW PAGE</div> }],
    // StrictMode remounts every component once, which is what development does
    // and what the guard inside the create form has to survive.
    shell: options.strict ? <StrictMode>{shell}</StrictMode> : shell,
  })
}

/** Open the create-workspace form from the menu and name the workspace. */
async function fillCreateForm(user: ReturnType<typeof userEvent.setup>) {
  await user.click(
    await screen.findByRole("button", { name: /^Switch workspace/ }),
  )
  await user.click(
    within(await screen.findByRole("dialog")).getByRole("button", {
      name: "Create workspace",
    }),
  )
  const form = await screen.findByRole("dialog", { name: "Create workspace" })
  await user.type(within(form).getByLabelText(/^Name/), "Staging")
  return form
}

describe("the workspace half of the scope switcher", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.localStorage.clear()
  })

  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  // The submit holds for `ENTER_HOLD_MS` before handing the page over, which is
  // inside `findBy*`'s own ceiling, so the cases below wait on the DOM rather
  // than on the clock. Fake timers are not an option: `userEvent` drives
  // react-aria's pointer events through real ones, and freezing the clock
  // deadlocks the interactions rather than the wait.

  // A workspace to start in, so "moved into the new one" is distinguishable
  // from "the new one is the only one there is": without a prior membership the
  // provider falls back to the single workspace it finds and every case passes.
  const startedInAWorkspace = {
    workspace_memberships: [
      {
        workspace_id: workspace().id,
        name: "Default Workspace",
        role: "owner",
      },
    ],
  }

  it("enters the workspace it just created", async () => {
    mockApi({ context: startedInAWorkspace })
    const user = userEvent.setup()
    await renderSwitcherOnAPage()

    const form = await fillCreateForm(user)
    // The label names the navigation, which is the only warning the operator
    // gets that the page is about to change under them.
    const submit = within(form).getByRole("button", { name: /Create and open/ })
    await user.click(submit)

    // The press is acknowledged before the page moves, rather than the create
    // landing them somewhere else with nothing in between. The button keeps its
    // name through the wait, so it is still the control it was.
    expect(submit).toHaveAttribute("data-pending", "true")
    expect(screen.queryByText("OVERVIEW PAGE")).toBeNull()

    // Both halves of "entering" it: the shell's scope moved to the new
    // workspace, and the flow ended on the page that scope reads.
    expect(await screen.findByText("OVERVIEW PAGE")).toBeInTheDocument()
    expect(
      await screen.findByRole("button", {
        name: /^Switch workspace, currently Staging/,
      }),
    ).toBeInTheDocument()
  })

  it("does not enter a workspace the operator dismissed the form over", async () => {
    mockApi({ context: startedInAWorkspace })
    const user = userEvent.setup()
    await renderSwitcherOnAPage()

    const form = await fillCreateForm(user)
    await user.click(
      within(form).getByRole("button", { name: /Create and open/ }),
    )
    // Escape rather than Cancel: Cancel is disabled while the create is in
    // flight, so dismissal is what is left, and it is the path that bypasses
    // every button.
    await user.keyboard("{Escape}")
    // The only case here that waits on the clock, because what it asserts is an
    // absence: there is no event to wait on when the completion is correctly
    // dropped. Bounded by the hold itself rather than by a copy of it.
    await new Promise((resolve) => {
      setTimeout(resolve, ENTER_HOLD_MS + 100)
    })

    // The workspace was created and the switcher will list it; what must not
    // happen is being taken there after saying not to.
    expect(screen.queryByText("OVERVIEW PAGE")).toBeNull()
    expect(screen.getByText("USAGE PAGE")).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: /^Switch workspace/ }),
    ).toHaveAccessibleName(/currently Default Workspace/)
  })

  it("still completes after StrictMode's development remount", async () => {
    // The guard that drops the completion after dismissal is a ref, and a
    // cleanup-only effect would leave it cleared by the remount: every create
    // would then hang with the modal open and the button spinning, in
    // development only, where `main.tsx` wraps the app in StrictMode.
    mockApi({ context: startedInAWorkspace })
    const user = userEvent.setup()
    await renderSwitcherOnAPage({ strict: true })

    const form = await fillCreateForm(user)
    await user.click(
      within(form).getByRole("button", { name: /Create and open/ }),
    )

    expect(await screen.findByText("OVERVIEW PAGE")).toBeInTheDocument()
  })
})
