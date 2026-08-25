import { Outlet } from "@tanstack/react-router"
import { screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { StrictMode } from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { WorkspaceSwitcher } from "@/app/nav/WorkspaceSwitcher"
import { Provider } from "@/app/provider"
import type { CallerOrganizationMembership } from "@/client"
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
/** A hold the test opens, so the beat is a gate rather than a duration. */
function pendingHold() {
  let release = () => {}
  const gate = new Promise<void>((resolve) => {
    release = resolve
  })
  return { hold: () => gate, release: () => release() }
}

function renderSwitcherOnAPage(
  options: { strict?: boolean; hold?: () => Promise<void> } = {},
) {
  const shell = (
    <Provider>
      <SelectedWorkspaceProvider>
        <WorkspaceSwitcher collapsed={false} createHold={options.hold} />
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

  // The submit holds for a beat before handing the page over, and these cases
  // supply that beat themselves (`pendingHold`) rather than waiting out the
  // shipped one, so the window the guard exists for is entered and left on
  // purpose. No duration in this file, and so nothing in it a slow runner can
  // make fail.
  //
  // Two shortcuts to the same end go vacuous, and both were checked by deleting
  // `if (!active.current) return` and watching them still pass. Deferring the
  // create response instead moves the dismissal ahead of `onSuccess`, and
  // TanStack Query drops `mutate`-level callbacks once the component has
  // unmounted, so `finish()` never runs at all. And fake timers fail twice over,
  // the second being the one that matters. While `userEvent` is driving they deadlock outright, since
  // react-aria's pointer events run on real timers. And in the dismissal case
  // below, where nothing is driving after the Escape, they look safe and are
  // worse than that: advancing the clock past the hold makes that case pass even
  // with the guard in `WorkspacesPage` deleted, because the router transition
  // never lands inside the fake-timer window. It would be vacuous rather than
  // fast.

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
    const beat = pendingHold()
    const user = userEvent.setup()
    await renderSwitcherOnAPage({ hold: beat.hold })

    const form = await fillCreateForm(user)
    // The label names the navigation, which is the only warning the operator
    // gets that the page is about to change under them.
    const submit = within(form).getByRole("button", { name: /Create and open/ })
    await user.click(submit)

    // The press is acknowledged before the page moves, rather than the create
    // landing them somewhere else with nothing in between. The button keeps its
    // name through the beat, so it is still the control it was.
    expect(submit).toHaveAttribute("data-pending", "true")
    expect(screen.queryByText("OVERVIEW PAGE")).toBeNull()

    beat.release()

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
    // The beat is a gate this test opens, so the window the guard exists for is
    // entered and left on purpose rather than by sleeping long enough to have
    // been inside it. Nothing here waits on a duration.
    mockApi({ context: startedInAWorkspace })
    const beat = pendingHold()
    const user = userEvent.setup()
    await renderSwitcherOnAPage({ hold: beat.hold })

    const form = await fillCreateForm(user)
    await user.click(
      within(form).getByRole("button", { name: /Create and open/ }),
    )
    // Escape rather than Cancel: Cancel is disabled while the create is in
    // flight, so dismissal is what is left, and it is the path that bypasses
    // every button.
    await user.keyboard("{Escape}")
    beat.release()

    // Reopening the switcher is a real interaction, and the workspace listed in
    // it is the visible result of the create. Getting there is what puts the
    // assertions below after the completion has had its turn.
    await user.click(screen.getByLabelText(/^Switch workspace/))
    const menu = await screen.findByRole("dialog", {
      name: "Switch workspace or organization",
    })
    const staging = await within(menu).findByRole("button", { name: /Staging/ })
    expect(staging).toBeVisible()

    // The workspace was created and the switcher offers it; what must not
    // happen is being taken there after saying not to.
    expect(screen.queryByText("OVERVIEW PAGE")).toBeNull()
    expect(screen.getByText("USAGE PAGE")).toBeInTheDocument()
    // Read from the menu rather than the trigger, which the open menu hides
    // from the accessibility tree: the scope never moved.
    expect(
      within(menu).getByRole("button", { name: /Default Workspace/ }),
    ).toHaveTextContent("Selected")
    expect(staging).not.toHaveTextContent("Selected")
  })

  it("still completes after StrictMode's development remount", async () => {
    // The guard that drops the completion after dismissal is a ref, and a
    // cleanup-only effect would leave it cleared by the remount: every create
    // would then hang with the modal open and the button spinning, in
    // development only, where `main.tsx` wraps the app in StrictMode.
    mockApi({ context: startedInAWorkspace })
    const beat = pendingHold()
    const user = userEvent.setup()
    await renderSwitcherOnAPage({ strict: true, hold: beat.hold })

    const form = await fillCreateForm(user)
    await user.click(
      within(form).getByRole("button", { name: /Create and open/ }),
    )
    beat.release()

    expect(await screen.findByText("OVERVIEW PAGE")).toBeInTheDocument()
  })
})
