import { screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
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
} from "@/tests/fixtures"
import { renderWithRouter } from "@/tests/router"

const SECOND_ORGANIZATION_ID = "99999999-9999-9999-9999-999999999999"

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
  vi.spyOn(apiClient, "apiFetch").mockImplementation(async (input, init) => {
    const url = String(input)
    requests.push({
      url,
      method: init?.method ?? "GET",
      body: init?.body ? JSON.parse(String(init.body)) : undefined,
    })
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
