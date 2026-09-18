import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { WorkspaceSwitcher } from "@/app/nav/WorkspaceSwitcher"
import type { MembershipRole, UsageSummary } from "@/client"
import { OverviewIndex } from "@/features/overview/OverviewPage"
import * as apiClient from "@/shared/api/client"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  bootstrap,
  callerOrganizationMembership,
  organization,
  organizationContext,
  organizationSpendCeiling,
  usageTotals,
} from "@/tests/fixtures"
import { withRouter } from "@/tests/router"

/**
 * What a switch of organization does to the page underneath it.
 *
 * The switcher and the page are mounted together because neither half shows the
 * defect alone: the switcher is what invalidates everything cached, and the page
 * is what asks again. A role-gated read is the case that matters, since the role
 * a gate reads is itself one of the things a switch moves.
 */

const SECOND_ORGANIZATION_ID = "99999999-9999-9999-9999-999999999999"
const WORKSPACE_HERE = "44444444-4444-4444-4444-444444444444"
const WORKSPACE_THERE = "55555555-5555-5555-5555-555555555555"

function emptySummary(): UsageSummary {
  return {
    start_date: "2026-06-22T00:00:00Z",
    end_date: "2026-07-22T00:00:00Z",
    bucket: "day",
    totals: usageTotals({ cost: 0, request_count: 0, error_count: 0 }),
    by_model: [],
    by_user: [],
    by_api_key: [],
    by_source: [],
    by_source_label: [],
    by_endpoint: [],
    by_provider: [],
    by_tool: [],
    errors_by_status_code: [],
    series: [],
  }
}

/**
 * Two organizations, the caller's role in the second one being what a test says.
 *
 * The transport is spied on rather than the hooks, so the real query keys, gates
 * and paths run. `/organizations/me` answers for whichever organization is
 * active, and the spend-ceilings read refuses anyone who is not an owner or an
 * admin of it, which is the gate `require_active_organization_management_access`
 * draws on the server.
 */
function mockApi({
  roleThere,
  holdCeilingsHere,
}: {
  roleThere: MembershipRole
  /** Holds this organization's ceilings walk open, so a switch can overtake it. */
  holdCeilingsHere?: { held: Promise<void>; answered: () => void }
}) {
  let active: "here" | "there" = "here"
  const requests: string[] = []

  const context = () =>
    active === "here"
      ? organizationContext({
          role: "admin",
          deployment_operator: false,
          workspace_memberships: [
            { workspace_id: WORKSPACE_HERE, name: "Default", role: "admin" },
          ],
        })
      : organizationContext({
          role: roleThere,
          deployment_operator: false,
          organization: organization({
            id: SECOND_ORGANIZATION_ID,
            name: "Research",
            slug: "research-1a2b3c4d",
          }),
          workspace_memberships: [
            {
              workspace_id: WORKSPACE_THERE,
              name: "Research",
              role: roleThere,
            },
          ],
        })

  vi.spyOn(apiClient, "apiFetch").mockImplementation(async (input, init) => {
    const url = String(input)
    requests.push(`${init?.method ?? "GET"} ${url}`)
    if (url === "/organizations/me/switch") {
      active = "there"
      return context() as never
    }
    if (url === "/organizations/me") {
      return context() as never
    }
    if (url.startsWith("/organizations/me/memberships")) {
      const rows = [
        callerOrganizationMembership({
          role: "admin",
          is_active_organization: active === "here",
        }),
        callerOrganizationMembership({
          organization_member_id: "88888888-8888-8888-8888-888888888888",
          organization: organization({
            id: SECOND_ORGANIZATION_ID,
            name: "Research",
            slug: "research-1a2b3c4d",
          }),
          role: roleThere,
          is_active_organization: active === "there",
        }),
      ]
      return { data: rows, count: rows.length } as never
    }
    if (url.startsWith("/organizations/me/spend-ceilings")) {
      if (active === "there") {
        if (roleThere !== "admin") {
          throw new apiClient.ApiError(
            403,
            "Not enough privileges to perform this action",
          )
        }
        // A different figure from the one here, so the cell says which
        // organization answered it: 10 of 100.
        return {
          data: [
            organizationSpendCeiling({ max_budget: 100, current_spend: 10 }),
          ],
          count: 1,
        } as never
      }
      if (holdCeilingsHere) {
        await holdCeilingsHere.held
        holdCeilingsHere.answered()
      }
      // 200 of 250, which the budget-health cell reads as 80.0%.
      return {
        data: [
          organizationSpendCeiling({ max_budget: 250, current_spend: 200 }),
        ],
        count: 1,
      } as never
    }
    if (url.startsWith("/organizations/me/usage/summary")) {
      return emptySummary() as never
    }
    if (url.startsWith("/models")) {
      return { object: "list", data: [] } as never
    }
    // The paged reads (`fetchAllPaged`) take an envelope, everything else an
    // array, so the fallthrough answers the shape the caller walks.
    if (
      url.startsWith("/organizations/me/pending-memberships") ||
      url.startsWith("/workspaces/")
    ) {
      return { data: [], count: 0 } as never
    }
    return [] as never
  })

  return requests
}

function renderShellOverPage() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <DeploymentProvider value={bootstrap()}>
        <SelectedWorkspaceProvider>
          <WorkspaceSwitcher isCollapsed={false} />
          <OverviewIndex />
        </SelectedWorkspaceProvider>
      </DeploymentProvider>
    </QueryClientProvider>,
    { wrapper: withRouter({ url: "/overview" }) },
  )
}

/** Open the scope menu and switch into the second organization. */
async function switchOrganization() {
  const user = userEvent.setup()
  await user.click(
    await screen.findByRole("button", { name: /^Switch workspace/ }),
  )
  const menu = await screen.findByRole("dialog")
  await user.click(
    await within(menu).findByRole("button", { name: /Research/ }),
  )
  // The switcher names the scope every query below it is now made in, so this
  // is the switch having landed rather than merely having been asked for.
  await screen.findByRole("button", { name: /in Research$/ })
}

/** A promise the test resolves by hand, for holding a request open. */
function deferred() {
  let resolve: () => void = () => {}
  const promise = new Promise<void>((r) => {
    resolve = r
  })
  return { promise, resolve }
}

/** What was asked for after the switch itself, which is the part at issue. */
function afterSwitch(requests: string[]): string[] {
  return requests.slice(requests.findIndex((r) => r.startsWith("POST")) + 1)
}

describe("switching organization", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("does not ask for a read the organization it moved to refuses", async () => {
    const requests = mockApi({ roleThere: "member" })
    renderShellOverPage()
    await screen.findByText(/At-a-glance spend/)

    await switchOrganization()

    // The page's own reads are made again, and the owners-and-admins-only one
    // is not: the role that opened it is the one in the organization just left.
    // Asking anyway is what left "Not enough privileges to perform this action"
    // on a page that was working, until the operator reloaded it.
    await waitFor(() => {
      expect(
        afterSwitch(requests).some((r) => r.includes("usage/summary")),
      ).toBe(true)
    })
    expect(
      afterSwitch(requests).some((r) => r.includes("spend-ceilings")),
    ).toBe(false)
    expect(screen.queryByText(/Not enough privileges/)).toBeNull()
  })

  it("takes the new role from the switch's own answer", async () => {
    const requests = mockApi({ roleThere: "member" })
    renderShellOverPage()
    await screen.findByText(/At-a-glance spend/)

    await switchOrganization()

    // The switch answers with the context it just made active, so asking
    // `/organizations/me` for it again would only hold the caller's new role a
    // round trip behind the reads already being made in it.
    expect(
      afterSwitch(requests).filter((r) => r === "GET /organizations/me"),
    ).toEqual([])
  })

  it("does not read the organization it left under the one it landed on", async () => {
    // A ceilings walk still in flight when the switch lands. The switch no
    // longer refetches it, so nothing cancels it either, and `apiFetch` does
    // not carry the query's signal: the answer arrives after the move and is
    // about the organization it was asked in. Keyed per organization, it lands
    // under that one rather than under the organization now on screen.
    const hold = deferred()
    const answered = deferred()
    mockApi({
      roleThere: "admin",
      holdCeilingsHere: { held: hold.promise, answered: answered.resolve },
    })
    renderShellOverPage()
    await screen.findByText(/At-a-glance spend/)

    await switchOrganization()
    // The organization switched into has answered, so the cell has a figure of
    // its own for the late one to overwrite.
    expect(await screen.findByText("10.0%")).toBeInTheDocument()

    hold.resolve()
    await answered.promise
    await act(async () => {})

    expect(screen.getByText("10.0%")).toBeInTheDocument()
    expect(screen.queryByText("80.0%")).toBeNull()
  })

  it("makes the gated read again where the role survives the switch", async () => {
    const requests = mockApi({ roleThere: "admin" })
    renderShellOverPage()
    await screen.findByText(/At-a-glance spend/)

    await switchOrganization()

    // The other half of the gate: it withholds the read while the role behind
    // it belongs to the organization just left, not for as long as the caller
    // is switching.
    await waitFor(() => {
      expect(
        afterSwitch(requests).some((r) => r.includes("spend-ceilings")),
      ).toBe(true)
    })
    expect(screen.queryByText(/Not enough privileges/)).toBeNull()
  })
})
