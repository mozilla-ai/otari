import { Outlet } from "@tanstack/react-router"
import { screen, waitFor, within } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import type { UsageEntry } from "@/client"
import { RefusedRequests } from "@/features/rules/RefusedRequests"
import type { Rule } from "@/features/rules/rules"
import { REFUSALS_SHOWN } from "@/features/rules/useRules"
import { API_ROOT } from "@/shared/api/client"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  bootstrap,
  organizationContext,
  organizationMember,
} from "@/tests/fixtures"
import { AppProviders as Provider } from "@/tests/providers"
import { renderWithRouter } from "@/tests/router"

const WEI = organizationMember({
  organization_member_id: "om-wei",
  user_id: "login-wei",
  attribution_user_id: "user-wei",
  full_name: "Wei Zhang",
  role: "member",
  status: "active",
})

const MAX = organizationMember({
  organization_member_id: "om-max",
  user_id: "login-max",
  attribution_user_id: "user-max",
  full_name: "Max Ruiz",
  role: "member",
  status: "active",
})

/** Wei, signed in as a member: no owner's or admin's reads. */
const AS_WEI = organizationContext({
  role: "member",
  organization_member_id: "om-wei",
  caller: {
    user_id: "login-wei",
    email: null,
    full_name: "Wei Zhang",
    has_password: true,
    claims_deployment: false,
  },
  deployment_operator: false,
})

/** The fields the table reads; the row is only ever served as JSON. */
function refusal(
  id: string,
  fields: Partial<UsageEntry> = {},
): Partial<UsageEntry> {
  return {
    id,
    timestamp: "2026-11-16T09:00:00Z",
    user_id: "user-wei",
    api_key_name: "laptop",
    error_message: "Spend limit reached",
    ...fields,
  }
}

function rule(subject: Rule["subject"]): Rule {
  return {
    id: "r",
    name: "Platform, monthly",
    type: "spend",
    scope: { kind: "team", team: "Platform" },
    mode: "Enforce",
    subject,
  }
}

/** The paths read, and the query of each usage read. */
async function renderRefusals(
  rows: Partial<UsageEntry>[],
  subject: Rule["subject"],
  context = organizationContext({ role: "owner" }),
) {
  const reads: string[] = []
  const usage: URLSearchParams[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const [path = "", query] = String(input).slice(API_ROOT.length).split("?")
    reads.push(path)
    if (path === "/organizations/me/usage")
      usage.push(new URLSearchParams(query))
    switch (path) {
      case "/organizations/me":
        return Response.json(context)
      case "/organizations/me/members":
        return Response.json({ data: [WEI, MAX], count: 2 })
      case "/organizations/me/usage": {
        // The server's own filter: rows for the people asked for, if any.
        const people = new URLSearchParams(query).getAll("user_id")
        return Response.json(
          people.length > 0
            ? rows.filter((row) => people.includes(row.user_id ?? ""))
            : rows,
        )
      }
      default:
        return Response.json({ data: [], count: 0 })
    }
  })
  await renderWithRouter(<RefusedRequests rule={rule(subject)} />, {
    url: "/rules/r",
    shell: (
      <Provider>
        <DeploymentProvider value={bootstrap()}>
          <SelectedWorkspaceProvider>
            <Outlet />
          </SelectedWorkspaceProvider>
        </DeploymentProvider>
      </Provider>
    ),
  })
  return { reads, usage }
}

describe("Refused requests", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("names whoever a refusal was for, as Home's People cut does", async () => {
    await renderRefusals(
      [
        refusal("1"),
        refusal("2", { user_id: "user-gone", user_alias: "ci-runner" }),
        refusal("3", { user_id: "0123456789abcdef", user_alias: null }),
        refusal("4", { user_id: null }),
      ],
      { workspaceId: "ws-platform" },
    )
    const table = await screen.findByRole("grid", {
      name: "Refusals for Platform, monthly",
    })
    expect(await within(table).findByText("Wei Zhang")).toBeVisible()
    expect(within(table).getByText("ci-runner")).toBeVisible()
    expect(within(table).getByText("01234567…")).toBeVisible()
    expect(within(table).getByText("No person")).toBeVisible()
    expect(screen.queryByText(/most recent refused requests/)).toBeNull()
  })

  it("says nothing was refused when the read came back empty", async () => {
    await renderRefusals([], { userIds: ["user-wei"] })
    expect(await screen.findByText("Nothing refused this week.")).toBeVisible()
  })

  it("claims nothing for a rule it has no one to read refusals for", async () => {
    const { reads } = await renderRefusals([refusal("1")], {})
    await screen.findByRole("grid", { name: "Refusals for Platform, monthly" })
    expect(screen.queryByText("Nothing refused this week.")).toBeNull()
    expect(reads).not.toContain("/organizations/me/usage")
  })

  it("reads the whole organization's refusals for a rule on everyone", async () => {
    const { usage } = await renderRefusals([refusal("1")], { everyone: true })
    const table = await screen.findByRole("grid", {
      name: "Refusals for Platform, monthly",
    })
    expect(await within(table).findByText("Wei Zhang")).toBeVisible()
    const [query] = usage
    expect(query?.get("status_code")).toBe("403")
    expect(query?.get("start_date")).toBeTruthy()
    for (const filter of ["user_id", "api_key_id", "workspace_id"]) {
      expect(query?.has(filter)).toBe(false)
    }
  })

  it("asks for one more refusal than it lists", async () => {
    const { usage } = await renderRefusals([], { userIds: ["user-wei"] })
    await screen.findByText("Nothing refused this week.")
    expect(usage[0]?.get("limit")).toBe(String(REFUSALS_SHOWN + 1))
  })

  it("lists the latest refusals and says so when there are more", async () => {
    await renderRefusals(
      Array.from({ length: REFUSALS_SHOWN + 1 }, (_, index) =>
        refusal(String(index)),
      ),
      { everyone: true },
    )
    expect(
      await screen.findByText("Showing the 50 most recent refused requests."),
    ).toBeVisible()
    const table = screen.getByRole("grid", {
      name: "Refusals for Platform, monthly",
    })
    // One header row, then the refusals.
    expect(within(table).getAllByRole("row")).toHaveLength(REFUSALS_SHOWN + 1)
  })

  it("says nothing of more refusals when all of them are listed", async () => {
    await renderRefusals(
      Array.from({ length: REFUSALS_SHOWN }, (_, index) =>
        refusal(String(index)),
      ),
      { userIds: ["user-wei"] },
    )
    const table = await screen.findByRole("grid", {
      name: "Refusals for Platform, monthly",
    })
    await waitFor(() =>
      expect(within(table).getAllByRole("row")).toHaveLength(
        REFUSALS_SHOWN + 1,
      ),
    )
    expect(screen.queryByText(/most recent refused requests/)).toBeNull()
  })

  it("shows a member only their own refusals, and nobody else's amounts", async () => {
    const { usage } = await renderRefusals(
      [
        refusal("1", {
          error_message:
            "Member has $0.40 of budget remaining, less than this request's estimated cost",
        }),
        refusal("2", {
          user_id: "user-max",
          error_message:
            "Member has $3.20 of budget remaining, less than this request's estimated cost",
        }),
      ],
      { userIds: ["user-wei", "user-max"] },
      AS_WEI,
    )
    const table = await screen.findByRole("grid", {
      name: "Refusals for Platform, monthly",
    })
    expect(await within(table).findByText(/\$0\.40/)).toBeVisible()
    expect(within(table).queryByText(/\$3\.20/)).toBeNull()
    expect(within(table).queryByText("Max Ruiz")).toBeNull()
    for (const query of usage) {
      expect(query.getAll("user_id")).toEqual(["user-wei"])
    }
  })

  it("reads a member's own refusals under a rule on everyone", async () => {
    const { usage } = await renderRefusals(
      [refusal("1"), refusal("2", { user_id: "user-max" })],
      { everyone: true },
      AS_WEI,
    )
    const table = await screen.findByRole("grid", {
      name: "Refusals for Platform, monthly",
    })
    expect(await within(table).findByText("Wei Zhang")).toBeVisible()
    expect(within(table).queryByText("Max Ruiz")).toBeNull()
    expect(usage.length).toBeGreaterThan(0)
    for (const query of usage) {
      expect(query.getAll("user_id")).toEqual(["user-wei"])
    }
  })

  it("reads nothing for a member a per-person rule does not reach", async () => {
    const { reads } = await renderRefusals(
      [refusal("1", { user_id: "user-max" })],
      { userIds: ["user-max"] },
      AS_WEI,
    )
    await screen.findByRole("grid", { name: "Refusals for Platform, monthly" })
    // The roster answering is what a read would wait on.
    await waitFor(() => expect(reads).toContain("/organizations/me/members"))
    expect(reads).not.toContain("/organizations/me/usage")
    expect(screen.queryByText("Max Ruiz")).toBeNull()
  })
})
