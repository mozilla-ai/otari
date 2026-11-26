import { Outlet } from "@tanstack/react-router"
import { screen, within } from "@testing-library/react"
import type { ReactElement } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"
import type { OrganizationMember } from "@/client"
import { PeoplePage } from "@/features/people/PeoplePage"
import { PersonPage } from "@/features/people/PersonPage"
import { API_ROOT } from "@/shared/api/client"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  apiKey,
  bootstrap,
  organizationContext,
  organizationMember,
  SURFACES,
  workspace,
} from "@/tests/fixtures"
import { AppProviders as Provider } from "@/tests/providers"
import { renderWithRouter } from "@/tests/router"
import { traceSummary } from "@/tests/traceFixtures"

const WEI = organizationMember({
  organization_member_id: "om-wei",
  user_id: "login-wei",
  attribution_user_id: "user-wei",
  email: "wei@example.com",
  full_name: "Wei Zhang",
  role: "member",
  status: "active",
  created_at: "2026-11-01T00:00:00Z",
  workspaces: [
    {
      workspace_id: "ws-platform",
      workspace_name: "Platform",
      workspace_member_id: "wm-wei",
      role: "member",
    },
  ],
})

/** A page as an owner sees it, with the read at `failing` refused. */
async function renderFailing(
  page: ReactElement,
  url: string,
  failing?: string,
  members: OrganizationMember[] = [WEI],
  /** What the reads that do answer return. */
  answers: {
    usageGroups?: unknown[]
    refusalGroups?: unknown[]
    sessions?: unknown[]
    keys?: unknown[]
    workspaces?: unknown[]
    /** Every request's path and query, in order. */
    seen?: string[]
    /** A deployment that records no agent sessions. */
    withoutTraces?: boolean
    /** A caller in no workspace: the whole organization is what they see. */
    noWorkspace?: boolean
    /** The workspace the rail has selected; Platform unless said. */
    selected?: "ws-platform" | "ws-data"
    /** Who is asking; an owner unless said. */
    role?: "owner" | "member"
  } = {},
) {
  if (!answers.noWorkspace) {
    window.localStorage.setItem(
      "otari.dashboard.selectedWorkspace",
      answers.selected ?? "ws-platform",
    )
  }
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    answers.seen?.push(String(input).slice(API_ROOT.length))
    const path = String(input).slice(API_ROOT.length).split("?")[0]
    // A 403, which the client does not retry.
    if (path === failing) {
      return Response.json({ detail: `Refused ${failing}` }, { status: 403 })
    }
    const paged = (data: unknown[]) =>
      Response.json({ data, count: data.length })
    switch (path) {
      case "/organizations/me":
        return Response.json(
          organizationContext({
            role: answers.role ?? "owner",
            deployment_operator: false,
            workspace_memberships: answers.noWorkspace
              ? []
              : [
                  {
                    workspace_id: "ws-platform",
                    name: "Platform",
                    role: answers.role ?? "owner",
                  },
                  {
                    workspace_id: "ws-data",
                    name: "Data",
                    role: answers.role ?? "owner",
                  },
                ],
          }),
        )
      case "/organizations/me/members":
        return paged(members)
      case "/organizations/me/traces":
        return Response.json({
          items: answers.sessions ?? [],
          total: (answers.sessions ?? []).length,
        })
      case "/organizations/me/traces/count":
        return Response.json({ count: (answers.sessions ?? []).length })
      case "/workspaces":
        return paged(answers.workspaces ?? [])
      case "/organizations/me/keys":
        return Response.json(answers.keys ?? [])
      case "/organizations/me/usage/summary": {
        const refusals = String(input).includes("status_code=403")
        return Response.json({
          totals: { cost: 0, request_count: 0, error_count: 0 },
          series: [],
          by_api_key: [],
          by_model: [],
          by_user: refusals
            ? (answers.refusalGroups ?? [])
            : (answers.usageGroups ?? []),
          groups: [],
        })
      }
      default:
        return paged([])
    }
  })
  return renderWithRouter(page, {
    url,
    shell: (
      <Provider>
        <DeploymentProvider
          value={bootstrap({
            surfaces: answers.withoutTraces
              ? [...SURFACES]
              : [...SURFACES, "traces"],
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

describe("People, when a read fails", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  for (const failing of ["/organizations/me/keys", "/workspaces"]) {
    it(`says ${failing} could not be read`, async () => {
      await renderFailing(<PeoplePage />, "/people", failing)
      expect(await screen.findByRole("alert")).toHaveTextContent(
        `Refused ${failing}`,
      )
    })
  }

  it("does not call a roster it could not read empty", async () => {
    const failing = "/organizations/me/members"
    await renderFailing(<PeoplePage />, "/people", failing)
    expect(await screen.findByRole("alert")).toHaveTextContent(
      `Refused ${failing}`,
    )
    expect(screen.queryByText("Nobody is here yet.")).toBeNull()
  })

  it("says nobody is here when the roster is read and has no one", async () => {
    await renderFailing(<PeoplePage />, "/people", undefined, [])
    expect(await screen.findByText("Nobody is here yet.")).toBeVisible()
    expect(screen.queryByRole("alert")).toBeNull()
  })
})

describe("A person's page, when a read fails", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("shows the failure and offers no limit to set", async () => {
    const failing = "/organizations/me/traces/count"
    await renderFailing(
      <PersonPage personId="om-wei" />,
      "/people/om-wei",
      failing,
    )
    expect(
      await screen.findByRole("heading", { name: "Wei Zhang" }),
    ).toBeVisible()
    expect(await screen.findByRole("alert")).toHaveTextContent(
      `Refused ${failing}`,
    )
    expect(screen.getByRole("heading", { name: "Spend limit" })).toBeVisible()
    expect(screen.queryByRole("button", { name: "Set spend limit" })).toBeNull()
    // Nothing read says no limit applies.
    expect(screen.queryByText("No spend limit applies to Wei.")).toBeNull()
  })

  it("offers a limit to set once every read has answered", async () => {
    await renderFailing(<PersonPage personId="om-wei" />, "/people/om-wei")
    expect(
      await screen.findByRole("button", { name: "Set spend limit" }),
    ).toBeVisible()
    expect(screen.getByText("No spend limit applies to Wei.")).toBeVisible()
    expect(screen.queryByRole("alert")).toBeNull()
  })
})

describe("People, when one read fails", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  const answers = {
    usageGroups: [
      {
        key: "user-wei",
        label: "Wei Zhang",
        cost: 12.5,
        requests: 40,
        tokens: 5000,
        is_other: false,
      },
    ],
    refusalGroups: [
      {
        key: "user-wei",
        label: "Wei Zhang",
        cost: 0,
        requests: 3,
        tokens: 0,
        is_other: false,
      },
    ],
    sessions: [
      traceSummary({ user_id: "user-wei", workspace_id: "ws-platform" }),
    ],
    keys: [apiKey({ id: "k1", user_id: "user-wei", is_active: true })],
    workspaces: [workspace({ id: "ws-platform", name: "Platform" })],
  }

  async function weiRow(failing?: string) {
    await renderFailing(<PeoplePage />, "/people", failing, [WEI], answers)
    const row = await screen.findByRole("row", { name: /Wei Zhang/ })
    // Settled: the keys column holds its figure.
    await within(row).findByText("1")
    return row
  }

  it("reads every column when every read answers", async () => {
    const row = await weiRow()
    expect(within(row).getByText("$12.50")).toBeVisible()
    expect(within(row).getByText("3")).toBeVisible()
  })

  it("shows the usage columns as unknown, not zero, and keeps the sessions and keys", async () => {
    const row = await weiRow("/organizations/me/usage/summary")
    expect(within(row).queryByText("$0.00")).toBeNull()
    expect(within(row).queryByText("0")).toBeNull()
    // Spend, tokens, top model, then sessions (read, so 1), then refusals.
    const cells = within(row)
      .getAllByRole("rowheader")
      .concat(within(row).getAllByRole("gridcell"))
    const text = cells.map((cell) => cell.textContent)
    // The last cell is the owner's row menu.
    expect(text.slice(-6, -1)).toEqual(["—", "—", "—", "1", "—"])
  })

  it("shows only the sessions as unknown when the session read fails", async () => {
    await renderFailing(
      <PeoplePage />,
      "/people",
      "/organizations/me/traces/count",
      [WEI],
      answers,
    )
    const row = await screen.findByRole("row", { name: /Wei Zhang/ })
    expect(await within(row).findByText("$12.50")).toBeVisible()
    expect(within(row).getByText("3")).toBeVisible()
    expect(within(row).getAllByText("—")).toHaveLength(1)
  })

  it("shows a person's page the same way", async () => {
    await renderFailing(
      <PersonPage personId="om-wei" />,
      "/people/om-wei",
      "/organizations/me/usage/summary",
      [WEI],
      answers,
    )
    const spent = (await screen.findByText("Spent this month")).closest("div")
    expect(spent).toHaveTextContent("—")
    const refused = screen.getByText("Refused requests").closest("div")
    expect(refused).toHaveTextContent("—")
    expect(refused).not.toHaveTextContent("0")
  })
})

describe("People, in a workspace", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  const FIGURES = [
    /\/usage\/summary\?.*dimensions=user/,
    /\/usage\/summary\?.*status_code=403/,
    /\/traces\?/,
    /\/traces\/count\?/,
  ]

  /** The figure reads among everything asked, by the pattern that names each. */
  const figureReads = (seen: string[]) =>
    FIGURES.flatMap((pattern) => seen.filter((url) => pattern.test(url)))

  it("reads spend, tokens, sessions and refusals for the selected workspace only", async () => {
    const seen: string[] = []
    await renderFailing(<PeoplePage />, "/people", undefined, [WEI], { seen })
    await screen.findByRole("row", { name: /Wei Zhang/ })
    const reads = figureReads(seen)
    expect(reads.length).toBeGreaterThanOrEqual(4)
    for (const url of reads) expect(url).toContain("workspace_id=ws-platform")
  })

  it("reads a person's page across workspaces, whatever the rail selects", async () => {
    const seen: string[] = []
    // Wei works in Platform only, and the rail has Data selected.
    await renderFailing(
      <PersonPage personId="om-wei" />,
      "/people/om-wei",
      undefined,
      [WEI],
      { seen, selected: "ws-data" },
    )
    await screen.findByRole("heading", { name: "Wei Zhang" })
    await screen.findByRole("heading", { name: "Models this month" })
    const reads = figureReads(seen)
    // Spend, refusals, the session count, the models and the sessions list.
    expect(reads.length).toBeGreaterThanOrEqual(5)
    expect(seen.some((url) => /\/traces\?/.test(url))).toBe(true)
    expect(seen.some((url) => /dimensions=model/.test(url))).toBe(true)
    for (const url of seen) expect(url).not.toContain("workspace_id")
  })

  it("asks for no sessions where the deployment records none", async () => {
    const seen: string[] = []
    await renderFailing(<PeoplePage />, "/people", undefined, [WEI], {
      seen,
      withoutTraces: true,
    })
    await screen.findByRole("row", { name: /Wei Zhang/ })
    expect(seen.some((url) => /\/traces/.test(url))).toBe(false)
  })

  it("shows a person no session figure where the deployment records none", async () => {
    await renderFailing(
      <PersonPage personId="om-wei" />,
      "/people/om-wei",
      undefined,
      [WEI],
      { withoutTraces: true },
    )
    await screen.findByRole("heading", { name: "Wei Zhang" })
    expect(await screen.findByText("Tokens this month")).toBeVisible()
    expect(screen.queryByText("Sessions this month")).toBeNull()
    expect(screen.queryByText(/sessions/i)).toBeNull()
  })

  it("counts a person's sessions in one read, not by paging the list", async () => {
    const seen: string[] = []
    await renderFailing(
      <PersonPage personId="om-wei" />,
      "/people/om-wei",
      undefined,
      [WEI],
      {
        seen,
        sessions: Array.from({ length: 250 }, (_, index) =>
          traceSummary({ trace_id: `s-${index}`, user_id: "user-wei" }),
        ),
      },
    )
    await screen.findByRole("heading", { name: "Wei Zhang" })
    const [label] = await screen.findAllByText("Sessions this month")
    const sessionsCard = label.closest("div")
    await within(sessionsCard as HTMLElement).findByText("250")
    const counts = seen.filter((url) => /\/traces\/count\?/.test(url))
    expect(counts).toHaveLength(1)
    expect(counts[0]).toContain("user_id=user-wei")
    // The one list read is the sessions table's first ten.
    const lists = seen.filter((url) => /\/traces\?/.test(url))
    expect(lists).toHaveLength(1)
    expect(lists[0]).toContain("limit=10")
  })

  it("reads the whole organization when no workspace is selected", async () => {
    const seen: string[] = []
    await renderFailing(<PeoplePage />, "/people", undefined, [WEI], {
      seen,
      noWorkspace: true,
    })
    await screen.findByRole("row", { name: /Wei Zhang/ })
    const reads = figureReads(seen)
    expect(reads.length).toBeGreaterThanOrEqual(4)
    for (const url of reads) expect(url).not.toContain("workspace_id")
  })
})

describe("A person's page, read by a member", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("names the roster's amount the workspace's limit for each person, not the person's", async () => {
    const wei = {
      ...WEI,
      workspaces: [
        {
          ...WEI.workspaces?.[0],
          ceiling: { id: "c1", budget_id: "b1", max_budget: 40 },
        },
      ],
    } as OrganizationMember
    await renderFailing(
      <PersonPage personId="om-wei" />,
      "/people/om-wei",
      undefined,
      [wei],
      { role: "member" },
    )
    expect(
      await screen.findByText("Each workspace’s limit for each person."),
    ).toBeVisible()
    expect(screen.getByText("$40.00")).toBeVisible()
    expect(
      screen.queryByText("The spend limit on this person in each workspace."),
    ).toBeNull()
  })
})
