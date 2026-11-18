import { Outlet } from "@tanstack/react-router"
import { screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
import type {
  ApiKey,
  OrganizationMember,
  UsageEntry,
  UsageSummary,
  WorkspaceBudgetDefault,
} from "@/client"
import { MemberHome } from "@/features/home/MemberHome"
import {
  LOWER_LIMIT_NOTE,
  UNSEEN_LIMIT_NOTE,
} from "@/features/home/memberHomeModel"
import { API_ROOT } from "@/shared/api/client"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  apiKey,
  bootstrap,
  organizationContext,
  organizationMember,
  SURFACES,
  usageTotals,
  workspaceBudgetDefault,
} from "@/tests/fixtures"
import { AppProviders as Provider } from "@/tests/providers"
import { renderWithRouter } from "@/tests/router"
import { findRowMenuItem } from "@/tests/rowMenu"

const PLATFORM = "ws-platform"

const WEI = organizationMember({
  organization_member_id: "om-wei",
  user_id: "login-wei",
  attribution_user_id: "user-wei",
  full_name: "Wei Zhang",
  role: "member",
  workspaces: [
    {
      workspace_id: PLATFORM,
      workspace_member_id: "wm-wei",
      workspace_name: "Platform",
      role: "member",
      ceiling: { id: "c-wei", budget_id: "b-weekly", max_budget: 40 },
    },
  ],
})

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

/** $40 a week for each person in Platform, which Wei's ceiling is drawn from. */
const WEEKLY = workspaceBudgetDefault({
  workspace_id: PLATFORM,
  budget_id: "b-weekly",
  max_budget: 40,
  budget_duration_sec: null,
  reset_alignment: "calendar_week",
})

function summary(start: string, cost: number, requestCount = 1): UsageSummary {
  return {
    // As the server writes it back, not as it was asked.
    start_date: new Date(start).toISOString().replace(".000Z", "+00:00"),
    end_date: new Date().toISOString(),
    bucket: "day",
    totals: usageTotals({ cost, request_count: requestCount }),
    series: [],
    by_api_key: [],
    by_endpoint: [],
    by_model: [],
    by_provider: [],
    by_source: [],
    by_source_label: [],
    by_tag: [],
    by_tool: [],
    by_user: [],
    errors_by_status_code: [],
  }
}

/** Wei's one key, with spend on it this month. */
const LAPTOP = apiKey({
  id: "k-laptop",
  key_name: "laptop",
  user_id: "user-wei",
  workspace_id: PLATFORM,
})

/**
 * Wei's Home on this month, refused over spend just now. The month's summary
 * says `windowCost`; one read from any other start says `periodCost`.
 */
async function renderMemberHome({
  windowCost,
  periodCost,
  member = WEI,
  defaults = [WEEKLY],
  refusalMessage = "Member has $0.01 of budget remaining, less than this request's estimated cost",
  hasTraces = false,
  keys = [LAPTOP],
  requestCount = 1,
  sessionCount = 3,
}: {
  windowCost: number
  periodCost: number
  member?: OrganizationMember
  defaults?: WorkspaceBudgetDefault[]
  refusalMessage?: string
  /** Whether the deployment captures sessions, as its bootstrap says. */
  hasTraces?: boolean
  keys?: ApiKey[]
  /** The member's requests in every summary read. */
  requestCount?: number
  sessionCount?: number
}) {
  const refusal: Partial<UsageEntry> = {
    id: "r-1",
    timestamp: new Date().toISOString(),
    user_id: "user-wei",
    error_message: refusalMessage,
  }
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const [path = "", search] = String(input).slice(API_ROOT.length).split("?")
    const query = new URLSearchParams(search)
    switch (path) {
      case "/organizations/me":
        return Response.json(AS_WEI)
      case "/organizations/me/members":
        return Response.json({ data: [member], count: 1 })
      case `/workspaces/${PLATFORM}/member-budget-policies`:
        return Response.json({ data: defaults, count: defaults.length })
      case "/organizations/me/keys":
        return Response.json(keys)
      case "/organizations/me/usage":
        return Response.json([refusal])
      case "/organizations/me/traces/count":
        return Response.json({ count: sessionCount })
      case "/organizations/me/usage/summary": {
        const start = query.get("start_date") ?? ""
        const isWindow =
          query.has("end_date") ||
          query.getAll("dimensions").some((one) => one !== "none")
        return Response.json({
          ...summary(start, isWindow ? windowCost : periodCost, requestCount),
          by_api_key: [
            {
              key: LAPTOP.id,
              label: "laptop",
              cost: 2,
              requests: 3,
              tokens: 900,
              is_other: false,
            },
          ],
        })
      }
      default:
        return Response.json({ data: [], count: 0 })
    }
  })
  await renderWithRouter(
    <MemberHome workspaceId={PLATFORM} workspace="Platform" />,
    {
      shell: (
        <Provider>
          <DeploymentProvider
            value={bootstrap(
              hasTraces ? { surfaces: [...SURFACES, "traces"] } : {},
            )}
          >
            <SelectedWorkspaceProvider>
              <Outlet />
            </SelectedWorkspaceProvider>
          </DeploymentProvider>
        </Provider>
      ),
    },
  )
}

describe("MemberHome, a lower limit", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("weighs a spend refusal against the week's spend, not the month's", async () => {
    // Over the weekly figure across the month, and well under it this week.
    await renderMemberHome({ windowCost: 100, periodCost: 0.48 })
    expect(await screen.findByText(LOWER_LIMIT_NOTE)).toBeVisible()
  })
})

/**
 * Wei as the owner's "Set spend limit…" leaves him: the limit is on his
 * membership of the organization, which the roster does not carry and a
 * member may not read, and Platform sets no limit for everyone.
 */
const WEI_LIMITED_ALONE = organizationMember({
  ...WEI,
  workspaces: [
    {
      workspace_id: PLATFORM,
      workspace_member_id: "wm-wei",
      workspace_name: "Platform",
      role: "member",
      ceiling: null,
    },
  ],
})

describe("MemberHome, a limit set on the member alone", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("says it is not shown to them, and that it refused them, rather than none", async () => {
    await renderMemberHome({
      windowCost: 41,
      periodCost: 41,
      member: WEI_LIMITED_ALONE,
      defaults: [],
      refusalMessage: "Member has exceeded budget limit",
    })
    expect(
      await screen.findByText(
        "A limit set on you refused requests this month.",
      ),
    ).toBeVisible()
    expect(screen.getByText("Your limit")).toBeVisible()
    expect(screen.getByText("Not shown")).toBeVisible()
    expect(screen.getByText("Only owners and admins see it")).toBeVisible()
    expect(screen.queryByText("None")).toBeNull()
    expect(screen.queryByText(/No spend limit on you/)).toBeNull()
  })

  it("does not read a workspace's or a key's refusal as a limit on them", async () => {
    await renderMemberHome({
      windowCost: 41,
      periodCost: 41,
      member: WEI_LIMITED_ALONE,
      defaults: [],
      refusalMessage: "Workspace has exceeded budget limit",
    })
    expect(await screen.findByText(UNSEEN_LIMIT_NOTE)).toBeVisible()
    expect(screen.getByText("None for everyone in Platform")).toBeVisible()
    expect(screen.queryByText(/A limit set on you refused/)).toBeNull()
    expect(screen.queryByText("None")).toBeNull()
  })
})

describe("MemberHome, sessions", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  async function spendTable() {
    const table = await screen.findByRole("grid", {
      name: "Your spend by api keys",
    })
    await within(table).findByText("laptop")
    return table
  }

  it("offers no session figure or link where the deployment records none", async () => {
    const user = userEvent.setup()
    await renderMemberHome({ windowCost: 2, periodCost: 2 })
    const table = await spendTable()
    expect(screen.queryByText("Your sessions")).toBeNull()
    expect(screen.queryByText(/A row opens its sessions/)).toBeNull()
    expect(await findRowMenuItem(user, /See its sessions/, table)).toBeNull()
    // The strip keeps four figures: the owner's Failures takes the place.
    expect(screen.getByText("Failures")).toBeVisible()
  })

  it("counts and links the member's sessions where the deployment records them", async () => {
    const user = userEvent.setup()
    await renderMemberHome({ windowCost: 2, periodCost: 2, hasTraces: true })
    const table = await spendTable()
    expect(screen.getByText("Your sessions")).toBeVisible()
    expect(
      await screen.findByRole("link", { name: "See your sessions" }),
    ).toBeVisible()
    expect(screen.getByText(/A row opens its sessions/)).toBeVisible()
    expect(
      await findRowMenuItem(user, /See its sessions/, table),
    ).not.toBeNull()
    expect(screen.queryByText("Failures")).toBeNull()
  })
})

describe("MemberHome, a first run", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  // The band in either form: a first key for who may make one, what a viewer
  // may do instead. This harness's caller has no workspace selected, so the second.
  const FIRST_RUN = /through Platform/

  it("opens on it with no key and no requests where the deployment records no sessions", async () => {
    await renderMemberHome({
      windowCost: 0,
      periodCost: 0,
      keys: [],
      requestCount: 0,
    })
    expect(await screen.findByText(FIRST_RUN)).toBeVisible()
  })

  it("does not open on it once a request has run, sessions or not", async () => {
    await renderMemberHome({ windowCost: 2, periodCost: 2, keys: [] })
    expect(await screen.findByText("1 requests")).toBeVisible()
    expect(screen.queryByText(FIRST_RUN)).toBeNull()
  })

  it("opens on it with no key and no sessions where the deployment records them", async () => {
    await renderMemberHome({
      windowCost: 0,
      periodCost: 0,
      keys: [],
      requestCount: 0,
      sessionCount: 0,
      hasTraces: true,
    })
    expect(await screen.findByText(FIRST_RUN)).toBeVisible()
  })
})
