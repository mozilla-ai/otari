import { Outlet } from "@tanstack/react-router"
import { screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
import type {
  OrganizationMember,
  UsageEntry,
  UsageGroupRow,
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
import { LocationProbe, renderWithRouter } from "@/tests/router"
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

function summary(start: string, cost: number): UsageSummary {
  return {
    // As the server writes it back, not as it was asked.
    start_date: new Date(start).toISOString().replace(".000Z", "+00:00"),
    end_date: new Date().toISOString(),
    bucket: "day",
    totals: usageTotals({ cost, request_count: 1 }),
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
 * Wei's Home on the default window, refused over spend just now. The window's
 * summary says `windowCost`; one read from any other start says `periodCost`.
 * Returns every URL the page asked.
 */
async function renderMemberHome({
  windowCost,
  periodCost,
  member = WEI,
  defaults = [WEEKLY],
  refusalMessage = "Member has $0.01 of budget remaining, less than this request's estimated cost",
  hasTraces = false,
  url = "/",
  byModel = [],
}: {
  windowCost: number
  periodCost: number
  member?: OrganizationMember
  defaults?: WorkspaceBudgetDefault[]
  refusalMessage?: string
  /** Whether the deployment captures sessions, as its bootstrap says. */
  hasTraces?: boolean
  /** Where the page opens, query string included. */
  url?: string
  /** The window's spend by model id. */
  byModel?: UsageGroupRow[]
}): Promise<URL[]> {
  const asked: URL[] = []
  const refusal: Partial<UsageEntry> = {
    id: "r-1",
    timestamp: new Date().toISOString(),
    user_id: "user-wei",
    error_message: refusalMessage,
  }
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    asked.push(new URL(String(input), "http://test"))
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
        return Response.json([LAPTOP])
      case "/organizations/me/usage":
        return Response.json([refusal])
      case "/organizations/me/traces/count":
        return Response.json({ count: 3 })
      case "/organizations/me/usage/summary": {
        const start = query.get("start_date") ?? ""
        const isWindow =
          query.has("end_date") ||
          query.getAll("dimensions").some((one) => one !== "none")
        return Response.json({
          ...summary(start, isWindow ? windowCost : periodCost),
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
          // Asked by a provider row, for its top model; else the window's.
          by_model: query.has("provider")
            ? [
                {
                  key: "claude-sonnet-5",
                  label: null,
                  cost: 1.5,
                  requests: 2,
                  tokens: 600,
                  is_other: false,
                },
              ]
            : byModel,
          by_provider: [
            {
              key: "anthropic",
              label: null,
              cost: 1.5,
              requests: 2,
              tokens: 600,
              is_other: false,
            },
            {
              key: null,
              label: null,
              cost: 0.5,
              requests: 1,
              tokens: 300,
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
      url,
      routes: [
        { path: "/requests", element: <LocationProbe /> },
        { path: "/sessions", element: <LocationProbe /> },
      ],
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
  return asked
}

describe("MemberHome, by model", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("lists one model once and opens the sessions of every id it ran as", async () => {
    const user = userEvent.setup()
    const glm = {
      label: "GLM 5.3",
      model_group: "z-ai/glm-5.3",
      is_other: false,
    }
    await renderMemberHome({
      windowCost: 2,
      periodCost: 2,
      hasTraces: true,
      url: "/?by=models",
      byModel: [
        { ...glm, key: "zai-org/GLM-5.3", cost: 1.5, requests: 2, tokens: 600 },
        {
          ...glm,
          key: "accounts/fireworks/models/glm-5p3",
          cost: 0.5,
          requests: 1,
          tokens: 300,
        },
      ],
    })
    const table = await screen.findByRole("grid", {
      name: "Your spend by models",
    })
    await within(table).findByRole("rowheader", { name: "GLM 5.3" })
    const [row, ...rest] = within(table).getAllByRole("row").slice(1)
    expect(rest).toHaveLength(0)
    expect(within(row).getByRole("rowheader")).toHaveTextContent("GLM 5.3")

    await user.click(within(row).getByRole("rowheader"))
    const landed = await screen.findByTestId("location")
    const [path, query] = (landed.textContent ?? "").split("?")
    expect(path).toBe("/sessions")
    expect(new URLSearchParams(query).getAll("model")).toEqual([
      "zai-org/GLM-5.3",
      "accounts/fireworks/models/glm-5p3",
    ])
  })
})

/** Where a row's press landed, with the search it carried. */

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
        "A limit set on you refused requests in the last 30 days.",
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

describe("MemberHome, by provider", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  async function providerTable() {
    const table = await screen.findByRole("grid", {
      name: "Your spend by providers",
    })
    await within(table).findByText("Anthropic")
    return table
  }

  it("offers Providers beside their keys and models", async () => {
    await renderMemberHome({ windowCost: 2, periodCost: 2 })
    const cuts = await screen.findByRole("radiogroup", {
      name: "Break down by",
    })
    expect(
      within(cuts)
        .getAllByRole("radio")
        .map((radio) => radio.closest("label")?.textContent),
    ).toEqual(["API keys", "Models", "Providers"])
  })

  it("lists their spend by provider, the unrecorded ones as one row", async () => {
    await renderMemberHome({
      windowCost: 2,
      periodCost: 2,
      url: "/?by=providers",
    })
    const table = await providerTable()
    const anthropic = within(table).getByRole("row", { name: /Anthropic/ })
    expect(within(anthropic).getByText("75%")).toBeVisible()
    expect(
      await within(anthropic).findByText("claude-sonnet-5"),
    ).toBeInTheDocument()
    const unknown = within(table).getByRole("row", {
      name: /Unknown provider/,
    })
    expect(within(unknown).getByText("25%")).toBeVisible()
    expect(screen.getByText(/A row opens its requests/)).toBeVisible()
  })

  it("opens their own requests to that provider", async () => {
    const user = userEvent.setup()
    await renderMemberHome({
      windowCost: 2,
      periodCost: 2,
      url: "/?by=providers",
    })
    const table = await providerTable()
    await user.click(within(table).getByText("Anthropic"))
    const landed = await screen.findByTestId("location")
    const [path, query] = (landed.textContent ?? "").split("?")
    expect(path).toBe("/requests")
    const search = new URLSearchParams(query)
    expect(search.get("provider")).toBe("anthropic")
    expect(search.get("user_id")).toBe("user-wei")
    expect(search.get("start_date")).toBeTruthy()
  })
})

describe("MemberHome, the window", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("ends every read of yesterday at midnight, refusals and sessions included", async () => {
    const asked = await renderMemberHome({
      windowCost: 41,
      periodCost: 41,
      member: WEI_LIMITED_ALONE,
      defaults: [],
      refusalMessage: "Member has exceeded budget limit",
      hasTraces: true,
      url: "/?period=yesterday",
    })
    expect(
      await screen.findByText("A limit set on you refused requests yesterday."),
    ).toBeVisible()
    expect(screen.getByText("Spent yesterday")).toBeVisible()
    expect(screen.getByText("yesterday in Platform")).toBeVisible()
    const reads = asked.filter(
      (url) =>
        url.searchParams.has("start_date") || url.searchParams.has("start"),
    )
    // The window, the one before it, the refusals and the session count.
    expect(reads.length).toBeGreaterThanOrEqual(4)
    for (const url of reads) {
      expect(
        url.searchParams.get("end_date") ?? url.searchParams.get("end"),
        String(url),
      ).toBeTruthy()
    }
  })

  it("labels its figures over three months, not over a count of days", async () => {
    await renderMemberHome({ windowCost: 2, periodCost: 2, url: "/?period=3m" })
    expect(await screen.findByText("Spent, 3 months")).toBeVisible()
    expect(screen.getByText("Tokens, 3 months")).toBeVisible()
  })
})
