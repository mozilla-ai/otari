import { useIsFetching } from "@tanstack/react-query"
import { Outlet } from "@tanstack/react-router"
import { screen, waitFor } from "@testing-library/react"
import type { ReactElement } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { RulePage } from "@/features/rules/RulePage"
import { RulesPage } from "@/features/rules/RulesPage"
import { API_ROOT } from "@/shared/api/client"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  apiKey,
  bootstrap,
  organizationContext,
  organizationMember,
  organizationSpendCeiling,
  workspace,
  workspaceBudgetDefault,
} from "@/tests/fixtures"
import { AppProviders as Provider } from "@/tests/providers"
import { renderWithRouter } from "@/tests/router"

type Role = NonNullable<
  NonNullable<Parameters<typeof organizationContext>[0]>["role"]
>

const PLATFORM = workspace({ id: "ws-platform", name: "Platform" })
const DEFAULTS_PATH = `/workspaces/${PLATFORM.id}/member-budget-policies`

const SHELL = (
  <Provider>
    <DeploymentProvider value={bootstrap()}>
      <SelectedWorkspaceProvider>
        <Outlet />
      </SelectedWorkspaceProvider>
    </DeploymentProvider>
  </Provider>
)

/** Rules as `role` sees it, with the read at `failing` refused. */
async function renderFailing(
  role: Role,
  failing: string,
  {
    url = "/rules",
    keys = [],
    page = <RulesPage />,
  }: { url?: string; keys?: unknown[]; page?: ReactElement } = {},
) {
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
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
            role,
            deployment_operator: false,
            workspace_memberships: [
              { workspace_id: PLATFORM.id, name: "Platform", role: "member" },
            ],
          }),
        )
      case "/organizations/me/keys":
        return Response.json(keys)
      case "/workspaces":
        return paged([PLATFORM])
      // A rule page lists the refusals its subject met.
      case "/organizations/me/usage":
        return Response.json([])
      default:
        return paged([])
    }
  })
  return renderWithRouter(page, { url, shell: SHELL })
}

describe("Rules, when a read fails", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  for (const failing of [
    "/organizations/me/members",
    "/organizations/me/keys",
    "/workspaces",
    DEFAULTS_PATH,
  ]) {
    it(`tells an owner that ${failing} could not be read`, async () => {
      await renderFailing("owner", failing)
      expect(await screen.findByRole("alert")).toHaveTextContent(
        `Refused ${failing}`,
      )
    })
  }

  it("tells a member that their workspace's default could not be read", async () => {
    await renderFailing("member", DEFAULTS_PATH)
    expect(await screen.findByRole("alert")).toHaveTextContent(
      `Refused ${DEFAULTS_PATH}`,
    )
  })

  it("does not call a tab empty when its read failed", async () => {
    const failing = "/organizations/me/spend-ceilings"
    await renderFailing("owner", failing, {
      url: "/rules?type=spend",
      // A rule on another tab, so the list is not empty as a whole.
      keys: [apiKey({ id: "key-ci", allowed_models: ["gpt-5"] })],
    })
    expect(await screen.findByRole("alert")).toHaveTextContent(
      `Refused ${failing}`,
    )
    expect(
      await screen.findByRole("grid", { name: "Rules" }),
    ).toBeInTheDocument()
    expect(screen.queryByText("No spend limits yet.")).toBeNull()
  })

  it("says a read failed on a rule it could still find", async () => {
    const failing = "/workspaces"
    await renderFailing("owner", failing, {
      url: "/rules/access-key-key-ci",
      keys: [
        apiKey({ id: "key-ci", key_name: "ci", allowed_models: ["gpt-5"] }),
      ],
      page: <RulePage ruleId="access-key-key-ci" />,
    })
    expect(
      await screen.findByRole("heading", {
        name: "ci: allowed models and tools",
      }),
    ).toBeVisible()
    expect(await screen.findByRole("alert")).toHaveTextContent(
      `Refused ${failing}`,
    )
  })

  it("says so when the organization context cannot be read, instead of loading for good", async () => {
    await renderFailing("owner", "/organizations/me")
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Refused /organizations/me",
    )
  })
})

const EACH = "Each member of Platform, monthly"
const EACH_BUDGET = "budget-platform-each"
const WEI_IN_PLATFORM = "wm-wei-platform"

/** How many reads besides the workspace defaults are in flight. */
let othersFetching = -1

function OthersFetching() {
  othersFetching = useIsFetching({
    predicate: ({ queryKey }) => queryKey[2] !== "budget-defaults",
  })
  return null
}

describe("Rules, while a workspace's defaults are read", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("shows a per-person rule to an owner only once its default is read", async () => {
    const requested: string[] = []
    let answerDefaults = () => {}
    const defaultsAnswered = new Promise<void>((resolve) => {
      answerDefaults = resolve
    })
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const path = String(input).slice(API_ROOT.length).split("?")[0]
      requested.push(path)
      const paged = (data: unknown[]) =>
        Response.json({ data, count: data.length })
      switch (path) {
        case "/organizations/me":
          return Response.json(
            organizationContext({ role: "owner", deployment_operator: false }),
          )
        case "/organizations/me/keys":
          return Response.json([])
        case "/workspaces":
          return paged([PLATFORM])
        case "/organizations/me/members":
          return paged([
            organizationMember({
              full_name: "Wei Zhang",
              workspaces: [
                {
                  workspace_id: PLATFORM.id,
                  workspace_member_id: WEI_IN_PLATFORM,
                  workspace_name: "Platform",
                  role: "member",
                },
              ],
            }),
          ])
        case "/organizations/me/spend-ceilings":
          return paged([
            organizationSpendCeiling({
              scope_type: "workspace_member",
              scope_id: WEI_IN_PLATFORM,
              budget_id: EACH_BUDGET,
              name: EACH,
              max_budget: 50,
            }),
          ])
        case DEFAULTS_PATH:
          await defaultsAnswered
          return paged([
            workspaceBudgetDefault({
              workspace_id: PLATFORM.id,
              budget_id: EACH_BUDGET,
              name: EACH,
              max_budget: 50,
              reset_alignment: "calendar_month",
              budget_duration_sec: null,
            }),
          ])
        default:
          return paged([])
      }
    })
    await renderWithRouter(
      <>
        <RulesPage />
        <OthersFetching />
      </>,
      { url: "/rules", shell: SHELL },
    )

    // Every other read has answered, so only the defaults hold the list back.
    await waitFor(() => {
      expect(requested).toContain(DEFAULTS_PATH)
      expect(othersFetching).toBe(0)
    })
    expect(screen.queryByRole("rowheader", { name: EACH })).toBeNull()

    answerDefaults()
    expect(await screen.findByRole("rowheader", { name: EACH })).toBeVisible()
    expect(
      screen.getByRole("button", { name: `Actions for ${EACH}` }),
    ).toBeVisible()
  })
})
