import { Outlet } from "@tanstack/react-router"
import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactElement } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { RoutingPage } from "@/features/routing/RoutingPage"
import { GuardrailsPage } from "@/features/rules/GuardrailsPage"
import { API_ROOT } from "@/shared/api/client"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  apiKey,
  bootstrap,
  organizationContext,
  organizationGuardrail,
  organizationMember,
  organizationSpendCeiling,
  workspace,
} from "@/tests/fixtures"
import { AppProviders as Provider } from "@/tests/providers"
import { renderWithRouter } from "@/tests/router"

type Role = NonNullable<
  NonNullable<Parameters<typeof organizationContext>[0]>["role"]
>

const ROLES: readonly Role[] = ["owner", "admin", "member", "viewer"]
const MANAGERS: readonly Role[] = ["owner", "admin"]

const PLATFORM = workspace({ id: "ws-platform", name: "Platform" })
const RESEARCH = workspace({ id: "ws-research", name: "Research" })

/** Moderation on Platform, on Research only, and paused. */
const GUARDRAILS = [
  organizationGuardrail({
    id: "g-pii",
    profile: "pii",
    mode: "block",
    applies_to_all_workspaces: true,
  }),
  organizationGuardrail({
    id: "g-injection",
    profile: "prompt-injection",
    mode: "monitor",
    workspace_ids: [PLATFORM.id, RESEARCH.id],
  }),
  organizationGuardrail({
    id: "g-research",
    profile: "secrets",
    workspace_ids: [RESEARCH.id],
  }),
  organizationGuardrail({
    id: "g-paused",
    profile: "toxicity",
    enabled: false,
    applies_to_all_workspaces: true,
  }),
]

const placement = (workspaceId: string, name: string, id: string) => ({
  workspace_id: workspaceId,
  workspace_member_id: id,
  workspace_name: name,
  role: "member",
})

const MEMBERS = [
  organizationMember({
    organization_member_id: "om-wei",
    user_id: "u-wei",
    full_name: "Wei Zhang",
    role: "owner",
    workspaces: [placement(PLATFORM.id, "Platform", "wm-wei")],
  }),
  organizationMember({
    organization_member_id: "om-elena",
    user_id: "u-elena",
    full_name: "Elena Petrova",
    role: "member",
    workspaces: [placement(PLATFORM.id, "Platform", "wm-elena")],
  }),
  organizationMember({
    organization_member_id: "om-hana",
    user_id: "u-hana",
    full_name: "Hana Kobayashi",
    role: "member",
    workspaces: [placement(RESEARCH.id, "Research", "wm-hana")],
  }),
]

const CEILINGS = [
  organizationSpendCeiling({
    id: "c-org",
    scope_type: "organization",
    scope_id: "org",
    budget_id: "b-org",
    name: "Northwind Labs, monthly",
    max_budget: 500,
  }),
  organizationSpendCeiling({
    id: "c-wei",
    scope_type: "workspace_member",
    scope_id: "wm-wei",
    budget_id: "b-each",
    name: "Each member of Platform, monthly",
    max_budget: 50,
  }),
  organizationSpendCeiling({
    id: "c-elena",
    scope_type: "org_member",
    scope_id: "om-elena",
    budget_id: "b-contractor",
    name: "Contractor cap",
    max_budget: 25,
  }),
  organizationSpendCeiling({
    id: "c-ci",
    scope_type: "api_token",
    scope_id: "key-ci",
    budget_id: "b-ci",
    name: "CI pipeline, daily",
    max_budget: 10,
  }),
]

const KEYS = [
  apiKey({
    id: "key-ci",
    key_name: "CI pipeline",
    workspace_id: PLATFORM.id,
    allowed_models: ["openai:gpt-5.1-mini"],
    allowed_tools: [],
  }),
  apiKey({
    id: "key-laptop",
    key_name: "Claude Code, laptop",
    workspace_id: PLATFORM.id,
    allowed_models: ["anthropic:*"],
  }),
  apiKey({
    id: "key-research",
    key_name: "Research notebook",
    workspace_id: RESEARCH.id,
  }),
]

interface Data {
  guardrails?: unknown[]
  members?: unknown[]
  ceilings?: unknown[]
  keys?: unknown[]
  /** Paths answered with a 403, which the client does not retry. */
  failing?: string[]
  /** Paths held until the test lets them answer. */
  held?: Record<string, Promise<void>>
}

const FULL: Data = {
  guardrails: GUARDRAILS,
  members: MEMBERS,
  ceilings: CEILINGS,
  keys: KEYS,
}

/** A page as `role` sees it, with Platform selected. */
async function renderAs(
  role: Role,
  page: ReactElement,
  url: string,
  data: Data = FULL,
) {
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const path = String(input).slice(API_ROOT.length).split("?")[0] ?? ""
    if (data.failing?.includes(path)) {
      return Response.json({ detail: `Refused ${path}` }, { status: 403 })
    }
    await data.held?.[path]
    const paged = (rows: unknown[] = []) =>
      Response.json({ data: rows, count: rows.length })
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
      case "/workspaces":
        return paged([PLATFORM, RESEARCH])
      case "/organizations/me/guardrails":
        return paged(data.guardrails)
      case "/organizations/me/members":
        return paged(data.members)
      case "/organizations/me/spend-ceilings":
        return paged(data.ceilings)
      case "/organizations/me/keys":
        return Response.json(data.keys ?? [])
      default:
        // The routing reads answer with bare lists.
        return /\/(routing-policies|aliases)$/.test(path)
          ? Response.json([])
          : paged([])
    }
  })
  return renderWithRouter(page, {
    url,
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
}

const views = () =>
  screen.queryByRole("navigation", { name: "Guardrails views" })

describe("Guardrails", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  for (const role of ROLES) {
    const isManager = MANAGERS.includes(role)

    it(`gives a ${role} ${isManager ? "both views and all three bands" : "budgets and key lists only"}`, async () => {
      await renderAs(role, <GuardrailsPage />, "/guardrails")
      await screen.findByRole("heading", { name: "Guardrails", level: 1 })
      await screen.findByRole("heading", { name: /^Budget/ })
      expect(
        screen.getByRole("heading", { name: /^Model & provider access/ }),
      ).toBeVisible()
      if (isManager) {
        await waitFor(() => expect(views()).not.toBeNull())
        expect(
          within(views() as HTMLElement)
            .getAllByRole("link")
            .map((tab) => tab.textContent),
        ).toEqual(["Guardrails", "Assignments"])
        expect(
          screen.getByRole("button", { name: "New guardrail" }),
        ).toBeVisible()
        expect(
          await screen.findByRole("heading", { name: /^Content moderation/ }),
        ).toBeVisible()
      } else {
        expect(
          await screen.findByText(
            /Owners and admins also see organization and workspace budgets/,
          ),
        ).toBeVisible()
        expect(views()).toBeNull()
        expect(screen.queryByRole("button", { name: "New guardrail" })).toBe(
          null,
        )
        expect(
          screen.queryByRole("heading", { name: /^Content moderation/ }),
        ).toBeNull()
      }
    })

    it(`titles the routing policies Routing, with no Guardrails views, for a ${role}`, async () => {
      await renderAs(role, <RoutingPage />, "/routing")
      expect(
        await screen.findByRole("heading", { name: "Routing" }),
      ).toBeInTheDocument()
      expect(screen.queryByRole("heading", { name: "Guardrails" })).toBeNull()
      expect(views()).toBeNull()
      expect(screen.queryByRole("button", { name: "New guardrail" })).toBeNull()
    })
  }

  it("lists the content moderation running in the selected workspace only", async () => {
    await renderAs("owner", <GuardrailsPage />, "/guardrails")
    const grid = await screen.findByRole("grid", {
      name: "Content moderation running in Platform",
    })
    const pii = await within(grid).findByRole("row", { name: /^pii/ })
    expect(pii).toHaveTextContent("Everyone")
    expect(pii).toHaveTextContent("Block")
    const injection = within(grid).getByRole("row", {
      name: /^prompt-injection/,
    })
    expect(injection).toHaveTextContent("Platform, Research")
    expect(injection).toHaveTextContent("Monitor")
    // Research only, and paused.
    expect(within(grid).queryByText("secrets")).toBeNull()
    expect(within(grid).queryByText("toxicity")).toBeNull()
    expect(
      screen.getByRole("heading", { name: /^Content moderation/ }),
    ).toHaveTextContent("Content moderation2")
  })

  it("shows a key's lists in the access band, a wildcard in words", async () => {
    await renderAs("owner", <GuardrailsPage />, "/guardrails")
    const grid = await screen.findByRole("grid", {
      name: "Model & provider access",
    })
    const laptop = await within(grid).findByRole("row", {
      name: /Claude Code, laptop/,
    })
    expect(laptop).toHaveTextContent("All Anthropic models")
    expect(laptop).toHaveTextContent("Any tool")
    const ci = within(grid).getByRole("row", { name: /CI pipeline/ })
    expect(ci).toHaveTextContent("openai:gpt-5.1-mini")
    expect(ci).toHaveTextContent("No tools")
  })

  it("offers an owner with nothing set one way to start", async () => {
    await renderAs("owner", <GuardrailsPage />, "/guardrails", {})
    expect(
      await screen.findByRole("heading", { name: "No guardrails yet" }),
    ).toBeVisible()
    expect(screen.queryByRole("heading", { name: /^Budget/ })).toBeNull()
    // The page's action, and the empty state's.
    expect(
      screen.getAllByRole("button", { name: "New guardrail" }),
    ).toHaveLength(2)
  })

  it("keeps the bands when only moderation is set, each empty band in a line", async () => {
    await renderAs("owner", <GuardrailsPage />, "/guardrails", {
      guardrails: GUARDRAILS,
    })
    expect(await screen.findByText("No budgets yet.")).toBeVisible()
    expect(screen.queryByText("No guardrails yet")).toBeNull()
  })
})

describe("Guardrails, Assignments", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("lists the selected workspace's members with their own budgets and its moderation", async () => {
    await renderAs("owner", <GuardrailsPage />, "/guardrails?view=assignments")
    const grid = await screen.findByRole("grid", {
      name: "Guardrails on each member",
    })
    const wei = await within(grid).findByRole("row", { name: /Wei Zhang/ })
    expect(wei).toHaveTextContent("Owner")
    expect(wei).toHaveTextContent("Each member of Platform, monthly")
    expect(wei).toHaveTextContent("pii, prompt-injection")
    expect(wei).toHaveTextContent("via Platform")
    const elena = within(grid).getByRole("row", { name: /Elena Petrova/ })
    expect(elena).toHaveTextContent("Contractor cap")
    // Research's member is not in Platform.
    expect(within(grid).queryByText("Hana Kobayashi")).toBeNull()
    // A member's model list is the deployment operator's to read.
    expect(
      within(grid)
        .getAllByRole("columnheader")
        .map((header) => header.textContent),
    ).toEqual(["Member", "Role", "Budget", "Content moderation"])
    expect(
      screen.getByText(/Everyone also stays under the organization’s budget/),
    ).toBeVisible()
  })

  it("finds a member by name", async () => {
    const user = userEvent.setup()
    await renderAs("owner", <GuardrailsPage />, "/guardrails?view=assignments")
    const grid = await screen.findByRole("grid", {
      name: "Guardrails on each member",
    })
    await within(grid).findByRole("row", { name: /Wei Zhang/ })
    await user.type(
      screen.getByRole("searchbox", { name: "Search people" }),
      "elena",
    )
    await waitFor(() =>
      expect(within(grid).queryByText("Wei Zhang")).toBeNull(),
    )
    expect(within(grid).getByText("Elena Petrova")).toBeVisible()
  })

  it("lists the selected workspace's API keys with their budgets and lists", async () => {
    const user = userEvent.setup()
    await renderAs("owner", <GuardrailsPage />, "/guardrails?view=assignments")
    await screen.findByRole("grid", { name: "Guardrails on each member" })
    await user.click(screen.getByRole("radio", { name: "API keys" }))
    const grid = await screen.findByRole("grid", {
      name: "Guardrails on each API key",
    })
    const ci = await within(grid).findByRole("row", { name: /CI pipeline/ })
    expect(ci).toHaveTextContent("CI pipeline, daily")
    expect(ci).toHaveTextContent("openai:gpt-5.1-mini")
    expect(ci).toHaveTextContent("No tools")
    expect(ci).toHaveTextContent("pii, prompt-injection")
    expect(within(grid).queryByText("Research notebook")).toBeNull()
  })

  it("opens a member on the Guardrails view, whatever the address asks", async () => {
    await renderAs("member", <GuardrailsPage />, "/guardrails?view=assignments")
    await screen.findByRole("heading", { name: /^Budget/ })
    expect(
      screen.queryByRole("grid", { name: "Guardrails on each member" }),
    ).toBeNull()
  })
})

describe("Guardrails, when a read fails or is slow", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("says a member's moderation could not be read, and says why once", async () => {
    await renderAs(
      "owner",
      <GuardrailsPage />,
      "/guardrails?view=assignments",
      {
        ...FULL,
        failing: ["/organizations/me/guardrails"],
      },
    )
    const grid = await screen.findByRole("grid", {
      name: "Guardrails on each member",
    })
    const wei = await within(grid).findByRole("row", { name: /Wei Zhang/ })
    expect(wei).toHaveTextContent("Could not be read")
    expect(wei).not.toHaveTextContent("None")
    expect(screen.getAllByRole("alert")).toHaveLength(1)
  })

  it("says a member's budget could not be read, and says why once", async () => {
    await renderAs(
      "owner",
      <GuardrailsPage />,
      "/guardrails?view=assignments",
      {
        ...FULL,
        failing: ["/organizations/me/spend-ceilings"],
      },
    )
    const grid = await screen.findByRole("grid", {
      name: "Guardrails on each member",
    })
    const wei = await within(grid).findByRole("row", { name: /Wei Zhang/ })
    expect(wei).toHaveTextContent("Could not be read")
    expect(screen.getAllByRole("alert")).toHaveLength(1)
  })

  it("draws no moderation row before the definitions and workspaces land", async () => {
    let answer = () => {}
    const held = new Promise<void>((resolve) => {
      answer = resolve
    })
    await renderAs("owner", <GuardrailsPage />, "/guardrails", {
      ...FULL,
      held: { "/workspaces": held },
    })
    await screen.findByRole("heading", { name: /^Content moderation/ })
    await new Promise((resolve) => setTimeout(resolve, 50))
    expect(screen.queryByText("prompt-injection")).toBeNull()
    expect(screen.queryByText(/A workspace/)).toBeNull()
    answer()
    const grid = await screen.findByRole("grid", {
      name: "Content moderation running in Platform",
    })
    expect(
      await within(grid).findByRole("row", { name: /^prompt-injection/ }),
    ).toHaveTextContent("Platform, Research")
  })

  it("names no budget in Assignments before the budgets land", async () => {
    let answer = () => {}
    const held = new Promise<void>((resolve) => {
      answer = resolve
    })
    await renderAs(
      "owner",
      <GuardrailsPage />,
      "/guardrails?view=assignments",
      {
        ...FULL,
        // Ceilings named by their budget alone, as a ceiling made without a name is.
        ceilings: CEILINGS.map((one) => ({ ...one, name: null })),
        held: { "/organizations/me/budgets": held },
      },
    )
    await screen.findByRole("grid", { name: "Guardrails on each member" })
    await new Promise((resolve) => setTimeout(resolve, 50))
    expect(screen.queryByText(/Spend limit/)).toBeNull()
    answer()
    expect(
      await screen.findByRole("row", { name: /Wei Zhang/ }),
    ).toBeInTheDocument()
  })

  for (const role of ["member", "viewer"] as const) {
    it(`asks nothing a ${role} would be refused`, async () => {
      await renderAs(role, <GuardrailsPage />, "/guardrails")
      await screen.findByRole("heading", { name: /^Budget/ })
      const asked = vi
        .mocked(globalThis.fetch)
        .mock.calls.map(
          ([input]) => String(input).slice(API_ROOT.length).split("?")[0],
        )
      for (const path of [
        "/organizations/me/guardrails",
        "/organizations/me/guardrail-definitions",
        "/organizations/me/spend-ceilings",
        "/workspaces",
      ]) {
        expect(asked, path).not.toContain(path)
      }
    })
  }
})
