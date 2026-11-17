import { Outlet } from "@tanstack/react-router"
import { screen, waitFor, within } from "@testing-library/react"
import type { ReactElement } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { RoutingPage } from "@/features/routing/RoutingPage"
import { RulesPage } from "@/features/rules/RulesPage"
import { API_ROOT } from "@/shared/api/client"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap, organizationContext } from "@/tests/fixtures"
import { AppProviders as Provider } from "@/tests/providers"
import { flushRouter, renderWithRouter } from "@/tests/router"

type Role = NonNullable<
  NonNullable<Parameters<typeof organizationContext>[0]>["role"]
>

const ROLES: readonly Role[] = ["owner", "admin", "member", "viewer"]
const MANAGERS: readonly Role[] = ["owner", "admin"]

/** A page as `role` sees it, on a deployment hosting every surface. */
async function renderAs(role: Role, page: ReactElement, url: string) {
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const path = String(input)
    if (path === `${API_ROOT}/organizations/me`) {
      return Response.json(
        organizationContext({ role, deployment_operator: false }),
      )
    }
    // The routing reads answer with bare lists.
    if (/\/(routing-policies|aliases)(\?|$)/.test(path)) {
      return Response.json([])
    }
    return Response.json({ data: [], count: 0 })
  })
  return renderWithRouter(page, {
    url,
    // The page the Rules line links to, so whether the caller may open it is
    // judged on a route the router serves, as in the dashboard.
    routes: [{ path: "/tools/guardrails", element: null }],
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

function tabRow() {
  return within(screen.getByRole("navigation", { name: "Rule types" }))
}

describe("Routing, under Rules", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  for (const role of ROLES) {
    it(`gives a ${role} a Routing tab after the rule types`, async () => {
      await renderAs(role, <RulesPage />, "/rules")
      await screen.findByRole("heading", { name: "Rules" })
      const tabs = tabRow().getAllByRole("link")
      const routing = tabs.at(-1)
      expect(routing).toHaveTextContent("Routing")
      expect(routing).toHaveAttribute("href", "/routing")
      expect(tabs.map((tab) => tab.textContent)).toEqual(
        MANAGERS.includes(role)
          ? ["All", "Spend", "Access", "Data", "Routing"]
          : ["All", "Spend", "Access", "Routing"],
      )
    })

    it(`renders the routing policies under Rules for a ${role}`, async () => {
      await renderAs(role, <RoutingPage />, "/routing")
      expect(
        await screen.findByRole("heading", { name: "Rules" }),
      ).toBeInTheDocument()
      expect(tabRow().getByRole("link", { name: "Routing" })).toHaveAttribute(
        "aria-current",
        "page",
      )
      // The tab names the view, so it has no title of its own.
      expect(screen.queryByRole("heading", { name: "Routing" })).toBeNull()
      expect(
        await screen.findByText("No routing policies yet"),
      ).toBeInTheDocument()
    })

    if (MANAGERS.includes(role)) {
      it(`links a ${role} to the workspace's guardrails from Rules`, async () => {
        await renderAs(role, <RulesPage />, "/rules?type=data")
        const link = await screen.findByRole("link", {
          name: "Workspace guardrails",
        })
        expect(link).toHaveAttribute("href", "/tools/guardrails")
      })
    } else {
      // The page is for the owners and admins who set the guardrails.
      it(`offers a ${role} no link to the workspace's guardrails`, async () => {
        await renderAs(role, <RulesPage />, "/rules")
        await screen.findByRole("heading", { name: "Rules" })
        await waitFor(() =>
          expect(
            vi
              .mocked(globalThis.fetch)
              .mock.calls.some(
                ([input]) => String(input) === `${API_ROOT}/organizations/me`,
              ),
          ).toBe(true),
        )
        await flushRouter()
        expect(
          screen.queryByRole("link", { name: "Workspace guardrails" }),
        ).toBeNull()
      })
    }
  }

  it("keeps the link off an owner's other tabs", async () => {
    await renderAs("owner", <RulesPage />, "/rules")
    await screen.findByRole("heading", { name: "Rules" })
    expect(
      screen.queryByRole("link", { name: "Workspace guardrails" }),
    ).toBeNull()
  })
})
