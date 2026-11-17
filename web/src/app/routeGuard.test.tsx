import {
  createMemoryHistory,
  createRootRoute,
  createRoute,
  createRouter,
  RouterProvider,
  redirect,
} from "@tanstack/react-router"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
import { AppShell } from "@/app/AppShell"
import { Provider } from "@/app/provider"
import { API_ROOT } from "@/shared/api/client"
import {
  parseSearch,
  stringifySearch,
  validateSearch,
} from "@/shared/helpers/search"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  BASE_CAPABILITIES,
  EntitlementProvider,
} from "@/shared/hooks/useEntitlements"
import { bootstrap, organizationContext } from "@/tests/fixtures"
import { flushRouter } from "@/tests/router"

type Context = NonNullable<Parameters<typeof organizationContext>[0]>
type Role = NonNullable<Context["role"]>
type Memberships = NonNullable<Context["workspace_memberships"]>

const PLATFORM: Memberships = [
  { workspace_id: "ws-platform", name: "Platform", role: "member" },
]

/**
 * The shell at `url` as `role` sees it. The page is served by a route declared
 * at `route` (by default the address's own path), as the real tree declares
 * it, so the router matches the address the way the dashboard's router does;
 * `@/tests/router` cannot say this, since its route is always the address
 * itself. Home is a destination of its own, so a redirect is visible, and an
 * address nothing serves goes Home as `routes/$.tsx` sends it. `context` holds
 * the caller's context back until it settles, as a cold reload does.
 */
async function renderAt(
  url: string,
  role: Role,
  {
    route = url.split("?")[0],
    workspace_memberships = PLATFORM,
    context,
  }: {
    route?: string
    workspace_memberships?: Memberships
    context?: Promise<unknown>
  } = {},
) {
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const path = String(input)
    if (path === `${API_ROOT}/organizations/me`) {
      await context
      return Response.json(
        organizationContext({
          role,
          deployment_operator: false,
          workspace_memberships,
        }),
      )
    }
    if (path.includes("/aliases")) return Response.json([])
    return Response.json({ data: [], count: 0 })
  })
  const shellRoute = createRootRoute({
    validateSearch,
    component: () => (
      <Provider>
        <DeploymentProvider value={bootstrap()}>
          <EntitlementProvider
            value={{ capabilities: BASE_CAPABILITIES, isLoading: false }}
          >
            <SelectedWorkspaceProvider>
              <AppShell />
            </SelectedWorkspaceProvider>
          </EntitlementProvider>
        </DeploymentProvider>
      </Provider>
    ),
  })
  const page = createRoute({
    getParentRoute: () => shellRoute,
    path: route,
    component: () => <div>PAGE CONTENT</div>,
  })
  const home = createRoute({
    getParentRoute: () => shellRoute,
    path: "/",
    component: () => <div>HOME PAGE</div>,
  })
  const unknown = createRoute({
    getParentRoute: () => shellRoute,
    path: "$",
    beforeLoad: () => {
      throw redirect({ to: "/", replace: true })
    },
  })
  const router = createRouter({
    routeTree: shellRoute.addChildren([page, home, unknown]),
    history: createMemoryHistory({ initialEntries: [url] }),
    parseSearch,
    stringifySearch,
  })
  render(<RouterProvider router={router} />)
  await flushRouter()
}

function aOrAn(role: Role): string {
  return role === "owner" || role === "admin" ? `an ${role}` : `a ${role}`
}

function setDesktop() {
  vi.stubGlobal(
    "matchMedia",
    vi.fn().mockReturnValue({
      matches: false,
      media: "",
      onchange: null,
      addListener: () => undefined,
      removeListener: () => undefined,
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
      dispatchEvent: () => true,
    }),
  )
}

async function expectSentHome() {
  expect(await screen.findByText("HOME PAGE")).toBeInTheDocument()
  expect(screen.queryByText("PAGE CONTENT")).toBeNull()
  // Home, not a panel claiming the deployment does not serve the page.
  expect(screen.queryByText(/is not available here/)).toBeNull()
}

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  window.localStorage.clear()
})

describe("a page the caller may not open", () => {
  for (const url of [
    "/organization/members",
    "/organization",
    "/budgets",
    "/workspaces",
    "/settings",
    "/admin/accounts",
    "/tools/guardrails",
  ]) {
    it(`sends a member who types ${url} Home`, async () => {
      setDesktop()
      await renderAt(url, "member")
      await expectSentHome()
    })
  }

  for (const url of [
    "/settings",
    "/health-checks",
    "/admin/hosted-providers",
  ]) {
    it(`sends an admin who types ${url} Home`, async () => {
      setDesktop()
      await renderAt(url, "admin")
      await expectSentHome()
    })
  }

  it("sends a viewer who types /playground Home, and offers them no Playground link", async () => {
    setDesktop()
    await renderAt("/playground", "viewer", {
      workspace_memberships: [
        { workspace_id: "ws-platform", name: "Platform", role: "viewer" },
      ],
    })
    await expectSentHome()
    expect(screen.queryByRole("link", { name: "Playground" })).toBeNull()
  })
})

describe("an address written another way", () => {
  // The router matches without regard to case, so each of these serves the
  // gated page and must go Home for the roles the lowercase form does.
  for (const [url, route, role] of [
    ["/Organization/members", "/organization/members", "member"],
    ["/ORGANIZATION/Domains", "/organization/domains", "member"],
    ["/Budgets", "/budgets", "member"],
    ["/Workspaces", "/workspaces", "member"],
    ["/Settings", "/settings", "member"],
    ["/Settings", "/settings", "admin"],
    ["/Admin/accounts", "/admin/accounts", "member"],
    ["/Admin/accounts", "/admin/accounts", "admin"],
    ["/tools/Guardrails", "/tools/guardrails", "member"],
    // A query string, a trailing slash, an escaped letter and a doubled slash.
    ["/organization/members?x=1", "/organization/members", "member"],
    ["/organization/members/", "/organization/members", "member"],
    ["/%6Frganization/members", "/organization/members", "member"],
    ["//organization/members", "/organization/members", "member"],
  ] as const) {
    it(`sends ${aOrAn(role)} who types ${url} Home`, async () => {
      setDesktop()
      await renderAt(url, role, { route })
      await expectSentHome()
    })
  }

  // The control for the cases above: each form is the page for someone who
  // may open it, so the member going Home is the guard and not the catch-all.
  for (const url of [
    "/Organization/members",
    "/organization/members?x=1",
    "/organization/members/",
    "/%6Frganization/members",
    "//organization/members",
  ]) {
    it(`serves ${url} to an owner`, async () => {
      setDesktop()
      await renderAt(url, "owner", { route: "/organization/members" })
      expect(await screen.findByText("PAGE CONTENT")).toBeInTheDocument()
    })
  }

  // An escaped slash, a dot segment and a parent segment are not the page to
  // the router at all, so they go Home as any unknown address does, for
  // everyone.
  for (const url of [
    "/organization%2Fmembers",
    "/./organization/members",
    "/people/../organization/members",
  ]) {
    it(`sends even an owner who types ${url} Home, as an address nothing serves`, async () => {
      setDesktop()
      await renderAt(url, "owner", { route: "/organization/members" })
      await expectSentHome()
    })
  }

  it("sends a viewer who types /Playground Home", async () => {
    setDesktop()
    await renderAt("/Playground", "viewer", {
      route: "/playground",
      workspace_memberships: [
        { workspace_id: "ws-platform", name: "Platform", role: "viewer" },
      ],
    })
    await expectSentHome()
  })

  it("keeps a workspace admin's page open under another case", async () => {
    setDesktop()
    await renderAt("/Workspaces/ws-data", "member", {
      route: "/workspaces/$workspaceId",
      workspace_memberships: [
        { workspace_id: "ws-data", name: "Data", role: "admin" },
      ],
    })
    expect(await screen.findByText("PAGE CONTENT")).toBeInTheDocument()
  })

  it("keeps the members page open to an owner under another case", async () => {
    setDesktop()
    await renderAt("/ORGANIZATION/members", "owner", {
      route: "/organization/members",
    })
    expect(await screen.findByText("PAGE CONTENT")).toBeInTheDocument()
  })
})

describe("a page the caller may open", () => {
  it("never sends an owner reloading an organization page Home while their role loads", async () => {
    setDesktop()
    let answer = (_: unknown) => {}
    await renderAt("/organization/members", "owner", {
      context: new Promise((resolve) => {
        answer = resolve
      }),
    })
    // Before the context answers, the page waits rather than leaving.
    expect(await screen.findByRole("status")).toBeInTheDocument()
    expect(screen.queryByText("HOME PAGE")).toBeNull()
    expect(screen.queryByText("PAGE CONTENT")).toBeNull()

    answer(undefined)
    expect(await screen.findByText("PAGE CONTENT")).toBeInTheDocument()
    expect(screen.queryByText("HOME PAGE")).toBeNull()
  })

  const DATA_ADMIN: Memberships = [
    { workspace_id: "ws-data", name: "Data", role: "admin" },
  ]

  it("opens a workspace's page to its own admin", async () => {
    setDesktop()
    await renderAt("/workspaces/ws-data", "member", {
      route: "/workspaces/$workspaceId",
      workspace_memberships: DATA_ADMIN,
    })
    expect(await screen.findByText("PAGE CONTENT")).toBeInTheDocument()
  })

  it("does not open the Workspaces list to that admin", async () => {
    setDesktop()
    await renderAt("/workspaces", "member", {
      workspace_memberships: DATA_ADMIN,
    })
    await expectSentHome()
  })

  it("does not open another workspace's page to that admin", async () => {
    setDesktop()
    await renderAt("/workspaces/ws-platform", "member", {
      route: "/workspaces/$workspaceId",
      workspace_memberships: DATA_ADMIN,
    })
    await expectSentHome()
  })

  it("offers a member the Playground link and opens it", async () => {
    setDesktop()
    await renderAt("/playground", "member")
    expect(await screen.findByText("PAGE CONTENT")).toBeInTheDocument()
    expect(
      await screen.findByRole("link", { name: "Playground" }),
    ).toBeInTheDocument()
  })

  it("opens the workspace's guardrails to an admin", async () => {
    setDesktop()
    await renderAt("/tools/guardrails", "admin")
    expect(await screen.findByText("PAGE CONTENT")).toBeInTheDocument()
  })

  it("keeps a member's way in from a rail row they may open", async () => {
    setDesktop()
    const user = userEvent.setup()
    await renderAt("/keys", "member")
    await screen.findByText("PAGE CONTENT")
    await user.click(screen.getByRole("link", { name: "Home" }))
    expect(await screen.findByText("HOME PAGE")).toBeInTheDocument()
  })
})

describe("the selected rail row", () => {
  for (const [url, row, route] of [
    ["/people/some-person", "People", "/people/$personId"],
    ["/rules/some-rule", "Rules", "/rules/$ruleId"],
    ["/routing/coding", "Rules", "/routing/$name"],
    ["/keys?state=disabled", "API keys", "/keys"],
    ["/sessions?range=30d", "Sessions", "/sessions"],
  ] as const) {
    it(`is announced as current on ${url}`, async () => {
      setDesktop()
      await renderAt(url, "member", { route })
      const sidebar = await screen.findByRole("navigation", { name: "Sidebar" })
      await waitFor(() =>
        expect(
          within(sidebar).getByRole("link", { name: row }),
        ).toHaveAttribute("aria-current", "page"),
      )
      expect(
        within(sidebar)
          .getAllByRole("link")
          .filter((link) => link.getAttribute("aria-current") === "page"),
      ).toHaveLength(1)
    })
  }
})
