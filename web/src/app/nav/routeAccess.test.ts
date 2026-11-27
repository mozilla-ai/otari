import {
  createMemoryHistory,
  createRootRoute,
  createRoute,
  createRouter,
} from "@tanstack/react-router"
import { describe, expect, it } from "vitest"
import {
  type Access,
  type Caller,
  requirementFor,
  routeAccess,
  targetOf,
} from "@/app/nav/useNavVisibility"
import { routeTree } from "@/routeTree.gen"
import { organizationContext } from "@/tests/fixtures"

// The real route tree, so an address is judged by the route the router
// serves for it, as the shell judges it.
const router = createRouter({ routeTree, history: createMemoryHistory() })
const at = (address: string) => targetOf(router, address)

const PLATFORM = "ws-platform"
const DATA = "ws-data"

/** One caller per role the access table was approved for. */
const ROLES = {
  // Owner of the organization and the deployment's operator.
  operator: {
    context: organizationContext({
      role: "owner",
      deployment_operator: true,
      workspace_memberships: [
        { workspace_id: PLATFORM, name: "Platform", role: "owner" },
        { workspace_id: DATA, name: "Data", role: "owner" },
      ],
    }),
    answer: "operator",
    selectedWorkspaceRole: "owner",
  },
  // An organization admin who does not operate the deployment.
  admin: {
    context: organizationContext({
      role: "admin",
      deployment_operator: false,
      workspace_memberships: [
        { workspace_id: PLATFORM, name: "Platform", role: "admin" },
      ],
    }),
    answer: "not-operator",
    selectedWorkspaceRole: "admin",
  },
  member: {
    context: organizationContext({
      role: "member",
      deployment_operator: false,
      workspace_memberships: [
        { workspace_id: PLATFORM, name: "Platform", role: "member" },
      ],
    }),
    answer: "not-operator",
    selectedWorkspaceRole: "member",
  },
  viewer: {
    context: organizationContext({
      role: "viewer",
      deployment_operator: false,
      workspace_memberships: [
        { workspace_id: PLATFORM, name: "Platform", role: "viewer" },
      ],
    }),
    answer: "not-operator",
    selectedWorkspaceRole: "viewer",
  },
  // An organization member who is an admin of one workspace.
  workspaceAdmin: {
    context: organizationContext({
      role: "member",
      deployment_operator: false,
      workspace_memberships: [
        { workspace_id: DATA, name: "Data", role: "admin" },
      ],
    }),
    answer: "not-operator",
    selectedWorkspaceRole: "admin",
  },
  // An organization member in no workspace.
  noWorkspace: {
    context: organizationContext({
      role: "member",
      deployment_operator: false,
      workspace_memberships: [],
    }),
    answer: "not-operator",
    selectedWorkspaceRole: null,
  },
} satisfies Record<string, Caller>

type Role = keyof typeof ROLES
const ORDER: readonly Role[] = [
  "operator",
  "admin",
  "member",
  "viewer",
  "workspaceAdmin",
  "noWorkspace",
]

const o: Access = "open"
const x: Access = "blocked"

/**
 * The approved route by role table, as implemented: operator, admin, member,
 * viewer, workspace admin, no workspace.
 */
const TABLE: readonly [string, readonly Access[]][] = [
  // Workspace pages.
  ["/", [o, o, o, o, o, o]],
  ["/sessions", [o, o, o, o, o, o]],
  ["/requests", [o, o, o, o, o, o]],
  ["/people", [o, o, o, o, o, o]],
  ["/people/some-person", [o, o, o, o, o, o]],
  ["/rules", [o, o, o, o, o, o]],
  ["/rules/some-rule", [o, o, o, o, o, o]],
  ["/routing", [o, o, o, o, o, o]],
  ["/routing/coding", [o, o, o, o, o, o]],
  ["/models", [o, o, o, o, o, o]],
  ["/models/anthropic/claude-sonnet-4.6", [o, o, o, o, o, o]],
  ["/tools", [o, o, o, o, o, o]],
  ["/tools/web-search", [o, o, o, o, o, o]],
  ["/tools/code-execution", [o, o, o, o, o, o]],
  ["/tools/mcp-servers", [o, o, o, o, o, o]],
  ["/tools/mcp-servers/some-server", [o, o, o, o, o, o]],
  ["/tools/guardrails", [o, o, x, x, x, x]],
  ["/mcp-connections/callback", [o, o, o, o, o, o]],
  ["/keys", [o, o, o, o, o, o]],
  ["/members", [o, o, o, o, o, o]],
  ["/playground", [o, o, o, x, o, o]],
  ["/account", [o, o, o, o, o, o]],
  ["/invitations", [o, o, o, o, o, o]],
  // Organization pages.
  ["/organization", [o, o, x, x, x, x]],
  ["/organization/members", [o, o, x, x, x, x]],
  ["/organization/domains", [o, o, x, x, x, x]],
  ["/organization/sso", [o, o, x, x, x, x]],
  ["/organization/provider-keys", [o, o, x, x, x, x]],
  ["/organization/web-search-keys", [o, o, x, x, x, x]],
  ["/organization/mcp-servers", [o, o, x, x, x, x]],
  ["/organization/guardrails", [o, o, x, x, x, x]],
  ["/organization/pricing", [o, o, x, x, x, x]],
  ["/budgets", [o, o, x, x, x, x]],
  ["/workspaces", [o, o, x, x, x, x]],
  [`/workspaces/${PLATFORM}`, [o, o, x, x, x, x]],
  [`/workspaces/${DATA}`, [o, o, x, x, o, x]],
  // Deployment pages.
  ["/settings", [o, x, x, x, x, x]],
  ["/health-checks", [o, x, x, x, x, x]],
  ["/admin/accounts", [o, x, x, x, x, x]],
  ["/admin/hosted-providers", [o, x, x, x, x, x]],
  ["/admin/recommender", [o, x, x, x, x, x]],
  ["/admin/budgets", [o, x, x, x, x, x]],
  ["/admin/web-search", [o, x, x, x, x, x]],
  ["/admin/code-execution", [o, x, x, x, x, x]],
  // Redirects that stay.
  ["/usage", [o, o, o, o, o, o]],
  ["/activity", [o, o, o, o, o, o]],
  ["/aliases", [o, o, o, o, o, o]],
]

describe("who may open which page", () => {
  for (const [path, expected] of TABLE) {
    it(`answers ${path} per role`, () => {
      expect(ORDER.map((role) => routeAccess(at(path), ROLES[role]))).toEqual(
        expected,
      )
    })
  }

  it("reads an address with a trailing slash as the page", () => {
    expect(routeAccess(at("/organization/members/"), ROLES.member)).toBe(
      "blocked",
    )
    expect(routeAccess(at("/settings/"), ROLES.member)).toBe("blocked")
  })

  it("leaves the MCP sign-in return to no requirement at all", () => {
    expect(at("/mcp-connections/callback").route).toBe(
      "/mcp-connections/callback",
    )
    expect(requirementFor("/mcp-connections/callback")).toBeUndefined()
  })

  it("prefers the exact entry over the one whose subtree contains it", () => {
    // Guardrails sits under Tools by address and asks more than Tools does.
    expect(requirementFor(at("/tools/guardrails").route ?? "")).toBe("manager")
    expect(requirementFor(at("/tools/web-search").route ?? "")).toBeUndefined()
    // A workspace's own page asks less than the list it sits under.
    expect(requirementFor(at("/workspaces").route ?? "")).toBe("manager")
    expect(requirementFor(at("/workspaces/ws-data").route ?? "")).toBe(
      "workspace-manager",
    )
  })

  it("keeps a viewer in no workspace out of the Playground, and a member in", () => {
    const viewer = {
      ...ROLES.noWorkspace,
      context: organizationContext({
        role: "viewer",
        deployment_operator: false,
      }),
    }
    expect(routeAccess(at("/playground"), viewer)).toBe("blocked")
    expect(routeAccess(at("/playground"), ROLES.noWorkspace)).toBe("open")
  })

  it("follows the selected workspace: a viewer there may not open the Playground", () => {
    const memberAndViewer = {
      ...ROLES.member,
      context: organizationContext({
        role: "member",
        deployment_operator: false,
        workspace_memberships: [
          { workspace_id: PLATFORM, name: "Platform", role: "member" },
          { workspace_id: DATA, name: "Data", role: "viewer" },
        ],
      }),
    }
    expect(
      routeAccess(at("/playground"), {
        ...memberAndViewer,
        selectedWorkspaceRole: "member",
      }),
    ).toBe("open")
    expect(
      routeAccess(at("/playground"), {
        ...memberAndViewer,
        selectedWorkspaceRole: "viewer",
      }),
    ).toBe("blocked")
  })
})

describe("an address written another way", () => {
  // The router matches without regard to case, so these are the gated pages
  // themselves and must be answered as they are.
  const VARIANTS: readonly [string, readonly Access[]][] = [
    ["/Organization/members", [o, o, x, x, x, x]],
    ["/ORGANIZATION/Domains", [o, o, x, x, x, x]],
    ["/Budgets", [o, o, x, x, x, x]],
    ["/Playground", [o, o, o, x, o, o]],
    ["/Settings", [o, x, x, x, x, x]],
    ["/Admin/accounts", [o, x, x, x, x, x]],
    ["/tools/Guardrails", [o, o, x, x, x, x]],
    ["/Workspaces", [o, o, x, x, x, x]],
    ["/Workspaces/ws-data", [o, o, x, x, o, x]],
    ["/organization/members/", [o, o, x, x, x, x]],
  ]
  for (const [address, expected] of VARIANTS) {
    it(`answers ${address} as the page it serves`, () => {
      expect(
        ORDER.map((role) => routeAccess(at(address), ROLES[role])),
      ).toEqual(expected)
    })
  }

  it("reads a workspace's id as written: another case is another id", () => {
    expect(routeAccess(at("/workspaces/WS-DATA"), ROLES.workspaceAdmin)).toBe(
      "blocked",
    )
  })
})

describe("an operator who does not manage the organization", () => {
  // Not in the approved table: pinned so a change to it is a decision.
  const operatorMember: Caller = {
    context: organizationContext({
      role: "member",
      deployment_operator: true,
      workspace_memberships: [
        { workspace_id: PLATFORM, name: "Platform", role: "member" },
      ],
    }),
    answer: "operator",
    selectedWorkspaceRole: "member",
  }

  it("opens the deployment's pages", () => {
    expect(routeAccess(at("/settings"), operatorMember)).toBe("open")
    expect(routeAccess(at("/admin/accounts"), operatorMember)).toBe("open")
  })

  it("does not open the organization's pages", () => {
    expect(routeAccess(at("/organization/members"), operatorMember)).toBe(
      "blocked",
    )
  })

  it("opens the deployment's budgets, not the organization's", () => {
    // Spend & budgets is the organization's own page, whoever opens it
    // (otari#2099); the deployment's budgets are a deployment page.
    expect(routeAccess(at("/admin/budgets"), operatorMember)).toBe("open")
    expect(routeAccess(at("/budgets"), operatorMember)).toBe("blocked")
  })
})

describe("while the caller is still being read", () => {
  const loading: Caller = {
    context: undefined,
    answer: "pending",
    selectedWorkspaceRole: undefined,
  }

  it("decides nothing about a gated page, so nobody is sent Home on the way", () => {
    for (const path of [
      "/organization/members",
      "/budgets",
      "/workspaces/ws-data",
      "/settings",
      "/playground",
      "/tools/guardrails",
    ]) {
      expect(routeAccess(at(path), loading), path).toBe("pending")
    }
  })

  it("opens an ungated page straight away", () => {
    expect(routeAccess(at("/sessions"), loading)).toBe("open")
  })

  it("waits for the selected workspace before the Playground", () => {
    expect(
      routeAccess(at("/playground"), {
        ...ROLES.member,
        selectedWorkspaceRole: undefined,
      }),
    ).toBe("pending")
  })
})

describe("when the caller could not be read", () => {
  it("opens the page, as the shell's Organization row fails open", () => {
    const failed: Caller = {
      context: undefined,
      answer: "unavailable",
      selectedWorkspaceRole: null,
    }
    expect(routeAccess(at("/organization/members"), failed)).toBe("open")
    expect(routeAccess(at("/settings"), failed)).toBe("open")
    expect(routeAccess(at("/playground"), failed)).toBe("open")
  })

  it("blocks the gated pages for someone in no organization", () => {
    const outside: Caller = {
      context: undefined,
      answer: "no-organization",
      selectedWorkspaceRole: null,
    }
    expect(routeAccess(at("/organization/members"), outside)).toBe("blocked")
    expect(routeAccess(at("/workspaces/ws-data"), outside)).toBe("blocked")
    expect(routeAccess(at("/settings"), outside)).toBe("blocked")
    // Home is where they are told how to get in.
    expect(routeAccess(at("/"), outside)).toBe("open")
  })

  it("keeps someone in no organization out of the rows their rail hides", () => {
    // The rail draws only the rows that opt in for this caller, so a page it
    // hides does not open by its address either.
    const outside: Caller = {
      context: undefined,
      answer: "no-organization",
      selectedWorkspaceRole: null,
    }
    for (const address of ["/people", "/rules", "/keys", "/sessions"]) {
      expect(routeAccess(at(address), outside), address).toBe("blocked")
    }
    // Pages no rail row governs, such as the caller's own account, still open.
    expect(routeAccess(at("/account"), outside)).toBe("open")
  })
})

describe("a route that declares its own access", () => {
  // No route in this build declares one; a build's own route below a gated
  // address is what the declaration is for.
  const root = createRootRoute()
  const settings = createRoute({
    getParentRoute: () => root,
    path: "/settings",
  })
  const moved = createRoute({
    getParentRoute: () => root,
    path: "/settings/keys",
    staticData: { access: "anyone" },
  })
  const declared = createRouter({
    routeTree: root.addChildren([settings, moved]),
    history: createMemoryHistory(),
  })

  it("is read before the entry its address falls under", () => {
    const target = targetOf(declared, "/settings/keys")
    expect(target).toMatchObject({ route: "/settings/keys", access: "anyone" })
    expect(requirementFor("/settings/keys")).toBe("operator")
    expect(routeAccess(target, ROLES.member)).toBe("open")
  })

  it("leaves the routes beside it to their entries", () => {
    expect(routeAccess(targetOf(declared, "/settings"), ROLES.member)).toBe(
      "blocked",
    )
  })

  const below = (access?: "anyone" | "manager") => ({
    route: "/settings/keys",
    params: {},
    ...(access && { access }),
  })

  it("opens to every role when it says anyone", () => {
    expect(
      ORDER.map((role) => routeAccess(below("anyone"), ROLES[role])),
    ).toEqual([o, o, o, o, o, o])
  })

  it("asks what it names in place of the entry above it", () => {
    expect(
      ORDER.map((role) => routeAccess(below("manager"), ROLES[role])),
    ).toEqual([o, o, x, x, x, x])
  })

  it("falls back to the entry above it when it declares nothing", () => {
    // `/settings` is the operator's.
    expect(ORDER.map((role) => routeAccess(below(), ROLES[role]))).toEqual([
      o,
      x,
      x,
      x,
      x,
      x,
    ])
  })

  it("still waits for the caller before a requirement it names", () => {
    const loading: Caller = {
      context: undefined,
      answer: "pending",
      selectedWorkspaceRole: undefined,
    }
    expect(routeAccess(below("manager"), loading)).toBe("pending")
    expect(routeAccess(below("anyone"), loading)).toBe("open")
  })
})
