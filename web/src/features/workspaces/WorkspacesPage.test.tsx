import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactElement } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import type {
  Budget,
  OrganizationContext,
  OrganizationMember,
  Workspace,
  WorkspaceBudgetDefault,
  WorkspaceMember,
} from "@/client"
import { WorkspacesPage } from "@/features/workspaces/WorkspacesPage"
import {
  budget,
  organizationContext,
  organizationMember,
  workspace,
  workspaceBudgetDefault,
  workspaceMember,
} from "@/tests/fixtures"

interface Request {
  url: string
  method: string
  body: unknown
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

function mockApi(
  opts: {
    context?: OrganizationContext
    workspaces?: Workspace[]
    members?: WorkspaceMember[]
    orgMembers?: OrganizationMember[]
    budgets?: Budget[]
    // Keyed by workspace id, so a test can give one workspace a default and
    // leave another without one.
    budgetDefaults?: Record<string, WorkspaceBudgetDefault[]>
    /** A refusal for `POST /v1/workspaces`, for the error paths. */
    createRefusal?: { status: number; detail: string }
  } = {},
) {
  const context = opts.context ?? organizationContext()
  const list = opts.workspaces ?? [workspace()]
  const members = opts.members ?? [workspaceMember()]
  const orgMembers = opts.orgMembers ?? [organizationMember()]
  const budgets = opts.budgets ?? []
  const budgetDefaults = opts.budgetDefaults ?? {}
  const requests: Request[] = []

  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
    requests.push({
      url,
      method,
      body: init?.body ? JSON.parse(String(init.body)) : undefined,
    })

    if (url.includes("/members") && url.includes("/v1/workspaces/")) {
      if (method === "GET") {
        return jsonResponse({ data: members, count: members.length })
      }
      return jsonResponse(members[0] ?? workspaceMember())
    }
    if (url.includes("member-budget-policies")) {
      const id = url.split("/v1/workspaces/")[1]?.split("/")[0] ?? ""
      const rows = budgetDefaults[id] ?? []
      return jsonResponse({ data: rows, count: rows.length })
    }
    if (url.includes("/v1/budgets")) {
      return jsonResponse(budgets)
    }
    if (url.includes("/v1/workspaces")) {
      if (method === "GET") {
        return jsonResponse({ data: list, count: list.length })
      }
      if (method === "DELETE") return jsonResponse({ message: "deleted" })
      if (method === "POST" && opts.createRefusal) {
        return jsonResponse(
          { detail: opts.createRefusal.detail },
          opts.createRefusal.status,
        )
      }
      return jsonResponse(workspace({ name: "Created" }))
    }
    if (url.includes("/v1/organizations/me/members")) {
      return jsonResponse({ data: orgMembers, count: orgMembers.length })
    }
    return jsonResponse(context)
  })

  return requests
}

function renderPage(ui: ReactElement) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>)
}

afterEach(() => {
  vi.restoreAllMocks()
})

const SECOND = "66666666-6666-6666-6666-666666666666"

describe("WorkspacesPage", () => {
  it("lists the organization's workspaces", async () => {
    mockApi({
      workspaces: [
        workspace({ name: "Production", description: "Live traffic" }),
      ],
    })
    renderPage(<WorkspacesPage />)

    expect(await screen.findByText("Production")).toBeInTheDocument()
    expect(screen.getByText("Live traffic")).toBeInTheDocument()
  })

  it("creates a workspace with a name and a description", async () => {
    const requests = mockApi({})
    const user = userEvent.setup()
    renderPage(<WorkspacesPage />)

    await user.click(
      await screen.findByRole("button", { name: "Create workspace" }),
    )
    await user.type(screen.getByLabelText("Name"), "Research")
    await user.type(
      screen.getByLabelText("Description (optional)"),
      "Experiments",
    )
    // The header action hides itself while the form is open, so the only
    // button left with this name is the form's own submit.
    await user.click(screen.getByRole("button", { name: "Create workspace" }))

    const post = requests.find((request) => request.method === "POST")
    expect(post?.url).toContain("/v1/workspaces")
    expect(post?.body).toEqual({
      name: "Research",
      description: "Experiments",
    })
  })

  it("puts a refused create on the name that caused it, not in a banner", async () => {
    // A banner above the form resizes whatever frames it, and every way this
    // endpoint refuses is about the name: taken, empty, too long.
    mockApi({
      createRefusal: {
        status: 409,
        detail: "A workspace named 'test' already exists in this organization",
      },
    })
    const user = userEvent.setup()
    renderPage(<WorkspacesPage />)

    await user.click(
      await screen.findByRole("button", { name: "Create workspace" }),
    )
    await user.type(screen.getByLabelText("Name"), "test")
    await user.click(screen.getByRole("button", { name: "Create workspace" }))

    const input = await screen.findByLabelText("Name")
    expect(input).toHaveAttribute("aria-invalid", "true")
    expect(await screen.findByText(/already exists/)).toBeInTheDocument()
    // The description it replaced, rather than a row added below it: the two
    // together would grow the form by a line.
    expect(screen.queryByText(/You become its owner/)).toBeNull()

    // Editing the name retires the refusal it produced, so the field is not
    // still red while the correction is being typed.
    await user.type(input, "2")
    expect(input).not.toHaveAttribute("aria-invalid", "true")
    expect(screen.getByText(/You become its owner/)).toBeInTheDocument()
  })

  it("keeps a failure the name cannot fix off the name", async () => {
    // A 500 or a dropped connection is not something the operator can retype
    // their way out of, so reddening the field would tell them (and assistive
    // tech) that the name was the problem.
    mockApi({
      createRefusal: { status: 500, detail: "Internal server error" },
    })
    const user = userEvent.setup()
    renderPage(<WorkspacesPage />)

    await user.click(
      await screen.findByRole("button", { name: "Create workspace" }),
    )
    await user.type(screen.getByLabelText("Name"), "test")
    await user.click(screen.getByRole("button", { name: "Create workspace" }))

    expect(await screen.findByRole("alert")).toHaveTextContent(/server error/i)
    expect(screen.getByLabelText("Name")).not.toHaveAttribute(
      "aria-invalid",
      "true",
    )
    // The field keeps its own description rather than borrowing the failure.
    expect(screen.getByText(/You become its owner/)).toBeInTheDocument()
  })

  it("confirms before deleting a workspace", async () => {
    // Two, because the last workspace cannot be deleted and its button says so.
    const requests = mockApi({
      workspaces: [workspace(), workspace({ id: SECOND, name: "Bravo" })],
    })
    const user = userEvent.setup()
    renderPage(<WorkspacesPage />)

    const deletes = await screen.findAllByRole("button", { name: "Delete" })
    await user.click(deletes[0])
    await user.click(screen.getByRole("button", { name: "Delete workspace" }))

    const remove = requests.find((request) => request.method === "DELETE")
    expect(remove?.url).toContain(
      "/v1/workspaces/44444444-4444-4444-4444-444444444444",
    )
  })

  it("refuses to offer the last workspace's deletion, and says why", async () => {
    // The server keeps every organization on at least one workspace, and first
    // boot is that state, so an enabled button here is one that always 400s.
    mockApi({})
    renderPage(<WorkspacesPage />)

    const remove = await screen.findByRole("button", {
      name: /^Delete Default Workspace \(/,
    })
    expect(remove).toBeDisabled()
    expect(remove).toHaveAccessibleName(/keeps at least one workspace/)
  })

  it("denies a non-manager every write control on this page", async () => {
    // Two workspaces, so the Delete button is about the caller's role rather
    // than about the last-workspace rule. The roster's own controls are the
    // members page's business now.
    mockApi({
      context: organizationContext({ role: "viewer" }),
      workspaces: [workspace(), workspace({ id: SECOND, name: "Bravo" })],
    })
    renderPage(<WorkspacesPage />)

    await screen.findByText("Default Workspace")
    expect(
      screen.queryByRole("button", { name: "Create workspace" }),
    ).toBeNull()
    expect(screen.getAllByRole("button", { name: "Edit" })[0]).toBeDisabled()
    expect(screen.getAllByRole("button", { name: "Delete" })[0]).toBeDisabled()
  })

  it("renames a workspace through the update endpoint", async () => {
    const requests = mockApi({})
    const user = userEvent.setup()
    renderPage(<WorkspacesPage />)

    await user.click(await screen.findByRole("button", { name: "Edit" }))
    const name = screen.getByLabelText("Name")
    await user.clear(name)
    await user.type(name, "Renamed")
    await user.click(screen.getByRole("button", { name: "Save changes" }))

    const patch = requests.find((request) => request.method === "PATCH")
    expect(patch?.url).toContain(
      "/v1/workspaces/44444444-4444-4444-4444-444444444444",
    )
    expect(patch?.body).toEqual({ name: "Renamed", description: null })
  })

  it("names the budget each workspace hands to its members", async () => {
    // The other half of the field on the edit form: a workspace's default is
    // readable from the list without opening it, and mirrors the budgets page's
    // own "Default for" column.
    mockApi({
      workspaces: [workspace(), workspace({ id: SECOND, name: "Bravo" })],
      budgets: [budget({ budget_id: "bud-team", name: "Team standard" })],
      budgetDefaults: {
        "44444444-4444-4444-4444-444444444444": [
          workspaceBudgetDefault({ budget_id: "bud-team" }),
        ],
      },
    })
    renderPage(<WorkspacesPage />)

    // Awaited, not read synchronously: the defaults are a fan-out over the
    // workspaces, so they land after the list the row itself comes from.
    const chip = await screen.findByText("Team standard")
    expect(chip.closest("tr")).toContainElement(
      await screen.findByText("Default Workspace"),
    )

    // Bravo has none, which reads as "None" rather than as an empty cell.
    const bravo = (await screen.findByText("Bravo")).closest("tr")!
    expect(within(bravo).getByText("None")).toBeInTheDocument()
  })

  it("withholds the budget column and its reads from a non-operator admin", async () => {
    // An organization admin who does not operate the deployment may manage
    // workspaces, but the budgets list answers them 403 (#821). Neither that
    // read nor the defaults fan-out that only feeds this column is made, and
    // the column goes with them rather than echoing UUID fragments.
    const requests = mockApi({
      context: organizationContext({ deployment_operator: false }),
      workspaces: [workspace(), workspace({ id: SECOND, name: "Bravo" })],
      budgetDefaults: {
        "44444444-4444-4444-4444-444444444444": [
          workspaceBudgetDefault({ budget_id: "bud-team" }),
        ],
      },
    })
    renderPage(<WorkspacesPage />)

    await screen.findByText("Bravo")
    expect(screen.queryByText("Default member budget")).toBeNull()
    expect(
      requests.some((request) => request.url.includes("/v1/budgets")),
    ).toBe(false)
    expect(
      requests.some((request) =>
        request.url.includes("member-budget-policies"),
      ),
    ).toBe(false)
  })

  it("offers a non-operator no default-budget picker on the create form", async () => {
    const requests = mockApi({
      context: organizationContext({ deployment_operator: false }),
    })
    const user = userEvent.setup()
    renderPage(<WorkspacesPage />)

    await user.click(
      await screen.findByRole("button", { name: "Create workspace" }),
    )
    await user.type(screen.getByLabelText("Name"), "Research")
    expect(
      screen.queryByRole("button", { name: /Default member budget/ }),
    ).toBeNull()
    await user.click(screen.getByRole("button", { name: "Create workspace" }))

    // The create itself is theirs to make; only the picker is withheld.
    const post = requests.find((request) => request.method === "POST")
    expect(post?.body).toEqual({ name: "Research", description: null })
    expect(
      requests.some((request) => request.url.includes("/v1/budgets")),
    ).toBe(false)
  })

  it("withholds the default-budget controls from a non-operator's edit form", async () => {
    const requests = mockApi({
      context: organizationContext({ deployment_operator: false }),
    })
    const user = userEvent.setup()
    renderPage(<WorkspacesPage />)

    await user.click(await screen.findByRole("button", { name: "Edit" }))
    const name = await screen.findByLabelText("Name")
    expect(
      screen.queryByRole("button", { name: /Default member budget/ }),
    ).toBeNull()
    expect(screen.queryByText("Per-provider defaults")).toBeNull()

    // The rename is theirs to make, and saving does not touch the withheld
    // default: no delete for a "none" the caller never picked.
    await user.clear(name)
    await user.type(name, "Renamed")
    await user.click(screen.getByRole("button", { name: "Save changes" }))

    const patch = requests.find((request) => request.method === "PATCH")
    expect(patch?.body).toEqual({ name: "Renamed", description: null })
    const operatorOnly = [
      "/v1/budgets",
      "/v1/providers",
      "member-budget-policies",
    ]
    expect(
      requests.filter((request) =>
        operatorOnly.some((path) => request.url.includes(path)),
      ),
    ).toEqual([])
  })
})
