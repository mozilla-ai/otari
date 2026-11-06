import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type {
  Budget,
  OrganizationContext,
  WorkspaceBudgetDefault,
} from "@/client"
import { WorkspaceDetailPage } from "@/features/workspaces/WorkspaceDetailPage"
import { API_ROOT } from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  bootstrap,
  budget,
  organizationContext,
  organizationMember,
  workspace,
  workspaceBudgetDefault,
  workspaceHostedProvider,
  workspaceMember,
} from "@/tests/fixtures"
import { withRouter } from "@/tests/router"

const WORKSPACE = "44444444-4444-4444-4444-444444444444"

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
    budgets?: Budget[]
    budgetDefaults?: WorkspaceBudgetDefault[]
  } = {},
) {
  const context = opts.context ?? organizationContext()
  const budgetDefaults = opts.budgetDefaults ?? []
  const requests: Request[] = []

  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
    requests.push({
      url,
      method,
      body: init?.body ? JSON.parse(String(init.body)) : undefined,
    })

    if (url.includes("/hosted-providers")) {
      return jsonResponse({ data: [workspaceHostedProvider()] })
    }
    if (url.includes("provider-keys")) {
      return jsonResponse({ data: [], count: 0 })
    }
    if (url.includes("member-budget-policies")) {
      if (method !== "GET") return jsonResponse({ message: "ok" })
      return jsonResponse({
        data: budgetDefaults,
        count: budgetDefaults.length,
      })
    }
    if (url.includes(`${API_ROOT}/workspaces/${WORKSPACE}/members`)) {
      const members = [workspaceMember()]
      return jsonResponse({ data: members, count: members.length })
    }
    if (url.includes(`${API_ROOT}/workspaces/${WORKSPACE}/`)) {
      // The tools cards' own reads, which this page only hosts.
      return jsonResponse({ data: [], count: 0 })
    }
    if (url.includes(`${API_ROOT}/budgets`)) {
      return jsonResponse(opts.budgets ?? [])
    }
    if (url.includes(`${API_ROOT}/workspaces`)) {
      if (method === "GET") {
        return jsonResponse({ data: [workspace()], count: 1 })
      }
      return jsonResponse(workspace({ name: "Renamed" }))
    }
    if (url.includes(`${API_ROOT}/organizations/me/members`)) {
      const members = [organizationMember()]
      return jsonResponse({ data: members, count: members.length })
    }
    if (url.includes(`${API_ROOT}/models`)) {
      return jsonResponse({ data: [] })
    }
    if (url.includes(`${API_ROOT}/search-tools/providers`)) {
      return jsonResponse([])
    }
    return jsonResponse(context)
  })

  return requests
}

function renderPage(url = `/workspaces/${WORKSPACE}`, workspaceId = WORKSPACE) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <DeploymentProvider value={bootstrap()}>
        <WorkspaceDetailPage workspaceId={workspaceId} />
      </DeploymentProvider>
    </QueryClientProvider>,
    { wrapper: withRouter({ url }) },
  )
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe("WorkspaceDetailPage", () => {
  it("names the workspace and opens on its settings", async () => {
    mockApi()
    renderPage()

    expect(
      await screen.findByRole("heading", { name: "Default Workspace" }),
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "General" })).toHaveAttribute(
      "aria-pressed",
      "true",
    )
    expect(screen.getByLabelText("Name")).toHaveValue("Default Workspace")
    expect(screen.getByRole("link", { name: "← Workspaces" })).toHaveAttribute(
      "href",
      "/workspaces",
    )
  })

  it("says a workspace it cannot find is not found", async () => {
    mockApi()
    renderPage(
      "/workspaces/99999999-9999-9999-9999-999999999999",
      "99999999-9999-9999-9999-999999999999",
    )

    expect(await screen.findByText("Workspace not found")).toBeInTheDocument()
  })

  it("renames a workspace through the update endpoint", async () => {
    const requests = mockApi()
    const user = userEvent.setup()
    renderPage()

    const name = await screen.findByLabelText("Name")
    await user.clear(name)
    await user.type(name, "Renamed")
    await user.click(screen.getByRole("button", { name: "Save" }))

    const patch = requests.find((request) => request.method === "PATCH")
    expect(patch?.url).toContain(`${API_ROOT}/workspaces/${WORKSPACE}`)
    expect(patch?.body).toEqual({ name: "Renamed", description: null })
  })

  it("holds Save until something changed", async () => {
    mockApi()
    renderPage()

    await screen.findByLabelText("Name")
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled()
  })

  it("confirms before removing a per-provider budget default", async () => {
    // otari-ai#2110: this Remove used to delete on the click, with no
    // confirmation of any kind.
    const requests = mockApi({
      budgets: [budget({ budget_id: "bud-team", name: "Team standard" })],
      budgetDefaults: [
        workspaceBudgetDefault({
          id: "default-openai",
          budget_id: "bud-team",
          provider_key_id: "openai",
        }),
      ],
    })
    const user = userEvent.setup()
    renderPage()

    await screen.findByText("Per-provider defaults")
    await user.click(
      await screen.findByRole("button", { name: "Remove default for openai" }),
    )

    const dialog = await screen.findByRole("alertdialog")
    expect(within(dialog).getByText(/openai fall back to/)).toBeVisible()
    expect(requests.some((request) => request.method === "DELETE")).toBe(false)

    await user.click(
      within(dialog).getByRole("button", { name: "Remove default" }),
    )

    const removed = requests.find((request) => request.method === "DELETE")
    expect(removed?.url).toContain("default-openai")
  })

  it("withholds the default-budget controls from a non-operator", async () => {
    const requests = mockApi({
      context: organizationContext({ deployment_operator: false }),
    })
    const user = userEvent.setup()
    renderPage()

    const name = await screen.findByLabelText("Name")
    expect(
      screen.queryByRole("button", { name: /Default member budget/ }),
    ).toBeNull()
    expect(screen.queryByText("Per-provider defaults")).toBeNull()

    // The rename is theirs to make, and saving does not touch the withheld
    // default: no delete for a "none" the caller never picked.
    await user.clear(name)
    await user.type(name, "Renamed")
    await user.click(screen.getByRole("button", { name: "Save" }))

    const patch = requests.find((request) => request.method === "PATCH")
    expect(patch?.body).toEqual({ name: "Renamed", description: null })
    const operatorOnly = [
      `${API_ROOT}/budgets`,
      `${API_ROOT}/providers`,
      "member-budget-policies",
    ]
    expect(
      requests.filter((request) =>
        operatorOnly.some((path) => request.url.includes(path)),
      ),
    ).toEqual([])
  })

  it("keeps the chosen tab in the URL", async () => {
    mockApi()
    const user = userEvent.setup()
    renderPage()

    await user.click(await screen.findByRole("button", { name: "Providers" }))

    expect(screen.getByRole("button", { name: "Providers" })).toHaveAttribute(
      "aria-pressed",
      "true",
    )
    expect(screen.queryByLabelText("Name")).toBeNull()
  })

  it("offers only its settings and its providers", async () => {
    mockApi()
    renderPage(`/workspaces/${WORKSPACE}?tab=providers`)

    expect(await screen.findByText("gemini")).toBeInTheDocument()
    for (const name of ["Members", "Web search", "Code execution"]) {
      expect(screen.queryByRole("button", { name })).toBeNull()
    }
  })
})
