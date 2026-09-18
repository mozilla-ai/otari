import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { OrganizationContext } from "@/client"
import { RoutingPage } from "@/features/routing/RoutingPage"
import { API_ROOT } from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap, organizationContext } from "@/tests/fixtures"
import { withRouter } from "@/tests/router"
import {
  ADMIN_WORKSPACE,
  adminContext,
  CHAIN,
  createTrigger,
  mockApi,
  policy,
  renderInWorkspace,
  renderPage,
} from "@/tests/routing"

afterEach(() => {
  vi.restoreAllMocks()
})

describe("RoutingPage for a caller who does not operate the deployment", () => {
  it("lists the tenant-scoped policies read-only for a non-operator", async () => {
    // The member half of otari-ai#1942: the page reads
    // /v1/organizations/me/routing-policies and offers nothing that writes.
    mockApi([], null, [], {
      context: organizationContext({
        deployment_operator: false,
        role: "member",
      }),
      memberPolicies: [policy("fast", CHAIN)],
    })
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("fast")).closest("tr")!
    expect(within(row).getByText(/openai:gpt-5-mini/)).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "New policy" }),
    ).not.toBeInTheDocument()
    expect(
      within(row).queryByRole("button", { name: "Edit" }),
    ).not.toBeInTheDocument()
    expect(
      within(row).queryByRole("button", { name: "Delete" }),
    ).not.toBeInTheDocument()
    // The Examples panel reads the operator-only /v1/routing/status, so its
    // opener goes with the rest of the actions column.
    expect(
      within(row).queryByRole("button", { name: "Examples" }),
    ).not.toBeInTheDocument()
  })

  it("keeps cached operator rows out of a member's table", async () => {
    // A disabled query still serves whatever sits under its key, so a caller
    // demoted mid-session would otherwise keep seeing the deployment-wide
    // policies and aliases they fetched as an operator. The pre-seeded client
    // stands in for that cache.
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    })
    client.setQueryData(["routing-policies"], [policy("operator-only", CHAIN)])
    client.setQueryData(
      ["aliases"],
      [
        {
          name: "operator-alias",
          target: "openai:gpt-5",
          source: "stored",
          user_id: null,
        },
      ],
    )
    mockApi([], null, [], {
      context: organizationContext({
        deployment_operator: false,
        role: "member",
      }),
      memberPolicies: [policy("mine", CHAIN)],
    })
    render(
      <DeploymentProvider value={bootstrap()}>
        <QueryClientProvider client={client}>
          <RoutingPage />
        </QueryClientProvider>
      </DeploymentProvider>,
      { wrapper: withRouter({ url: "/" }) },
    )

    expect(await screen.findByText("mine")).toBeInTheDocument()
    expect(screen.queryByText("operator-only")).not.toBeInTheDocument()
    expect(screen.queryByText("operator-alias")).not.toBeInTheDocument()
  })

  it("never fires the operator reads for a non-operator", async () => {
    const { calls } = mockApi([], null, [], {
      context: organizationContext({
        deployment_operator: false,
        role: "member",
      }),
      memberPolicies: [policy("fast", CHAIN)],
    })
    renderPage(<RoutingPage />)
    await screen.findByText("fast")

    const urls = calls.map((call) => call.url)
    expect(
      urls.some((url) => url.endsWith(`${API_ROOT}/routing/policies`)),
    ).toBe(false)
    expect(urls.some((url) => url.includes(`${API_ROOT}/aliases`))).toBe(false)
    expect(urls.some((url) => url.includes(`${API_ROOT}/tool-settings`))).toBe(
      false,
    )
    expect(urls.some((url) => url.includes(`${API_ROOT}/users`))).toBe(false)
  })

  it("withholds the deep-linked add form from a member", async () => {
    // ?target= seeds `adding` before the membership context settles, so the
    // role has to be applied at render time. Without that a member on this URL
    // gets a create form whose only outcome is a refusal, and the read-only
    // empty state is suppressed behind it.
    mockApi([], null, [], {
      context: organizationContext({
        deployment_operator: false,
        role: "member",
      }),
      memberPolicies: [],
    })
    renderPage(<RoutingPage />, "/routing?target=openai:gpt-4o")

    expect(
      await screen.findByText(/once your organization's admins define them/),
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: /Create policy/i }),
    ).not.toBeInTheDocument()
  })

  it("keeps the deep-linked add form for an operator", async () => {
    mockApi([], null, [], { context: organizationContext() })
    renderPage(<RoutingPage />, "/routing?target=openai:gpt-4o")

    expect(
      await screen.findByRole("button", { name: /Create policy/i }),
    ).toBeInTheDocument()
  })

  it("tells a member with no policies who defines them", async () => {
    mockApi([], null, [], {
      context: organizationContext({
        deployment_operator: false,
        role: "member",
      }),
      memberPolicies: [],
    })
    renderPage(<RoutingPage />)

    expect(
      await screen.findByText(/once your organization's admins define them/),
    ).toBeInTheDocument()
    // The operator's numbered getting-started walkthrough is not for them.
    expect(screen.queryByText(/Create a policy/)).not.toBeInTheDocument()
  })
})

// otari-ai#2087: the page showed an operator every tenant's stored rows and an
// admin every workspace of their organization's, while resolution is scoped to
// one workspace. Both reads and both writes name the selected workspace now,
// and the surface they land on is still the caller's role.
describe("RoutingPage scoped to the selected workspace", () => {
  const OPERATOR_WORKSPACE = ADMIN_WORKSPACE

  function operatorInWorkspace(): OrganizationContext {
    return organizationContext({
      workspace_memberships: [
        { workspace_id: OPERATOR_WORKSPACE, name: "Alpha one", role: "owner" },
      ],
    })
  }

  it("names the selected workspace on the operator's list read", async () => {
    const { calls } = mockApi([policy("fast", CHAIN)], null, [], {
      context: operatorInWorkspace(),
    })
    renderInWorkspace(<RoutingPage />)

    await screen.findByText("fast")
    const listed = calls.filter(
      (call) =>
        call.method === "GET" &&
        call.url.includes(`${API_ROOT}/routing/policies`),
    )
    expect(listed.length).toBeGreaterThan(0)
    for (const call of listed) {
      expect(call.url).toContain(`workspace_id=${OPERATOR_WORKSPACE}`)
    }
  })

  it("names the selected workspace on an admin's list read", async () => {
    const { calls } = mockApi([], null, [], {
      context: adminContext(),
      memberPolicies: [policy("tenant-fast", CHAIN)],
    })
    renderInWorkspace(<RoutingPage />)

    await screen.findByText("tenant-fast")
    const listed = calls.filter(
      (call) =>
        call.method === "GET" &&
        call.url.includes(`${API_ROOT}/organizations/me/routing-policies`),
    )
    expect(listed.length).toBeGreaterThan(0)
    for (const call of listed) {
      expect(call.url).toContain(`workspace_id=${ADMIN_WORKSPACE}`)
    }
  })

  it("lands an operator's create in the workspace they are looking at", async () => {
    // Without this the write omitted the workspace, so the row went to the
    // deployment's default one and the page it was created from never showed it.
    const { calls } = mockApi([], null, [], { context: operatorInWorkspace() })
    const user = userEvent.setup()
    renderInWorkspace(<RoutingPage />)

    await user.click(await createTrigger())
    await user.type(
      screen.getByRole("textbox", { name: /policy name/i }),
      "scoped",
    )
    await user.type(
      screen.getByRole("combobox", { name: /^serves$/i }),
      "openai:gpt-5-mini",
    )
    await user.keyboard("{Escape}")
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", {
        name: "Create policy",
      }),
    )

    const written = calls.find(
      (call) =>
        call.method === "POST" &&
        call.url.endsWith(`${API_ROOT}/routing/policies`),
    )
    expect(written?.body).toMatchObject({
      name: "scoped",
      workspace_id: OPERATOR_WORKSPACE,
    })
  })

  it("names the row's workspace on an operator's delete", async () => {
    const OTHER_WORKSPACE = "66666666-6666-6666-6666-666666666666"
    const { calls } = mockApi(
      [policy("doomed", CHAIN, { workspace_id: OTHER_WORKSPACE })],
      null,
      [],
      { context: operatorInWorkspace() },
    )
    const user = userEvent.setup()
    renderInWorkspace(<RoutingPage />)

    await screen.findByText("doomed")
    await user.click(screen.getByRole("button", { name: "Delete" }))
    await user.click(
      within(await screen.findByRole("alertdialog")).getByRole("button", {
        name: "Delete policy",
      }),
    )

    const deleted = calls.find((call) => call.method === "DELETE")
    expect(deleted?.url).toContain(`${API_ROOT}/routing/policies/doomed`)
    expect(deleted?.url).toContain(`workspace_id=${OTHER_WORKSPACE}`)
  })
})

// otari-ai#1969: the Build pages are Edit for admins. An organization admin
// writes the tenant-scoped routers, which name the workspace and take no user
// scope; an operator keeps the deployment-wide ones, unchanged above.
describe("RoutingPage for an organization admin", () => {
  it("writes a new policy through the tenant-scoped router", async () => {
    const { calls } = mockApi([], null, [], {
      context: adminContext(),
      memberPolicies: [],
    })
    const user = userEvent.setup()
    renderInWorkspace(<RoutingPage />)

    await user.click(await createTrigger())
    await user.type(
      screen.getByRole("textbox", { name: /policy name/i }),
      "tenant-fast",
    )
    await user.type(
      screen.getByRole("combobox", { name: /^serves$/i }),
      "openai:gpt-5-mini",
    )
    // Close the combobox popover, which otherwise aria-hides the submit button.
    await user.keyboard("{Escape}")
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", {
        name: "Create policy",
      }),
    )

    const written = calls.find(
      (call) =>
        call.method === "POST" &&
        call.url.includes(`${API_ROOT}/organizations/me/routing-policies`),
    )
    expect(written).toBeDefined()
    expect(written?.body).toMatchObject({
      name: "tenant-fast",
      workspace_id: ADMIN_WORKSPACE,
    })
    // The deployment-wide router is never reached, whatever the form did.
    expect(
      calls.some(
        (call) =>
          call.method === "POST" &&
          call.url.endsWith(`${API_ROOT}/routing/policies`),
      ),
    ).toBe(false)
  })

  it("offers no user scope, which the tenant router refuses", async () => {
    mockApi([], null, [], { context: adminContext(), memberPolicies: [] })
    const user = userEvent.setup()
    renderInWorkspace(<RoutingPage />)

    await user.click(await createTrigger())
    expect(
      await screen.findByText(/applies to everyone in the selected workspace/i),
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Specific users" }),
    ).not.toBeInTheDocument()
  })

  it("deletes through the tenant-scoped router, naming the workspace", async () => {
    const { calls } = mockApi([], null, [], {
      context: adminContext(),
      memberPolicies: [policy("doomed", CHAIN)],
    })
    const user = userEvent.setup()
    renderInWorkspace(<RoutingPage />)

    await screen.findByText("doomed")
    await user.click(screen.getByRole("button", { name: "Delete" }))
    await user.click(
      within(await screen.findByRole("alertdialog")).getByRole("button", {
        name: "Delete policy",
      }),
    )

    const deleted = calls.find((call) => call.method === "DELETE")
    expect(deleted?.url).toContain(
      `${API_ROOT}/organizations/me/routing-policies/`,
    )
    expect(deleted?.url).toContain(`workspace_id=${ADMIN_WORKSPACE}`)
  })

  it("writes an edit back to the row's own workspace, not the selected one", async () => {
    // An admin's list spans the organization, so the row being edited need not
    // live in the workspace the switcher points at. Writing to the selection
    // would create a second policy of that name and leave this one untouched.
    const OTHER_WORKSPACE = "55555555-5555-5555-5555-555555555555"
    const { calls } = mockApi([], null, [], {
      context: adminContext(),
      memberPolicies: [
        policy("elsewhere", CHAIN, { workspace_id: OTHER_WORKSPACE }),
      ],
    })
    const user = userEvent.setup()
    renderInWorkspace(<RoutingPage />)

    await screen.findByText("elsewhere")
    await user.click(screen.getByRole("button", { name: "Edit" }))
    await user.click(await screen.findByRole("button", { name: "Save" }))

    const written = calls.find(
      (call) =>
        call.method === "POST" &&
        call.url.includes(`${API_ROOT}/organizations/me/routing-policies`),
    )
    expect(written?.body).toMatchObject({ workspace_id: OTHER_WORKSPACE })
  })

  it("lists the tenant-scoped aliases beside the policies", async () => {
    mockApi([], null, [], {
      context: adminContext(),
      memberPolicies: [policy("mine", CHAIN)],
      memberAliases: [
        {
          name: "tenant-alias",
          target: "openai:gpt-5",
          source: "stored",
          user_id: null,
        },
      ],
    })
    renderInWorkspace(<RoutingPage />)

    expect(await screen.findByText("mine")).toBeInTheDocument()
    expect(screen.getByText("tenant-alias")).toBeInTheDocument()
  })

  it("writes an alias edit back to the alias's own workspace", async () => {
    // The sibling of the policy case above, and it was the one that regressed:
    // `aliasAsRow` dropped `workspace_id`, so every alias row fell back to the
    // selected workspace and an edit landed in the wrong one silently.
    const OTHER_WORKSPACE = "66666666-6666-6666-6666-666666666666"
    const { calls } = mockApi([], null, [], {
      context: adminContext(),
      memberPolicies: [],
      memberAliases: [
        {
          name: "elsewhere-alias",
          target: "openai:gpt-5",
          source: "stored",
          user_id: null,
          workspace_id: OTHER_WORKSPACE,
        },
      ],
    })
    const user = userEvent.setup()
    renderInWorkspace(<RoutingPage />)

    await screen.findByText("elsewhere-alias")
    await user.click(screen.getByRole("button", { name: "Edit" }))
    await user.click(await screen.findByRole("button", { name: "Save" }))

    const written = calls.find(
      (call) =>
        call.method === "POST" &&
        call.url.includes(`${API_ROOT}/organizations/me/aliases`),
    )
    expect(written?.body).toMatchObject({ workspace_id: OTHER_WORKSPACE })
  })

  it("deletes an alias from the alias's own workspace", async () => {
    const OTHER_WORKSPACE = "77777777-7777-7777-7777-777777777777"
    const { calls } = mockApi([], null, [], {
      context: adminContext(),
      memberPolicies: [],
      memberAliases: [
        {
          name: "doomed-alias",
          target: "openai:gpt-5",
          source: "stored",
          user_id: null,
          workspace_id: OTHER_WORKSPACE,
        },
      ],
    })
    const user = userEvent.setup()
    renderInWorkspace(<RoutingPage />)

    await screen.findByText("doomed-alias")
    await user.click(screen.getByRole("button", { name: "Delete" }))
    await user.click(
      within(await screen.findByRole("alertdialog")).getByRole("button", {
        name: "Delete alias",
      }),
    )

    const deleted = calls.find((call) => call.method === "DELETE")
    expect(deleted?.url).toContain(`${API_ROOT}/organizations/me/aliases/`)
    expect(deleted?.url).toContain(`workspace_id=${OTHER_WORKSPACE}`)
  })

  it("withholds the write affordances from an admin in no workspace", async () => {
    // The switcher is seeded from the caller's own memberships, so an admin who
    // joined none has no workspace to scope a write to and gets the read-only
    // page rather than a form whose only outcome is a 422.
    mockApi([], null, [], {
      context: organizationContext({
        deployment_operator: false,
        role: "admin",
      }),
      memberPolicies: [policy("mine", CHAIN)],
    })
    renderInWorkspace(<RoutingPage />)

    await screen.findByText("mine")
    expect(
      screen.queryByRole("button", { name: "New policy" }),
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Delete" }),
    ).not.toBeInTheDocument()
  })

  it("still withholds the write affordances from a member", async () => {
    mockApi([], null, [], {
      context: organizationContext({
        deployment_operator: false,
        role: "member",
        workspace_memberships: [
          { workspace_id: ADMIN_WORKSPACE, name: "Alpha one", role: "member" },
        ],
      }),
      memberPolicies: [policy("mine", CHAIN)],
    })
    renderInWorkspace(<RoutingPage />)

    await screen.findByText("mine")
    expect(
      screen.queryByRole("button", { name: "New policy" }),
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Delete" }),
    ).not.toBeInTheDocument()
  })
})
