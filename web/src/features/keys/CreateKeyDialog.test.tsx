import { act, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { type RefObject, useEffect, useState } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"
import type { Budget, WorkspaceBudgetDefault } from "@/client"
import { CreateKeyDialog } from "@/features/keys/CreateKeyDialog"
import { API_ROOT } from "@/shared/api/client"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  apiKey,
  bootstrap,
  budget,
  organizationContext,
  organizationMember,
  workspace,
  workspaceBudgetDefault,
} from "@/tests/fixtures"
import { AppProviders as Provider } from "@/tests/providers"
import { renderWithRouter } from "@/tests/router"
import { pickOption, selectTrigger } from "@/tests/select"

const SECRET = "gw-NEWSECRET0000000000000000000000000000000000000000000000"

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

type Role = "owner" | "admin" | "member"

const WORKSPACE_ID = "ws-data"

/**
 * The gateway as a caller of `role` sees it. `isOperator` is the deployment's
 * operator, who reads the deployment's budgets; anyone else reads the
 * organization's, and only if they own or administer it. One workspace, whose
 * per-person defaults are `defaults`.
 */
function mockApi({
  role,
  isOperator,
  budgets = [],
  defaults = [],
}: {
  role: Role
  isOperator: boolean
  budgets?: Budget[]
  defaults?: WorkspaceBudgetDefault[]
}) {
  const writes: { url: string; body: Record<string, unknown> }[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
    if (url === `${API_ROOT}/organizations/me`) {
      return json(
        organizationContext({ role, deployment_operator: isOperator }),
      )
    }
    if (method === "POST") {
      const body = JSON.parse(String(init?.body ?? "{}")) as Record<
        string,
        unknown
      >
      writes.push({ url, body })
      if (/\/keys$/.test(url)) {
        return json({
          ...apiKey({
            id: "key-new",
            key_name: String(body.key_name),
            key_prefix: SECRET.slice(0, 10),
            key_suffix: SECRET.slice(-4),
          }),
          key: SECRET,
        })
      }
      return json({ id: "ceiling-1", ...body }, 201)
    }
    if (url.includes("/organizations/me/budgets")) {
      return json({ data: isOperator ? [] : budgets, count: budgets.length })
    }
    if (url.includes(`${API_ROOT}/budgets`))
      return json(isOperator ? budgets : [])
    if (url.includes(`/workspaces/${WORKSPACE_ID}/member-budget-policies`)) {
      return json({ data: defaults, count: defaults.length })
    }
    if (url.startsWith(`${API_ROOT}/workspaces?`)) {
      return json({
        data: [workspace({ id: WORKSPACE_ID, name: "Data" })],
        count: 1,
      })
    }
    // The caller's own row, which is what gives them an owner id to bill through.
    if (url.includes("/members")) {
      return json({
        data: [organizationMember({ attribution_user_id: "owner-1" })],
        count: 1,
      })
    }
    return json([])
  })
  return writes
}

async function openDialog(returnFocusRef?: RefObject<HTMLElement | null>) {
  const user = userEvent.setup()
  await renderWithRouter(<div />, {
    shell: (
      <Provider>
        <DeploymentProvider value={bootstrap()}>
          <SelectedWorkspaceProvider>
            <CreateKeyDialog
              isOpen
              onOpenChange={() => undefined}
              returnFocusRef={returnFocusRef}
            />
          </SelectedWorkspaceProvider>
        </DeploymentProvider>
      </Provider>
    ),
  })
  const dialog = await screen.findByRole("dialog")
  return { user, dialog }
}

/**
 * The dialog as the shell holds it: open and closed by a plain setter, the way
 * the shell closes it when Back leaves the page it was opened on.
 */
async function openHeldDialog() {
  const shell: { setOpen: (open: boolean) => void } = {
    setOpen: () => undefined,
  }
  function Held() {
    const [isOpen, setOpen] = useState(true)
    useEffect(() => {
      shell.setOpen = setOpen
    }, [])
    return <CreateKeyDialog isOpen={isOpen} onOpenChange={setOpen} />
  }
  const user = userEvent.setup()
  await renderWithRouter(<div />, {
    shell: (
      <Provider>
        <DeploymentProvider value={bootstrap()}>
          <SelectedWorkspaceProvider>
            <Held />
          </SelectedWorkspaceProvider>
        </DeploymentProvider>
      </Provider>
    ),
  })
  const dialog = await screen.findByRole("dialog")
  const reopen = async () => {
    act(() => shell.setOpen(false))
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull())
    act(() => shell.setOpen(true))
    return screen.findByRole("dialog")
  }
  return { user, dialog, reopen }
}

async function createKey(
  user: ReturnType<typeof userEvent.setup>,
  dialog: HTMLElement,
) {
  const submit = within(dialog).getByRole("button", { name: "Create API key" })
  await waitFor(() => expect(submit).toBeEnabled())
  await user.click(submit)
}

describe("CreateKeyDialog", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("hands over the secret alone, with no example requests", async () => {
    mockApi({ role: "owner", isOperator: false })
    const { user, dialog } = await openDialog()
    await user.type(within(dialog).getByLabelText("Name"), "ci-bot")
    await createKey(user, dialog)

    const created = await screen.findByRole("dialog", {
      name: "API key created",
    })
    expect(within(created).getByLabelText("Secret key")).toBeInTheDocument()
    expect(within(created).queryByText("Send your first request")).toBeNull()
    expect(within(created).queryByLabelText("curl")).toBeNull()
    expect(
      within(created).getByRole("button", { name: "Create another" }),
    ).toBeVisible()
  })

  it("offers an existing budget to the operator under More options and caps the new key by it", async () => {
    const writes = mockApi({
      role: "owner",
      isOperator: true,
      budgets: [budget({ budget_id: "b-team", name: "Team standard" })],
    })
    const { user, dialog } = await openDialog()
    await user.type(within(dialog).getByLabelText("Name"), "ci-bot")
    await user.click(
      within(dialog).getByRole("button", { name: "More options" }),
    )
    await pickOption(user, "Budget", /^Team standard/, dialog)
    await createKey(user, dialog)

    await screen.findByRole("dialog", { name: "API key created" })
    await waitFor(() =>
      expect(writes.map((write) => write.url)).toContain(
        `${API_ROOT}/scoped-budgets`,
      ),
    )
    const ceiling = writes.find(
      (write) => write.url === `${API_ROOT}/scoped-budgets`,
    )
    expect(ceiling?.body).toMatchObject({
      scope_type: "api_token",
      scope_id: "key-new",
      budget_id: "b-team",
    })
    // No budget of its own is made when an existing one is picked.
    expect(writes.some((write) => write.url.endsWith("/budgets"))).toBe(false)
  })

  it("keeps focus on the secret when the form gives way to it", async () => {
    mockApi({ role: "owner", isOperator: false })
    const elsewhere = document.createElement("button")
    document.body.append(elsewhere)
    const { user, dialog } = await openDialog({ current: elsewhere })
    await user.type(within(dialog).getByLabelText("Name"), "ci-bot")
    await createKey(user, dialog)

    const created = await screen.findByRole("dialog", {
      name: "API key created",
    })
    await new Promise((resolve) => requestAnimationFrame(resolve))
    expect(within(created).getByLabelText("Secret key")).toHaveFocus()
    elsewhere.remove()
  })

  it("opens empty after the shell closed it over a draft", async () => {
    mockApi({ role: "owner", isOperator: false })
    const { user, dialog, reopen } = await openHeldDialog()
    await user.type(within(dialog).getByLabelText("Name"), "ci-bot")

    const reopened = await reopen()
    expect(within(reopened).getByLabelText("Name")).toHaveValue("")
  })

  it("never reopens on the last key's secret after the shell closed it", async () => {
    mockApi({ role: "owner", isOperator: false })
    const { user, dialog, reopen } = await openHeldDialog()
    await user.type(within(dialog).getByLabelText("Name"), "ci-bot")
    await createKey(user, dialog)
    await screen.findByRole("dialog", { name: "API key created" })

    const reopened = await reopen()
    expect(reopened).toHaveAccessibleName("Create API key")
    expect(within(reopened).queryByLabelText("Secret key")).toBeNull()
  })

  it("offers an organization owner the organization's budgets", async () => {
    mockApi({
      role: "admin",
      isOperator: false,
      budgets: [budget({ budget_id: "b-org", name: "Org standard" })],
    })
    const { user, dialog } = await openDialog()
    await user.click(
      within(dialog).getByRole("button", { name: "More options" }),
    )
    expect(await within(dialog).findByText("Budget")).toBeVisible()
  })

  it("does not offer a budget that backs a workspace's per-person default", async () => {
    mockApi({
      role: "admin",
      isOperator: false,
      budgets: [
        budget({ budget_id: "b-default", name: "Each Data member, weekly" }),
        budget({ budget_id: "b-team", name: "Team standard" }),
      ],
      defaults: [
        workspaceBudgetDefault({
          workspace_id: WORKSPACE_ID,
          budget_id: "b-default",
        }),
      ],
    })
    const { user, dialog } = await openDialog()
    await user.click(
      within(dialog).getByRole("button", { name: "More options" }),
    )
    await waitFor(() => expect(selectTrigger("Budget", dialog)).toBeEnabled())
    await user.click(selectTrigger("Budget", dialog))

    expect(
      await screen.findByRole("option", { name: /^Team standard/ }),
    ).toBeVisible()
    expect(
      screen.queryByRole("option", { name: /^Each Data member/ }),
    ).not.toBeInTheDocument()
  })

  it("offers a member no budget and says who sets one", async () => {
    mockApi({
      role: "member",
      isOperator: false,
      budgets: [budget({ budget_id: "b-org", name: "Org standard" })],
    })
    const { dialog } = await openDialog()
    expect(await within(dialog).findByText("Spend limit")).toBeVisible()
    expect(within(dialog).getByText(/Set by owners and admins/)).toBeVisible()
    expect(
      within(dialog).queryByRole("button", { name: /Budget/ }),
    ).not.toBeInTheDocument()
    expect(
      within(dialog).queryByLabelText("Spend limit, in dollars"),
    ).not.toBeInTheDocument()
  })
})

describe("what a new key carries", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  /** The one key the dialog posted, and where it posted it. */
  function posted(writes: ReturnType<typeof mockApi>) {
    const post = writes.find((write) => /\/keys$/.test(write.url))
    if (!post) throw new Error("no key was posted")
    return post
  }

  async function moreOptions(
    user: ReturnType<typeof userEvent.setup>,
    dialog: HTMLElement,
  ) {
    await user.click(
      within(dialog).getByRole("button", { name: "More options" }),
    )
  }

  it("sends the owner the operator names, and the access narrowed under More options", async () => {
    const writes = mockApi({ role: "owner", isOperator: true })
    const { user, dialog } = await openDialog()
    await moreOptions(user, dialog)
    await user.type(within(dialog).getByPlaceholderText(/Pick a user/), "alice")
    // Off the picker first: its open list hides the rest of the dialog.
    await user.click(within(dialog).getByLabelText("Name"))
    await user.click(within(dialog).getByRole("button", { name: "Block all" }))
    await user.click(within(dialog).getByRole("button", { name: "No tool" }))
    await createKey(user, dialog)

    await screen.findByRole("dialog", { name: "API key created" })
    expect(posted(writes).body).toMatchObject({
      user_id: "alice",
      allowed_models: [],
      allowed_tools: [],
    })
  })

  it("sends the default scopes and the deployment's user setting when left alone", async () => {
    const writes = mockApi({ role: "owner", isOperator: false })
    const { user, dialog } = await openDialog()
    await createKey(user, dialog)

    await screen.findByRole("dialog", { name: "API key created" })
    expect(posted(writes).body).toMatchObject({
      scopes: [
        "inference:run",
        "web:search",
        "web:fetch",
        "code:execute",
        "mcp:read",
        "mcp:execute",
      ],
      reject_user_mismatch: null,
      allowed_models: null,
      allowed_tools: null,
    })
  })

  it("sends the scopes checked", async () => {
    const writes = mockApi({ role: "owner", isOperator: false })
    const { user, dialog } = await openDialog()
    await moreOptions(user, dialog)
    await user.click(
      within(dialog).getByRole("button", { name: /Advanced scopes/ }),
    )
    await user.click(
      within(dialog).getByRole("checkbox", {
        name: "All coding agents scopes",
      }),
    )
    await createKey(user, dialog)

    await screen.findByRole("dialog", { name: "API key created" })
    expect(posted(writes).body.scopes).toEqual([
      "inference:run",
      "web:search",
      "web:fetch",
      "code:execute",
      "mcp:read",
      "mcp:execute",
      "telemetry:write",
      "agents:write",
    ])
  })

  it("pins a key to accept a mismatched user field", async () => {
    const writes = mockApi({ role: "owner", isOperator: false })
    const { user, dialog } = await openDialog()
    await moreOptions(user, dialog)
    await pickOption(
      user,
      "When a request names a different user",
      "Always accept",
      dialog,
    )
    await createKey(user, dialog)

    await screen.findByRole("dialog", { name: "API key created" })
    expect(posted(writes).body.reject_user_mismatch).toBe(false)
  })

  it("makes the operator a service key, exempt from budget", async () => {
    const writes = mockApi({ role: "owner", isOperator: true })
    const { user, dialog } = await openDialog()
    await moreOptions(user, dialog)
    await user.click(within(dialog).getByLabelText("Service key"))
    await user.click(within(dialog).getByLabelText("Exempt from budget"))
    await createKey(user, dialog)

    await screen.findByRole("dialog", { name: "API key created" })
    expect(posted(writes).body).toMatchObject({
      is_service_key: true,
      end_user_budget_ids: [],
      end_user_budget_id: null,
      max_end_users: null,
      exclude_from_budget: true,
    })
  })

  it("makes an organization owner the organization's service key", async () => {
    const writes = mockApi({ role: "owner", isOperator: false })
    const { user, dialog } = await openDialog()
    await moreOptions(user, dialog)
    await user.click(within(dialog).getByLabelText("Service key"))
    await createKey(user, dialog)

    await screen.findByRole("dialog", { name: "API key created" })
    const post = posted(writes)
    expect(post.url).toBe(`${API_ROOT}/organizations/me/keys`)
    expect(post.body).toMatchObject({
      is_service_key: true,
      end_user_budget_ids: [],
      end_user_budget_id: null,
      max_end_users: 10000,
    })
  })

  it("offers a member no service key, and sends neither an owner nor an exemption", async () => {
    const writes = mockApi({ role: "member", isOperator: false })
    const { user, dialog } = await openDialog()
    expect(within(dialog).queryByPlaceholderText(/Pick a user/)).toBeNull()
    await moreOptions(user, dialog)
    expect(within(dialog).queryByLabelText("Service key")).toBeNull()
    expect(within(dialog).queryByLabelText("Exempt from budget")).toBeNull()
    await createKey(user, dialog)

    await screen.findByRole("dialog", { name: "API key created" })
    const post = posted(writes)
    expect(post.url).toBe(`${API_ROOT}/organizations/me/keys`)
    expect(post.body).not.toHaveProperty("user_id")
    expect(post.body).not.toHaveProperty("exclude_from_budget")
  })
})
