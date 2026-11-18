import { Outlet } from "@tanstack/react-router"
import { screen, waitFor, within } from "@testing-library/react"
import userEvent, { type UserEvent } from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type {
  ApiKey,
  Budget,
  DeploymentBootstrap,
  MembershipRole,
  ScopedBudget,
} from "@/client"
import { KeysPage } from "@/features/keys/KeysPage"
import {
  type CreateKeyEntryPoint,
  OpenCreateKeyContext,
} from "@/features/keys/openCreateKey"
import { API_ROOT } from "@/shared/api/client"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  apiKey,
  bootstrap,
  budget,
  organizationContext,
  organizationMember,
  scopedBudget,
  workspace,
} from "@/tests/fixtures"
import { AppProviders as Provider } from "@/tests/providers"
import { renderWithRouter } from "@/tests/router"
import { getRowMenuItem } from "@/tests/rowMenu"
import { pickOption } from "@/tests/select"

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

const REFUSED = () => json({ detail: "Forbidden" }, 403)

const ME = "33333333-3333-3333-3333-333333333333"
const WORKSPACE_ID = "11111111-1111-1111-1111-111111111111"
const REGEN_SECRET =
  "gw-REGEN00000000000000000000000000000000000000000000000000"

// Both key surfaces answer identical shapes, so one handler serves them: the
// operator's /api/v1/keys and everyone else's /api/v1/organizations/me/keys.
const KEYS_URL = /\/api\/v1\/(?:organizations\/me\/)?keys(?:\/|\?|$)/

interface Api {
  keys?: ApiKey[]
  /** The deployment operator reads every key; anyone else their own. */
  deploymentOperator?: boolean
  role?: MembershipRole
  /** The caller's role in the keys' workspace; a contributor's by default. */
  workspaceRole?: MembershipRole
  members?: ReturnType<typeof organizationMember>[]
  /** This month's spend per key id, as the deployment's summary reads it. */
  spend?: Record<string, { cost: number; tokens: number }>
  /** The same, as the organization's summary reads it; `spend` by default. */
  organizationSpend?: Record<string, { cost: number; tokens: number }>
  budgets?: Budget[]
  /** The keys' own ceilings, as the deployment's list answers them. */
  ceilings?: ScopedBudget[]
  refuseUsage?: boolean
  refuseCeilings?: boolean
  /** Holds the deployment's ceiling list until it settles. */
  holdCeilings?: Promise<void>
}

/** The gateway as a caller sees it, with keys a write changes. */
function mockApi(opts: Api = {}) {
  let list = [...(opts.keys ?? [])]
  const operator = opts.deploymentOperator ?? true
  const role = opts.role ?? "owner"
  const members = opts.members ?? [
    organizationMember({ attribution_user_id: ME }),
  ]
  return vi
    .spyOn(globalThis, "fetch")
    .mockImplementation(async (input, init) => {
      const url = String(input)
      const method = (init?.method ?? "GET").toUpperCase()

      if (url === `${API_ROOT}/organizations/me`) {
        return json(
          organizationContext({
            role,
            deployment_operator: operator,
            workspace_memberships: [
              {
                workspace_id: WORKSPACE_ID,
                name: "Platform",
                role: opts.workspaceRole ?? "member",
              },
            ],
          }),
        )
      }
      if (/\/keys\/[^/]+\/end-user-count/.test(url)) {
        return json({ count: 0, max_end_users: 10000 })
      }
      if (/\/keys\/[^/]+\/end-users/.test(url)) return json([])
      if (/\/keys\/[^/]+\/federation-rules/.test(url)) return json([])
      if (KEYS_URL.test(url)) {
        const id = decodeURIComponent(
          url.split("?")[0].split("/keys/")[1]?.split("/")[0] ?? "",
        )
        if (url.endsWith("/rotate") && method === "POST") {
          const prefix = REGEN_SECRET.slice(0, 10)
          const suffix = REGEN_SECRET.slice(-4)
          list = list.map((k) =>
            k.id === id ? { ...k, key_prefix: prefix, key_suffix: suffix } : k,
          )
          const row = list.find((k) => k.id === id) ?? apiKey({ id })
          return json({ ...row, key: REGEN_SECRET })
        }
        if (method === "PATCH") {
          const body = JSON.parse(String(init?.body)) as Partial<ApiKey>
          list = list.map((k) => (k.id === id ? { ...k, ...body } : k))
          return json(list.find((k) => k.id === id))
        }
        if (method === "DELETE") {
          list = list.filter((k) => k.id !== id)
          return new Response(null, { status: 204 })
        }
        return json(list)
      }
      const summary = url.match(
        /\/api\/v1\/(organizations\/me\/)?usage\/summary/,
      )
      if (summary) {
        if (opts.refuseUsage) return REFUSED()
        const spend = summary[1]
          ? (opts.organizationSpend ?? opts.spend)
          : opts.spend
        return json({
          by_api_key: Object.entries(spend ?? {}).map(([key, used]) => ({
            key,
            requests: 1,
            ...used,
          })),
        })
      }
      if (url.includes(`${API_ROOT}/organizations/me/spend-ceilings`)) {
        if (opts.refuseCeilings) return REFUSED()
        return json({ data: [], count: 0 })
      }
      // An existing key's cap moves or goes through its ceiling's own path.
      if (url.includes(`${API_ROOT}/scoped-budgets/`)) {
        if (method === "DELETE") return new Response(null, { status: 204 })
        return json({
          ...scopedBudget({ scope_type: "api_token" }),
          ...JSON.parse(String(init?.body ?? "{}")),
        })
      }
      if (url.includes(`${API_ROOT}/scoped-budgets`)) {
        if (method === "POST") {
          return json(
            { id: "ceiling-new", ...JSON.parse(String(init?.body)) },
            201,
          )
        }
        if (opts.refuseCeilings) return REFUSED()
        await opts.holdCeilings
        return json(opts.ceilings ?? [])
      }
      if (url.includes(`${API_ROOT}/organizations/me/budgets`)) {
        return json({ data: [], count: 0 })
      }
      if (url.includes(`${API_ROOT}/budgets`)) return json(opts.budgets ?? [])
      if (url.includes(`${API_ROOT}/organizations/me/members`)) {
        return json({ data: members, count: members.length })
      }
      if (url.includes("/member-budget-policies")) {
        return json({ data: [], count: 0 })
      }
      if (url.startsWith(`${API_ROOT}/workspaces?`)) {
        return json({
          data: [workspace({ id: WORKSPACE_ID, name: "Platform" })],
          count: 1,
        })
      }
      return json([])
    })
}

/** The page at `url`, with the shell's Create API key dialog stood in for by `openCreateKey`. */
async function renderPage({
  url = "/",
  deployment = {},
  openCreateKey = () => undefined,
}: {
  url?: string
  deployment?: Partial<DeploymentBootstrap>
  openCreateKey?: (entryPoint: CreateKeyEntryPoint) => void
} = {}) {
  await renderWithRouter(<KeysPage />, {
    url,
    routes: [{ path: "/sessions", element: <p>Sessions</p> }],
    shell: (
      <Provider>
        <DeploymentProvider value={bootstrap(deployment)}>
          <SelectedWorkspaceProvider>
            <OpenCreateKeyContext value={openCreateKey}>
              <Outlet />
            </OpenCreateKeyContext>
          </SelectedWorkspaceProvider>
        </DeploymentProvider>
      </Provider>
    ),
  })
}

async function rowOf(name: string): Promise<HTMLElement> {
  return (await screen.findByText(name)).closest("tr") as HTMLElement
}

async function choose(user: UserEvent, row: HTMLElement, action: string) {
  await user.click(await getRowMenuItem(user, new RegExp(`^${action}`), row))
}

/** The calls `fetchMock` saw with `method` to a path ending in `suffix`. */
function calls(
  fetchMock: ReturnType<typeof mockApi>,
  method: string,
  suffix: string,
) {
  return fetchMock.mock.calls.filter(
    ([u, init]) =>
      String(u).split("?")[0].endsWith(suffix) &&
      (init?.method ?? "GET").toUpperCase() === method,
  )
}

function bodyOf(call: unknown[] | undefined): Record<string, unknown> {
  return JSON.parse(String((call?.[1] as RequestInit | undefined)?.body))
}

describe("KeysPage", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  describe("the list", () => {
    it("names each key with the last four of its secret, its state and its dates", async () => {
      mockApi({ keys: [apiKey({ id: "key-1", key_name: "ci-bot" })] })
      await renderPage()

      const row = await rowOf("ci-bot")
      expect(within(row).getByText("…7xKp")).toBeInTheDocument()
      expect(within(row).getByText("Active")).toBeInTheDocument()
      // Never used, and never expires.
      expect(within(row).getAllByText("Never")).toHaveLength(2)
    })

    describe("on a phone", () => {
      const realMatchMedia = window.matchMedia

      afterEach(() => {
        window.matchMedia = realMatchMedia
      })

      it("lists each key as a card, with its state and its menu", async () => {
        window.matchMedia = ((query: string) => ({
          matches: query === "(max-width: 767px)",
          media: query,
          onchange: null,
          addEventListener: () => undefined,
          removeEventListener: () => undefined,
          addListener: () => undefined,
          removeListener: () => undefined,
          dispatchEvent: () => false,
        })) as typeof window.matchMedia
        mockApi({ keys: [apiKey({ id: "key-1", key_name: "ci-bot" })] })
        await renderPage()

        const card = (await screen.findByText("ci-bot")).closest(
          "li",
        ) as HTMLElement
        expect(screen.queryByRole("grid")).toBeNull()
        expect(within(card).getByText("…7xKp")).toBeInTheDocument()
        expect(within(card).getByText("Active")).toBeInTheDocument()
        expect(
          within(card).getByRole("button", { name: /^Actions for ci-bot/ }),
        ).toBeInTheDocument()
      })
    })

    it("reads a key's spend, tokens and budget once they are read", async () => {
      mockApi({
        keys: [apiKey({ id: "key-1", key_name: "ci-bot" })],
        spend: { "key-1": { cost: 4.2, tokens: 1200 } },
      })
      await renderPage()

      const row = await rowOf("ci-bot")
      expect(await within(row).findByText("$4.20")).toBeInTheDocument()
      expect(within(row).getByText("None")).toBeInTheDocument()
      expect(screen.queryByRole("alert")).toBeNull()
    })

    it("says it could not read the spend, rather than that nothing was spent", async () => {
      mockApi({
        keys: [apiKey({ id: "key-1", key_name: "ci-bot" })],
        refuseUsage: true,
      })
      await renderPage()

      const row = await rowOf("ci-bot")
      expect(await screen.findByRole("alert")).toBeInTheDocument()
      await waitFor(() => expect(within(row).getAllByText("—")).toHaveLength(2))
      expect(within(row).queryByText("$0.00")).toBeNull()
    })

    it("says it could not read the budget, rather than that there is none", async () => {
      mockApi({
        keys: [apiKey({ id: "key-1", key_name: "ci-bot" })],
        refuseCeilings: true,
      })
      await renderPage()

      const row = await rowOf("ci-bot")
      expect(await screen.findByRole("alert")).toBeInTheDocument()
      await waitFor(() =>
        expect(within(row).getByText("—")).toBeInTheDocument(),
      )
      expect(within(row).queryByText("None")).toBeNull()
    })

    it("opens on active keys and lists the others under Every state", async () => {
      mockApi({
        keys: [
          apiKey({ id: "k1", key_name: "live" }),
          apiKey({ id: "k2", key_name: "off", is_active: false }),
          apiKey({
            id: "k3",
            key_name: "old",
            expires_at: "2020-01-01T00:00:00+00:00",
          }),
        ],
      })
      const user = userEvent.setup()
      await renderPage()

      await rowOf("live")
      expect(screen.queryByText("off")).toBeNull()
      expect(screen.queryByText("old")).toBeNull()
      expect(screen.getByText("1 active of 3")).toBeInTheDocument()

      await pickOption(user, "State", "Every state")
      expect(within(await rowOf("off")).getByText("Disabled")).toBeVisible()
      expect(within(await rowOf("old")).getByText("Expired")).toBeVisible()
    })

    it("offers every state when the filter leaves nothing", async () => {
      mockApi({ keys: [apiKey({ id: "k1", key_name: "live" })] })
      const user = userEvent.setup()
      await renderPage({ url: "/?state=disabled" })

      expect(await screen.findByText("No disabled keys.")).toBeVisible()
      await user.click(screen.getByRole("button", { name: "Show every state" }))
      expect(await rowOf("live")).toBeInTheDocument()
    })

    it("shows the whole fingerprint and what the key reaches in its menu", async () => {
      mockApi({
        keys: [
          apiKey({
            id: "k1",
            key_name: "scoped",
            allowed_models: ["openai:*", "openai:gpt-4o"],
            exclude_from_budget: true,
            reject_user_mismatch: false,
          }),
          apiKey({ id: "k2", key_name: "locked", allowed_models: [] }),
        ],
      })
      const user = userEvent.setup()
      await renderPage()

      await user.click(
        within(await rowOf("scoped")).getByRole("button", {
          name: /^Actions for scoped/,
        }),
      )
      // The head sits above the menu's items, in the same popover.
      await screen.findByRole("menu")
      expect(screen.getByText("gw-AbC3dE…7xKp")).toBeInTheDocument()
      expect(
        await screen.findByText(/^Created .+ in Platform$/),
      ).toBeInTheDocument()
      expect(screen.getByText("openai:*, openai:gpt-4o")).toBeInTheDocument()
      expect(screen.getByText("Exempt")).toBeInTheDocument()
      expect(screen.getByText("Lenient")).toBeInTheDocument()
      await user.keyboard("{Escape}")

      await user.click(
        within(await rowOf("locked")).getByRole("button", {
          name: /^Actions for locked/,
        }),
      )
      await screen.findByRole("menu")
      expect(screen.getByText("No models")).toBeInTheDocument()
    })

    it("marks a service key in its row", async () => {
      mockApi({
        keys: [apiKey({ id: "k1", key_name: "mlpa", is_service_key: true })],
      })
      await renderPage()

      expect(
        within(await rowOf("mlpa")).getByText("Service key"),
      ).toBeInTheDocument()
    })
  })

  describe("the operator's view", () => {
    it("names each key's owner, from the roster or by the id it bills to", async () => {
      mockApi({
        keys: [
          apiKey({ id: "k1", key_name: "alice-laptop", user_id: "alice-id" }),
          apiKey({ id: "k2", key_name: "ci", user_id: "ci-bot" }),
        ],
        members: [
          organizationMember({ attribution_user_id: ME }),
          organizationMember({
            organization_member_id: "om-alice",
            user_id: "alice-user",
            attribution_user_id: "alice-id",
            full_name: "Alice Doe",
          }),
        ],
      })
      await renderPage()

      expect(
        await screen.findByRole("columnheader", { name: "Owner" }),
      ).toBeInTheDocument()
      const alice = await rowOf("alice-laptop")
      expect(await within(alice).findByText("Alice Doe")).toBeInTheDocument()
      expect(within(alice).queryByText("alice-id")).toBeNull()
      // Nobody on the roster bills through `ci-bot`, so the id is the name.
      expect(within(await rowOf("ci")).getByText("ci-bot")).toHaveAttribute(
        "title",
        "Not in the organization",
      )
    })

    it("reads another organization's key's spend over the deployment, not as nothing", async () => {
      mockApi({
        keys: [
          apiKey({
            id: "k1",
            key_name: "theirs",
            user_id: "someone-else",
            workspace_id: "99999999-9999-9999-9999-999999999999",
          }),
        ],
        spend: { k1: { cost: 7.5, tokens: 3000 } },
        // Their organization is not the operator's, so its summary has no row.
        organizationSpend: {},
      })
      await renderPage()

      const row = await rowOf("theirs")
      expect(await within(row).findByText("$7.50")).toBeInTheDocument()
      expect(within(row).queryByText("$0.00")).toBeNull()
    })

    it("lists one person's keys from their page, and every key again on dismiss", async () => {
      mockApi({
        keys: [
          apiKey({ id: "k1", key_name: "alice-laptop", user_id: "alice-id" }),
          apiKey({ id: "k2", key_name: "ci", user_id: "ci-bot" }),
        ],
      })
      const user = userEvent.setup()
      await renderPage({ url: "/?owner=alice-id" })

      await rowOf("alice-laptop")
      expect(screen.queryByText("ci")).toBeNull()
      await user.click(screen.getByRole("button", { name: "Show every key" }))
      expect(await rowOf("ci")).toBeInTheDocument()
    })
  })

  describe("as an organization member", () => {
    it("lists through the member surface, without the Owner column", async () => {
      const fetchMock = mockApi({
        deploymentOperator: false,
        role: "member",
        keys: [apiKey({ id: "key-1", key_name: "mine", user_id: ME })],
      })
      await renderPage()

      expect(within(await rowOf("mine")).getByText("Active")).toBeVisible()
      const listCalls = fetchMock.mock.calls
        .map(([u]) => String(u))
        .filter((u) => KEYS_URL.test(u))
      expect(listCalls.length).toBeGreaterThan(0)
      for (const u of listCalls) {
        expect(u).toContain(`${API_ROOT}/organizations/me/keys`)
      }
      expect(
        screen.queryByRole("columnheader", { name: "Owner" }),
      ).not.toBeInTheDocument()
    })

    it("does not send a member to a Home that lists only their own keys", async () => {
      mockApi({ deploymentOperator: false, role: "member" })
      await renderPage({ url: "/?owner=someone-else" })

      expect(
        await screen.findByText(/Each person sees and changes their own keys/),
      ).toBeVisible()
      expect(screen.queryByText(/Their spend per key is on Home/)).toBeNull()
    })

    it("still points an owner to Home, which lists every key's spend", async () => {
      mockApi({ deploymentOperator: false, role: "owner" })
      await renderPage({ url: "/?owner=someone-else" })

      expect(
        await screen.findByText(/Their spend per key is on Home/),
      ).toBeVisible()
    })
  })

  describe("creating a key", () => {
    it("opens the Create API key dialog from the page's action", async () => {
      mockApi({ keys: [apiKey({ id: "key-1", key_name: "ci-bot" })] })
      const openCreateKey = vi.fn()
      const user = userEvent.setup()
      await renderPage({ openCreateKey })

      await rowOf("ci-bot")
      await user.click(screen.getByRole("button", { name: "Create API key" }))
      expect(openCreateKey).toHaveBeenCalledWith("api_keys_page")
    })

    it("offers one from an empty list", async () => {
      mockApi({ keys: [] })
      const openCreateKey = vi.fn()
      const user = userEvent.setup()
      await renderPage({ openCreateKey })

      const empty = (await screen.findByText("No API keys yet.")).closest(
        "td",
      ) as HTMLElement
      await user.click(
        within(empty).getByRole("button", { name: "Create API key" }),
      )
      expect(openCreateKey).toHaveBeenCalledWith("api_keys_empty")
    })
  })

  describe("a viewer", () => {
    it("is offered their keys to read and nothing that changes one", async () => {
      mockApi({
        deploymentOperator: false,
        role: "viewer",
        workspaceRole: "viewer",
        keys: [apiKey({ id: "key-1", key_name: "issued", user_id: ME })],
      })
      const user = userEvent.setup()
      await renderPage()

      const row = await rowOf("issued")
      expect(
        screen.queryByRole("button", { name: "Create API key" }),
      ).not.toBeInTheDocument()
      await getRowMenuItem(user, /^See its sessions/, row)
      expect(screen.getAllByRole("menuitem")).toHaveLength(1)
      expect(
        await screen.findByText(
          "A viewer cannot change or delete a key, including their own.",
        ),
      ).toBeVisible()
    })
  })

  describe("switching a key off and on", () => {
    it("disables an active key via PATCH, and will not delete one until then", async () => {
      const fetchMock = mockApi({
        keys: [
          apiKey({ id: "key-1", key_name: "ci-bot" }),
          apiKey({ id: "key-2", key_name: "other" }),
        ],
      })
      const user = userEvent.setup()
      await renderPage()

      const row = await rowOf("ci-bot")
      const refused = await getRowMenuItem(user, /^Delete/, row)
      expect(refused).toHaveAttribute("aria-disabled", "true")
      expect(refused).toHaveTextContent("Disable it first")
      await user.keyboard("{Escape}")

      await choose(user, row, "Disable")
      await waitFor(() =>
        expect(bodyOf(calls(fetchMock, "PATCH", "/keys/key-1")[0])).toEqual({
          is_active: false,
        }),
      )
      // The row leaves the Active list, and focus goes to the filter that
      // says which keys the list holds.
      await waitFor(() => expect(screen.queryByText("ci-bot")).toBeNull())
      expect(document.activeElement).toBe(
        screen.getByRole("button", { name: /Active/ }),
      )
    })

    it("enables a disabled key via PATCH", async () => {
      const fetchMock = mockApi({
        keys: [apiKey({ id: "key-1", key_name: "off", is_active: false })],
      })
      const user = userEvent.setup()
      await renderPage({ url: "/?state=all" })

      await choose(user, await rowOf("off"), "Enable")
      await waitFor(() =>
        expect(bodyOf(calls(fetchMock, "PATCH", "/keys/key-1")[0])).toEqual({
          is_active: true,
        }),
      )
    })

    it.each([
      [
        "a member, on a key an operator disabled",
        false,
        { disabled_reason: "operator" as const },
        "Disabled by an operator",
      ],
      [
        "the operator, on a revoked key",
        true,
        {
          disabled_reason: "revoked" as const,
          operator_can_enable: false,
        },
        "Revoked permanently",
      ],
      [
        "the operator, on a reason the server says is not theirs",
        true,
        {
          disabled_reason: "operator" as const,
          operator_can_enable: false,
        },
        "Disabled by an operator",
      ],
    ])("keeps %s from enabling it", async (_case, operator, key, hint) => {
      mockApi({
        deploymentOperator: operator,
        role: "member",
        keys: [
          apiKey({
            id: "key-1",
            key_name: "held",
            user_id: ME,
            is_active: false,
            ...key,
          }),
        ],
      })
      const user = userEvent.setup()
      await renderPage({ url: "/?state=all" })

      const enable = await getRowMenuItem(user, /^Enable/, await rowOf("held"))
      expect(enable).toHaveAttribute("aria-disabled", "true")
      expect(within(enable).getByText(hint)).toBeInTheDocument()
    })

    it("lets the operator enable a key an operator disabled", async () => {
      mockApi({
        keys: [
          apiKey({
            id: "key-1",
            key_name: "held",
            is_active: false,
            disabled_reason: "operator",
          }),
        ],
      })
      const user = userEvent.setup()
      await renderPage({ url: "/?state=all" })

      const enable = await getRowMenuItem(user, /^Enable/, await rowOf("held"))
      expect(enable).not.toHaveAttribute("aria-disabled", "true")
    })
  })

  it("permanently deletes a disabled key after a confirm that names it", async () => {
    const fetchMock = mockApi({
      keys: [apiKey({ id: "key-1", key_name: "legacy", is_active: false })],
    })
    const user = userEvent.setup()
    await renderPage({ url: "/?state=all" })

    await choose(user, await rowOf("legacy"), "Delete")
    const dialog = await screen.findByRole("alertdialog")
    expect(within(dialog).getByText(/unlinks its usage history/)).toBeVisible()
    expect(within(dialog).getByText("legacy")).toBeInTheDocument()
    await user.click(
      within(dialog).getByRole("button", { name: "Delete permanently" }),
    )

    await waitFor(() =>
      expect(calls(fetchMock, "DELETE", "/keys/key-1")).toHaveLength(1),
    )
    await waitFor(() => expect(screen.queryByText("legacy")).toBeNull())
    // Its row is gone, so focus goes to the filter over the list.
    expect(document.activeElement).toBe(
      screen.getByRole("button", { name: /Every state/ }),
    )
  })

  describe("regenerating a key", () => {
    it("hands over the new secret once, after an explicit confirm", async () => {
      mockApi({ keys: [apiKey({ id: "key-1", key_name: "ci-bot" })] })
      const user = userEvent.setup()
      await renderPage()

      await choose(user, await rowOf("ci-bot"), "Regenerate")
      const confirm = await screen.findByRole("alertdialog")
      expect(within(confirm).getByText("ci-bot")).toBeInTheDocument()
      await user.click(
        within(confirm).getByRole("button", { name: "Regenerate" }),
      )

      const reveal = await screen.findByRole("alert", {
        name: "New secret for ci-bot",
      })
      expect(within(reveal).getByLabelText("Secret key")).toHaveValue(
        "gw-REGEN00••••••••0000",
      )
      const dialog = screen.getByRole("dialog")
      // The key already exists, so there is nothing to create again.
      expect(
        within(dialog).queryByRole("button", { name: "Create another" }),
      ).toBeNull()

      // A stray Escape does not lose it; only the acknowledgement does.
      await user.keyboard("{Escape}")
      expect(screen.getByRole("dialog")).toBeInTheDocument()
      await user.click(
        within(dialog).getByRole("button", { name: /I.?ve saved this key/ }),
      )
      await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull())
    })

    it("gives focus back to the row's menu when the confirm is cancelled", async () => {
      mockApi({ keys: [apiKey({ id: "key-1", key_name: "ci-bot" })] })
      const user = userEvent.setup()
      await renderPage()

      const row = await rowOf("ci-bot")
      await choose(user, row, "Regenerate")
      const confirm = await screen.findByRole("alertdialog")
      await user.click(within(confirm).getByRole("button", { name: "Cancel" }))

      await waitFor(() => expect(screen.queryByRole("alertdialog")).toBeNull())
      expect(
        within(row).getByRole("button", { name: /^Actions for ci-bot/ }),
      ).toHaveFocus()
    })
  })

  describe("workload identity", () => {
    it.each([
      [
        "the deployment does not let organizations manage rules",
        false,
        "owner",
      ],
      ["the caller is a member", true, "member"],
    ] as const)("is not offered when %s", async (_case, rules, role) => {
      mockApi({
        deploymentOperator: false,
        role,
        keys: [apiKey({ id: "key-1", key_name: "ci-bot", user_id: ME })],
      })
      const user = userEvent.setup()
      await renderPage({ deployment: { federation_member_rules: rules } })

      await getRowMenuItem(user, /^Edit/, await rowOf("ci-bot"))
      expect(
        screen.queryByRole("menuitem", { name: /^Workload identity/ }),
      ).toBeNull()
    })

    it.each([
      ["an admin's own key", "admin", false],
      ["any key, for the operator", "member", true],
    ] as const)(
      "opens the federation rules of %s from its menu",
      async (_case, role, operator) => {
        mockApi({
          deploymentOperator: operator,
          role,
          keys: [apiKey({ id: "key-1", key_name: "ci-bot", user_id: ME })],
        })
        const user = userEvent.setup()
        await renderPage({ deployment: { federation_member_rules: true } })

        await choose(user, await rowOf("ci-bot"), "Workload identity")
        const dialog = await screen.findByRole("dialog", {
          name: "Workload identity",
        })
        expect(within(dialog).getByText("ci-bot")).toBeInTheDocument()
      },
    )
  })

  it("opens a service key's end users from its menu, and offers them for no other key", async () => {
    mockApi({
      deploymentOperator: false,
      role: "owner",
      keys: [
        apiKey({
          id: "key-1",
          key_name: "mlpa",
          is_service_key: true,
          is_organization_service_key: true,
        }),
        apiKey({ id: "key-2", key_name: "plain", user_id: ME }),
      ],
    })
    const user = userEvent.setup()
    await renderPage()

    await getRowMenuItem(user, /^Edit/, await rowOf("plain"))
    expect(screen.queryByRole("menuitem", { name: /^End users/ })).toBeNull()
    await user.keyboard("{Escape}")
    await choose(user, await rowOf("mlpa"), "End users")
    const dialog = await screen.findByRole("dialog", { name: "End users" })
    expect(within(dialog).getByText("mlpa")).toBeInTheDocument()
  })

  it("opens a key's sessions in its own workspace", async () => {
    mockApi({ keys: [apiKey({ id: "key-1", key_name: "ci-bot" })] })
    const user = userEvent.setup()
    await renderPage()

    await choose(user, await rowOf("ci-bot"), "See its sessions")
    expect(await screen.findByText("Sessions")).toBeVisible()
  })

  describe("editing a key", () => {
    async function editKey(user: UserEvent, name = "ci-bot") {
      await choose(user, await rowOf(name), "Edit")
      return screen.findByRole("dialog", { name: "Edit API key" })
    }

    it("opens the form in a dialog naming the key, and gives focus back to its row", async () => {
      mockApi({ keys: [apiKey({ id: "key-1", key_name: "ci-bot" })] })
      const user = userEvent.setup()
      await renderPage()

      const dialog = await editKey(user)
      expect(within(dialog).getByText("ci-bot")).toBeInTheDocument()
      expect(within(dialog).getByLabelText("Name")).toHaveValue("ci-bot")
      await user.keyboard("{Escape}")
      await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull())
      expect(
        within(await rowOf("ci-bot")).getByRole("button", {
          name: /^Actions for ci-bot/,
        }),
      ).toHaveFocus()
    })

    it("asks before discarding an edited field", async () => {
      mockApi({ keys: [apiKey({ id: "key-1", key_name: "ci-bot" })] })
      const user = userEvent.setup()
      await renderPage()

      const dialog = await editKey(user)
      await user.type(within(dialog).getByLabelText("Name"), "-2")
      await user.keyboard("{Escape}")

      expect(screen.getByRole("button", { name: "Keep editing" })).toBeVisible()
      await user.click(screen.getByRole("button", { name: "Discard" }))
      await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull())
    })

    it("opens each key on its own values", async () => {
      mockApi({
        keys: [
          apiKey({ id: "k1", key_name: "alpha" }),
          apiKey({ id: "k2", key_name: "bravo" }),
        ],
      })
      const user = userEvent.setup()
      await renderPage()

      const alpha = await editKey(user, "alpha")
      expect(within(alpha).getByLabelText("Name")).toHaveValue("alpha")
      await user.click(within(alpha).getByRole("button", { name: "Cancel" }))
      await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull())

      const bravo = await editKey(user, "bravo")
      expect(within(bravo).getByLabelText("Name")).toHaveValue("bravo")
    })

    it("narrows an existing key's scopes via PATCH", async () => {
      const fetchMock = mockApi({
        keys: [
          apiKey({
            id: "key-1",
            key_name: "ci-bot",
            scopes: ["inference:run", "usage:read"],
          }),
        ],
      })
      const user = userEvent.setup()
      await renderPage()

      const dialog = await editKey(user)
      // A scope under Advanced scopes, which opens by itself for a key holding one.
      await user.click(
        await within(dialog).findByRole("checkbox", {
          name: /Read its own budget/,
        }),
      )
      await user.click(within(dialog).getByRole("button", { name: "Save" }))

      await waitFor(() =>
        expect(
          bodyOf(calls(fetchMock, "PATCH", "/keys/key-1")[0]).scopes,
        ).toEqual(["inference:run"]),
      )
    })

    it("will not save a key with no scope", async () => {
      mockApi({
        keys: [
          apiKey({ id: "key-1", key_name: "ci-bot", scopes: ["web:search"] }),
        ],
      })
      const user = userEvent.setup()
      await renderPage()

      const dialog = await editKey(user)
      await user.click(
        await within(dialog).findByRole("checkbox", { name: /Search the web/ }),
      )

      expect(within(dialog).getByText("Choose at least one")).toBeVisible()
      expect(
        within(dialog).getByRole("button", { name: "Save" }),
      ).toBeDisabled()
    })

    it("takes the key out of budget for the operator via PATCH", async () => {
      const fetchMock = mockApi({
        keys: [apiKey({ id: "key-1", key_name: "ci-bot" })],
      })
      const user = userEvent.setup()
      await renderPage()

      const dialog = await editKey(user)
      await user.click(within(dialog).getByLabelText("Exempt from budget"))
      await user.click(within(dialog).getByRole("button", { name: "Save" }))

      await waitFor(() =>
        expect(
          bodyOf(calls(fetchMock, "PATCH", "/keys/key-1")[0])
            .exclude_from_budget,
        ).toBe(true),
      )
    })

    it.each([
      [null, "Always accept", false],
      [false, "Use the deployment setting (default)", null],
    ])(
      "moves a key's mismatched user choice from %s via PATCH",
      async (from, option, sent) => {
        const fetchMock = mockApi({
          keys: [
            apiKey({
              id: "key-1",
              key_name: "ci-bot",
              reject_user_mismatch: from,
            }),
          ],
        })
        const user = userEvent.setup()
        await renderPage()

        const dialog = await editKey(user)
        await pickOption(
          user,
          "When a request names a different user",
          option,
          dialog,
        )
        await user.click(within(dialog).getByRole("button", { name: "Save" }))

        // An explicit null is what clears the override.
        await waitFor(() =>
          expect(
            bodyOf(calls(fetchMock, "PATCH", "/keys/key-1")[0]),
          ).toHaveProperty("reject_user_mismatch", sent),
        )
      },
    )

    it("edits a member's own key through the member surface, with no budget exemption", async () => {
      const fetchMock = mockApi({
        deploymentOperator: false,
        role: "member",
        keys: [apiKey({ id: "key-1", key_name: "mine", user_id: ME })],
      })
      const user = userEvent.setup()
      await renderPage()

      const dialog = await editKey(user, "mine")
      expect(within(dialog).queryByLabelText("Exempt from budget")).toBeNull()
      await user.type(within(dialog).getByLabelText("Name"), "-2")
      await user.click(within(dialog).getByRole("button", { name: "Save" }))

      await waitFor(() =>
        expect(
          calls(fetchMock, "PATCH", "/organizations/me/keys/key-1"),
        ).toHaveLength(1),
      )
      expect(
        bodyOf(calls(fetchMock, "PATCH", "/organizations/me/keys/key-1")[0]),
      ).not.toHaveProperty("exclude_from_budget")
    })

    it("leaves an operator's service key to the operator on an owner's form", async () => {
      mockApi({
        deploymentOperator: false,
        role: "owner",
        keys: [
          apiKey({
            id: "key-1",
            key_name: "issued",
            user_id: ME,
            is_service_key: true,
            end_user_budget_ids: ["deployment-budget"],
          }),
        ],
      })
      const user = userEvent.setup()
      await renderPage()

      const dialog = await editKey(user, "issued")
      expect(within(dialog).getByLabelText("Name")).toBeInTheDocument()
      expect(within(dialog).queryByLabelText("Service key")).toBeNull()
    })
  })

  describe("a key's own budget", () => {
    const team = budget({ budget_id: "b-1", name: "Team standard" })
    const other = budget({ budget_id: "b-2", name: "CI bots" })
    const capped = scopedBudget({
      id: "ceiling-1",
      scope_type: "api_token",
      scope_id: "key-1",
      budget_id: "b-1",
      max_budget: 100,
      current_spend: 2.5,
    })

    async function editKey(user: UserEvent) {
      const row = await rowOf("ci-bot")
      // The edit form seeds from the ceilings the page read for the column.
      await within(row).findByText(/^\$100\.00 a month$|^None$/)
      await choose(user, row, "Edit")
      return screen.findByRole("dialog", { name: "Edit API key" })
    }

    it("shows its cap in a column, and its spend against it in the menu", async () => {
      mockApi({
        keys: [
          apiKey({ id: "key-1", key_name: "ci-bot" }),
          apiKey({ id: "key-2", key_name: "uncapped" }),
        ],
        budgets: [team],
        ceilings: [capped],
      })
      const user = userEvent.setup()
      await renderPage()

      const row = await rowOf("ci-bot")
      expect(
        await within(row).findByText("$100.00 a month"),
      ).toBeInTheDocument()
      expect(within(await rowOf("uncapped")).getByText("None")).toBeVisible()
      await user.click(
        within(row).getByRole("button", { name: /^Actions for ci-bot/ }),
      )
      await screen.findByRole("menu")
      expect(
        screen.getByText(/^\$2\.50 of \$100\.00 \(Team standard/),
      ).toBeInTheDocument()
    })

    it("shows the budget field once the ceilings land after the form opened", async () => {
      let release = () => {}
      const held = new Promise<void>((resolve) => {
        release = resolve
      })
      mockApi({
        keys: [apiKey({ id: "key-1", key_name: "ci-bot" })],
        budgets: [team],
        ceilings: [capped],
        holdCeilings: held,
      })
      const user = userEvent.setup()
      await renderPage()

      await choose(user, await rowOf("ci-bot"), "Edit")
      const dialog = await screen.findByRole("dialog", { name: "Edit API key" })
      expect(within(dialog).queryByText("Budget (optional)")).toBeNull()

      release()
      // It opens on the key's budget, and that is the seed rather than an edit.
      expect(
        await within(dialog).findByRole("button", {
          name: /Team standard.*Budget \(optional\)$/,
        }),
      ).toBeInTheDocument()
      await user.keyboard("{Escape}")
      await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull())
    })

    it("puts an uncapped key under a budget from its edit form", async () => {
      const fetchMock = mockApi({
        keys: [apiKey({ id: "key-1", key_name: "ci-bot" })],
        budgets: [team],
      })
      const user = userEvent.setup()
      await renderPage()

      const dialog = await editKey(user)
      await pickOption(user, "Budget (optional)", /^Team standard/, dialog)
      await user.click(within(dialog).getByRole("button", { name: "Save" }))

      await waitFor(() =>
        expect(bodyOf(calls(fetchMock, "POST", "/scoped-budgets")[0])).toEqual({
          scope_type: "api_token",
          scope_id: "key-1",
          budget_id: "b-1",
        }),
      )
    })

    it("moves a capped key to another budget", async () => {
      const fetchMock = mockApi({
        keys: [apiKey({ id: "key-1", key_name: "ci-bot" })],
        budgets: [team, other],
        ceilings: [capped],
      })
      const user = userEvent.setup()
      await renderPage()

      const dialog = await editKey(user)
      await pickOption(user, "Budget (optional)", /^CI bots/, dialog)
      await user.click(within(dialog).getByRole("button", { name: "Save" }))

      await waitFor(() =>
        expect(
          bodyOf(calls(fetchMock, "PATCH", "/scoped-budgets/ceiling-1")[0]),
        ).toEqual({ budget_id: "b-2" }),
      )
      expect(calls(fetchMock, "POST", "/scoped-budgets")).toHaveLength(0)
    })

    it("takes a key's budget off", async () => {
      const fetchMock = mockApi({
        keys: [apiKey({ id: "key-1", key_name: "ci-bot" })],
        budgets: [team],
        ceilings: [capped],
      })
      const user = userEvent.setup()
      await renderPage()

      const dialog = await editKey(user)
      await pickOption(user, "Budget (optional)", "No budget", dialog)
      await user.click(within(dialog).getByRole("button", { name: "Save" }))

      await waitFor(() =>
        expect(
          calls(fetchMock, "DELETE", "/scoped-budgets/ceiling-1"),
        ).toHaveLength(1),
      )
    })

    it("leaves the cap alone when only the key changed", async () => {
      const fetchMock = mockApi({
        keys: [apiKey({ id: "key-1", key_name: "ci-bot" })],
        budgets: [team],
        ceilings: [capped],
      })
      const user = userEvent.setup()
      await renderPage()

      const dialog = await editKey(user)
      await user.type(within(dialog).getByLabelText("Name"), "-2")
      await user.click(within(dialog).getByRole("button", { name: "Save" }))

      await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull())
      expect(
        fetchMock.mock.calls.some(([u]) =>
          String(u).includes(`${API_ROOT}/scoped-budgets/`),
        ),
      ).toBe(false)
      expect(calls(fetchMock, "POST", "/scoped-budgets")).toHaveLength(0)
    })
  })
})
