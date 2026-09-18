import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, screen, within } from "@testing-library/react"
import type { UserEvent } from "@testing-library/user-event"
import type { ReactElement } from "react"
import { vi } from "vitest"

import type { ApiKey, DeploymentBootstrap, User } from "@/client"
import { API_ROOT } from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { apiKey, bootstrap, type organizationMember } from "@/tests/fixtures"
import { renderWithRouter } from "@/tests/router"

export function user(overrides: Partial<User> = {}): User {
  return {
    user_id: "alice",
    // An alias is what makes the owner option's label differ from its user_id,
    // which is the case the picker has to get right.
    alias: "Alice",
    spend: 0,
    reserved: 0,
    current_tokens: 0,
    reserved_tokens: 0,
    current_requests: 0,
    reserved_requests: 0,
    budget_id: null,
    allowed_models: null,
    budget_started_at: null,
    next_budget_reset_at: null,
    blocked: false,
    created_at: "2026-01-01T00:00:00+00:00",
    updated_at: "2026-01-01T00:00:00+00:00",
    metadata: {},
    ...overrides,
  }
}

// The page picks its layout from the region it is given, which jsdom reports as
// 0 wide. Feeding the observer a width is what selects wide, compact or the list.
export function stubRegionWidth(width: number) {
  const OriginalObserver = globalThis.ResizeObserver
  vi.stubGlobal(
    "ResizeObserver",
    class extends OriginalObserver {
      constructor(callback: ResizeObserverCallback) {
        // Acted: the callback drives `setLayout`, and forwarding it raw left
        // every layout change as an unacted update.
        super((entries, observer) => {
          act(() => {
            callback(
              entries.map((entry) => ({
                ...entry,
                contentRect: { ...entry.contentRect, width },
              })),
              observer,
            )
          })
        })
      }
    },
  )
}

export async function chooseAction(
  user: UserEvent,
  row: HTMLElement,
  action: string,
) {
  await user.click(within(row).getByRole("button", { name: /^Actions for / }))
  await user.click(
    await screen.findByRole("menuitem", { name: new RegExp(`^${action}`) }),
  )
}

export function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

export const NEW_SECRET =
  "gw-NEWSECRET0000000000000000000000000000000000000000000000"
export const REGEN_SECRET =
  "gw-REGEN00000000000000000000000000000000000000000000000000"

// Both key surfaces answer identical shapes, so one handler serves them: the
// operator's /api/v1/keys and the member's /api/v1/organizations/me/keys
// (otari-ai#1941). Which one the page asked is what the member-view cases
// assert, off the spy's recorded URLs.
export const KEYS_URL = /\/api\/v1\/(?:organizations\/me\/)?keys(?:\/|\?|$)/

export function mockApi(
  opts: {
    keys?: ApiKey[]
    users?: User[]
    members?: ReturnType<typeof organizationMember>[]
    deploymentOperator?: boolean
  } = {},
) {
  let list = [...(opts.keys ?? [])]
  const users = opts.users ?? []
  const members = opts.members ?? []

  return vi
    .spyOn(globalThis, "fetch")
    .mockImplementation(async (input, init) => {
      const url = String(input)
      const method = (init?.method ?? "GET").toUpperCase()

      if (KEYS_URL.test(url)) {
        if (url.endsWith("/rotate") && method === "POST") {
          const id = url.split("/").slice(-2)[0]
          const prefix = REGEN_SECRET.slice(0, 10)
          const suffix = REGEN_SECRET.slice(-4)
          list = list.map((k) =>
            k.id === id ? { ...k, key_prefix: prefix, key_suffix: suffix } : k,
          )
          const row = list.find((k) => k.id === id) ?? apiKey({ id })
          return jsonResponse({
            ...row,
            key: REGEN_SECRET,
            key_prefix: prefix,
            key_suffix: suffix,
          })
        }
        if (method === "POST") {
          const body = JSON.parse(String(init?.body)) as {
            key_name?: string | null
            user_id?: string | null
            allowed_models?: string[] | null
            reject_user_mismatch?: boolean | null
          }
          const row = apiKey({
            id: "key-new",
            key_prefix: NEW_SECRET.slice(0, 10),
            key_suffix: NEW_SECRET.slice(-4),
            key_name: body.key_name ?? null,
            user_id: body.user_id ?? "apikey-key-new",
            allowed_models: body.allowed_models ?? null,
            reject_user_mismatch: body.reject_user_mismatch ?? null,
          })
          list = [...list, row]
          return jsonResponse({ ...row, key: NEW_SECRET })
        }
        if (method === "PATCH") {
          const id = decodeURIComponent(url.split("/").pop() ?? "")
          const body = JSON.parse(String(init?.body)) as Partial<ApiKey>
          list = list.map((k) => (k.id === id ? { ...k, ...body } : k))
          return jsonResponse(list.find((k) => k.id === id))
        }
        if (method === "DELETE") {
          const id = decodeURIComponent(url.split("/").pop() ?? "")
          list = list.filter((k) => k.id !== id)
          return new Response(null, { status: 204 })
        }
        return jsonResponse(list)
      }
      // Before /api/v1/users, and paged: the owner picker names members through
      // this, and `fetchAllPaged` reads `data`/`count` rather than a bare list.
      if (url.includes(`${API_ROOT}/organizations/me/members`)) {
        return jsonResponse({ data: members, count: members.length })
      }
      // Seeds the scope: `deployment_operator` is what routes the page onto the
      // operator surface or the member one. These suites default to the
      // operator's view, and the member cases flip it.
      if (url.endsWith(`${API_ROOT}/organizations/me`)) {
        return jsonResponse({
          organization_member_id: "om-1",
          role: "member",
          status: "active",
          organization: {
            id: "org-1",
            name: "Acme",
            slug: "acme",
            created_by_user_id: null,
            created_at: "2026-01-01T00:00:00+00:00",
            updated_at: null,
          },
          byo_provider_keys_allowed: true,
          deployment_operator: opts.deploymentOperator ?? true,
          provider_key_encryption_available: true,
          workspace_memberships: [],
        })
      }
      if (url.includes(`${API_ROOT}/users`)) {
        return jsonResponse(users)
      }
      if (url.includes(`${API_ROOT}/models/discoverable`)) {
        return jsonResponse({
          providers: [
            {
              provider: "openai",
              ok: true,
              error: null,
              models: [{ id: "gpt-4o", key: "openai:gpt-4o" }],
            },
          ],
        })
      }
      if (url.includes(`${API_ROOT}/providers`)) {
        return jsonResponse({ providers: [{ instance: "openai" }] })
      }
      if (url.includes(`${API_ROOT}/aliases`)) {
        return jsonResponse([])
      }
      return jsonResponse([])
    })
}

export function renderPage(
  ui: ReactElement,
  deployment: DeploymentBootstrap = bootstrap(),
) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  // The page reads the deployment's surfaces to decide whether it may name key
  // owners from the organization roster, and links to the pages that own a key's
  // budget, so it needs both the deployment context and a router around it.
  return renderWithRouter(
    <QueryClientProvider client={client}>
      <DeploymentProvider value={deployment}>{ui}</DeploymentProvider>
    </QueryClientProvider>,
  )
}

/**
 * Press the create dialog's submit.
 *
 * Scoped to the dialog because "Create key" is deliberately on screen twice
 * while it is open: the labels rule makes the page's trigger and the dialog's
 * submit the same string, and the trigger no longer hides behind the form.
 */
export async function submitTheCreateDialog(user: UserEvent) {
  const dialog = await screen.findByRole("dialog")
  await user.click(within(dialog).getByRole("button", { name: "Create key" }))
}
