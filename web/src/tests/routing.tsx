import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, within } from "@testing-library/react"
import type userEvent from "@testing-library/user-event"
import type { ReactElement } from "react"
import { vi } from "vitest"

import type {
  OrganizationContext,
  OrganizationMember,
  PolicySpec,
  RoutingPolicyResponse,
} from "@/client"
import { API_ROOT } from "@/shared/api/client"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  bootstrap,
  organizationContext,
  organizationMember,
} from "@/tests/fixtures"
import { withRouter } from "@/tests/router"

export const policy = (
  name: string,
  spec: PolicySpec,
  overrides: Partial<RoutingPolicyResponse> = {},
): RoutingPolicyResponse => ({
  name,
  spec,
  source: "stored",
  user_id: null,
  is_dynamic: false,
  created_at: null,
  updated_at: null,
  ...overrides,
})

export const CHAIN: PolicySpec = {
  select: [{ default: "openai:gpt-5-mini" }],
  on_failure: ["anthropic:claude-haiku-4-5"],
}

export const LEARNED: PolicySpec = {
  select: [
    { router: "knn", candidates: ["openai:gpt-5-nano", "openai:gpt-5"] },
    { default: "openai:gpt-5" },
  ],
}

export const WEIGHTED: PolicySpec = {
  select: [
    {
      router: "weighted",
      candidates: ["openai:gpt-5", "anthropic:claude-sonnet-4-5"],
      weights: { "openai:gpt-5": 70, "anthropic:claude-sonnet-4-5": 30 },
    },
    { default: "openai:gpt-5" },
  ],
}

export const POLICIES: RoutingPolicyResponse[] = [
  policy("fast", CHAIN),
  policy(
    "auto",
    {
      select: [
        { when: { budget_used_pct: { gte: 80 } }, target: "openai:gpt-5-nano" },
        { default: "openai:gpt-5-mini" },
      ],
    },
    { source: "config", is_dynamic: true },
  ),
]

export const USERS = [
  { user_id: "alice", alias: "alice", spend: 0, is_blocked: false },
  { user_id: "u-bob", alias: "bob", spend: 0, is_blocked: false },
  { user_id: "u-carol", alias: "carol", spend: 0, is_blocked: false },
]

// One of the three has a roster row behind them, so the same list exercises both
// halves of the picker's labeling: a person the organization can name, and an
// owner id nobody named, whose id is its name.
export const MEMBERS = [
  organizationMember({
    organization_member_id: "55555555-5555-5555-5555-555555555555",
    user_id: "u-bob",
    attribution_user_id: "u-bob",
    full_name: "Bob Builder",
    role: "member",
  }),
]

export function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  })
}

export function mockApi(
  policies: RoutingPolicyResponse[] = POLICIES,
  guardrailsUrl: string | null = "http://guardrails:8000",
  aliases: {
    name: string
    target: string
    source: string
    user_id: string | null
  }[] = [],
  opts: {
    // The caller's membership context, an operator's by default: the page picks
    // its list read off `deployment_operator`, and most tests here are about
    // the management view.
    context?: OrganizationContext
    // What /api/v1/organizations/me/routing-policies answers, for member tests.
    memberPolicies?: RoutingPolicyResponse[]
    // What /api/v1/organizations/me/aliases answers, its sibling.
    memberAliases?: {
      name: string
      target: string
      source: string
      user_id: string | null
      workspace_id?: string
    }[]
    // What a delete of a deployment-wide policy answers, for the error path.
    deleteBody?: { status: number; detail: string }
    // The same for a save, which is the path the form's own banner reports.
    saveBody?: { status: number; detail: string }
    // The organization roster, which is what names a person behind an owner id.
    members?: OrganizationMember[]
    // Owner ids whose *first* policy write is refused. N scopes are N writes, so
    // the retry has to be able to succeed for the one that failed while leaving
    // the ones that landed alone.
    refuseFirstWriteFor?: string[]
  } = {},
) {
  let list = [...policies]
  let aliasList = [...aliases]
  let memberList = [...(opts.memberPolicies ?? [])]
  let memberAliasList = [...(opts.memberAliases ?? [])]
  const refuseOnce = new Set(opts.refuseFirstWriteFor ?? [])
  const calls: { url: string; method: string; body: unknown }[] = []
  const spy = vi
    .spyOn(globalThis, "fetch")
    .mockImplementation(async (input, init) => {
      const url = String(input)
      const method = (init?.method ?? "GET").toUpperCase()
      const body =
        init?.body === undefined ? undefined : JSON.parse(String(init.body))
      calls.push({ url, method, body })

      if (url.includes(`${API_ROOT}/organizations/me/routing-policies`)) {
        if (method === "POST") {
          const row = policy(body.name, body.spec)
          memberList = [
            ...memberList.filter((item) => item.name !== row.name),
            row,
          ]
          return jsonResponse(row)
        }
        if (method === "DELETE") {
          const name = decodeURIComponent(
            url.split("?")[0].split("/").pop() ?? "",
          )
          memberList = memberList.filter((item) => item.name !== name)
          return new Response(null, { status: 204 })
        }
        return jsonResponse(memberList)
      }
      if (url.includes(`${API_ROOT}/organizations/me/aliases`)) {
        if (method === "POST") {
          const row = {
            name: body.name as string,
            target: body.target as string,
            source: "stored",
            user_id: null,
            workspace_id: body.workspace_id as string | undefined,
          }
          memberAliasList = [
            ...memberAliasList.filter((item) => item.name !== row.name),
            row,
          ]
          return jsonResponse(row)
        }
        if (method === "DELETE") {
          const name = decodeURIComponent(
            url.split("?")[0].split("/").pop() ?? "",
          )
          memberAliasList = memberAliasList.filter((item) => item.name !== name)
          return new Response(null, { status: 204 })
        }
        return jsonResponse(memberAliasList)
      }
      if (url.endsWith(`${API_ROOT}/organizations/me`)) {
        return jsonResponse(opts.context ?? organizationContext())
      }
      // The user picker's roster read, which names the person behind an owner
      // id. Paged, like the real endpoint: `fetchAllPaged` reads `data`, so a
      // bare array here throws rather than answering an empty roster.
      if (url.includes(`${API_ROOT}/organizations/me/members`)) {
        const members = opts.members ?? []
        return jsonResponse({ data: members, count: members.length })
      }
      if (url.includes(`${API_ROOT}/routing/policies/explain`)) {
        return jsonResponse({
          name: "fast",
          selection_reason: "default",
          is_dynamic: false,
          candidates: [
            {
              position: 1,
              instance: "openai",
              model: "gpt-5-mini",
              selection_reason: "default",
              dispatch_model: "openai:gpt-5-mini",
            },
          ],
          dropped: [
            {
              selector: "anthropic:claude-haiku-4-5",
              reason: "not_allowed",
              detail: "is not in allowed_models for this caller",
            },
          ],
          guardrails: [],
        })
      }
      if (url.includes(`${API_ROOT}/routing/status`)) {
        return jsonResponse({
          user_id: "alice",
          embedding_model: "openai:text-embedding-3-small",
          seed_count: 20,
          granularity: "trace_sticky",
          alpha: 0.3,
          k: 5,
          confidence_floor: 0,
          default_pool: { records: 6, warm: false },
          tasks: [{ task_id: "summaries", records: 21, warm: true }],
          policies: [
            {
              name: "smart",
              backend: "knn",
              candidates: ["openai:gpt-5-nano", "openai:gpt-5"],
              default_target: "openai:gpt-5",
            },
          ],
        })
      }
      if (url.includes(`${API_ROOT}/routing/preferences/rank`)) {
        return jsonResponse({
          recorded: (body as { examples: unknown[] }).examples.length,
          seed_count: 20,
          pools: [{ task_id: null, records: 7, warm: false }],
        })
      }
      if (url.includes(`${API_ROOT}/routing/policies`)) {
        if (method === "POST") {
          const scope = (body.user_id ?? null) as string | null
          if (scope !== null && refuseOnce.has(scope)) {
            refuseOnce.delete(scope)
            return new Response(
              JSON.stringify({ detail: `No room for ${scope}` }),
              { status: 409, headers: { "Content-Type": "application/json" } },
            )
          }
          if (opts.saveBody) {
            // Not `jsonResponse`, which is a 200 by construction.
            return new Response(
              JSON.stringify({ detail: opts.saveBody.detail }),
              {
                status: opts.saveBody.status,
                headers: { "Content-Type": "application/json" },
              },
            )
          }
          // An upsert, like the real endpoint: appending would put two rows under
          // one name and scope, which is a state the API cannot produce. And
          // `rename_from` moves the row rather than keying on `name`, so the old
          // name has to leave the list; a mock that only added the new one would
          // pass a test that the real API would fail.
          const row = policy(body.name, body.spec, {
            user_id: body.user_id ?? null,
          })
          const vacated: string[] = [
            row.name,
            ...(body.rename_from ? [body.rename_from as string] : []),
          ]
          list = [
            ...list.filter(
              (item) =>
                item.user_id !== row.user_id || !vacated.includes(item.name),
            ),
            row,
          ]
          return jsonResponse(row)
        }
        if (method === "DELETE") {
          if (opts.deleteBody) {
            // Not `jsonResponse`, which is a 200 by construction.
            return new Response(
              JSON.stringify({ detail: opts.deleteBody.detail }),
              {
                status: opts.deleteBody.status,
                headers: { "Content-Type": "application/json" },
              },
            )
          }
          const name = decodeURIComponent(
            url.split("?")[0].split("/").pop() ?? "",
          )
          list = list.filter((item) => item.name !== name)
          return new Response(null, { status: 204 })
        }
        return jsonResponse(list)
      }
      if (url.includes(`${API_ROOT}/aliases`)) {
        if (method === "DELETE") {
          aliasList = []
          return new Response(null, { status: 204 })
        }
        return jsonResponse(aliasList)
      }
      if (url.includes(`${API_ROOT}/tool-settings`)) {
        return jsonResponse({
          fields: [
            {
              key: "guardrails_url",
              service: "guardrails",
              type: "url",
              value: guardrailsUrl,
            },
          ],
        })
      }
      if (url.includes(`${API_ROOT}/users`)) return jsonResponse(USERS)
      if (url.includes(`${API_ROOT}/models`))
        return jsonResponse({ object: "list", data: [] })
      return jsonResponse([])
    })
  return { spy, calls }
}

// The user picker asks the organization roster what to call each owner, and that
// read is gated on the `organizations` surface, so these pages need the
// deployment context the shell always gives them.
export function renderPage(ui: ReactElement, url = "/") {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <DeploymentProvider value={bootstrap()}>
      <QueryClientProvider client={client}>{ui}</QueryClientProvider>
    </DeploymentProvider>,
    { wrapper: withRouter({ url }) },
  )
}

// The workspace a tenant admin's write is scoped to comes from the shell's
// switcher, so the provider is part of the harness for those tests. Outside it
// `useSelectedWorkspace` answers "none selected", which is the state that
// leaves an admin with the read-only page.
export const ADMIN_WORKSPACE = "44444444-4444-4444-4444-444444444444"

export function renderInWorkspace(ui: ReactElement, url = "/") {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <DeploymentProvider value={bootstrap()}>
      <QueryClientProvider client={client}>
        <SelectedWorkspaceProvider>{ui}</SelectedWorkspaceProvider>
      </QueryClientProvider>
    </DeploymentProvider>,
    { wrapper: withRouter({ url }) },
  )
}

export function adminContext(): OrganizationContext {
  return organizationContext({
    deployment_operator: false,
    role: "admin",
    workspace_memberships: [
      { workspace_id: ADMIN_WORKSPACE, name: "Alpha one", role: "admin" },
    ],
  })
}

/**
 * The page's own create action, scoped to the heading's own header.
 *
 * Scoped rather than resolved by name, because "Create policy" is on screen
 * twice once the dialog is open: this one and the submit.
 *
 * It was three until the empty state's action took its own words
 * ("Create your first policy", matching keys and budgets), and that third copy
 * is why these calls were passing by accident: they ran while the list was
 * still loading, so the empty state had not rendered and the name was
 * momentarily unique. Awaiting the empty state before any one of them turned it
 * red with "Found multiple elements". The label fix removes that copy and the
 * scoping removes the dependence on when anything renders, which is why both
 * are here. The inner query awaits as well, because the action is gated on a
 * query and so arrives after the heading.
 */
export const createTrigger = async () => {
  const heading = await screen.findByRole("heading", { name: "Routing" })
  const header = heading.closest("header")
  if (!header) throw new Error("PageIntro's header is gone")
  return within(header).findByRole("button", { name: "Create policy" })
}

export type TestUser = ReturnType<typeof userEvent.setup>

/** The two fields every policy needs, with the model popover dismissed after.
 *
 *  An open React Aria popover aria-hides the submit button, so leaving it up
 *  makes every later query in the dialog miss.
 */
export async function nameAndServe(
  user: TestUser,
  name: string,
  target = "openai:gpt-5-nano",
) {
  await user.type(screen.getByRole("textbox", { name: /policy name/i }), name)
  await user.type(screen.getByRole("combobox", { name: /^serves$/i }), target)
  await user.keyboard("{Escape}")
}

/** Switch to the scoped tab and choose people, the way an operator does.
 *
 *  Each query is typed rather than the list being opened cold: the picker's
 *  menu triggers on input, and it matches the owner id as well as the label, so
 *  "bob" reaches a person the roster calls something else entirely.
 */
export async function pickUsers(user: TestUser, queries: string[]) {
  const dialog = within(screen.getByRole("dialog"))
  await user.click(dialog.getByRole("button", { name: "Specific users" }))
  for (const query of queries) {
    // Named by the field's own label: the multi-select primitive labels its
    // search box that way rather than carrying an `aria-label` of its own.
    const search = dialog.getByLabelText("Users")
    // Cleared between queries, because the primitive keeps the query after a
    // pick on purpose (see `MultiSelect`: clearing it would refill the list
    // under the pointer). Typing the next name onto the last one matches
    // nobody.
    await user.clear(search)
    await user.type(search, query)
    await user.click(await screen.findByRole("option"))
  }
  await user.keyboard("{Escape}")
}

export async function submitDialog(user: TestUser, label = "Create policy") {
  await user.click(
    within(screen.getByRole("dialog")).getByRole("button", { name: label }),
  )
}

/** The bodies of the deployment-wide policy writes, in the order they were sent.
 *
 *  Matched on the exact path, not a prefix: `/routing/policies/explain` is a
 *  POST too, and counting one of those as a write would put a phantom row in
 *  every assertion about how many writes a submit made.
 */
export function policyWrites(
  calls: { url: string; method: string; body: unknown }[],
): { name: string; spec: PolicySpec; user_id?: string | null }[] {
  return calls
    .filter(
      (call) =>
        call.method === "POST" &&
        call.url.endsWith(`${API_ROOT}/routing/policies`),
    )
    .map(
      (call) =>
        call.body as {
          name: string
          spec: PolicySpec
          user_id?: string | null
        },
    )
}
