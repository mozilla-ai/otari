import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactElement } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import type {
  OrganizationContext,
  OrganizationMember,
  PolicySpec,
  RoutingPolicyResponse,
} from "@/client"
import { RoutingPage } from "@/features/routing/RoutingPage"
import { API_ROOT } from "@/shared/api/client"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  bootstrap,
  organizationContext,
  organizationMember,
} from "@/tests/fixtures"
import { withRouter } from "@/tests/router"

const policy = (
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

const CHAIN: PolicySpec = {
  select: [{ default: "openai:gpt-5-mini" }],
  on_failure: ["anthropic:claude-haiku-4-5"],
}

const LEARNED: PolicySpec = {
  select: [
    { router: "knn", candidates: ["openai:gpt-5-nano", "openai:gpt-5"] },
    { default: "openai:gpt-5" },
  ],
}

const WEIGHTED: PolicySpec = {
  select: [
    {
      router: "weighted",
      candidates: ["openai:gpt-5", "anthropic:claude-sonnet-4-5"],
      weights: { "openai:gpt-5": 70, "anthropic:claude-sonnet-4-5": 30 },
    },
    { default: "openai:gpt-5" },
  ],
}

const POLICIES: RoutingPolicyResponse[] = [
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

const USERS = [
  { user_id: "alice", alias: "alice", spend: 0, is_blocked: false },
  { user_id: "u-bob", alias: "bob", spend: 0, is_blocked: false },
  { user_id: "u-carol", alias: "carol", spend: 0, is_blocked: false },
]

// One of the three has a roster row behind them, so the same list exercises both
// halves of the picker's labeling: a person the organization can name, and an
// owner id nobody named, whose id is its name.
const MEMBERS = [
  organizationMember({
    organization_member_id: "55555555-5555-5555-5555-555555555555",
    user_id: "u-bob",
    attribution_user_id: "u-bob",
    full_name: "Bob Builder",
    role: "member",
  }),
]

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  })
}

function mockApi(
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
function renderPage(ui: ReactElement, url = "/") {
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
const ADMIN_WORKSPACE = "44444444-4444-4444-4444-444444444444"

function renderInWorkspace(ui: ReactElement, url = "/") {
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

function adminContext(): OrganizationContext {
  return organizationContext({
    deployment_operator: false,
    role: "admin",
    workspace_memberships: [
      { workspace_id: ADMIN_WORKSPACE, name: "Alpha one", role: "admin" },
    ],
  })
}

afterEach(() => {
  vi.restoreAllMocks()
})

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
const createTrigger = async () => {
  const heading = await screen.findByRole("heading", { name: "Routing" })
  const header = heading.closest("header")
  if (!header) throw new Error("PageIntro's header is gone")
  return within(header).findByRole("button", { name: "Create policy" })
}

type TestUser = ReturnType<typeof userEvent.setup>

/** The two fields every policy needs, with the model popover dismissed after.
 *
 *  An open React Aria popover aria-hides the submit button, so leaving it up
 *  makes every later query in the dialog miss.
 */
async function nameAndServe(
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
async function pickUsers(user: TestUser, queries: string[]) {
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

async function submitDialog(user: TestUser, label = "Create policy") {
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
function policyWrites(
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

describe("RoutingPage", () => {
  it("lists policies with what they serve and where they come from", async () => {
    mockApi()
    renderPage(<RoutingPage />)

    const fastRow = (await screen.findByText("fast")).closest("tr")!
    // The chain is summarised rather than hidden: an operator scanning the table
    // needs to see that a fallback exists without opening the policy.
    expect(within(fastRow).getByText(/openai:gpt-5-mini/)).toBeInTheDocument()
    expect(within(fastRow).getByText(/\+1 on failure/)).toBeInTheDocument()
    expect(within(fastRow).getByText("STORED")).toBeInTheDocument()
  })

  it("marks a policy that decides per request, since it has no single target", async () => {
    mockApi()
    renderPage(<RoutingPage />)

    const autoRow = (await screen.findByText("auto")).closest("tr")!
    expect(within(autoRow).getByText("DYNAMIC")).toBeInTheDocument()
    expect(within(autoRow).getByText(/Chosen per request/)).toBeInTheDocument()
  })

  it("does not offer to edit or delete a policy that lives in config.yml", async () => {
    mockApi()
    renderPage(<RoutingPage />)

    const autoRow = (await screen.findByText("auto")).closest("tr")!
    expect(within(autoRow).getByText("set in config.yml")).toBeInTheDocument()
    expect(
      within(autoRow).queryByRole("button", { name: "Delete" }),
    ).not.toBeInTheDocument()
  })

  it("keeps the page's create action visible while the dialog is open", async () => {
    // It used to hide itself while the inline form was on the page. The form is
    // over the page now, so hiding the control that opened it would take the
    // heading's action away mid-task for no reason.
    mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const trigger = await createTrigger()
    await user.click(trigger)
    await screen.findByRole("dialog")
    expect(trigger).toBeInTheDocument()
  })

  it("offers the same dialog from the empty state", async () => {
    // The empty state's explanation is the page's onboarding and stays; what it
    // gained is the action, so a first policy does not have to be started from
    // the heading a reader has already scrolled past.
    mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    // Two of them on screen deliberately, the heading's and this one, so the
    // press is scoped to the empty state rather than picked by position, which
    // it no longer strictly needs now that its label is its own, and which is
    // kept because scoping is the right query either way.
    const empty = (
      await screen.findByRole("heading", {
        name: "No routing policies yet",
      })
    ).closest("div")!.parentElement!
    await user.click(
      within(empty).getByRole("button", { name: "Create your first policy" }),
    )
    const dialog = await screen.findByRole("dialog")
    expect(dialog).toHaveAccessibleName("New policy")
  })

  it("names the object in the title and the policy in the description when editing", async () => {
    // `title` is a string, so the old heading's `<code>` name moved into the
    // description rather than being dropped: it is the policy's identity and it
    // is what tells an operator which row they pressed Edit on.
    mockApi()
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const fastRow = (await screen.findByText("fast")).closest("tr")!
    await user.click(within(fastRow).getByRole("button", { name: "Edit" }))
    const dialog = await screen.findByRole("dialog")
    expect(dialog).toHaveAccessibleName("Edit policy")
    expect(dialog).toHaveTextContent("fast")
    expect(
      within(dialog).getByRole("button", { name: "Save" }),
    ).toBeInTheDocument()
  })

  it("guards a half-built policy against a stray Escape", async () => {
    // This form grows a fallback chain, a condition tier and a guardrail list as
    // they are asked for, so it is exactly the one where ten minutes of work sits
    // behind one keystroke.
    mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await createTrigger())
    await user.type(
      screen.getByRole("textbox", { name: /policy name/i }),
      "cheap",
    )
    await user.keyboard("{Escape}")

    const dialog = screen.getByRole("dialog")
    expect(dialog).toHaveTextContent("Unsaved changes")
    expect(
      within(dialog).getByRole("button", { name: "Keep editing" }),
    ).toBeInTheDocument()
    await user.click(
      within(dialog).getByRole("button", { name: "Keep editing" }),
    )
    expect(
      within(dialog).getByRole("button", { name: "Create policy" }),
    ).toBeInTheDocument()
  })

  it("creates a one-target policy from three fields", async () => {
    const { calls } = mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await createTrigger())
    await user.type(
      screen.getByRole("textbox", { name: /policy name/i }),
      "cheap",
    )
    await user.type(
      screen.getByRole("combobox", { name: /^serves$/i }),
      "openai:gpt-5-nano",
    )
    // Close the combobox popover, which otherwise aria-hides the submit button.
    await user.keyboard("{Escape}")
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", {
        name: "Create policy",
      }),
    )

    const post = calls.find((call) => call.method === "POST")
    expect(post).toBeDefined()
    const body = post!.body as { name: string; spec: PolicySpec }
    expect(body.name).toBe("cheap")
    // The fallthrough is explicit and last, which is what the schema requires.
    expect(body.spec.select).toEqual([{ default: "openai:gpt-5-nano" }])
    expect(body.spec.on_failure).toBeUndefined()
  })

  it("writes exactly one unscoped policy for every caller", async () => {
    // The default tab, asserted rather than assumed: "every caller" and "scoped
    // but nobody chosen" are two states, and only this one is one write.
    const { calls } = mockApi([], null, [], { members: MEMBERS })
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await createTrigger())
    await nameAndServe(user, "cheap")
    await submitDialog(user)

    const writes = policyWrites(calls)
    expect(writes).toHaveLength(1)
    expect(writes[0].user_id ?? null).toBeNull()
  })

  it("writes one policy per chosen user, each carrying its own scope", async () => {
    // A policy's key is its name plus its user, so N people are N rows of the
    // same name and spec. There is no batch endpoint; this is the whole design.
    const { calls } = mockApi([], null, [], { members: MEMBERS })
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await createTrigger())
    await nameAndServe(user, "cheap")
    await pickUsers(user, ["bob", "carol"])
    await submitDialog(user)

    const writes = policyWrites(calls)
    expect(writes).toHaveLength(2)
    expect(writes.map((write) => write.user_id)).toEqual(["u-bob", "u-carol"])
    // Same name and same spec on each: only the scope differs.
    expect(new Set(writes.map((write) => write.name))).toEqual(
      new Set(["cheap"]),
    )
    for (const write of writes) {
      expect(write.spec.select).toEqual([{ default: "openai:gpt-5-nano" }])
    }
  })

  it("shows the chosen people by name above the input, not as bare owner ids", async () => {
    mockApi([], null, [], { members: MEMBERS })
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await createTrigger())
    await nameAndServe(user, "cheap")
    await pickUsers(user, ["bob"])

    const dialog = within(screen.getByRole("dialog"))
    // The roster names this one, so the chip says the person rather than the id
    // the request plane bills to.
    expect(
      dialog.getByRole("button", { name: "Remove Bob Builder" }),
    ).toBeInTheDocument()
  })

  it("will not submit a scoped policy with nobody chosen", async () => {
    const { calls } = mockApi([], null, [], { members: MEMBERS })
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await createTrigger())
    await nameAndServe(user, "cheap")
    const dialog = within(screen.getByRole("dialog"))
    await user.click(dialog.getByRole("button", { name: "Specific users" }))

    expect(dialog.getByRole("button", { name: "Create policy" })).toBeDisabled()
    await submitDialog(user)
    expect(policyWrites(calls)).toHaveLength(0)
  })

  it("keeps the dialog open on a part-written save, naming who landed and who did not", async () => {
    // Three writes with no transaction over them, so a refusal partway leaves
    // the earlier rows in place. Saying "it failed" would leave the operator to
    // work out which of the three exist by reading the table. The refusal is in
    // the middle, so this also pins that the writes after it are still
    // attempted rather than one conflict standing in for everyone behind it.
    const { calls } = mockApi([], null, [], {
      members: MEMBERS,
      refuseFirstWriteFor: ["u-carol"],
    })
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await createTrigger())
    await nameAndServe(user, "cheap")
    await pickUsers(user, ["bob", "carol", "alice"])
    await submitDialog(user)

    const alert = await screen.findByRole("alert")
    expect(alert).toHaveTextContent("Created for Bob Builder, alice (alice).")
    // By its id, since nobody named this one, and with the refusal's own reason
    // rather than a swallowed "something went wrong".
    expect(alert).toHaveTextContent("Not created for u-carol (carol)")
    expect(alert).toHaveTextContent("No room for u-carol")
    expect(screen.getByRole("dialog")).toBeInTheDocument()

    // Pressing again rewrites nothing that already landed.
    await submitDialog(user)
    const scopes = policyWrites(calls).map((write) => write.user_id)
    expect(scopes).toEqual(["u-bob", "u-carol", "alice", "u-carol"])
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
  })

  it("fixes who a policy applies to once a write for somebody has landed", async () => {
    // Nothing here can take a policy back, so a scope that can still be edited
    // after a partial write is a way to end up with rows the operator did not
    // ask for: drop a written person from the selection and their policy lives
    // on unmentioned, or switch to every caller and it lives on ALSO outranking
    // the global one for exactly them, which is the precedence the field's own
    // description promises. The controls are withheld instead, and say why.
    mockApi([], null, [], {
      members: MEMBERS,
      refuseFirstWriteFor: ["u-carol"],
    })
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await createTrigger())
    await nameAndServe(user, "cheap")
    await pickUsers(user, ["bob", "carol"])
    await submitDialog(user)
    await screen.findByRole("alert")

    const dialog = within(screen.getByRole("dialog"))
    // Neither tab, and no picker: the selection is settled.
    expect(
      dialog.queryByRole("button", { name: "Specific users" }),
    ).not.toBeInTheDocument()
    expect(
      dialog.queryByRole("button", { name: "Every caller" }),
    ).not.toBeInTheDocument()
    expect(dialog.queryByLabelText("Users")).not.toBeInTheDocument()
    // And the reason, since a control that vanishes without one teaches
    // nothing. The route back out is naming the list, not this form.
    expect(
      dialog.getByText(/already been created/, { exact: false }),
    ).toHaveTextContent("delete it from the list")
  })

  it("writes every scope again when the payload changed after a part-written save", async () => {
    // The ids that landed are remembered against the payload they landed under,
    // so correcting the name first is not the same policy: skipping them then
    // would leave the corrected one unwritten for exactly the people the old
    // one already reached.
    const { calls } = mockApi([], null, [], {
      members: MEMBERS,
      refuseFirstWriteFor: ["u-carol"],
    })
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await createTrigger())
    await nameAndServe(user, "cheap")
    await pickUsers(user, ["bob", "carol"])
    await submitDialog(user)
    expect(await screen.findByRole("alert")).toBeInTheDocument()

    await user.type(screen.getByRole("textbox", { name: /policy name/i }), "er")
    await submitDialog(user)

    const writes = policyWrites(calls)
    expect(writes.map((write) => [write.name, write.user_id])).toEqual([
      ["cheap", "u-bob"],
      ["cheap", "u-carol"],
      ["cheaper", "u-bob"],
      ["cheaper", "u-carol"],
    ])
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
  })

  it("lists a row per user for a policy written to several of them", async () => {
    mockApi(
      [
        policy("cheap", CHAIN, { user_id: "u-bob" }),
        policy("cheap", CHAIN, { user_id: "u-carol" }),
      ],
      null,
      [],
      { members: MEMBERS },
    )
    renderPage(<RoutingPage />)

    // Two rows under one name: the scope is half the row's identity, so they do
    // not collapse into one.
    expect(
      await screen.findAllByRole("rowheader", { name: "cheap" }),
    ).toHaveLength(2)
  })

  it("keeps the failure chain and guardrails out of the way until asked for", async () => {
    mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await createTrigger())
    // Naming one model must stay a short task, so neither section is present yet.
    expect(screen.queryByText("If that fails, try")).not.toBeInTheDocument()
    expect(screen.queryByText("Always check")).not.toBeInTheDocument()

    await user.click(
      screen.getByRole("button", { name: /Add a fallback chain/ }),
    )
    expect(screen.getByText("If that fails, try")).toBeInTheDocument()
    // Adding another one belongs inside the section it extends, not in the row of
    // links that start a section.
    const section = screen.getByText("If that fails, try").closest("div")!
      .parentElement!
    expect(
      within(section).getByRole("button", { name: /Another fallback/ }),
    ).toBeInTheDocument()
  })

  it("disables the guardrails affordance when no guardrails service is configured", async () => {
    mockApi([], null)
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await createTrigger())
    const add = await screen.findByRole("button", { name: /Add guardrails/ })

    // Disabled, and never silently: the reason and the route to fixing it sit next
    // to the control as text, so it works on touch and for a screen reader.
    expect(add).toBeDisabled()
    expect(
      screen.getByText(/No guardrails service is configured/),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("link", { name: /Tools & Guardrails/ }),
    ).toHaveAttribute("href", "/tools")
  })

  it("refuses a policy name that would shadow a real model selector", async () => {
    mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await createTrigger())
    await user.type(
      screen.getByRole("textbox", { name: /policy name/i }),
      "openai:gpt-4o",
    )
    await user.type(
      screen.getByRole("combobox", { name: /^serves$/i }),
      "openai:gpt-5-nano",
    )
    await user.keyboard("{Escape}")

    expect(screen.getByText(/cannot contain/)).toBeInTheDocument()
    expect(
      within(screen.getByRole("dialog")).getByRole("button", {
        name: "Create policy",
      }),
    ).toBeDisabled()
  })

  it("warns when a guardrail makes the guardrails service a hard dependency", async () => {
    mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await createTrigger())
    await user.click(screen.getByRole("button", { name: /Add guardrails/ }))

    // block + block is the honest default, and its cost has to be visible where
    // the choice is made: an outage then refuses every request through the policy.
    expect(
      screen.getByText(/rejects every request through this policy/),
    ).toBeInTheDocument()
  })

  it("refuses a tier-down threshold that could never fire", async () => {
    mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await createTrigger())
    await user.type(
      screen.getByRole("textbox", { name: /policy name/i }),
      "thrifty",
    )
    await user.type(
      screen.getByRole("combobox", { name: /^serves$/i }),
      "openai:gpt-5-mini",
    )
    await user.keyboard("{Escape}")
    await user.click(screen.getByRole("button", { name: /Tier down/ }))

    const threshold = screen.getByRole("textbox", {
      name: /budget used at least/i,
    })
    await user.clear(threshold)
    await user.type(threshold, "100")

    // The budget gate refuses the request before selection at 100%, so such a rule
    // is dead config. Saying so here beats a 400 from the server.
    expect(screen.getByText("Must be under 100.")).toBeInTheDocument()
    expect(
      within(screen.getByRole("dialog")).getByRole("button", {
        name: "Create policy",
      }),
    ).toBeDisabled()
  })

  it("renames a policy through the name field, sending rename_from", async () => {
    // The name is the key, so an edit that changes it has to say which row it moves.
    // Posting the new name alone would create a second policy and leave the old one
    // serving callers.
    const { calls } = mockApi([policy("fast", CHAIN)])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("fast")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Edit" }))

    const nameField = screen.getByRole("textbox", { name: /policy name/i })
    await user.clear(nameField)
    await user.type(nameField, "speedy")
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", { name: "Save" }),
    )

    const post = calls.find(
      (call) =>
        call.method === "POST" &&
        call.url.includes(`${API_ROOT}/routing/policies`),
    )
    const body = post!.body as {
      name: string
      rename_from?: string
      spec: PolicySpec
    }
    expect(body.name).toBe("speedy")
    expect(body.rename_from).toBe("fast")
    // The rest of the policy rides along on the same write, so a rename cannot land
    // half-applied.
    expect(body.spec.on_failure).toEqual(["anthropic:claude-haiku-4-5"])
    expect(await screen.findByText("speedy")).toBeInTheDocument()
    expect(screen.queryByText("fast")).not.toBeInTheDocument()
  })

  it("omits rename_from when an edit leaves the name alone", async () => {
    // Sending it unchanged would be harmless server-side, but a plain spec edit
    // reading as a rename in the audit log is not.
    const { calls } = mockApi([policy("fast", CHAIN)])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("fast")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Edit" }))
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", { name: "Save" }),
    )

    const post = calls.find(
      (call) =>
        call.method === "POST" &&
        call.url.includes(`${API_ROOT}/routing/policies`),
    )
    expect((post!.body as { rename_from?: string }).rename_from).toBeUndefined()
  })

  it("says what a pending rename will do before it is saved", async () => {
    mockApi([policy("fast", CHAIN)])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("fast")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Edit" }))

    const nameField = screen.getByRole("textbox", { name: /policy name/i })
    await user.clear(nameField)
    await user.type(nameField, "speedy")

    // Renaming changes what callers must send and splits historical usage, so the
    // consequence belongs next to the field rather than in a release note.
    expect(
      screen.getByText(/Callers have to send the new name/),
    ).toBeInTheDocument()
    expect(
      screen.getByText(/usage already recorded keeps the old one/),
    ).toBeInTheDocument()
  })

  it("refuses a renamed policy whose new name carries a delimiter", async () => {
    // Same rule as a create: ":" or "/" would shadow a real model selector.
    const { calls } = mockApi([policy("fast", CHAIN)])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("fast")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Edit" }))

    const nameField = screen.getByRole("textbox", { name: /policy name/i })
    await user.clear(nameField)
    await user.type(nameField, "openai:gpt-5")

    expect(screen.getByText(/cannot contain/)).toBeInTheDocument()
    expect(
      within(screen.getByRole("dialog")).getByRole("button", { name: "Save" }),
    ).toBeDisabled()
    expect(calls.some((call) => call.method === "POST")).toBe(false)
  })

  it("keeps an alias name fixed, since the alias API cannot rename", async () => {
    mockApi([], "http://guardrails:8000", [
      {
        name: "gpt",
        target: "openai:gpt-5-mini",
        source: "stored",
        user_id: null,
      },
    ])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("gpt")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Edit" }))

    expect(
      screen.queryByRole("textbox", { name: /policy name/i }),
    ).not.toBeInTheDocument()
    expect(screen.getByText(/An alias name is its key/)).toBeInTheDocument()
  })

  it("does not offer Edit for a policy the form would silently truncate", async () => {
    // The editor models only a `budget_used_pct.gte` condition. Offering Edit on a
    // policy built through the API with anything else would drop it on save, which
    // is worse than not offering the button.
    mockApi([
      policy("api-authored", {
        select: [
          { when: { key_id: "k-1" }, target: "openai:gpt-5-nano" },
          { default: "openai:gpt-5-mini" },
        ],
      }),
    ])
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("api-authored")).closest("tr")!
    expect(
      within(row).queryByRole("button", { name: "Edit" }),
    ).not.toBeInTheDocument()
    expect(within(row).getByText(/cannot show yet/)).toBeInTheDocument()
    // Delete stays available: removing a policy is never lossy.
    expect(
      within(row).getByRole("button", { name: "Delete" }),
    ).toBeInTheDocument()
  })

  it("lists stored aliases alongside policies, so nothing is unmanageable", async () => {
    // Aliases were folded into this page. If they were not listed here they would
    // be invisible and undeletable from the dashboard, since the Aliases tab is gone.
    mockApi([], "http://guardrails:8000", [
      {
        name: "legacy",
        target: "openai:gpt-4o-mini",
        source: "stored",
        user_id: null,
      },
    ])
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("legacy")).closest("tr")!
    expect(within(row).getByText("openai:gpt-4o-mini")).toBeInTheDocument()
    expect(within(row).getByText("alias")).toBeInTheDocument()
    expect(
      within(row).getByRole("button", { name: "Delete" }),
    ).toBeInTheDocument()
  })

  it("names the policy in a confirm dialog before deleting it", async () => {
    // otari-ai#2110: the confirmation used to arm inside the row, where it read
    // as part of the table rather than as a decision. It is a modal now, and
    // the policy it is about has to be named in it: the row is behind the
    // backdrop, so the name on the row is no longer the operator's reference.
    const { calls } = mockApi([policy("fast", CHAIN)])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("fast")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Delete" }))

    const dialog = await screen.findByRole("alertdialog")
    expect(within(dialog).getByText(/^fast stops resolving/)).toBeVisible()
    // Nothing is sent by opening it.
    expect(calls.some((call) => call.method === "DELETE")).toBe(false)

    await user.click(
      within(dialog).getByRole("button", { name: "Delete policy" }),
    )

    const deletes = calls.filter((call) => call.method === "DELETE")
    expect(deletes).toHaveLength(1)
    expect(deletes[0].url).toContain(`${API_ROOT}/routing/policies/fast`)
  })

  it("deletes nothing when the confirm dialog is cancelled", async () => {
    const { calls } = mockApi([policy("fast", CHAIN)])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("fast")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Delete" }))
    const dialog = await screen.findByRole("alertdialog")
    await user.click(within(dialog).getByRole("button", { name: "Cancel" }))

    expect(calls.some((call) => call.method === "DELETE")).toBe(false)
    expect(screen.getByText("fast")).toBeInTheDocument()
  })

  it("returns focus to the page's action when the empty state's dialog closes", async () => {
    // Creating the first policy fills the table, so the empty state unmounts and
    // the node react-aria stored for focus restoration is gone: focus resets to
    // `document.body` and the next Tab starts at the top of the document. The
    // page's own trigger is where it lands instead.
    mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const empty = (
      await screen.findByRole("heading", { name: "No routing policies yet" })
    ).closest("div")!.parentElement!
    await user.click(
      within(empty).getByRole("button", { name: "Create your first policy" }),
    )
    await user.type(
      screen.getByRole("textbox", { name: /policy name/i }),
      "cheap",
    )
    await user.type(
      screen.getByRole("combobox", { name: /^serves$/i }),
      "openai:gpt-5-nano",
    )
    await user.keyboard("{Escape}")
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", {
        name: "Create policy",
      }),
    )

    const trigger = await createTrigger()
    await waitFor(() => expect(trigger).toHaveFocus())
  })

  it("does not fetch the form's own reads until the dialog opens", async () => {
    // The form stays mounted while closed so the frame can play its exit, which
    // puts its queries on the page unless they are gated: the roster and the
    // tool settings are the form's, not the table's.
    const { calls } = mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await screen.findByText("No routing policies yet")
    const formReads = () =>
      calls.filter(
        (call) =>
          call.url.includes(`${API_ROOT}/users`) ||
          call.url.includes(`${API_ROOT}/tool-settings`),
      )
    expect(formReads()).toHaveLength(0)

    await user.click(await createTrigger())
    await waitFor(() => expect(formReads().length).toBeGreaterThan(0))
  })

  it("offers a fresh draft on each open of the create dialog", async () => {
    // Reset on the way in, not on the way out: the frame keeps its content
    // while it animates out, so clearing on close blanks the body in front of
    // the operator. The page keys the form on an open counter instead.
    mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await createTrigger())
    await user.type(
      screen.getByRole("textbox", { name: /policy name/i }),
      "half-typed",
    )

    // Out through the guard, which is the only way out of a dirty form.
    await user.keyboard("{Escape}")
    await user.click(screen.getByRole("button", { name: "Discard" }))

    await user.click(await createTrigger())
    expect(screen.getByRole("textbox", { name: /policy name/i })).toHaveValue(
      "",
    )
  })

  it("reports a refused save inside the dialog, leaving the form filled", async () => {
    // The failure this guards against is the silent one: the mutation refuses,
    // the dialog stays, and nothing on screen says why. Its delete equivalent
    // is below; a page-level banner is no use here, because the operator is
    // looking at the modal and a message behind the backdrop is unread.
    mockApi([], "http://guardrails:8000", [], {
      saveBody: { status: 400, detail: "cheap already names an alias" },
    })
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await createTrigger())
    await user.type(
      screen.getByRole("textbox", { name: /policy name/i }),
      "cheap",
    )
    await user.type(
      screen.getByRole("combobox", { name: /^serves$/i }),
      "openai:gpt-5-nano",
    )
    await user.keyboard("{Escape}")
    const dialog = screen.getByRole("dialog")
    await user.click(
      within(dialog).getByRole("button", { name: "Create policy" }),
    )

    expect(
      await within(dialog).findByText(/already names an alias/),
    ).toBeVisible()
    // Still open with the work intact, so the operator can correct the name
    // rather than retyping the policy.
    expect(screen.getByRole("dialog")).toBeInTheDocument()
    expect(screen.getByRole("textbox", { name: /policy name/i })).toHaveValue(
      "cheap",
    )
  })

  it("reports a refused delete inside the dialog, leaving the row", async () => {
    // The page banner no longer carries this: the operator is looking at the
    // modal, and a message behind the backdrop is a message they do not read.
    mockApi([policy("fast", CHAIN)], "http://guardrails:8000", [], {
      deleteBody: { status: 409, detail: "fast is referenced by an alias" },
    })
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("fast")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Delete" }))
    const dialog = await screen.findByRole("alertdialog")
    await user.click(
      within(dialog).getByRole("button", { name: "Delete policy" }),
    )

    expect(
      await within(dialog).findByText(/referenced by an alias/),
    ).toBeVisible()
    // Still open, so the operator can retry or back out rather than being
    // returned to a table that looks unchanged for no stated reason.
    expect(screen.getByRole("alertdialog")).toBeInTheDocument()
    expect(screen.getByText("fast")).toBeInTheDocument()
  })

  it("does not greet the next row's confirm with the last row's refusal", async () => {
    // The mutation holds its error until the next call, and the dialog reads it,
    // so without clearing it on close the second row opens already reporting a
    // failure that was about the first.
    mockApi([policy("fast", CHAIN), policy("smart", LEARNED)], undefined, [], {
      deleteBody: { status: 409, detail: "fast is referenced by an alias" },
    })
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const fast = (await screen.findByText("fast")).closest("tr")!
    await user.click(within(fast).getByRole("button", { name: "Delete" }))
    const first = await screen.findByRole("alertdialog")
    await user.click(
      within(first).getByRole("button", { name: "Delete policy" }),
    )
    await within(first).findByText(/referenced by an alias/)
    await user.click(within(first).getByRole("button", { name: "Cancel" }))

    const smart = screen.getByText("smart").closest("tr")!
    await user.click(within(smart).getByRole("button", { name: "Delete" }))

    const second = await screen.findByRole("alertdialog")
    expect(within(second).getByText(/^smart stops resolving/)).toBeVisible()
    expect(within(second).queryByText(/referenced by an alias/)).toBeNull()
  })

  it("deletes an alias through the alias endpoint, not the policy one", async () => {
    const { calls } = mockApi([], "http://guardrails:8000", [
      {
        name: "legacy",
        target: "openai:gpt-4o-mini",
        source: "stored",
        user_id: null,
      },
    ])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("legacy")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Delete" }))
    await user.click(
      within(await screen.findByRole("alertdialog")).getByRole("button", {
        name: "Delete alias",
      }),
    )

    const deletes = calls.filter((call) => call.method === "DELETE")
    expect(deletes).toHaveLength(1)
    // An alias still lives in model_aliases; deleting it as a policy would 404 and
    // leave the row in place.
    expect(deletes[0].url).toContain(`${API_ROOT}/aliases/legacy`)
  })

  it("will not let an alias grow options an alias cannot hold", async () => {
    mockApi([], "http://guardrails:8000", [
      {
        name: "legacy",
        target: "openai:gpt-4o-mini",
        source: "stored",
        user_id: null,
      },
    ])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("legacy")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Edit" }))
    await user.click(
      await screen.findByRole("button", { name: /Add a fallback chain/ }),
    )

    // Saving it as a policy would leave the alias row behind under the same name,
    // and the API refuses that collision, so the form says so instead of failing.
    expect(screen.getByText(/An alias holds one target/)).toBeInTheDocument()
    expect(
      within(screen.getByRole("dialog")).getByRole("button", { name: "Save" }),
    ).toBeDisabled()
  })

  it("summarises a learned policy by its pool rather than as an opaque dynamic row", async () => {
    // "Chosen per request" is true of a tier-down too. What an operator needs to
    // see here is that a router picks between named models, and which one serves
    // when it declines.
    mockApi([policy("smart", LEARNED, { is_dynamic: true })])
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("smart")).closest("tr")!
    expect(
      within(row).getByText(/Learned . 2 candidates, openai:gpt-5 by default/),
    ).toBeInTheDocument()
  })

  it("puts the fallback in the pool rather than asking for it twice", async () => {
    // The fallback is always one of the models the router may choose, so the form
    // shows one list with the safe one marked. A stored spec that omitted its default
    // target from `candidates` still shows it, because the gateway appends it.
    mockApi([
      policy("smart", {
        select: [
          { router: "knn", candidates: ["openai:gpt-5-nano"] },
          { default: "openai:gpt-5" },
        ],
      }),
    ])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("smart")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Edit" }))

    // Both models in one list, and the default target marked.
    expect(screen.getByRole("combobox", { name: /model 1/i })).toHaveValue(
      "openai:gpt-5-nano",
    )
    expect(screen.getByRole("combobox", { name: /model 2/i })).toHaveValue(
      "openai:gpt-5",
    )
    const marks = screen.getAllByRole("radio", { name: /serves when unsure/i })
    expect(marks[1]).toBeChecked()
    // ...and no second field asking for the same model again.
    expect(
      screen.queryByRole("combobox", { name: /^serves$/i }),
    ).not.toBeInTheDocument()
  })

  it("marking a different model as the fallback changes the saved default", async () => {
    const { calls } = mockApi([policy("smart", LEARNED, { is_dynamic: true })])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("smart")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Edit" }))
    await user.click(
      screen.getAllByRole("radio", { name: /serves when unsure/i })[0],
    )
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", { name: "Save" }),
    )

    const post = calls.find(
      (call) =>
        call.method === "POST" &&
        call.url.includes(`${API_ROOT}/routing/policies`),
    )
    const spec = (post!.body as { spec: PolicySpec }).spec
    expect(spec.select[1]).toEqual({ default: "openai:gpt-5-nano" })
    // The pool is unchanged: marking a fallback is not reordering.
    expect(spec.select[0]).toEqual({
      router: "knn",
      candidates: ["openai:gpt-5-nano", "openai:gpt-5"],
    })
  })

  it("edits a learned policy without losing its candidate pool", async () => {
    // A router entry the form can represent must be editable: showing it read-only
    // would mean the only way to change a candidate is the API.
    const { calls } = mockApi([policy("smart", LEARNED, { is_dynamic: true })])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("smart")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Edit" }))
    expect(screen.getByRole("combobox", { name: /model 1/i })).toHaveValue(
      "openai:gpt-5-nano",
    )
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", { name: "Save" }),
    )

    const post = calls.find(
      (call) =>
        call.method === "POST" &&
        call.url.includes(`${API_ROOT}/routing/policies`),
    )
    const spec = (post!.body as { spec: PolicySpec }).spec
    expect(spec.select[0]).toEqual({
      router: "knn",
      candidates: ["openai:gpt-5-nano", "openai:gpt-5"],
    })
    expect(spec.select[1]).toEqual({ default: "openai:gpt-5" })
  })

  it("will not save a pool of one, which is not a routing decision", async () => {
    const { calls } = mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await createTrigger())
    await user.type(
      screen.getByRole("textbox", { name: /policy name/i }),
      "smart",
    )
    await user.type(
      screen.getByRole("combobox", { name: /^serves$/i }),
      "openai:gpt-5",
    )
    await user.keyboard("{Escape}")
    await user.click(screen.getByRole("button", { name: /let a router pick/i }))
    // The pool is seeded with the policy's target plus one empty row, so removing
    // the empty row leaves a single candidate.
    await user.click(screen.getAllByRole("button", { name: "Remove" })[1])

    expect(screen.getByText(/at least two models/i)).toBeInTheDocument()
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", {
        name: "Create policy",
      }),
    )
    expect(calls.some((call) => call.method === "POST")).toBe(false)
  })

  it("summarises a weighted policy by its split, not by its pool size", async () => {
    // Two provider:model strings do not fit the cell, and the shares are what tells
    // one weighted policy from another at a glance.
    mockApi([policy("balanced", WEIGHTED, { is_dynamic: true })])
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("balanced")).closest("tr")!
    expect(within(row).getByText("WEIGHTED")).toBeInTheDocument()
    expect(
      within(row).getByText(/70% \/ 30% across 2 models/),
    ).toBeInTheDocument()
  })

  it("creates a weighted policy from the split control", async () => {
    const { calls } = mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await createTrigger())
    await user.type(
      screen.getByRole("textbox", { name: /policy name/i }),
      "balanced",
    )
    await user.type(
      screen.getByRole("combobox", { name: /^serves$/i }),
      "openai:gpt-5",
    )
    await user.keyboard("{Escape}")
    await user.click(
      screen.getByRole("button", { name: /split traffic across providers/i }),
    )

    // Seeded as an even split of the policy's own target plus one empty row, so the
    // operator names the second provider and skews the shares.
    await user.type(
      screen.getByRole("combobox", { name: /model 2/i }),
      "anthropic:claude-sonnet-4-5",
    )
    await user.keyboard("{Escape}")
    const shares = screen.getAllByRole("textbox", { name: /share/i })
    await user.clear(shares[0])
    await user.type(shares[0], "70")
    await user.clear(shares[1])
    await user.type(shares[1], "30")
    // Relative weights are hard to read, so the form says what they come to.
    expect(screen.getByText("70% of requests")).toBeInTheDocument()
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", {
        name: "Create policy",
      }),
    )

    const post = calls.find(
      (call) =>
        call.method === "POST" &&
        call.url.includes(`${API_ROOT}/routing/policies`),
    )
    const spec = (post!.body as { spec: PolicySpec }).spec
    expect(spec.select[0]).toEqual({
      router: "weighted",
      candidates: ["openai:gpt-5", "anthropic:claude-sonnet-4-5"],
      weights: { "openai:gpt-5": 70, "anthropic:claude-sonnet-4-5": 30 },
    })
    expect(spec.select[1]).toEqual({ default: "openai:gpt-5" })
  })

  it("edits a weighted policy without losing its split", async () => {
    const { calls } = mockApi([
      policy("balanced", WEIGHTED, { is_dynamic: true }),
    ])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("balanced")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Edit" }))

    const shares = screen.getAllByRole("textbox", { name: /share/i })
    expect(shares[0]).toHaveValue("70")
    expect(shares[1]).toHaveValue("30")
    // Drain the second provider without deleting it, which is what a zero share is
    // for. The form has to say the model is still there, or a zero reads as removal.
    await user.clear(shares[1])
    await user.type(shares[1], "0")
    expect(
      screen.getByText(/No weighted traffic; still tried if another fails/),
    ).toBeInTheDocument()
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", { name: "Save" }),
    )

    const post = calls.find(
      (call) =>
        call.method === "POST" &&
        call.url.includes(`${API_ROOT}/routing/policies`),
    )
    const spec = (post!.body as { spec: PolicySpec }).spec
    expect(spec.select[0]).toEqual({
      router: "weighted",
      candidates: ["openai:gpt-5", "anthropic:claude-sonnet-4-5"],
      weights: { "openai:gpt-5": 70, "anthropic:claude-sonnet-4-5": 0 },
    })
  })

  it("keeps a fractional share typeable and refuses a non-numeric one", async () => {
    // The field holds what was typed, so a decimal point survives the keystroke that
    // follows it. "Infinity" and a negative parse but are refused, matching the API's
    // finite, non-negative rule rather than being coerced to something else on save.
    const { calls } = mockApi([
      policy("balanced", WEIGHTED, { is_dynamic: true }),
    ])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("balanced")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Edit" }))

    const shares = screen.getAllByRole("textbox", { name: /share/i })
    await user.clear(shares[0])
    await user.type(shares[0], "Infinity")
    expect(
      screen.getByText(/Every share is a number of zero or more/),
    ).toBeInTheDocument()
    expect(
      within(screen.getByRole("dialog")).getByRole("button", { name: "Save" }),
    ).toBeDisabled()

    await user.clear(shares[0])
    await user.type(shares[0], "-5")
    expect(
      within(screen.getByRole("dialog")).getByRole("button", { name: "Save" }),
    ).toBeDisabled()

    // The decimal has to survive the round trip, not only the keystroke: a field
    // that renders "7.5" but posts 7 would be the same bug one layer down.
    await user.clear(shares[0])
    await user.type(shares[0], "7.5")
    expect(shares[0]).toHaveValue("7.5")
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", { name: "Save" }),
    )

    const post = calls.find(
      (call) =>
        call.method === "POST" &&
        call.url.includes(`${API_ROOT}/routing/policies`),
    )
    const spec = (post!.body as { spec: PolicySpec }).spec
    expect(spec.select[0].weights).toEqual({
      "openai:gpt-5": 7.5,
      "anthropic:claude-sonnet-4-5": 30,
    })
  })

  it("edits a weighted policy whose backend name is spelled loosely", async () => {
    // The gateway resolves a backend on `name.strip().lower()`, so " Weighted " is a
    // working policy. Reading it as an unknown backend would show it read-only and
    // label it wrong on a page that otherwise offers to edit it.
    const loose: PolicySpec = {
      select: [
        {
          router: " Weighted ",
          candidates: ["openai:gpt-5", "anthropic:claude-sonnet-4-5"],
          weights: { "openai:gpt-5": 70, "anthropic:claude-sonnet-4-5": 30 },
        },
        { default: "openai:gpt-5" },
      ],
    }
    const { calls } = mockApi([policy("balanced", loose, { is_dynamic: true })])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("balanced")).closest("tr")!
    expect(within(row).getByText("WEIGHTED")).toBeInTheDocument()
    await user.click(within(row).getByRole("button", { name: "Edit" }))

    // Loading it as weighted is half the claim; saving it back unchanged is the
    // other half. The spelling is normalized on the way out, which is what the
    // gateway would have resolved it to anyway.
    const shares = screen.getAllByRole("textbox", { name: /share/i })
    expect(shares[0]).toHaveValue("70")
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", { name: "Save" }),
    )

    const post = calls.find(
      (call) =>
        call.method === "POST" &&
        call.url.includes(`${API_ROOT}/routing/policies`),
    )
    const spec = (post!.body as { spec: PolicySpec }).spec
    expect(spec.select[0]).toEqual({
      router: "weighted",
      candidates: ["openai:gpt-5", "anthropic:claude-sonnet-4-5"],
      weights: { "openai:gpt-5": 70, "anthropic:claude-sonnet-4-5": 30 },
    })
  })

  it("will not save a split where every share is zero", async () => {
    // It would select nothing and the policy would always serve its marked model,
    // which is a load balancer that balances nothing. The API refuses it too.
    const { calls } = mockApi([
      policy("balanced", WEIGHTED, { is_dynamic: true }),
    ])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("balanced")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Edit" }))
    for (const share of screen.getAllByRole("textbox", { name: /share/i })) {
      await user.clear(share)
      await user.type(share, "0")
    }

    expect(
      screen.getByText(/at least one model a share above zero/i),
    ).toBeInTheDocument()
    expect(
      within(screen.getByRole("dialog")).getByRole("button", { name: "Save" }),
    ).toBeDisabled()
    expect(calls.some((call) => call.method === "POST")).toBe(false)
  })

  it("adds a model to a split with no traffic until a share is set", async () => {
    // Adding a provider must not silently move traffic onto it.
    mockApi([policy("balanced", WEIGHTED, { is_dynamic: true })])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("balanced")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Edit" }))
    await user.click(screen.getByRole("button", { name: "+ Another model" }))

    const shares = screen.getAllByRole("textbox", { name: /share/i })
    expect(shares[2]).toHaveValue("0")
  })

  it("will not save a split that names the same model twice", async () => {
    // Two rows collapse to one key in the weight map, so the split saved would not be
    // the split shown (and the API refuses a repeated candidate regardless).
    const { calls } = mockApi([
      policy("balanced", WEIGHTED, { is_dynamic: true }),
    ])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("balanced")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Edit" }))
    const second = screen.getByRole("combobox", { name: /model 2/i })
    await user.clear(second)
    await user.type(second, "openai:gpt-5")
    await user.keyboard("{Escape}")

    expect(screen.getByText(/name each model once/i)).toBeInTheDocument()
    expect(
      within(screen.getByRole("dialog")).getByRole("button", { name: "Save" }),
    ).toBeDisabled()
    expect(calls.some((call) => call.method === "POST")).toBe(false)
  })

  it("names a router backend it does not know without claiming it learns", async () => {
    // Only "knn" learns. Labelling every other backend "Learned" would make the table
    // lie about the first backend added after this line was written.
    mockApi([
      policy("future", {
        select: [
          {
            router: "cheapest",
            candidates: ["openai:gpt-5-nano", "openai:gpt-5"],
          },
          { default: "openai:gpt-5" },
        ],
      }),
    ])
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("future")).closest("tr")!
    expect(within(row).getByText("ROUTED")).toBeInTheDocument()
    expect(within(row).queryByText("Learned")).not.toBeInTheDocument()
  })

  it("does not offer the examples panel for a weighted policy, which learns nothing", async () => {
    mockApi([policy("balanced", WEIGHTED, { is_dynamic: true })])
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("balanced")).closest("tr")!
    expect(
      within(row).queryByRole("button", { name: "Examples" }),
    ).not.toBeInTheDocument()
    expect(
      within(row).getByRole("button", { name: "Edit" }),
    ).toBeInTheDocument()
  })

  it("does not offer Edit for a weighted policy with no split to show", async () => {
    // The form would have to invent the shares, and the API refuses such a spec
    // anyway, so this one is only reachable as an older or hand-written document.
    mockApi([
      policy("legacy", {
        select: [
          {
            router: "weighted",
            candidates: ["openai:gpt-5", "anthropic:claude-sonnet-4-5"],
          },
          { default: "openai:gpt-5" },
        ],
      }),
    ])
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("legacy")).closest("tr")!
    expect(
      within(row).queryByRole("button", { name: "Edit" }),
    ).not.toBeInTheDocument()
  })

  it("does not offer Edit for a router backend the form cannot write", async () => {
    mockApi([
      policy("future", {
        select: [
          {
            router: "cheapest",
            candidates: ["openai:gpt-5-nano", "openai:gpt-5"],
          },
          { default: "openai:gpt-5" },
        ],
      }),
    ])
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("future")).closest("tr")!
    expect(
      within(row).queryByRole("button", { name: "Edit" }),
    ).not.toBeInTheDocument()
  })

  it("offers the examples panel only on a policy that actually uses a router", async () => {
    // Readiness is a per-policy question, so it belongs on the row like Edit does
    // rather than in a panel that is always on the page.
    mockApi([
      policy("smart", LEARNED, { is_dynamic: true }),
      policy("fast", CHAIN),
    ])
    renderPage(<RoutingPage />)

    const learnedRow = (await screen.findByText("smart")).closest("tr")!
    const plainRow = (await screen.findByText("fast")).closest("tr")!
    expect(
      within(learnedRow).getByRole("button", { name: "Examples" }),
    ).toBeInTheDocument()
    expect(
      within(plainRow).queryByRole("button", { name: "Examples" }),
    ).not.toBeInTheDocument()
    // Nothing about learned routing is on the page until asked for.
    expect(screen.queryByText(/Whose memory/)).not.toBeInTheDocument()
  })

  it("offers the examples panel for a config.yml policy, which cannot be edited", async () => {
    // Reading readiness is safe for a policy this page cannot change, and without it
    // a config-defined learned policy would be entirely opaque here.
    mockApi([policy("smart", LEARNED, { is_dynamic: true, source: "config" })])
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("smart")).closest("tr")!
    expect(
      within(row).getByRole("button", { name: "Examples" }),
    ).toBeInTheDocument()
    expect(
      within(row).queryByRole("button", { name: "Edit" }),
    ).not.toBeInTheDocument()
    expect(within(row).getByText("set in config.yml")).toBeInTheDocument()
  })

  it("names the pool and what serves when the router declines", async () => {
    mockApi([policy("smart", LEARNED, { is_dynamic: true })])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("smart")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Examples" }))

    expect(
      await screen.findByText(/ranks openai:gpt-5-nano, openai:gpt-5/),
    ).toBeInTheDocument()
    expect(screen.getByText(/serves whenever it declines/)).toBeInTheDocument()
    // The honest empty state: no user picked yet, so no warmth claim.
    expect(screen.getByText(/Pick a user to see how warm/)).toBeInTheDocument()
  })

  it("reports each pool's warmth for the chosen user, since memory is per user", async () => {
    mockApi([policy("smart", LEARNED, { is_dynamic: true })])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("smart")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Examples" }))
    await user.type(
      screen.getByRole("combobox", { name: /whose memory/i }),
      "alice",
    )
    await user.keyboard("{Escape}")

    expect(await screen.findByText("6 / 20 examples")).toBeInTheDocument()
    expect(screen.getByText("WARMING UP")).toBeInTheDocument()
    // A task partition warms on its own, so it gets its own line.
    expect(screen.getByText("summaries")).toBeInTheDocument()
    expect(screen.getByText("21 / 20 examples")).toBeInTheDocument()
    expect(screen.getByText("ROUTING")).toBeInTheDocument()
  })

  it("says where examples come from instead of offering to collect them", async () => {
    // Recording examples is an API job in this release. The panel has to say so, or
    // an operator reads "0 examples" as a bug with no next step.
    mockApi([policy("smart", LEARNED, { is_dynamic: true })])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("smart")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Examples" }))

    expect(
      await screen.findByText(/POST \/api\/v1\/routing\/preferences\/rank/),
    ).toBeInTheDocument()
    expect(screen.getByRole("link", { name: /teach it/i })).toBeInTheDocument()
    // No write affordance anywhere in it.
    expect(
      screen.queryByRole("button", { name: /ask all/i }),
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: /record these scores/i }),
    ).not.toBeInTheDocument()
  })

  it("does not ask whose memory for a user-scoped policy", async () => {
    // A policy scoped to one user can only use that user's memory, so asking would
    // be a question with one answer, and a wrong answer would be accepted.
    mockApi([policy("smart", LEARNED, { is_dynamic: true, user_id: "alice" })])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("smart")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Examples" }))

    expect(
      screen.queryByRole("combobox", { name: /whose memory/i }),
    ).not.toBeInTheDocument()
    expect(await screen.findByText("6 / 20 examples")).toBeInTheDocument()
  })

  it("will not author a policy that dispatches more models than the server allows", async () => {
    // The cap counts the routed pool plus the fallback chain. Authoring past it and
    // finding out via a 400 on Save is the form lying about its own rules.
    const { calls } = mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await createTrigger())
    await user.type(
      screen.getByRole("textbox", { name: /policy name/i }),
      "wide",
    )
    await user.type(
      screen.getByRole("combobox", { name: /^serves$/i }),
      "openai:gpt-5",
    )
    await user.keyboard("{Escape}")
    await user.click(screen.getByRole("button", { name: /let a router pick/i }))
    // Seeded with 2 candidates; add three more to reach the cap of 5.
    for (let i = 0; i < 3; i += 1) {
      await user.click(screen.getByRole("button", { name: "+ Another model" }))
    }

    expect(
      screen.getByRole("button", { name: "+ Another model" }),
    ).toBeDisabled()
    expect(screen.getByText(/dispatches at most 5 models/i)).toBeInTheDocument()
    expect(calls.some((call) => call.method === "POST")).toBe(false)
  })

  it("refuses to edit a policy whose router sits before its conditions", async () => {
    // Selection is order-sensitive server-side and the form always re-emits
    // conditions first, so editing this spec would silently change what it does.
    mockApi([
      policy("api-authored", {
        select: [
          { router: "knn", candidates: ["openai:gpt-5-nano", "openai:gpt-5"] },
          {
            when: { budget_used_pct: { gte: 80 } },
            target: "openai:gpt-5-nano",
          },
          { default: "openai:gpt-5" },
        ],
      }),
    ])
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("api-authored")).closest("tr")!
    expect(
      within(row).queryByRole("button", { name: "Edit" }),
    ).not.toBeInTheDocument()
    expect(within(row).getByText(/cannot show yet/)).toBeInTheDocument()
    // Reading its readiness is still fine.
    expect(
      within(row).getByRole("button", { name: "Examples" }),
    ).toBeInTheDocument()
  })

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
