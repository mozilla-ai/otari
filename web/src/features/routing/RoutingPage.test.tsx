import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactElement } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import type {
  OrganizationContext,
  PolicySpec,
  RoutingPolicyResponse,
} from "@/client"
import { RoutingPage } from "@/features/routing/RoutingPage"
import { organizationContext } from "@/tests/fixtures"
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
    // What `/v1/organizations/me/routing-policies` answers, for member tests.
    memberPolicies?: RoutingPolicyResponse[]
  } = {},
) {
  let list = [...policies]
  let aliasList = [...aliases]
  const calls: { url: string; method: string; body: unknown }[] = []
  const spy = vi
    .spyOn(globalThis, "fetch")
    .mockImplementation(async (input, init) => {
      const url = String(input)
      const method = (init?.method ?? "GET").toUpperCase()
      const body =
        init?.body === undefined ? undefined : JSON.parse(String(init.body))
      calls.push({ url, method, body })

      if (url.includes("/v1/organizations/me/routing-policies")) {
        return jsonResponse(opts.memberPolicies ?? [])
      }
      if (url.endsWith("/v1/organizations/me")) {
        return jsonResponse(opts.context ?? organizationContext())
      }
      if (url.includes("/v1/routing/policies/explain")) {
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
      if (url.includes("/v1/routing/status")) {
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
      if (url.includes("/v1/routing/preferences/rank")) {
        return jsonResponse({
          recorded: (body as { examples: unknown[] }).examples.length,
          seed_count: 20,
          pools: [{ task_id: null, records: 7, warm: false }],
        })
      }
      if (url.includes("/v1/routing/policies")) {
        if (method === "POST") {
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
          const name = decodeURIComponent(
            url.split("?")[0].split("/").pop() ?? "",
          )
          list = list.filter((item) => item.name !== name)
          return new Response(null, { status: 204 })
        }
        return jsonResponse(list)
      }
      if (url.includes("/v1/aliases")) {
        if (method === "DELETE") {
          aliasList = []
          return new Response(null, { status: 204 })
        }
        return jsonResponse(aliasList)
      }
      if (url.includes("/v1/tool-settings")) {
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
      if (url.includes("/v1/users")) return jsonResponse(USERS)
      if (url.includes("/v1/models"))
        return jsonResponse({ object: "list", data: [] })
      return jsonResponse([])
    })
  return { spy, calls }
}

function renderPage(ui: ReactElement, url = "/") {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>{ui}</QueryClientProvider>,
    { wrapper: withRouter({ url }) },
  )
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe("RoutingPage", () => {
  it("lists policies with what they serve and where they come from", async () => {
    mockApi()
    renderPage(<RoutingPage />)

    const fastRow = (await screen.findByText("fast")).closest("tr")!
    // The chain is summarised rather than hidden: an operator scanning the table
    // needs to see that a fallback exists without opening the policy.
    expect(within(fastRow).getByText(/openai:gpt-5-mini/)).toBeInTheDocument()
    expect(within(fastRow).getByText(/\+1 on failure/)).toBeInTheDocument()
    expect(within(fastRow).getByText("stored")).toBeInTheDocument()
  })

  it("marks a policy that decides per request, since it has no single target", async () => {
    mockApi()
    renderPage(<RoutingPage />)

    const autoRow = (await screen.findByText("auto")).closest("tr")!
    expect(within(autoRow).getByText("Dynamic")).toBeInTheDocument()
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

  it("creates a one-target policy from three fields", async () => {
    const { calls } = mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await screen.findByRole("button", { name: "New policy" }))
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
    await user.click(screen.getByRole("button", { name: "Create policy" }))

    const post = calls.find((call) => call.method === "POST")
    expect(post).toBeDefined()
    const body = post!.body as { name: string; spec: PolicySpec }
    expect(body.name).toBe("cheap")
    // The fallthrough is explicit and last, which is what the schema requires.
    expect(body.spec.select).toEqual([{ default: "openai:gpt-5-nano" }])
    expect(body.spec.on_failure).toBeUndefined()
  })

  it("keeps the failure chain and guardrails out of the way until asked for", async () => {
    mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await screen.findByRole("button", { name: "New policy" }))
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

    await user.click(await screen.findByRole("button", { name: "New policy" }))
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

    await user.click(await screen.findByRole("button", { name: "New policy" }))
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
    expect(screen.getByRole("button", { name: "Create policy" })).toBeDisabled()
  })

  it("warns when a guardrail makes the guardrails service a hard dependency", async () => {
    mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await screen.findByRole("button", { name: "New policy" }))
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

    await user.click(await screen.findByRole("button", { name: "New policy" }))
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
    expect(screen.getByRole("button", { name: "Create policy" })).toBeDisabled()
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
    await user.click(screen.getByRole("button", { name: "Save" }))

    const post = calls.find(
      (call) =>
        call.method === "POST" && call.url.includes("/v1/routing/policies"),
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
    await user.click(screen.getByRole("button", { name: "Save" }))

    const post = calls.find(
      (call) =>
        call.method === "POST" && call.url.includes("/v1/routing/policies"),
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
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled()
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
    await user.click(within(row).getByRole("button", { name: "Confirm" }))

    const deletes = calls.filter((call) => call.method === "DELETE")
    expect(deletes).toHaveLength(1)
    // An alias still lives in model_aliases; deleting it as a policy would 404 and
    // leave the row in place.
    expect(deletes[0].url).toContain("/v1/aliases/legacy")
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
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled()
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
    await user.click(screen.getByRole("button", { name: "Save" }))

    const post = calls.find(
      (call) =>
        call.method === "POST" && call.url.includes("/v1/routing/policies"),
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
    await user.click(screen.getByRole("button", { name: "Save" }))

    const post = calls.find(
      (call) =>
        call.method === "POST" && call.url.includes("/v1/routing/policies"),
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

    await user.click(await screen.findByRole("button", { name: "New policy" }))
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
    await user.click(screen.getByRole("button", { name: "Create policy" }))
    expect(calls.some((call) => call.method === "POST")).toBe(false)
  })

  it("summarises a weighted policy by its split, not by its pool size", async () => {
    // Two provider:model strings do not fit the cell, and the shares are what tells
    // one weighted policy from another at a glance.
    mockApi([policy("balanced", WEIGHTED, { is_dynamic: true })])
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("balanced")).closest("tr")!
    expect(within(row).getByText("Weighted")).toBeInTheDocument()
    expect(
      within(row).getByText(/70% \/ 30% across 2 models/),
    ).toBeInTheDocument()
  })

  it("creates a weighted policy from the split control", async () => {
    const { calls } = mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await screen.findByRole("button", { name: "New policy" }))
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
    await user.click(screen.getByRole("button", { name: "Create policy" }))

    const post = calls.find(
      (call) =>
        call.method === "POST" && call.url.includes("/v1/routing/policies"),
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
    await user.click(screen.getByRole("button", { name: "Save" }))

    const post = calls.find(
      (call) =>
        call.method === "POST" && call.url.includes("/v1/routing/policies"),
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
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled()

    await user.clear(shares[0])
    await user.type(shares[0], "-5")
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled()

    // The decimal has to survive the round trip, not only the keystroke: a field
    // that renders "7.5" but posts 7 would be the same bug one layer down.
    await user.clear(shares[0])
    await user.type(shares[0], "7.5")
    expect(shares[0]).toHaveValue("7.5")
    await user.click(screen.getByRole("button", { name: "Save" }))

    const post = calls.find(
      (call) =>
        call.method === "POST" && call.url.includes("/v1/routing/policies"),
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
    expect(within(row).getByText("Weighted")).toBeInTheDocument()
    await user.click(within(row).getByRole("button", { name: "Edit" }))

    // Loading it as weighted is half the claim; saving it back unchanged is the
    // other half. The spelling is normalized on the way out, which is what the
    // gateway would have resolved it to anyway.
    const shares = screen.getAllByRole("textbox", { name: /share/i })
    expect(shares[0]).toHaveValue("70")
    await user.click(screen.getByRole("button", { name: "Save" }))

    const post = calls.find(
      (call) =>
        call.method === "POST" && call.url.includes("/v1/routing/policies"),
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
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled()
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
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled()
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
    expect(within(row).getByText("Routed")).toBeInTheDocument()
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
    expect(screen.getByText("warming up")).toBeInTheDocument()
    // A task partition warms on its own, so it gets its own line.
    expect(screen.getByText("summaries")).toBeInTheDocument()
    expect(screen.getByText("21 / 20 examples")).toBeInTheDocument()
    expect(screen.getByText("routing")).toBeInTheDocument()
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
      await screen.findByText(/POST \/v1\/routing\/preferences\/rank/),
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

    await user.click(await screen.findByRole("button", { name: "New policy" }))
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
      <QueryClientProvider client={client}>
        <RoutingPage />
      </QueryClientProvider>,
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
    expect(urls.some((url) => url.endsWith("/v1/routing/policies"))).toBe(false)
    expect(urls.some((url) => url.includes("/v1/aliases"))).toBe(false)
    expect(urls.some((url) => url.includes("/v1/tool-settings"))).toBe(false)
    expect(urls.some((url) => url.includes("/v1/users"))).toBe(false)
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
      await screen.findByText(/once a deployment operator defines them/),
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
      await screen.findByText(/once a deployment operator defines them/),
    ).toBeInTheDocument()
    // The operator's numbered getting-started walkthrough is not for them.
    expect(screen.queryByText(/Create a policy/)).not.toBeInTheDocument()
  })
})
