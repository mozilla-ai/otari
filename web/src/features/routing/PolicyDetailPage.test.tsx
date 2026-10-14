import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { PolicySpec } from "@/client"
import { PolicyDetailPage } from "@/features/routing/PolicyDetailPage"
import { API_ROOT } from "@/shared/api/client"
import {
  adminContext,
  CHAIN,
  LEARNED,
  mockApi,
  policy,
  policyWrites,
  renderInWorkspace,
  renderPage,
  WEIGHTED,
} from "@/tests/routing"

afterEach(() => {
  vi.restoreAllMocks()
})

type TestUser = ReturnType<typeof userEvent.setup>

/** Every section a policy can combine, at once. */
const COMBINED: PolicySpec = {
  select: [
    { when: { budget_used_pct: { gte: 80 } }, target: "openai:gpt-5-nano" },
    {
      router: "weighted",
      candidates: ["openai:gpt-5", "anthropic:claude-sonnet-4-5"],
      weights: { "openai:gpt-5": 70, "anthropic:claude-sonnet-4-5": 30 },
    },
    { default: "openai:gpt-5" },
  ],
  on_failure: ["anthropic:claude-haiku-4-5"],
  guardrails: [{ profile: "pii", mode: "monitor", on_unavailable: "monitor" }],
}

function renderPolicy(name: string, url = `/routing/${name}`) {
  return renderPage(<PolicyDetailPage name={name} />, url, [
    { path: "/routing/$name", element: <p>Moved</p> },
  ])
}

/** The section named `title`, for queries inside it. */
async function section(title: string) {
  return within(await screen.findByRole("region", { name: title }))
}

async function save(user: TestUser, label: string) {
  await user.click(screen.getByRole("button", { name: label }))
}

function lastWrite(calls: { url: string; method: string; body: unknown }[]) {
  const writes = policyWrites(calls)
  expect(writes.length).toBeGreaterThan(0)
  return writes[writes.length - 1]
}

describe("PolicyDetailPage", () => {
  it("shows every section of a policy and where its requests go", async () => {
    mockApi([policy("fast", CHAIN)])
    renderPolicy("fast")

    expect(
      await screen.findByRole("heading", { level: 1, name: "fast" }),
    ).toBeInTheDocument()
    for (const title of [
      "Name",
      "Main selection",
      "Fallback chain",
      "Budget tier-down",
      "Guardrails",
    ]) {
      expect(screen.getByRole("heading", { name: title })).toBeInTheDocument()
    }
    expect(
      screen.getByRole("region", { name: "Where requests go" }),
    ).toHaveTextContent(/Requests to fast go to .*If that fails/)
    expect(screen.getByRole("link", { name: "← Routing" })).toHaveAttribute(
      "href",
      "/routing",
    )
    // Nothing to save until something changes.
    expect(
      screen.getByRole("button", { name: "Save fallback chain" }),
    ).toBeDisabled()
    // The same rule the create form states, in the same words.
    expect(
      within(screen.getByRole("region", { name: "Fallback chain" })).getByText(
        /missing or rejected provider key, or a request it refuses as invalid\. Not once the answer has started streaming/,
      ),
    ).toBeInTheDocument()
  })

  it("opens a policy combining every section and saves one without touching the rest", async () => {
    const { calls } = mockApi([policy("mixed", COMBINED)])
    const user = userEvent.setup()
    renderPolicy("mixed")

    const shares = await screen.findAllByRole("textbox", { name: /share/i })
    expect(shares.map((share) => (share as HTMLInputElement).value)).toEqual([
      "70",
      "30",
    ])
    expect(
      screen.getByRole("textbox", { name: /budget used at least/i }),
    ).toHaveValue("80")
    expect(screen.getByRole("textbox", { name: /profile/i })).toHaveValue("pii")

    // Unlike a new fallback, an existing policy may keep no backup at all.
    await user.click(screen.getByRole("button", { name: "Remove backup 1" }))
    await save(user, "Save fallback chain")

    const { on_failure, ...rest } = COMBINED
    expect(on_failure).toHaveLength(1)
    expect(lastWrite(calls).spec).toEqual(rest)
    expect(lastWrite(calls).name).toBe("mixed")
  })

  it("renames a policy, sending rename_from, and follows it to its new address", async () => {
    const { calls } = mockApi([policy("fast", CHAIN)])
    const user = userEvent.setup()
    renderPolicy("fast")

    const name = await screen.findByRole("textbox", { name: /^name$/i })
    await user.clear(name)
    await user.type(name, "speedy")
    // The consequence sits next to the field, before anything is saved.
    expect(
      screen.getByText(/Callers have to send the new name/),
    ).toBeInTheDocument()
    await save(user, "Rename")

    const body = lastWrite(calls) as { name: string; rename_from?: string }
    expect(body.name).toBe("speedy")
    expect(body.rename_from).toBe("fast")
    // The rest of the policy rides on the same write.
    expect(lastWrite(calls).spec).toEqual(CHAIN)
    expect(await screen.findByText("Moved")).toBeInTheDocument()
  })

  it("omits rename_from when a section is saved under the same name", async () => {
    const { calls } = mockApi([policy("fast", CHAIN)])
    const user = userEvent.setup()
    renderPolicy("fast")

    await user.click(
      await screen.findByRole("button", { name: "Remove backup 1" }),
    )
    await save(user, "Save fallback chain")

    const body = lastWrite(calls) as { rename_from?: string; spec: PolicySpec }
    expect(body.rename_from).toBeUndefined()
    expect(body.spec).toEqual({ select: [{ default: "openai:gpt-5-mini" }] })
  })

  it("refuses a new name that carries a delimiter", async () => {
    const { calls } = mockApi([policy("fast", CHAIN)])
    const user = userEvent.setup()
    renderPolicy("fast")

    const name = await screen.findByRole("textbox", { name: /^name$/i })
    await user.clear(name)
    await user.type(name, "openai:gpt-5")

    expect(screen.getByText(/cannot contain/)).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Rename" })).toBeDisabled()
    expect(policyWrites(calls)).toHaveLength(0)
  })

  it("adds a split to a plain fallback, starting from its main model", async () => {
    const { calls } = mockApi([policy("fast", CHAIN)])
    const user = userEvent.setup()
    renderPolicy("fast")

    await user.click(
      await screen.findByRole("radio", { name: "Split traffic" }),
    )
    expect(screen.getByRole("combobox", { name: /model 1/i })).toHaveValue(
      "openai:gpt-5-mini",
    )
    await user.type(
      screen.getByRole("combobox", { name: /model 2/i }),
      "anthropic:claude-sonnet-4-5",
    )
    await user.keyboard("{Escape}")
    await save(user, "Save main selection")

    expect(lastWrite(calls).spec).toEqual({
      select: [
        {
          router: "weighted",
          candidates: ["openai:gpt-5-mini", "anthropic:claude-sonnet-4-5"],
          weights: {
            "openai:gpt-5-mini": 50,
            "anthropic:claude-sonnet-4-5": 50,
          },
        },
        { default: "openai:gpt-5-mini" },
      ],
      // The chain is another section's, written back as stored.
      on_failure: ["anthropic:claude-haiku-4-5"],
    })
  })

  it("goes back to one model, the one the pool had marked", async () => {
    const { calls } = mockApi([policy("smart", LEARNED, { is_dynamic: true })])
    const user = userEvent.setup()
    renderPolicy("smart")

    await user.click(await screen.findByRole("radio", { name: "One model" }))
    await save(user, "Save main selection")

    expect(lastWrite(calls).spec).toEqual({
      select: [{ default: "openai:gpt-5" }],
    })
  })

  it("puts the fallthrough in the pool rather than asking for it twice", async () => {
    // A stored spec that omitted its default from `candidates` still shows
    // it, because the gateway appends it.
    mockApi([
      policy("smart", {
        select: [
          { router: "knn", candidates: ["openai:gpt-5-nano"] },
          { default: "openai:gpt-5" },
        ],
      }),
    ])
    renderPolicy("smart")

    expect(
      await screen.findByRole("combobox", { name: /model 1/i }),
    ).toHaveValue("openai:gpt-5-nano")
    expect(screen.getByRole("combobox", { name: /model 2/i })).toHaveValue(
      "openai:gpt-5",
    )
    expect(
      screen.getAllByRole("radio", { name: /serves when unsure/i })[1],
    ).toBeChecked()
    expect(
      screen.queryByRole("combobox", { name: /^main model$/i }),
    ).not.toBeInTheDocument()
  })

  it("marking a different model changes the saved default, not the pool", async () => {
    const { calls } = mockApi([policy("smart", LEARNED, { is_dynamic: true })])
    const user = userEvent.setup()
    renderPolicy("smart")

    await user.click(
      (await screen.findAllByRole("radio", { name: /serves when unsure/i }))[0],
    )
    await save(user, "Save main selection")

    expect(lastWrite(calls).spec.select).toEqual([
      { router: "knn", candidates: ["openai:gpt-5-nano", "openai:gpt-5"] },
      { default: "openai:gpt-5-nano" },
    ])
  })

  it("keeps a learned policy's pool when another section is saved", async () => {
    const { calls } = mockApi([policy("smart", LEARNED, { is_dynamic: true })])
    const user = userEvent.setup()
    renderPolicy("smart")

    const chain = await section("Fallback chain")
    await user.click(chain.getByRole("button", { name: "Add a backup" }))
    await user.type(
      screen.getByRole("combobox", { name: /^backup 1$/i }),
      "anthropic:claude-haiku-4-5",
    )
    await user.keyboard("{Escape}")
    await save(user, "Save fallback chain")

    expect(lastWrite(calls).spec).toEqual({
      ...LEARNED,
      on_failure: ["anthropic:claude-haiku-4-5"],
    })
  })

  it("orders a spill-over policy's models", async () => {
    mockApi([
      policy(
        "spill",
        {
          select: [
            {
              router: "priority",
              candidates: ["openai:gpt-5-mini", "openai:gpt-5"],
            },
            { default: "openai:gpt-5" },
          ],
        },
        { is_dynamic: true },
      ),
    ])
    renderPolicy("spill")

    expect(
      await screen.findByText("Send each request to the first with room"),
    ).toBeInTheDocument()
    expect(screen.getByRole("radio", { name: "Spill over" })).toBeChecked()
    expect(
      screen.getByRole("button", { name: "Move model 1 up" }),
    ).toBeDisabled()
  })

  it("drains a model from a split without removing it", async () => {
    const { calls } = mockApi([
      policy("balanced", WEIGHTED, { is_dynamic: true }),
    ])
    const user = userEvent.setup()
    renderPolicy("balanced")

    const shares = await screen.findAllByRole("textbox", { name: /share/i })
    await user.clear(shares[1])
    await user.type(shares[1], "0")
    expect(
      screen.getByText(/No weighted traffic; still tried if another fails/),
    ).toBeInTheDocument()
    await save(user, "Save main selection")

    expect(lastWrite(calls).spec.select[0]).toEqual({
      router: "weighted",
      candidates: ["openai:gpt-5", "anthropic:claude-sonnet-4-5"],
      weights: { "openai:gpt-5": 70, "anthropic:claude-sonnet-4-5": 0 },
    })
  })

  it("keeps a fractional share typeable and refuses a non-numeric one", async () => {
    const { calls } = mockApi([
      policy("balanced", WEIGHTED, { is_dynamic: true }),
    ])
    const user = userEvent.setup()
    renderPolicy("balanced")

    const shares = await screen.findAllByRole("textbox", { name: /share/i })
    const saveButton = screen.getByRole("button", {
      name: "Save main selection",
    })
    await user.clear(shares[0])
    await user.type(shares[0], "Infinity")
    expect(
      screen.getByText(/Every share is a number of zero or more/),
    ).toBeInTheDocument()
    expect(saveButton).toBeDisabled()

    await user.clear(shares[0])
    await user.type(shares[0], "-5")
    expect(saveButton).toBeDisabled()

    await user.clear(shares[0])
    await user.type(shares[0], "7.5")
    expect(shares[0]).toHaveValue("7.5")
    await user.click(saveButton)

    expect(lastWrite(calls).spec.select[0].weights).toEqual({
      "openai:gpt-5": 7.5,
      "anthropic:claude-sonnet-4-5": 30,
    })
  })

  it("saves a loosely spelled backend back under its normal name", async () => {
    // The gateway resolves a backend on `name.strip().lower()`.
    const loose: PolicySpec = {
      select: [
        { ...WEIGHTED.select[0], router: " Weighted " },
        WEIGHTED.select[1],
      ],
    }
    const { calls } = mockApi([policy("balanced", loose, { is_dynamic: true })])
    const user = userEvent.setup()
    renderPolicy("balanced")

    expect(
      await screen.findByRole("radio", { name: "Split traffic" }),
    ).toBeChecked()
    const chain = await section("Fallback chain")
    await user.click(chain.getByRole("button", { name: "Add a backup" }))
    await user.type(
      screen.getByRole("combobox", { name: /^backup 1$/i }),
      "openai:gpt-5-mini",
    )
    await user.keyboard("{Escape}")
    await save(user, "Save fallback chain")

    expect(lastWrite(calls).spec.select[0]).toEqual(WEIGHTED.select[0])
  })

  it("will not save a split where every share is zero, or one naming a model twice", async () => {
    const { calls } = mockApi([
      policy("balanced", WEIGHTED, { is_dynamic: true }),
    ])
    const user = userEvent.setup()
    renderPolicy("balanced")

    for (const share of await screen.findAllByRole("textbox", {
      name: /share/i,
    })) {
      await user.clear(share)
      await user.type(share, "0")
    }
    expect(
      screen.getByText(/at least one model a share above zero/i),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Save main selection" }),
    ).toBeDisabled()

    const second = screen.getByRole("combobox", { name: /model 2/i })
    await user.clear(second)
    await user.type(second, "openai:gpt-5")
    await user.keyboard("{Escape}")
    expect(screen.getByText(/name each model once/i)).toBeInTheDocument()
    expect(policyWrites(calls)).toHaveLength(0)
  })

  it("adds a model to a split with no traffic until a share is set", async () => {
    mockApi([policy("balanced", WEIGHTED, { is_dynamic: true })])
    const user = userEvent.setup()
    renderPolicy("balanced")

    await user.click(await screen.findByRole("button", { name: "Add a model" }))
    expect(screen.getAllByRole("textbox", { name: /share/i })[2]).toHaveValue(
      "0",
    )
  })

  it("counts the pool and the backups against one cap", async () => {
    mockApi([
      policy("wide", {
        select: [
          {
            router: "knn",
            candidates: [
              "openai:gpt-5-nano",
              "openai:gpt-5-mini",
              "openai:gpt-5",
            ],
          },
          { default: "openai:gpt-5" },
        ],
        on_failure: ["anthropic:claude-haiku-4-5", "openai:gpt-4o"],
      }),
    ])
    renderPolicy("wide")

    const chain = await section("Fallback chain")
    expect(chain.getByRole("button", { name: "Add a backup" })).toBeDisabled()
    const main = await section("Main selection")
    expect(main.getByRole("button", { name: "Add a model" })).toBeDisabled()
    expect(
      main.getByText(/dispatches at most 5 models, counting the backups/),
    ).toBeInTheDocument()
  })

  it("adds a budget tier-down and refuses one that could never fire", async () => {
    const { calls } = mockApi([policy("fast", CHAIN)])
    const user = userEvent.setup()
    renderPolicy("fast")

    await user.click(
      await screen.findByRole("button", { name: "Add a tier-down" }),
    )
    await user.type(
      screen.getByRole("combobox", { name: /^use instead$/i }),
      "openai:gpt-5-nano",
    )
    await user.keyboard("{Escape}")
    const threshold = screen.getByRole("textbox", {
      name: /budget used at least/i,
    })
    await user.clear(threshold)
    await user.type(threshold, "100")
    expect(screen.getByText("Must be under 100.")).toBeInTheDocument()
    const saveButton = screen.getByRole("button", {
      name: "Save budget tier-down",
    })
    expect(saveButton).toBeDisabled()

    await user.clear(threshold)
    await user.type(threshold, "90")
    await user.click(saveButton)
    expect(lastWrite(calls).spec).toEqual({
      select: [
        { when: { budget_used_pct: { gte: 90 } }, target: "openai:gpt-5-nano" },
        { default: "openai:gpt-5-mini" },
      ],
      on_failure: ["anthropic:claude-haiku-4-5"],
    })
  })

  it("adds a guardrail, warning when it makes the service a hard dependency", async () => {
    const { calls } = mockApi([policy("fast", CHAIN)])
    const user = userEvent.setup()
    renderPolicy("fast")

    const guardrails = await section("Guardrails")
    expect(
      guardrails.getByRole("link", { name: "Guardrails page" }),
    ).toHaveAttribute("href", "/tools/guardrails")
    await waitFor(() =>
      expect(
        guardrails.getByRole("button", { name: "Add a guardrail" }),
      ).toBeEnabled(),
    )
    await user.click(
      guardrails.getByRole("button", { name: "Add a guardrail" }),
    )
    expect(
      screen.getByText(/rejects every request through this policy/),
    ).toBeInTheDocument()
    await user.type(
      screen.getByRole("textbox", { name: /profile/i }),
      "prompt-injection",
    )
    await save(user, "Save guardrails")

    expect(lastWrite(calls).spec.guardrails).toEqual([
      { profile: "prompt-injection", mode: "block", on_unavailable: "block" },
    ])
  })

  it("disables adding a guardrail when no guardrails service is configured", async () => {
    mockApi([policy("fast", CHAIN)], null)
    renderPolicy("fast")

    const guardrails = await section("Guardrails")
    await waitFor(() =>
      expect(
        guardrails.getByRole("button", { name: "Add a guardrail" }),
      ).toBeDisabled(),
    )
    expect(
      guardrails.getByText(/No guardrails service is configured/),
    ).toBeInTheDocument()
    expect(
      guardrails.getByRole("link", { name: "Guardrails page" }),
    ).toHaveAttribute("href", "/tools/guardrails")
  })

  it("opens the user's own row when the address names one", async () => {
    mockApi([
      policy("cheap", CHAIN),
      policy("cheap", LEARNED, { user_id: "u-bob" }),
    ])
    renderPolicy("cheap", "/routing/cheap?user=u-bob")

    expect(
      await screen.findByRole("combobox", { name: /model 1/i }),
    ).toHaveValue("openai:gpt-5-nano")
    expect(screen.getByText("u-bob")).toBeInTheDocument()
  })

  it("says so when the policy is not there", async () => {
    mockApi([policy("fast", CHAIN)])
    renderPolicy("gone")

    expect(
      await screen.findByRole("heading", { name: "Policy not found" }),
    ).toBeInTheDocument()
  })
})

describe("PolicyDetailPage for an alias", () => {
  const ALIAS = {
    name: "legacy",
    target: "openai:gpt-4o-mini",
    source: "stored",
    user_id: null,
  }

  it("keeps the name fixed and offers only the one target", async () => {
    mockApi([], "http://guardrails:8000", [ALIAS])
    renderPolicy("legacy")

    expect(
      await screen.findByText(/An alias name is its key/),
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("textbox", { name: /^name$/i }),
    ).not.toBeInTheDocument()
    expect(screen.getByText(/An alias holds one target/)).toBeInTheDocument()
    expect(
      screen.queryByRole("heading", { name: "Fallback chain" }),
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole("radio", { name: "Split traffic" }),
    ).not.toBeInTheDocument()
  })

  it("writes an alias edit back to the alias's own workspace", async () => {
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
    renderInWorkspace(
      <PolicyDetailPage name="elsewhere-alias" />,
      "/routing/elsewhere-alias",
    )

    const main = await screen.findByRole("combobox", { name: /^main model$/i })
    await user.clear(main)
    await user.type(main, "openai:gpt-5-mini")
    await user.keyboard("{Escape}")
    await save(user, "Save main selection")

    const written = calls.find(
      (call) =>
        call.method === "POST" &&
        call.url.includes(`${API_ROOT}/organizations/me/aliases`),
    )
    expect(written?.body).toMatchObject({
      name: "elsewhere-alias",
      target: "openai:gpt-5-mini",
      workspace_id: OTHER_WORKSPACE,
    })
  })
})

describe("PolicyDetailPage read-only", () => {
  it("shows a config.yml policy without a control that writes", async () => {
    mockApi()
    renderPolicy("auto")

    expect(
      await screen.findByText(/Set in config.yml, so it is read-only here/),
    ).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: /^Save/ })).toBeNull()
    expect(screen.queryByRole("textbox")).toBeNull()
    expect(
      screen.getByText(/From 80% of budget: gpt-5-nano · OpenAI/),
    ).toBeInTheDocument()
  })

  it("shows a policy it cannot edit without loss, with its spec in full", async () => {
    mockApi([
      policy("api-authored", {
        select: [
          { when: { key_id: "k-1" }, target: "openai:gpt-5-nano" },
          { default: "openai:gpt-5-mini" },
        ],
      }),
    ])
    renderPolicy("api-authored")

    expect(
      await screen.findByText(/Uses options this page cannot show yet/),
    ).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: /^Save/ })).toBeNull()
    expect(screen.getByText(/"key_id": "k-1"/)).toBeInTheDocument()
  })
})

describe("PolicyDetailPage for an organization admin", () => {
  it("writes an admin's edit back to the row's own workspace", async () => {
    const OTHER_WORKSPACE = "55555555-5555-5555-5555-555555555555"
    const { calls } = mockApi([], null, [], {
      context: adminContext(),
      memberPolicies: [
        policy("elsewhere", CHAIN, { workspace_id: OTHER_WORKSPACE }),
      ],
    })
    const user = userEvent.setup()
    renderInWorkspace(
      <PolicyDetailPage name="elsewhere" />,
      "/routing/elsewhere",
    )

    await user.click(
      await screen.findByRole("button", { name: "Remove backup 1" }),
    )
    await save(user, "Save fallback chain")

    const written = calls.find(
      (call) =>
        call.method === "POST" &&
        call.url.includes(`${API_ROOT}/organizations/me/routing-policies`),
    )
    expect(written?.body).toMatchObject({ workspace_id: OTHER_WORKSPACE })
  })
})
