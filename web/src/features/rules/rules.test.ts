import { describe, expect, it } from "vitest"
import type {
  OrganizationMember,
  OrganizationSpendCeiling,
  WorkspaceBudgetDefault,
} from "@/client"
import {
  accessWords,
  buildMemberRules,
  buildRules,
  defaultRuleSort,
  emptyCopy,
  peopleBandWords,
  peopleShown,
  type Rule,
  type RulePerson,
  type RuleSources,
  ruleActions,
  ruleColumns,
  ruleIdForCeiling,
  rulesOnTab,
  ruleTabOf,
  ruleTabs,
  sortRules,
  stopDefaultWords,
  type TeamDefault,
  type WorkspaceDefaultRow,
} from "@/features/rules/rules"
import {
  apiKey,
  organizationGuardrail,
  organizationMember,
  organizationSpendCeiling,
  workspace,
} from "@/tests/fixtures"

const NOW = Date.parse("2026-11-11T12:00:00Z")

function ceiling(
  id: string,
  overrides: Partial<OrganizationSpendCeiling>,
): OrganizationSpendCeiling {
  return organizationSpendCeiling({
    id,
    scope_type: "organization",
    scope_id: "org-1",
    provider_key_id: null,
    budget_id: `budget-${id}`,
    name: null,
    max_budget: 100,
    current_spend: 10,
    budget_duration_sec: 604_800,
    reset_alignment: null,
    period_end: "2026-11-15T00:00:00Z",
    created_at: "2026-11-01T00:00:00Z",
    ...overrides,
  })
}

function member(
  id: string,
  places: {
    workspaceMemberId: string
    team?: string
    workspaceId?: string
    ceiling?: { budget_id: string; max_budget: number }
  }[],
): OrganizationMember {
  return organizationMember({
    organization_member_id: `om-${id}`,
    user_id: `login-${id}`,
    attribution_user_id: null,
    full_name: id,
    email: `${id}@example.com`,
    role: "member",
    status: "active",
    created_at: "2026-11-01T00:00:00Z",
    workspaces: places.map((place) => ({
      workspace_id: place.workspaceId ?? "ws-platform",
      workspace_name: place.team ?? "Platform",
      workspace_member_id: place.workspaceMemberId,
      role: "member",
      ceiling: place.ceiling
        ? { id: `c-${place.workspaceMemberId}`, ...place.ceiling }
        : null,
    })),
  })
}

function policy(
  workspaceId: string,
  budgetId: string,
  overrides: Partial<WorkspaceBudgetDefault> = {},
): WorkspaceDefaultRow {
  return {
    workspaceId,
    default: {
      id: `policy-${budgetId}`,
      workspace_id: workspaceId,
      budget_id: budgetId,
      provider_key_id: null,
      name: null,
      max_budget: 1,
      budget_duration_sec: null,
      reset_alignment: "calendar_month",
      token_limit: null,
      request_limit: null,
      created_at: "2026-11-01T00:00:00Z",
      updated_at: "2026-11-01T00:00:00Z",
      ...overrides,
    },
  }
}

function sources(overrides: Partial<RuleSources>): RuleSources {
  return {
    ceilings: [],
    budgets: [],
    members: [],
    keys: [],
    workspaces: [
      workspace({ id: "ws-platform", name: "Platform" }),
      workspace({ id: "ws-data", name: "Data" }),
    ],
    guardrails: [],
    defaults: [],
    organizationName: "Northwind Labs",
    now: NOW,
    ...overrides,
  }
}

const byId = (rules: Rule[]) => new Map(rules.map((rule) => [rule.id, rule]))

describe("buildRules, as an owner or admin reads it", () => {
  const perPerson = sources({
    ceilings: [
      ceiling("a", {
        scope_type: "workspace_member",
        scope_id: "wm-wei",
        budget_id: "each",
        max_budget: 40,
        current_spend: 25,
      }),
      ceiling("b", {
        scope_type: "workspace_member",
        scope_id: "wm-zainab",
        budget_id: "each",
        max_budget: 40,
        current_spend: 2,
      }),
      // Someone who left: their placement is on no roster row.
      ceiling("c", {
        scope_type: "workspace_member",
        scope_id: "wm-gone",
        budget_id: "each",
        max_budget: 40,
        current_spend: 30,
      }),
    ],
    budgets: [{ budget_id: "each", name: "Each Platform member, weekly" }],
    members: [
      member("wei", [{ workspaceMemberId: "wm-wei" }]),
      member("zainab", [{ workspaceMemberId: "wm-zainab" }]),
    ],
  })

  it("makes one per-person rule of a budget's ceilings, counting only the people still here", () => {
    const [rule] = buildRules(perPerson)
    expect(rule).toMatchObject({
      id: "spend-each-each",
      name: "Each Platform member, weekly",
      scope: { kind: "each", teams: ["Platform"], count: 2 },
      subject: { userIds: ["login-wei", "login-zainab"] },
      spend: { max: 40, per: "a week", isEach: true, spent: 27 },
      formerCount: 1,
    })
    expect(rule?.spend?.used).toBeCloseTo(25 / 40)
    expect(rule?.people?.map((one) => one.name)).toEqual(["wei", "zainab"])
    // Each person's ceiling follows the workspace default, set elsewhere.
    expect(rule?.spend?.target).toBeUndefined()
    expect(rule?.teamDefault).toBeUndefined()
    expect(
      ruleIdForCeiling({
        id: "a",
        scope_type: "workspace_member",
        budget_id: "each",
      }),
    ).toBe("spend-each-each")
  })

  it("compares each of a person's placements with its own ceiling, never their sum", () => {
    const twoPlaces = sources({
      ceilings: [
        ceiling("p", {
          scope_type: "workspace_member",
          scope_id: "wm-ethan-platform",
          budget_id: "each",
          max_budget: 10,
          current_spend: 6,
        }),
        ceiling("d", {
          scope_type: "workspace_member",
          scope_id: "wm-ethan-data",
          budget_id: "each",
          max_budget: 10,
          current_spend: 6,
        }),
      ],
      budgets: [{ budget_id: "each", name: "Each member, weekly" }],
      members: [
        member("ethan", [
          { workspaceMemberId: "wm-ethan-platform", team: "Platform" },
          {
            workspaceMemberId: "wm-ethan-data",
            team: "Data",
            workspaceId: "ws-data",
          },
        ]),
      ],
    })
    const [rule] = buildRules(twoPlaces)
    // Together they spent $12; neither workspace refused them at $6 of $10.
    expect(rule?.spend?.spent).toBe(12)
    expect(rule?.spend?.used).toBeCloseTo(0.6)
    expect(rule?.spend?.usedSpent).toBe(6)
    expect(rule?.spend?.usedMax).toBe(10)
    expect(rule?.people?.[0]).toMatchObject({
      spent: 12,
      peak: { spent: 6, max: 10 },
    })
  })

  it("shows the placement that is closest to its own ceiling", () => {
    const [rule] = buildRules(
      sources({
        ceilings: [
          ceiling("p", {
            scope_type: "workspace_member",
            scope_id: "wm-platform",
            budget_id: "each",
            max_budget: 100,
            current_spend: 50,
          }),
          ceiling("d", {
            scope_type: "workspace_member",
            scope_id: "wm-data",
            budget_id: "each",
            max_budget: 10,
            current_spend: 9,
          }),
        ],
        members: [
          member("ethan", [
            { workspaceMemberId: "wm-platform", team: "Platform" },
            {
              workspaceMemberId: "wm-data",
              team: "Data",
              workspaceId: "ws-data",
            },
          ]),
        ],
      }),
    )
    expect(rule?.spend?.used).toBeCloseTo(0.9)
    expect(rule?.spend?.usedSpent).toBe(9)
    expect(rule?.spend?.usedMax).toBe(10)
  })

  it("marks a per-person rule that is a workspace's current default", () => {
    const [rule] = buildRules({
      ...perPerson,
      defaults: [
        policy("ws-platform", "each", {
          max_budget: 40,
          reset_alignment: "calendar_week",
        }),
      ],
    })
    expect(rule?.teamDefault).toEqual({
      workspaceId: "ws-platform",
      team: "Platform",
      policyId: "policy-each",
      budgetId: "each",
      max: 40,
      // What the limit dialog says it resets on.
      period: { reset_alignment: "calendar_week", budget_duration_sec: null },
    })
  })

  it("keeps a $0 limit, which refuses every request, as fully used once spent", () => {
    const rules = byId(
      buildRules(
        sources({
          ceilings: [
            ceiling("zero", { max_budget: 0, current_spend: 0 }),
            ceiling("spent", {
              scope_type: "api_token",
              scope_id: "key-1",
              max_budget: 0,
              current_spend: 3,
            }),
            ceiling("each-zero", {
              scope_type: "workspace_member",
              scope_id: "wm-wei",
              budget_id: "zero-each",
              max_budget: 0,
              current_spend: 0,
            }),
          ],
          members: [member("wei", [{ workspaceMemberId: "wm-wei" }])],
        }),
      ),
    )
    expect(rules.get("spend-zero")?.spend).toMatchObject({ max: 0, used: 0 })
    expect(rules.get("spend-spent")?.spend).toMatchObject({
      max: 0,
      used: 1,
      usedSpent: 3,
    })
    expect(rules.get("spend-each-zero-each")?.spend).toMatchObject({
      max: 0,
      used: 0,
      isEach: true,
    })
  })

  it("reads a period that ended before now as nothing spent, as Home does", () => {
    const rules = buildRules(
      sources({
        ceilings: [
          ceiling("ended", {
            current_spend: 27.87,
            period_end: "2026-11-08T21:41:14Z",
          }),
          ceiling("running", { current_spend: 21 }),
        ],
      }),
    )
    const rule = byId(rules)
    expect(rule.get("spend-ended")?.spend).toMatchObject({ spent: 0, used: 0 })
    expect(rule.get("spend-running")?.spend).toMatchObject({ spent: 21 })
    // The dialog's own figure reads the same.
    expect(rule.get("spend-ended")?.spend?.target?.own?.spent).toBe(0)
  })

  it("says who each ceiling reaches, and lets the four scopes it can set be changed", () => {
    const rules = byId(
      buildRules(
        sources({
          ceilings: [
            ceiling("org", { name: "Northwind, monthly" }),
            ceiling("team", {
              scope_type: "workspace",
              scope_id: "ws-platform",
            }),
            ceiling("person", { scope_type: "org_member", scope_id: "om-wei" }),
            ceiling("key", { scope_type: "api_token", scope_id: "key-1" }),
            ceiling("hidden", { scope_type: "api_token", scope_id: "key-2" }),
            ceiling("narrow", {
              scope_type: "org_member",
              scope_id: "om-wei",
              provider_key_id: "pk-1",
            }),
          ],
          budgets: [{ budget_id: "budget-team", name: "Platform, monthly" }],
          members: [member("wei", [{ workspaceMemberId: "wm-wei" }])],
          keys: [apiKey({ id: "key-1", key_name: "nightly-evals" })],
        }),
      ),
    )
    expect(rules.get("spend-org")).toMatchObject({
      name: "Northwind, monthly",
      scope: { kind: "everyone", organization: "Northwind Labs" },
      subject: { everyone: true },
      spend: { target: { scope: "organization", name: "Northwind Labs" } },
    })
    expect(rules.get("spend-team")).toMatchObject({
      name: "Platform, monthly",
      scope: { kind: "team", team: "Platform" },
      subject: { workspaceId: "ws-platform" },
    })
    expect(rules.get("spend-person")).toMatchObject({
      scope: { kind: "person", name: "wei", avatarUserId: "login-wei" },
      subject: { userIds: ["login-wei"] },
      spend: { target: { scope: "org_member", scopeId: "om-wei" } },
    })
    expect(rules.get("spend-key")).toMatchObject({
      scope: { kind: "key", name: "nightly-evals" },
      subject: { apiKeyIds: ["key-1"] },
    })
    // An owner who does not run the deployment cannot read someone else's key.
    expect(rules.get("spend-hidden")).toMatchObject({
      scope: { kind: "key", name: undefined },
    })
    // A limit narrowed to one provider key is not the one the dialog sets.
    expect(rules.get("spend-narrow")?.spend?.target).toBeUndefined()
  })

  it("lists key allow-lists as access rules and guardrails as data rules, and no per-person allow-list", () => {
    const withAccess: OrganizationMember = {
      ...member("wei", [{ workspaceMemberId: "wm-wei" }]),
      // Filled for the deployment's operator only; a hosted owner never gets it.
      attribution: {
        spend: 0,
        reserved: 0,
        blocked: false,
        allowed_models: ["anthropic:claude-haiku-4-5"],
        allowed_tools: null,
      },
    }
    const rules = buildRules(
      sources({
        members: [withAccess],
        keys: [
          apiKey({
            id: "key-1",
            key_name: "ci",
            is_active: true,
            allowed_models: ["openai:gpt-5.4-mini"],
            allowed_tools: null,
          }),
          apiKey({
            id: "key-expired",
            key_name: "expired",
            is_active: true,
            expires_at: "2026-11-10T00:00:00Z",
            allowed_models: ["openai:gpt-5.4-mini"],
          }),
          apiKey({
            id: "key-off",
            key_name: "off",
            is_active: false,
            allowed_models: ["openai:gpt-5.4-mini"],
          }),
        ],
        guardrails: [
          organizationGuardrail({
            id: "g1",
            profile: "pii",
            mode: "block",
            enabled: true,
            applies_to_all_workspaces: true,
            workspace_ids: [],
          }),
          organizationGuardrail({
            id: "g2",
            profile: "toxicity",
            mode: "monitor",
            enabled: true,
            applies_to_all_workspaces: false,
            workspace_ids: ["ws-platform"],
          }),
          organizationGuardrail({
            id: "g3",
            profile: "secrets",
            mode: "block",
            enabled: false,
            applies_to_all_workspaces: true,
            workspace_ids: [],
          }),
        ],
      }),
    )
    expect(rules.map((rule) => [rule.type, rule.scope, rule.mode])).toEqual([
      ["access", { kind: "key", name: "ci" }, "Enforce"],
      ["data", { kind: "everyone" }, "Enforce"],
      ["data", { kind: "everyone" }, "Off"],
      ["data", { kind: "teams", teams: ["Platform"] }, "Watch"],
    ])
    expect(rules[0]?.access).toEqual({
      models: ["openai:gpt-5.4-mini"],
      tools: null,
    })
  })
})

describe("buildMemberRules, as a member or viewer reads it", () => {
  it("keeps a $0 limit on the roster", () => {
    const [rule] = buildMemberRules({
      now: NOW,
      members: [
        member("wei", [
          {
            workspaceMemberId: "wm-wei",
            ceiling: { budget_id: "zero", max_budget: 0 },
          },
        ]),
      ],
      keys: [],
      defaults: [],
    })
    expect(rule).toMatchObject({ id: "spend-each-zero", spend: { max: 0 } })
  })

  const platform = { budget_id: "each-platform", max_budget: 40 }
  const data = { budget_id: "each-data", max_budget: 25 }
  const roster = [
    member("max", [{ workspaceMemberId: "wm-max", ceiling: platform }]),
    member("wei", [{ workspaceMemberId: "wm-wei", ceiling: platform }]),
    member("daniel", [
      {
        workspaceMemberId: "wm-daniel",
        team: "Data",
        workspaceId: "ws-data",
        ceiling: data,
      },
    ]),
    member("hana", [{ workspaceMemberId: "wm-hana", team: "Mobile" }]),
  ]

  it("makes one rule per budget on the roster, with no spend and no stored record", () => {
    const rules = byId(
      buildMemberRules({
        now: NOW,
        members: roster,
        keys: [],
        defaults: [],
        callerMemberId: "om-max",
      }),
    )
    expect([...rules.keys()]).toEqual([
      "spend-each-each-data",
      "spend-each-each-platform",
    ])
    expect(rules.get("spend-each-each-platform")).toMatchObject({
      name: "Per person in Platform, $40.00",
      scope: { kind: "each", count: 2, includesYou: true },
      spend: { max: 40, per: "", isEach: true },
      subject: { userIds: ["login-max", "login-wei"] },
    })
    expect(rules.get("spend-each-each-data")).toMatchObject({
      name: "Per person in Data, $25.00",
      scope: { kind: "each", count: 1, includesYou: false },
    })
    for (const rule of rules.values()) {
      expect(rule.spend?.spent).toBeUndefined()
      expect(rule.spend?.used).toBeUndefined()
      expect(rule.spend?.target).toBeUndefined()
      expect(rule.definition).toBeUndefined()
      expect(rule.people?.every((one) => one.spent === undefined)).toBe(true)
    }
  })

  it("names and times a rule that is the default of a workspace the caller reads", () => {
    const [rule] = buildMemberRules({
      now: NOW,
      members: roster.slice(0, 2),
      keys: [],
      defaults: [
        policy("ws-platform", "each-platform", {
          name: "Each member of Platform, monthly",
          max_budget: 40,
        }),
      ],
      callerMemberId: "om-max",
    })
    expect(rule).toMatchObject({
      name: "Each member of Platform, monthly",
      spend: { per: "a month" },
      teamDefault: {
        workspaceId: "ws-platform",
        policyId: "policy-each-platform",
      },
    })
  })

  it("lists the caller's own keys' allow-lists", () => {
    const rules = buildMemberRules({
      now: NOW,
      members: [],
      keys: [
        apiKey({
          id: "k",
          key_name: "mine",
          is_active: true,
          allowed_models: null,
          allowed_tools: ["web_search"],
        }),
      ],
      defaults: [],
    })
    expect(rules.map((rule) => [rule.id, rule.scope])).toEqual([
      ["access-key-k", { kind: "key", name: "mine" }],
    ])
  })

  it("does not list the allow-list of a key past its expiry", () => {
    const keys = [
      apiKey({
        id: "old",
        key_name: "old",
        is_active: true,
        expires_at: "2026-11-10T00:00:00Z",
        allowed_models: ["gpt-4o"],
        allowed_tools: null,
      }),
      apiKey({
        id: "new",
        key_name: "new",
        is_active: true,
        expires_at: "2026-12-10T00:00:00Z",
        allowed_models: ["gpt-4o"],
        allowed_tools: null,
      }),
    ]
    const rules = buildMemberRules({
      now: NOW,
      members: [],
      keys,
      defaults: [],
    })
    expect(rules.map((rule) => rule.id)).toEqual(["access-key-new"])
  })
})

/** A rule with what a test names; the rest reaches everyone, enforced. */
function rule(
  fields: Pick<Rule, "id" | "name" | "type"> & Partial<Rule>,
): Rule {
  return {
    scope: { kind: "everyone" },
    mode: "Enforce",
    subject: {},
    ...fields,
  }
}

describe("what each role may do and see", () => {
  const single = rule({
    id: "spend-1",
    name: "Northwind, monthly",
    type: "spend",
    spend: {
      max: 700,
      per: "a month",
      isEach: false,
      spent: 150,
      used: 150 / 700,
      target: { scope: "organization", scopeId: "org-1", name: "N", own: null },
    },
  })
  const dataDefault: TeamDefault = {
    workspaceId: "ws-data",
    team: "Data",
    policyId: "p",
    budgetId: "d",
    max: 25,
    period: { reset_alignment: "calendar_week", budget_duration_sec: null },
  }
  const teamDefault = rule({
    id: "spend-each-d",
    name: "Each Data member, weekly",
    type: "spend",
    spend: { max: 25, per: "a week", isEach: true, spent: 0, used: 0 },
    teamDefault: dataDefault,
  })
  const oldBudget = rule({
    id: "spend-each-o",
    name: "Each Platform member, weekly",
    type: "spend",
    spend: { max: 40, per: "a week", isEach: true, spent: 1, used: 0.5 },
  })
  const none = () => false
  const dataAdmin = (id: string) => id === "ws-data"

  it("gives an owner or admin the limit writes and a workspace's default", () => {
    const owner = { isManager: true, canManageWorkspace: () => true }
    expect(ruleActions(single, owner)).toEqual(["change-limit", "remove-limit"])
    expect(ruleActions(teamDefault, owner)).toEqual(["change-default"])
    expect(ruleActions(oldBudget, owner)).toEqual([])
  })

  it("gives a workspace's own admin only the stop on that workspace's default", () => {
    const teamAdmin = { isManager: false, canManageWorkspace: dataAdmin }
    expect(ruleActions(teamDefault, teamAdmin)).toEqual(["stop-default"])
    expect(ruleActions(single, teamAdmin)).toEqual([])
    expect(
      ruleActions(
        {
          ...teamDefault,
          teamDefault: { ...dataDefault, workspaceId: "ws-platform" },
        },
        teamAdmin,
      ),
    ).toEqual([])
  })

  it("gives a member or viewer nothing to do", () => {
    const reader = { isManager: false, canManageWorkspace: none }
    for (const rule of [single, teamDefault, oldBudget]) {
      expect(ruleActions(rule, reader)).toEqual([])
    }
  })

  it("shows spend columns and the Data tab to owners and admins only", () => {
    expect(ruleTabs(true)).toEqual(["all", "spend", "access", "data"])
    expect(ruleTabs(false)).toEqual(["all", "spend", "access"])
    expect(ruleTabOf("data", false)).toBe("all")
    expect(ruleTabOf("spend", false)).toBe("spend")
    expect(ruleTabOf("actions", true)).toBe("all")
    expect(ruleColumns("all", true, true)).toEqual([
      "name",
      "type",
      "scope",
      "limit",
      "spent",
      "used",
      "mode",
      "actions",
    ])
    expect(ruleColumns("spend", true, true)).not.toContain("type")
    expect(ruleColumns("all", false, false)).toEqual([
      "name",
      "type",
      "scope",
      "limit",
      "mode",
    ])
    expect(ruleColumns("all", false, true)).toContain("actions")
    expect(ruleColumns("access", false, false)).toEqual([
      "name",
      "scope",
      "models",
      "tools",
      "mode",
    ])
    expect(ruleColumns("data", true, true)).toEqual([
      "name",
      "scope",
      "mode",
      "actions",
    ])
  })

  it("says what is empty in the caller's terms", () => {
    expect(emptyCopy("all", true).title).toBe("No rules yet.")
    expect(emptyCopy("all", false).title).toBe(
      "No spend limit or model list reaches your workspaces or your API keys.",
    )
    expect(emptyCopy("data", true).link?.to).toBe("/organization/guardrails")
  })

  it("speaks for the keys the caller reads on an empty Access tab", () => {
    const operator = emptyCopy("access", true, {
      isDeploymentWide: true,
      canCreateKey: true,
    })
    expect(operator.title).toBe(
      "No API key has a list of allowed models or tools.",
    )
    expect(operator.body).toBe(
      "A key without one can use every model and tool its workspace offers. Pick them under More options when you create a key.",
    )
    expect(
      emptyCopy("access", true, {
        isDeploymentWide: false,
        canCreateKey: true,
        organizationName: "Northwind",
      }).title,
    ).toBe(
      "None of your API keys or Northwind’s service keys has a list of allowed models or tools.",
    )
    expect(
      emptyCopy("access", false, {
        isDeploymentWide: false,
        canCreateKey: true,
      }).title,
    ).toBe("None of your API keys has a list of allowed models or tools.")
    // A viewer cannot create a key, so is not told how to.
    expect(
      emptyCopy("access", false, {
        isDeploymentWide: false,
        canCreateKey: false,
      }).body,
    ).toBe(
      "A key without one can use every model and tool its workspace offers.",
    )
  })

  it("orders by use for owners and admins, by name otherwise, with unspent rules last", () => {
    const access = rule({ id: "a", name: "Aardvark key", type: "access" })
    const rules = [access, oldBudget, single, teamDefault]
    expect(
      sortRules(rules, defaultRuleSort(true)).map((one) => one.id),
    ).toEqual(["spend-each-o", "spend-1", "spend-each-d", "a"])
    expect(
      sortRules(rules, { column: "used", direction: "ascending" }).map(
        (one) => one.id,
      ),
    ).toEqual(["spend-each-d", "spend-1", "spend-each-o", "a"])
    expect(
      sortRules(rules, defaultRuleSort(false)).map((one) => one.name),
    ).toEqual([
      "Aardvark key",
      "Each Data member, weekly",
      "Each Platform member, weekly",
      "Northwind, monthly",
    ])
    expect(rulesOnTab(rules, "access")).toEqual([access])
  })
})

describe("the rule page", () => {
  const people = Array.from(
    { length: 12 },
    (_, index): RulePerson => ({
      name: `Person ${String(index).padStart(2, "0")}`,
      userId: `u${index}`,
      team: "Platform",
      spent: index === 5 ? 3 : 0,
      isYou: false,
    }),
  )

  it("lists ten people, the most spent first where spend is shown", () => {
    const shown = peopleShown(people, true, false)
    expect(shown).toHaveLength(10)
    expect(shown[0]?.name).toBe("Person 05")
    expect(peopleShown(people, false, false)[0]?.name).toBe("Person 00")
    expect(peopleShown(people, true, true)).toHaveLength(12)
  })

  it("says who keeps a stopped default in their own number", () => {
    expect(stopDefaultWords(1)).toBe(
      "The 1 person under it keeps theirs. A new amount is set by an owner or admin.",
    )
    expect(stopDefaultWords(5)).toContain("The 5 people under it keep theirs.")
  })

  it("counts the people who left rather than listing them", () => {
    expect(peopleBandWords(0)).toBe(
      "Each person has the whole limit to themselves.",
    )
    expect(peopleBandWords(8)).toBe(
      "Each person has the whole limit to themselves. 8 people who have left still carry it and are not listed.",
    )
  })

  it("sizes an allow-list in words", () => {
    expect(accessWords({ models: ["a", "b"], tools: ["c"] })).toBe(
      "2 models, 1 tool",
    )
    expect(accessWords({ models: null, tools: [] })).toBe("Any model, 0 tools")
  })
})
