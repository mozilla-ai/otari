import { describe, expect, it } from "vitest"

import type { OrganizationMember } from "@/client"
import { namedAppliedEntity, organizationSpendCeiling } from "@/tests/fixtures"

import {
  describeEntity,
  entityFromKey,
  entityGroups,
  entityKey,
  entityNamer,
  takenEntities,
} from "./appliedEntities"

const ORG = "org-1"

function member(overrides: Partial<OrganizationMember>): OrganizationMember {
  return {
    created_at: "2026-01-01T00:00:00+00:00",
    role: "member",
    status: "active",
    ...overrides,
  }
}

const SOURCES = {
  organizationId: ORG,
  workspaces: [{ id: "ws-1", name: "Platform" }],
  members: [
    member({
      organization_member_id: "om-1",
      full_name: "Pat Okafor",
      email: "pat@example.com",
      workspaces: [
        {
          role: "member",
          workspace_id: "ws-1",
          workspace_member_id: "wm-1",
          workspace_name: "Platform",
        },
      ],
    }),
    // A pending invitation: no membership row yet, so nothing to cap.
    member({ email: "invited@example.com", status: "invited" }),
  ],
  keys: [
    { id: "k-1", key_name: "ci", key_prefix: "sk-a", workspace_id: "ws-1" },
  ],
  modelIds: ["openai:gpt-4o", "openai:o3", "anthropic:claude", "my-alias"],
}

const optionsOf = (groupId: string, taken = new Map<string, string>()) =>
  entityGroups(SOURCES, taken).find((group) => group.id === groupId)?.options ??
  []

describe("entityKey", () => {
  it("round-trips an entity, with an absent axis read as null", () => {
    const key = entityKey({ scope_type: "workspace", scope_id: "ws-1" })
    expect(entityFromKey(key)).toEqual({
      scope_type: "workspace",
      scope_id: "ws-1",
      provider_key_id: null,
      model: null,
    })
  })

  it("matches a held ceiling to the entity the picker offers for it", () => {
    const ceiling = organizationSpendCeiling({
      scope_type: "organization",
      scope_id: ORG,
      provider_key_id: "openai",
      model: "gpt-4o",
    })
    expect(optionsOf("model").map((option) => option.id)).toContain(
      entityKey(ceiling),
    )
  })
})

describe("entityGroups", () => {
  it("offers each kind of entity under the scope id its ceiling stores", () => {
    expect(entityFromKey(optionsOf("org_member")[0].id).scope_id).toBe("om-1")
    expect(entityFromKey(optionsOf("workspace_member")[0].id).scope_id).toBe(
      "wm-1",
    )
    expect(entityFromKey(optionsOf("api_token")[0].id).scope_id).toBe("k-1")
  })

  it("leaves out an invitation that has no membership to cap", () => {
    expect(optionsOf("org_member")).toHaveLength(1)
  })

  it("tells two placements of one person apart by workspace", () => {
    expect(optionsOf("workspace_member")[0]).toMatchObject({
      label: "Pat Okafor",
      hint: "Platform",
    })
  })

  it("reads providers and models off prefixed catalog ids only", () => {
    expect(optionsOf("provider").map((option) => option.label)).toEqual([
      "anthropic",
      "openai",
    ])
    expect(optionsOf("model").map((option) => option.label)).toEqual([
      "anthropic:claude",
      "openai:gpt-4o",
      "openai:o3",
    ])
    expect(entityFromKey(optionsOf("provider")[1].id)).toEqual({
      scope_type: "organization",
      scope_id: ORG,
      provider_key_id: "openai",
      model: null,
    })
  })

  it("disables an entity another budget carries, and says which", () => {
    const workspaceKey = entityKey({
      scope_type: "workspace",
      scope_id: "ws-1",
    })
    const [option] = optionsOf(
      "workspace",
      new Map([[workspaceKey, "On Research"]]),
    )
    expect(option).toMatchObject({ isDisabled: true, hint: "On Research" })
  })
})

describe("takenEntities", () => {
  const entity = (scope_id: string) => namedAppliedEntity({ scope_id })
  const budgets = [
    { budget_id: "mine", name: "Mine", applied_to: [entity("ws-1")] },
    { budget_id: "theirs", name: "Research", applied_to: [entity("ws-2")] },
  ]

  it("skips the budget being edited and names the rest", () => {
    const taken = takenEntities(budgets, "mine", (budget) => budget.name)
    expect([...taken.entries()]).toEqual([
      [entityKey(entity("ws-2")), "On Research"],
    ])
  })
})

describe("describeEntity", () => {
  const names = { organizationName: "Acme", workspaces: SOURCES.workspaces }

  it("names a combination no picker group offers", () => {
    expect(
      describeEntity(
        {
          scope_type: "workspace",
          scope_id: "ws-1",
          provider_key_id: "openai",
        },
        names,
      ),
    ).toBe("Platform, on openai")
    expect(
      describeEntity(
        {
          scope_type: "workspace_member",
          scope_id: "wm-9",
          provider_key_id: "openai",
          model: "o3",
        },
        names,
      ),
    ).toBe("Workspace member wm-9, openai:o3")
  })
})

describe("entityNamer", () => {
  const name = entityNamer(SOURCES, "Acme")

  it("names an entity the way the picker offered it, with its kind", () => {
    expect(
      name(entityKey({ scope_type: "workspace_member", scope_id: "wm-1" })),
    ).toEqual({
      name: "Pat Okafor (Platform)",
      kind: "Workspace member",
    })
    expect(
      name(entityKey({ scope_type: "organization", scope_id: ORG })),
    ).toEqual({
      name: "Acme",
      kind: "Whole organization",
    })
  })

  it("falls back to a description for an entity no group offers", () => {
    expect(
      name(
        entityKey({
          scope_type: "workspace",
          scope_id: "ws-1",
          provider_key_id: "openai",
        }),
      ),
    ).toEqual({ name: "Platform, on openai", kind: "Narrowed" })
  })
})
