import { describe, expect, it } from "vitest"

import {
  APPLIED_TO_NOTHING,
  type AppliedCeiling,
  type AppliedToContext,
  appliedToLabel,
  ceilingsByBudget,
} from "./appliedTo"

const CONTEXT: AppliedToContext = {
  organizationName: "Acme",
  workspaces: [
    { id: "ws-1", name: "Platform" },
    { id: "ws-2", name: "Research" },
  ],
}

function ceiling(partial: Partial<AppliedCeiling>): AppliedCeiling {
  return {
    scope_type: "workspace",
    scope_id: "ws-1",
    provider_key_id: null,
    model: null,
    budget_id: "b-1",
    ...partial,
  }
}

describe("appliedToLabel", () => {
  it("says so when a budget applies to nothing", () => {
    expect(appliedToLabel([], CONTEXT)).toBe(APPLIED_TO_NOTHING)
  })

  it("names the organization rather than counting it", () => {
    const label = appliedToLabel(
      [ceiling({ scope_type: "organization", scope_id: "org-1" })],
      CONTEXT,
    )
    expect(label).toBe("Acme (organization)")
  })

  it("names every workspace", () => {
    expect(appliedToLabel([ceiling({ scope_id: "ws-2" })], CONTEXT)).toBe(
      "Research",
    )
    expect(
      appliedToLabel(
        [ceiling({ scope_id: "ws-1" }), ceiling({ scope_id: "ws-2" })],
        CONTEXT,
      ),
    ).toBe("Platform, Research")
  })

  it("counts a workspace the page has not read", () => {
    expect(appliedToLabel([ceiling({ scope_id: "ws-gone" })], CONTEXT)).toBe(
      "1 workspace",
    )
    expect(
      appliedToLabel(
        [ceiling({ scope_id: "ws-1" }), ceiling({ scope_id: "ws-gone" })],
        CONTEXT,
      ),
    ).toBe("Platform, 1 other workspace")
  })

  it("reads a narrowed ceiling as its provider, not as its scope", () => {
    const label = appliedToLabel(
      [
        ceiling({
          scope_type: "organization",
          scope_id: "org-1",
          provider_key_id: "openai",
        }),
      ],
      CONTEXT,
    )
    expect(label).toBe("openai")
  })

  it("keeps the scope and its narrowed sibling apart", () => {
    const label = appliedToLabel(
      [
        ceiling({ scope_type: "organization", scope_id: "org-1" }),
        ceiling({
          scope_type: "organization",
          scope_id: "org-1",
          provider_key_id: "openai",
        }),
      ],
      CONTEXT,
    )
    expect(label).toBe("Acme (organization), openai")
  })

  it("reads a model-narrowed ceiling as the model on its provider", () => {
    const onOpenai = {
      scope_type: "organization",
      scope_id: "org-1",
      provider_key_id: "openai",
    } as const
    expect(
      appliedToLabel([ceiling({ ...onOpenai, model: "gpt-4o" })], CONTEXT),
    ).toBe("gpt-4o on openai")
    expect(
      appliedToLabel(
        [
          ceiling(onOpenai),
          ceiling({ ...onOpenai, model: "gpt-4o" }),
          ceiling({ ...onOpenai, model: "o3" }),
        ],
        CONTEXT,
      ),
    ).toBe("openai, 2 models")
  })

  it("counts members and keys without inventing names for them", () => {
    const label = appliedToLabel(
      [
        ceiling({ scope_type: "org_member", scope_id: "m-1" }),
        ceiling({ scope_type: "org_member", scope_id: "m-2" }),
        ceiling({ scope_type: "api_token", scope_id: "k-1" }),
      ],
      CONTEXT,
    )
    expect(label).toBe("2 organization members, 1 API key")
  })

  it("orders phrases widest scope first", () => {
    const label = appliedToLabel(
      [
        ceiling({ scope_type: "api_token", scope_id: "k-1" }),
        ceiling({ scope_id: "ws-1" }),
        ceiling({ scope_type: "organization", scope_id: "org-1" }),
      ],
      CONTEXT,
      5,
    )
    expect(label).toBe("Acme (organization), Platform, 1 API key")
  })

  it("overflows by hidden entities, not by hidden phrases", () => {
    const label = appliedToLabel(
      [
        ceiling({ scope_type: "organization", scope_id: "org-1" }),
        ceiling({ scope_id: "ws-1" }),
        ceiling({ scope_id: "ws-2" }),
        ceiling({ scope_type: "org_member", scope_id: "m-1" }),
        ceiling({ scope_type: "workspace_member", scope_id: "wm-1" }),
        ceiling({ scope_type: "workspace_member", scope_id: "wm-2" }),
        ceiling({ scope_type: "api_token", scope_id: "k-1" }),
      ],
      CONTEXT,
    )
    // Three phrases shown, covering four entities; the three behind the two
    // hidden phrases are what the overflow counts.
    expect(label).toBe(
      "Acme (organization), Platform, Research, 1 organization member, +3",
    )
  })
})

describe("ceilingsByBudget", () => {
  it("groups every ceiling under the budget it enforces", () => {
    const grouped = ceilingsByBudget([
      ceiling({ budget_id: "b-1", scope_id: "ws-1" }),
      ceiling({ budget_id: "b-2", scope_id: "ws-2" }),
      ceiling({ budget_id: "b-1", scope_type: "org_member", scope_id: "m-1" }),
    ])
    expect(grouped.get("b-1")).toHaveLength(2)
    expect(grouped.get("b-2")).toHaveLength(1)
    expect(grouped.get("b-3")).toBeUndefined()
  })
})
