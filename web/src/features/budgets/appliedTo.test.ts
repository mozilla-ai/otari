import { describe, expect, it } from "vitest"

import type { AppliedEntity } from "@/client"

import { APPLIED_TO_NOTHING, formatAppliedTo } from "./appliedTo"

const NAMES: Record<string, string> = {
  "org-1": "Acme",
  "ws-1": "Platform",
  "ws-2": "Research",
}

/** An entity as the server lists it: the organization and workspaces named, the rest not. */
function entity(partial: Partial<AppliedEntity>): AppliedEntity {
  const fields = {
    scope_type: "workspace",
    scope_id: "ws-1",
    provider_key_id: null,
    model: null,
    ...partial,
  }
  const named = ["organization", "workspace"].includes(fields.scope_type)
  return { name: named ? (NAMES[fields.scope_id] ?? null) : null, ...fields }
}

describe("formatAppliedTo", () => {
  it("says so when a budget applies to nothing", () => {
    expect(formatAppliedTo([])).toBe(APPLIED_TO_NOTHING)
  })

  it("names the organization rather than counting it", () => {
    const label = formatAppliedTo([
      entity({ scope_type: "organization", scope_id: "org-1" }),
    ])
    expect(label).toBe("Acme (organization)")
  })

  it("names a lone workspace and counts several", () => {
    expect(formatAppliedTo([entity({ scope_id: "ws-2" })])).toBe("Research")
    expect(
      formatAppliedTo([
        entity({ scope_id: "ws-1" }),
        entity({ scope_id: "ws-2" }),
      ]),
    ).toBe("2 workspaces")
  })

  it("falls back to a count for a workspace the server could not name", () => {
    expect(formatAppliedTo([entity({ scope_id: "ws-gone" })])).toBe(
      "1 workspace",
    )
  })

  it("reads a narrowed ceiling as its provider, not as its scope", () => {
    const label = formatAppliedTo([
      entity({
        scope_type: "organization",
        scope_id: "org-1",
        provider_key_id: "openai",
      }),
    ])
    expect(label).toBe("openai")
  })

  it("keeps the scope and its narrowed sibling apart", () => {
    const label = formatAppliedTo([
      entity({ scope_type: "organization", scope_id: "org-1" }),
      entity({
        scope_type: "organization",
        scope_id: "org-1",
        provider_key_id: "openai",
      }),
    ])
    expect(label).toBe("Acme (organization), openai")
  })

  it("reads a model-narrowed entity as the model on its provider", () => {
    const onOpenai = {
      scope_type: "organization",
      scope_id: "org-1",
      provider_key_id: "openai",
    } as const
    expect(formatAppliedTo([entity({ ...onOpenai, model: "gpt-4o" })])).toBe(
      "gpt-4o on openai",
    )
    expect(
      formatAppliedTo([
        entity(onOpenai),
        entity({ ...onOpenai, model: "gpt-4o" }),
        entity({ ...onOpenai, model: "o3" }),
      ]),
    ).toBe("openai, 2 models")
  })

  it("counts members and keys without inventing names for them", () => {
    const label = formatAppliedTo([
      entity({ scope_type: "org_member", scope_id: "m-1" }),
      entity({ scope_type: "org_member", scope_id: "m-2" }),
      entity({ scope_type: "api_token", scope_id: "k-1" }),
    ])
    expect(label).toBe("2 organization members, 1 API key")
  })

  it("orders phrases widest scope first", () => {
    const label = formatAppliedTo(
      [
        entity({ scope_type: "api_token", scope_id: "k-1" }),
        entity({ scope_id: "ws-1" }),
        entity({ scope_type: "organization", scope_id: "org-1" }),
      ],
      5,
    )
    expect(label).toBe("Acme (organization), Platform, 1 API key")
  })

  it("overflows by hidden entities, not by hidden phrases", () => {
    const label = formatAppliedTo([
      entity({ scope_type: "organization", scope_id: "org-1" }),
      entity({ scope_id: "ws-1" }),
      entity({ scope_id: "ws-2" }),
      entity({ scope_type: "org_member", scope_id: "m-1" }),
      entity({ scope_type: "workspace_member", scope_id: "wm-1" }),
      entity({ scope_type: "workspace_member", scope_id: "wm-2" }),
      entity({ scope_type: "api_token", scope_id: "k-1" }),
    ])
    // Three phrases shown, covering four entities; the three behind the two
    // hidden phrases are what the overflow counts.
    expect(label).toBe(
      "Acme (organization), 2 workspaces, 1 organization member, +3",
    )
  })
})
