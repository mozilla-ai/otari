import { describe, expect, it } from "vitest"

import type { CreateKeyResponse } from "@/client"
import { formatDate } from "@/shared/helpers/format"
import { secretCaption } from "./secretCaption"

const result = (over: Partial<CreateKeyResponse> = {}): CreateKeyResponse => ({
  allowed_models: null,
  capture_agent_telemetry: null,
  created_at: "2026-09-10T00:00:00Z",
  exclude_from_budget: false,
  expires_at: null,
  id: "key-1",
  is_active: true,
  key: "gw-secret",
  key_name: "ci-bot",
  key_prefix: "gw-secret…",
  metadata: {},
  reject_user_mismatch: null,
  user_id: "alice",
  ...over,
})

const labels = new Map([["alice", "Alice Chen"]])

describe("secretCaption", () => {
  it("names the owner, the access and the expiry", () => {
    // The date goes through the shared formatter rather than being spelled
    // here: `formatDate` is `toLocaleDateString()`, so the rendered string is
    // the reader's locale and a literal in this file would pin the suite to
    // whichever one CI happens to run under.
    const expires = "2027-01-31T00:00:00Z"
    expect(
      secretCaption(
        result({ expires_at: expires, allowed_models: null }),
        labels,
      ),
    ).toBe(`Owner Alice Chen · All models · expires ${formatDate(expires)}`)
    expect(
      secretCaption(result({ expires_at: expires }), labels),
    ).not.toContain(expires)
  })

  it("drops the owner rather than printing an id", () => {
    // A member creating their own key cannot resolve a name: `/v1/users` is
    // operator-only, so the map is empty on that surface. An identifier in a
    // sentence is noise to the person who just chose the owner from a list.
    expect(secretCaption(result(), new Map())).toBe(
      "All models · never expires",
    )
  })

  it("drops the owner for a virtual user, which has no person behind it", () => {
    expect(secretCaption(result({ user_id: "apikey-abc" }), labels)).toBe(
      "All models · never expires",
    )
  })

  it("reuses the table's own access wording rather than restating it", () => {
    expect(secretCaption(result({ allowed_models: ["gpt-4o"] }), labels)).toBe(
      "Owner Alice Chen · Selected models · never expires",
    )
    expect(secretCaption(result({ allowed_models: [] }), labels)).toBe(
      "Owner Alice Chen · No models · never expires",
    )
  })

  it("carries no spend figure, because a key has no budget of its own", () => {
    // The frame this sits in is where an operator would look for one, so this
    // is a rule rather than an omission: a key spends against its owner's
    // budget, which the page says and links to above the table.
    const caption = secretCaption(result({ exclude_from_budget: true }), labels)
    expect(caption).not.toMatch(/\$|budget|per month/)
    expect(caption).toBe("Owner Alice Chen · All models · never expires")
  })
})
