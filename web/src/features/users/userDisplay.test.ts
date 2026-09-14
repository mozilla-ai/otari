import { describe, expect, it } from "vitest"

import { aliasesByUserId, userDisplay } from "@/features/users/userDisplay"
import { user } from "@/tests/fixtures"

const UUID = "81e24d08-7d1e-4287-a074-54aa57d9debc"
const ROSTER = new Map([[UUID, "Alice Example"]])

describe("userDisplay", () => {
  it("leads with the roster name and keeps the id reachable", () => {
    expect(userDisplay(UUID, null, ROSTER)).toEqual({
      label: "Alice Example",
      id: UUID,
    })
  })

  it("prefers the roster name to the alias the request plane carries", () => {
    // The alias is whatever the gateway was told; the roster is who the person
    // is, so it wins, the same way it does in a picker row.
    expect(userDisplay(UUID, "svc-alice", ROSTER)).toEqual({
      label: "Alice Example",
      id: UUID,
    })
  })

  it("falls back to the alias the server resolved", () => {
    expect(userDisplay(UUID, "Bob Example", new Map())).toEqual({
      label: "Bob Example",
      id: UUID,
    })
  })

  it("leaves an id nobody named alone, with no id to render beside it", () => {
    expect(userDisplay("ci-bot", null, ROSTER)).toEqual({ label: "ci-bot" })
    // An alias equal to the id is the same case: one copy, not two.
    expect(userDisplay("ci-bot", "ci-bot", ROSTER)).toEqual({ label: "ci-bot" })
  })
})

describe("aliasesByUserId", () => {
  it("keys the request plane's aliases by user id, skipping the unnamed", () => {
    const aliases = aliasesByUserId([
      user({ user_id: UUID, alias: "Alice" }),
      user({ user_id: "ci-bot", alias: null }),
    ])
    expect(aliases.get(UUID)).toBe("Alice")
    expect(aliases.has("ci-bot")).toBe(false)
  })

  it("is empty while the users list is still loading", () => {
    expect(aliasesByUserId(undefined).size).toBe(0)
  })
})
