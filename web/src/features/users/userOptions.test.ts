import { describe, expect, it } from "vitest"

import { userOptionText } from "@/features/users/userOptions"
import { user } from "@/tests/fixtures"

const UUID = "33333333-3333-3333-3333-333333333333"
const ROSTER = new Map([[UUID, "Alice Example"]])

describe("userOptionText", () => {
  it("leads with the roster name and keeps the billing id as the hint", () => {
    expect(userOptionText(user({ user_id: UUID }), ROSTER)).toEqual({
      label: "Alice Example",
      hint: UUID,
    })
  })

  it("prefers the roster name to the alias the request plane carries", () => {
    // The alias is whatever the gateway was told; the roster is who the person
    // is, so it wins and the alias drops out of the row entirely.
    expect(
      userOptionText(user({ user_id: UUID, alias: "svc-alice" }), ROSTER),
    ).toEqual({ label: "Alice Example", hint: UUID })
  })

  it("leaves an id nobody named alone, with no hint repeating it", () => {
    expect(userOptionText(user({ user_id: "ci-bot" }), ROSTER)).toEqual({
      label: "ci-bot",
    })
    expect(
      userOptionText(user({ user_id: "ci-bot", alias: "release" }), ROSTER),
    ).toEqual({ label: "ci-bot (release)" })
  })
})
