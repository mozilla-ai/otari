import { describe, expect, it } from "vitest"

import { userOptionText, userOptionTexts } from "@/features/users/userOptions"
import { user } from "@/tests/fixtures"

const UUID = "33333333-3333-3333-3333-333333333333"
const ROSTER = new Map([[UUID, "Alice Example"]])
const EMAILS = new Map([[UUID, "alice@example.com"]])

describe("userOptionText", () => {
  it("leads with the roster name and shows the email as the hint", () => {
    expect(userOptionText(user({ user_id: UUID }), ROSTER, EMAILS)).toEqual({
      label: "Alice Example",
      hint: "alice@example.com",
    })
  })

  it("never shows the billing id on its own", () => {
    expect(userOptionText(user({ user_id: UUID }), ROSTER)).toEqual({
      label: "Alice Example",
    })
  })

  it("drops an email that only repeats the label", () => {
    const named = new Map([[UUID, "alice@example.com"]])
    expect(userOptionText(user({ user_id: UUID }), named, EMAILS)).toEqual({
      label: "alice@example.com",
    })
  })

  it("prefers the roster name to the alias the request plane carries", () => {
    // The alias is whatever the gateway was told; the roster is who the person
    // is, so it wins and the alias drops out of the row entirely.
    expect(
      userOptionText(user({ user_id: UUID, alias: "svc-alice" }), ROSTER),
    ).toEqual({ label: "Alice Example" })
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

describe("userOptionTexts", () => {
  const OTHER = "44444444-4444-4444-4444-444444444444"

  it("falls back to the id for two members who read the same", () => {
    const roster = new Map([
      [UUID, "Alex Smith"],
      [OTHER, "Alex Smith"],
    ])
    expect(
      userOptionTexts(
        [user({ user_id: UUID }), user({ user_id: OTHER })],
        roster,
      ),
    ).toEqual([
      { label: "Alex Smith", hint: UUID },
      { label: "Alex Smith", hint: OTHER },
    ])
  })

  it("keeps the email when it already tells two members apart", () => {
    const roster = new Map([
      [UUID, "Alex Smith"],
      [OTHER, "Alex Smith"],
    ])
    const emails = new Map([
      [UUID, "alex@a.example"],
      [OTHER, "alex@b.example"],
    ])
    expect(
      userOptionTexts(
        [user({ user_id: UUID }), user({ user_id: OTHER })],
        roster,
        emails,
      ),
    ).toEqual([
      { label: "Alex Smith", hint: "alex@a.example" },
      { label: "Alex Smith", hint: "alex@b.example" },
    ])
  })
})
