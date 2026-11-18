import { describe, expect, it } from "vitest"

import {
  foldEffects,
  offerEachCopyOnce,
  settleAtMost,
} from "@/features/tools/mcpFold"
import { mcpServerFoldCandidate } from "@/tests/fixtures"

describe("foldEffects", () => {
  it("says a copy with no credential of its own changes account when it is not on the same one", () => {
    const keyless = mcpServerFoldCandidate({
      credential_mode: "shared",
      same_credential: false,
      signed_in: 0,
      keep_sign_in: 0,
    })
    expect(foldEffects(keyless)).toEqual([
      "The account its calls use will change.",
    ])
    expect(foldEffects({ ...keyless, same_credential: true })).toEqual([
      "No one's sign-in changes.",
    ])
  })
})

describe("foldEffects, people already signed in to the organization's server", () => {
  it("counts them apart from those who keep their sign-in or sign in again", () => {
    const personal = mcpServerFoldCandidate({
      credential_mode: "per_user",
      same_credential: true,
      signed_in: 4,
      keep_sign_in: 1,
      switch_account: 2,
      sign_in_again: 1,
    })
    expect(foldEffects(personal)).toEqual([
      "1 person will keep their sign-in.",
      "2 people will use the account they already signed in with on the organization's server.",
      "1 person will need to sign in again.",
    ])
  })
})

describe("offerEachCopyOnce", () => {
  it("keeps a copy under the first list that found it", () => {
    const a = mcpServerFoldCandidate({ server_id: "a" })
    const b = mcpServerFoldCandidate({ server_id: "b" })
    expect(offerEachCopyOnce([[a], [a, b], []])).toEqual([[a], [b], []])
  })
})

describe("settleAtMost", () => {
  it("runs no more than the limit at once and reports each outcome in order", async () => {
    let running = 0
    let most = 0
    const outcomes = await settleAtMost([1, 2, 3, 4, 5], 2, async (n) => {
      running += 1
      most = Math.max(most, running)
      await new Promise((resolve) => setTimeout(resolve, 1))
      running -= 1
      if (n === 3) throw new Error("three")
      return n * 10
    })
    expect(most).toBe(2)
    expect(outcomes.map((outcome) => outcome.status)).toEqual([
      "fulfilled",
      "fulfilled",
      "rejected",
      "fulfilled",
      "fulfilled",
    ])
    expect(outcomes[4]).toEqual({ status: "fulfilled", value: 50 })
  })
})
