import { describe, expect, it } from "vitest"

import {
  defaultFoldIds,
  foldableIds,
  foldBlocker,
  foldEffects,
  offerEachCopyOnce,
  settleAtMost,
} from "@/features/tools/mcpFold"
import { mcpServerFoldCandidate } from "@/tests/fixtures"

describe("defaultFoldIds", () => {
  it("ticks only copies whose folding changes nobody's account or sign-in", () => {
    const quiet = mcpServerFoldCandidate({
      server_id: "quiet",
      workspace_id: "a",
    })
    const switches = mcpServerFoldCandidate({
      server_id: "switches",
      workspace_id: "b",
      switch_account: 1,
    })
    const again = mcpServerFoldCandidate({
      server_id: "again",
      workspace_id: "c",
      sign_in_again: 1,
    })
    const other = mcpServerFoldCandidate({
      server_id: "other",
      workspace_id: "d",
      same_credential: false,
    })
    expect(defaultFoldIds([quiet, switches, again, other], true)).toEqual([
      "quiet",
    ])
  })

  it("ticks none of a workspace's copies when it could fold another, even one that changes an account", () => {
    const quiet = mcpServerFoldCandidate({ server_id: "quiet" })
    const ownToken = mcpServerFoldCandidate({
      server_id: "own-token",
      same_credential: false,
      replaces_shared_credential: true,
    })
    expect(defaultFoldIds([quiet, ownToken], true)).toEqual([])
    // A copy the server cannot replace is no choice to make.
    const onCopy = { ...ownToken, enabled: true }
    const offQuiet = { ...quiet, enabled: false }
    expect(defaultFoldIds([offQuiet, onCopy], false)).toEqual(["quiet"])
  })

  it("ticks none of a workspace's copies when it has several, for the admin to pick", () => {
    const first = mcpServerFoldCandidate({ server_id: "first" })
    const second = mcpServerFoldCandidate({ server_id: "second" })
    const elsewhere = mcpServerFoldCandidate({
      server_id: "elsewhere",
      workspace_id: "other",
    })
    expect(defaultFoldIds([first, second, elsewhere], true)).toEqual([
      "elsewhere",
    ])
  })

  it("leaves a copy that is on unticked for a server that is off", () => {
    const on = mcpServerFoldCandidate({ server_id: "on", workspace_id: "a" })
    const off = mcpServerFoldCandidate({
      server_id: "off",
      workspace_id: "b",
      enabled: false,
    })
    expect(defaultFoldIds([on, off], false)).toEqual(["off"])
    expect(foldBlocker(on, false)).not.toBe("")
    expect(foldBlocker(on, true)).toBe("")
  })
})

describe("foldableIds", () => {
  it("sends only ticked copies still listed and foldable into the server as it is", () => {
    const on = mcpServerFoldCandidate({ server_id: "on", workspace_id: "a" })
    const off = mcpServerFoldCandidate({
      server_id: "off",
      workspace_id: "b",
      enabled: false,
    })
    expect(foldableIds([on, off], ["on", "off", "gone"], false)).toEqual([
      "off",
    ])
    expect(foldableIds([on, off], ["on", "gone"], true)).toEqual(["on"])
  })
})

describe("foldEffects", () => {
  it("says a copy with no credential of its own changes account when it is not on the same one", () => {
    const keyless = mcpServerFoldCandidate({
      credential_mode: "shared",
      same_credential: false,
      signed_in: 0,
      keep_sign_in: 0,
    })
    expect(foldEffects(keyless, "shared")).toEqual([
      "The account its calls use will change.",
    ])
    expect(
      foldEffects({ ...keyless, same_credential: true }, "shared"),
    ).toEqual(["No one's sign-in changes."])
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
    expect(foldEffects(personal, "per_user")).toEqual([
      "1 person will keep their sign-in.",
      "2 people will use the account they already signed in with on the organization's server.",
      "1 person will need to sign in again.",
    ])
  })

  it("asks nobody to sign in again on a shared server, whose credential serves them", () => {
    const personal = mcpServerFoldCandidate({
      credential_mode: "per_user",
      same_credential: false,
      signed_in: 2,
      keep_sign_in: 0,
      sign_in_again: 2,
    })
    expect(foldEffects(personal, "shared")).toEqual([
      "2 people will stop using their own sign-in, and use the organization's account instead.",
      "The account its calls use will change.",
    ])
  })
})

describe("offerEachCopyOnce", () => {
  it("keeps a copy under the first list that found it", () => {
    const a = mcpServerFoldCandidate({ server_id: "a" })
    const b = mcpServerFoldCandidate({ server_id: "b" })
    expect(offerEachCopyOnce([[a], [a, b], []], [true, true, true])).toEqual([
      [a],
      [b],
      [],
    ])
  })

  it("offers a copy under the first pick that can fold it, else the first that found it", () => {
    const on = mcpServerFoldCandidate({ server_id: "on" })
    const off = mcpServerFoldCandidate({ server_id: "off", enabled: false })
    expect(
      offerEachCopyOnce(
        [
          [on, off],
          [on, off],
        ],
        [false, true],
      ),
    ).toEqual([[off], [on]])
    expect(offerEachCopyOnce([[on], [on]], [false, false])).toEqual([[on], []])
  })
})

describe("settleAtMost", () => {
  it("runs no more than the limit at once and reports each outcome in input order", async () => {
    let running = 0
    let most = 0
    const finish = new Map<number, () => void>()
    const started: number[] = []
    const settled = settleAtMost([1, 2, 3, 4, 5], 2, async (n) => {
      running += 1
      most = Math.max(most, running)
      started.push(n)
      await new Promise<void>((resolve) => finish.set(n, resolve))
      running -= 1
      if (n === 3) throw new Error("three")
      return n * 10
    })
    const flush = () => new Promise((resolve) => setTimeout(resolve, 0))
    await flush()
    expect(started).toEqual([1, 2])
    // The second finishes first, and the third starts in its place.
    finish.get(2)?.()
    await flush()
    expect(started).toEqual([1, 2, 3])
    for (const n of [3, 1, 5, 4]) {
      await flush()
      finish.get(n)?.()
    }
    const outcomes = await settled
    expect(most).toBe(2)
    expect(outcomes).toEqual([
      { status: "fulfilled", value: 10 },
      { status: "fulfilled", value: 20 },
      { status: "rejected", reason: new Error("three") },
      { status: "fulfilled", value: 40 },
      { status: "fulfilled", value: 50 },
    ])
  })
})
