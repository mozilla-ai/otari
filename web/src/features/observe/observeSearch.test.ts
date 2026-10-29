import { describe, expect, it } from "vitest"

import {
  carrySearch,
  legacySearch,
  statusApplies,
  viewForPath,
  windowFor,
} from "./observeSearch"

const NOW = Date.parse("2026-10-23T12:00:00Z")

describe("carrySearch", () => {
  it("carries the shared filters and drops a view's own keys", () => {
    expect(
      carrySearch(
        "requests",
        "sessions",
        {
          range: "7d",
          model: ["a", "b"],
          user_id: "u1",
          status: "error",
          page: "3",
          peek: "t1",
        },
        NOW,
      ),
    ).toEqual({
      range: "7d",
      model: ["a", "b"],
      user_id: "u1",
      status: "error",
    })
  })

  it("pins an absent range to the source view's default", () => {
    expect(carrySearch("usage", "requests", {}, NOW)).toEqual({
      range: "30d",
    })
    expect(carrySearch("requests", "usage", {}, NOW)).toEqual({
      range: "24h",
    })
  })

  it("omits a range that is the destination's default", () => {
    expect(carrySearch("requests", "sessions", {}, NOW)).toEqual({})
    expect(carrySearch("requests", "usage", { range: "30d" }, NOW)).toEqual({})
  })

  it("restates a preset the destination lacks as an explicit start", () => {
    expect(carrySearch("usage", "requests", { range: "90d" }, NOW)).toEqual({
      start_date: new Date(NOW - 90 * 86_400_000).toISOString(),
    })
  })

  it("opens the unbounded window on Usage as the year windowFor frames it", () => {
    expect(carrySearch("requests", "usage", { range: "all" }, NOW)).toEqual({
      range: "12mo",
    })
    expect(windowFor("requests", { range: "all" }, NOW).start).toBe(
      windowFor("usage", { range: "12mo" }, NOW).start,
    )
    expect(carrySearch("requests", "sessions", { range: "all" }, NOW)).toEqual({
      range: "all",
    })
  })

  it("reads a bound-less custom or unknown range as the source's default", () => {
    // What the source view is showing, which is what windowFor reads: not the
    // widest window the destination has.
    for (const range of ["custom", "nope", "90d"]) {
      const carried = carrySearch("requests", "usage", { range }, NOW)
      expect(carried).toEqual({ range: "24h" })
      expect(windowFor("usage", carried, NOW)).toEqual(
        windowFor("requests", { range }, NOW),
      )
    }
    expect(carrySearch("usage", "requests", { range: "custom" }, NOW)).toEqual({
      range: "30d",
    })
  })

  it("leaves explicit bounds alone", () => {
    const search = {
      range: "custom",
      start_date: "2026-10-01T00:00:00.000Z",
      end_date: "2026-10-02T00:00:00.000Z",
    }
    expect(carrySearch("usage", "sessions", search, NOW)).toEqual(search)
  })
})

describe("viewForPath", () => {
  it("names the view a path renders", () => {
    expect(viewForPath("/")).toBe("usage")
    expect(viewForPath("/")).toBe("usage")
    expect(viewForPath("/requests")).toBe("requests")
    expect(viewForPath("/sessions")).toBe("sessions")
  })
})

describe("windowFor", () => {
  const ago = (ms: number) => new Date(NOW - ms).toISOString()

  it("reads each view's own default window", () => {
    expect(windowFor("usage", {}, NOW)).toEqual({
      start: ago(30 * 86_400_000),
      end: "",
    })
    expect(windowFor("requests", {}, NOW).start).toBe(ago(86_400_000))
    expect(windowFor("sessions", {}, NOW).start).toBe(ago(86_400_000))
  })

  it("reads the preset the view offers", () => {
    expect(windowFor("requests", { range: "7d" }, NOW).start).toBe(
      ago(7 * 86_400_000),
    )
    expect(windowFor("usage", { range: ["90d", "7d"] }, NOW).start).toBe(
      ago(90 * 86_400_000),
    )
  })

  it("falls back to the default for a range the view lacks", () => {
    expect(windowFor("requests", { range: "90d" }, NOW).start).toBe(
      ago(86_400_000),
    )
    expect(windowFor("usage", { range: "custom" }, NOW).start).toBe(
      ago(30 * 86_400_000),
    )
  })

  it("frames the unbounded window as a year", () => {
    expect(windowFor("requests", { range: "all" }, NOW).start).toBe(
      ago(365 * 86_400_000),
    )
  })

  it("lets explicit bounds win", () => {
    expect(
      windowFor(
        "usage",
        { range: "7d", start_date: "2026-10-01T00:00:00.000Z" },
        NOW,
      ),
    ).toEqual({ start: "2026-10-01T00:00:00.000Z", end: "" })
    expect(
      windowFor("requests", { end_date: "2026-10-02T00:00:00.000Z" }, NOW),
    ).toEqual({ start: "", end: "2026-10-02T00:00:00.000Z" })
  })
})

describe("statusApplies", () => {
  it("reads every status on Requests, outcomes on Sessions, none on Usage", () => {
    expect(statusApplies("requests", "absorbed")).toBe(true)
    expect(statusApplies("sessions", "error")).toBe(true)
    expect(statusApplies("sessions", "success")).toBe(true)
    expect(statusApplies("sessions", "absorbed")).toBe(false)
    expect(statusApplies("usage", "error")).toBe(false)
  })
})

describe("carrySearch scope", () => {
  it("carries the scope between views", () => {
    expect(carrySearch("usage", "requests", { scope: "mine" }).scope).toBe(
      "mine",
    )
  })
})

describe("legacySearch", () => {
  it("opens an old link that names a user on Everyone", () => {
    expect(legacySearch({ user_id: "u1", range: "7d" })).toEqual({
      user_id: "u1",
      range: "7d",
      scope: "everyone",
    })
    expect(legacySearch({ user_id: ["u1", "u2"] }).scope).toBe("everyone")
  })

  it("keeps a scope the link already chose, and leaves the rest alone", () => {
    expect(legacySearch({ user_id: "u1", scope: "mine" })).toEqual({
      user_id: "u1",
      scope: "mine",
    })
    expect(legacySearch({ range: "7d" })).toEqual({ range: "7d" })
    expect(legacySearch({ user_id: [] })).toEqual({ user_id: [] })
  })
})
