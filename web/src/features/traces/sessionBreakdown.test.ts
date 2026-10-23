import { describe, expect, it } from "vitest"

import {
  costParts,
  IDLE_DRAWN_AT,
  TOOLS_SHOWN,
  timeBreakdown,
  toolRows,
} from "./sessionBreakdown"
import type { SessionStats, ToolTally } from "./sessionStats"

const s = 1000
const m = 60 * s
const h = 60 * m

function stats(overrides: Partial<SessionStats> = {}): SessionStats {
  return {
    turns: 1,
    lengthMs: 0,
    requests: 1,
    toolCalls: 0,
    inSubagents: 0,
    tools: [],
    gates: { runs: 0, verdicts: {} },
    judges: { calls: 0, models: [], cost: 0, priced: 0 },
    time: { modelMs: 0, toolsMs: 0, waitingMs: 0, hooksMs: 0 },
    activeMs: 0,
    toolsInModelMs: 0,
    tokens: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    cost: null,
    cacheSaved: null,
    pricedRequests: 0,
    models: [],
    ...overrides,
  }
}

const keys = (parts: Array<{ key: string }>) => parts.map((part) => part.key)
const drawnTotal = (parts: Array<{ drawn: number }>) =>
  parts.reduce((total, part) => total + part.drawn, 0)

describe("timeBreakdown", () => {
  // A coding agent's session: four kinds of work that never overlapped, and a
  // few seconds between its turns.
  const session = stats({
    turns: 4,
    lengthMs: 6 * m + 12 * s,
    activeMs: 6 * m,
    time: {
      modelMs: 221 * s,
      toolsMs: 78 * s,
      waitingMs: 52 * s,
      hooksMs: 9 * s,
    },
  })

  it("lists the work by time and idle last", () => {
    const time = timeBreakdown(session)

    expect(keys(time?.parts ?? [])).toEqual([
      "model",
      "tools",
      "waiting",
      "hooks",
      "idle",
    ])
    expect(time?.parts.at(-1)).toMatchObject({
      value: 12 * s,
      hint: "between turns",
    })
  })

  it("draws every part to scale when the session was mostly busy", () => {
    const time = timeBreakdown(session)

    for (const part of time?.parts ?? [])
      expect(part.drawn).toBeCloseTo(part.share)
    expect(time?.idleShortened).toBe(false)
    expect(time?.overlaps).toBe(false)
    expect(time?.lane).toBeNull()
  })

  it("draws idle shorter when it was most of the session, and keeps its share true", () => {
    // Left open for six days, with 8h 54m of work in it.
    const time = timeBreakdown(
      stats({
        lengthMs: 520_604 * s,
        activeMs: 32_038 * s,
        time: {
          modelMs: 245 * s,
          toolsMs: 31_793 * s,
          waitingMs: 0,
          hooksMs: 0,
        },
      }),
    )
    const idle = time?.parts.find((part) => part.key === "idle")

    expect(time?.idleShortened).toBe(true)
    expect(idle?.share).toBeCloseTo(0.938, 3)
    expect(idle?.drawn).toBe(IDLE_DRAWN_AT)
    expect(drawnTotal(time?.parts ?? [])).toBeCloseTo(1)
    // The work keeps its proportions in what idle leaves of the bar.
    const [tools, model] = time?.parts ?? []
    expect((tools?.drawn ?? 0) / (model?.drawn ?? 1)).toBeCloseTo(31_793 / 245)
  })

  it("says when the work overlapped, and still fits it in the bar", () => {
    // A subagent's requests ran alongside the main agent's tools.
    const time = timeBreakdown(
      stats({
        lengthMs: 7 * m,
        activeMs: 6 * m,
        time: { modelMs: 5 * m, toolsMs: 4 * m, waitingMs: 0, hooksMs: 0 },
      }),
    )

    expect(time?.overlaps).toBe(true)
    expect(drawnTotal(time?.parts ?? [])).toBeCloseTo(1)
    expect(time?.parts[0]?.share).toBeCloseTo(5 / 7)
  })

  it("gives tools that ran inside the model's requests a lane of their own", () => {
    // A gateway's built-in web search and code execution, all within one request.
    const time = timeBreakdown(
      stats({
        lengthMs: 40_170,
        activeMs: 40_150,
        time: { modelMs: 40_150, toolsMs: 19_620, waitingMs: 0, hooksMs: 0 },
        toolsInModelMs: 19_620,
      }),
    )

    // Drawn as a share of the model's segment, which it runs under.
    expect(time?.lane?.drawn).toBeCloseTo(19_620 / 40_150)
    expect(time?.parts.find((part) => part.key === "tools")?.drawn).toBe(0)
    // The lane accounts for the doubled time, so there is no overlap to explain.
    expect(time?.overlaps).toBe(false)
  })

  it("keeps tools on the bar when only some of their time was inside a request", () => {
    const time = timeBreakdown(
      stats({
        lengthMs: 10 * m,
        activeMs: 10 * m,
        time: { modelMs: 4 * m, toolsMs: 6 * m, waitingMs: 0, hooksMs: 0 },
        toolsInModelMs: 10 * s,
      }),
    )

    expect(time?.lane).toBeNull()
  })

  it("leaves out idle time too short to be a pause, and says where it fell in one turn", () => {
    const busy = timeBreakdown(
      stats({
        lengthMs: 40_170,
        activeMs: 40_150,
        time: { modelMs: 40_150, toolsMs: 0, waitingMs: 0, hooksMs: 0 },
      }),
    )
    const paused = timeBreakdown(
      stats({
        lengthMs: 5_360,
        activeMs: 5_107,
        time: { modelMs: 5_107, toolsMs: 0, waitingMs: 0, hooksMs: 0 },
      }),
    )

    expect(keys(busy?.parts ?? [])).toEqual(["model"])
    expect(paused?.parts.at(-1)).toMatchObject({
      key: "idle",
      hint: "between steps",
    })
  })

  it("has nothing to draw for a session with no length or no activity", () => {
    expect(timeBreakdown(stats())).toBeNull()
    expect(timeBreakdown(stats({ lengthMs: 5 * s }))).toBeNull()
  })
})

describe("costParts", () => {
  it("splits by meter when the requests were priced by the token, the costliest first", () => {
    const { by, parts } = costParts(
      stats({
        cost: { input: 0.07, output: 0.44, cacheRead: 0.62, cacheWrite: 0.71 },
        tokens: {
          input: 13_402,
          output: 17_610,
          cacheRead: 1_240_388,
          cacheWrite: 114_020,
        },
      }),
    )

    expect(by).toBe("meter")
    expect(keys(parts)).toEqual(["cacheWrite", "cacheRead", "output", "input"])
    expect(parts[1]).toMatchObject({
      label: "Reading the cache",
      hint: "1,240,388 tokens",
    })
    expect(drawnTotal(parts)).toBeCloseTo(1)
  })

  it("leaves out a meter nothing was paid on", () => {
    const { parts } = costParts(
      stats({
        cost: { input: 0.092, output: 0.003, cacheRead: 0, cacheWrite: 0 },
      }),
    )

    expect(keys(parts)).toEqual(["input", "output"])
  })

  it("splits by model when the requests were priced whole", () => {
    const { by, parts } = costParts(
      stats({
        models: [
          { model: "claude-sonnet-5-5", requests: 44, cost: 5.600345 },
          { model: "claude-haiku-5-5", requests: 3, cost: 0.4 },
        ],
      }),
    )

    expect(by).toBe("model")
    expect(parts.map(({ label, hint }) => [label, hint])).toEqual([
      ["claude-sonnet-5-5", "44 requests"],
      ["claude-haiku-5-5", "3 requests"],
    ])
  })

  it("falls back to the models when every meter came to nothing", () => {
    const { by } = costParts(
      stats({
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        models: [{ model: "kimi", requests: 1, cost: 0.16 }],
      }),
    )

    expect(by).toBe("model")
  })

  it("has no parts when nothing was priced", () => {
    const { parts } = costParts(
      stats({
        models: [{ model: "claude-opus-5-5", requests: 3, cost: null }],
      }),
    )

    expect(parts).toEqual([])
  })
})

describe("toolRows", () => {
  const tool = (name: string, runningMs: number, calls = 1): ToolTally => ({
    tool: name,
    calls,
    runningMs,
  })

  it("shows the slowest tools first, and folds the rest into one row", () => {
    const tools = Array.from({ length: 11 }, (_, i) =>
      tool(`tool-${i}`, (i + 1) * s, 2),
    )
    const rows = toolRows(tools, 66 * s)

    expect(rows.shown).toHaveLength(TOOLS_SHOWN)
    expect(rows.shown[0]?.tool).toBe("tool-10")
    expect(rows.rest).toEqual({ tools: 6, calls: 12, runningMs: 21 * s })
  })

  it("does not fold a single tool into a row of its own", () => {
    const tools = Array.from({ length: TOOLS_SHOWN + 1 }, (_, i) =>
      tool(`tool-${i}`, s),
    )

    expect(toolRows(tools, 6 * s)).toMatchObject({ rest: null })
    expect(toolRows(tools, 6 * s).shown).toHaveLength(TOOLS_SHOWN + 1)
  })

  it("says when the tools' own times add up to more than the tools ran", () => {
    const tools = [
      tool("github_task", 8 * h + 29 * m, 17),
      tool("github_create_issue", 20 * m, 3),
    ]

    expect(toolRows(tools, 8 * h + 30 * m).overlap).toBe(true)
    expect(toolRows(tools, 8 * h + 49 * m).overlap).toBe(false)
  })
})
