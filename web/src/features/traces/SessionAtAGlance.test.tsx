import { render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it } from "vitest"

import { SessionAtAGlance } from "./SessionAtAGlance"
import type { SessionStats } from "./sessionStats"

const s = 1000
const m = 60 * s

function stats(overrides: Partial<SessionStats> = {}): SessionStats {
  return {
    turns: 4,
    lengthMs: 6 * m + 12 * s,
    requests: 31,
    toolCalls: 22,
    inSubagents: 0,
    tools: [
      { tool: "Bash", calls: 9, runningMs: 71 * s },
      { tool: "Read", calls: 7, runningMs: 4 * s },
      { tool: "Edit", calls: 5, runningMs: 2 * s },
      { tool: "Grep", calls: 1, runningMs: 1 * s },
    ],
    gates: { runs: 9, verdicts: { passed: 8, flagged: 1 } },
    judges: { calls: 0, cost: 0, priced: 0 },
    time: {
      modelMs: 221 * s,
      toolsMs: 78 * s,
      waitingMs: 52 * s,
      hooksMs: 9 * s,
    },
    activeMs: 6 * m,
    toolsInModelMs: 0,
    tokens: {
      input: 13_402,
      output: 17_610,
      cacheRead: 1_240_388,
      cacheWrite: 114_020,
    },
    cost: { input: 0.07, output: 0.44, cacheRead: 0.62, cacheWrite: 0.71 },
    cacheSaved: 5.58,
    pricedRequests: 31,
    models: [{ model: "claude-opus-5-5", requests: 31, cost: 1.84 }],
    ...overrides,
  }
}

// A cell or row of one of the panel's lists, by the name it leads with.
const row = (name: string) => {
  const item = screen.getByText(name).closest("li")
  if (!item) throw new Error(`no row named ${name}`)
  return item
}

describe("SessionAtAGlance", () => {
  it("leads with how long the session took and what it did, then a cell a part, idle last", () => {
    render(<SessionAtAGlance stats={stats()} cost={1.84} />)

    expect(
      screen.getByRole("heading", { name: "Took 6m 12s" }),
    ).toBeInTheDocument()
    expect(screen.getByText("4 turns · 22 tool calls")).toBeInTheDocument()
    expect(
      within(row("Model requests")).getByText("3m 41s"),
    ).toBeInTheDocument()
    expect(within(row("Idle")).getByText("between turns")).toBeInTheDocument()
    expect(within(row("Idle")).getByText("12.00 s")).toBeInTheDocument()
    // The bar shows the proportions; the cells give the times alone.
    expect(screen.queryByText("59%")).not.toBeInTheDocument()
    expect(screen.queryByText(/draws it shorter/)).not.toBeInTheDocument()
  })

  it("counts the tool calls subagents made", () => {
    render(<SessionAtAGlance stats={stats({ inSubagents: 5 })} cost={1.84} />)

    expect(
      screen.getByText("4 turns · 22 tool calls, 5 in subagents"),
    ).toBeInTheDocument()
  })

  it("keeps the tools closed until the cell is opened, by pointer or by keyboard", async () => {
    const user = userEvent.setup()
    render(<SessionAtAGlance stats={stats()} cost={1.84} />)
    const tools = screen.getByRole("button", { name: /^Running tools/ })

    expect(tools).toHaveAccessibleName(/4 tools/)
    expect(tools).toHaveAttribute("aria-expanded", "false")
    expect(screen.getByText("Bash")).not.toBeVisible()

    await user.click(tools)
    expect(tools).toHaveAttribute("aria-expanded", "true")
    expect(within(row("Bash")).getByText("9 calls")).toBeVisible()
    expect(within(row("Bash")).getByText("1m 11s")).toBeVisible()

    tools.focus()
    await user.keyboard("{Enter}")
    expect(tools).toHaveAttribute("aria-expanded", "false")
    expect(screen.getByText("Bash")).not.toBeVisible()
  })

  it("names a lone tool in its cell, with nothing to open", () => {
    render(
      <SessionAtAGlance
        stats={stats({ tools: [{ tool: "Read", calls: 4, runningMs: 900 }] })}
        cost={1.84}
      />,
    )

    expect(screen.queryByRole("button")).not.toBeInTheDocument()
    expect(
      within(row("Running tools")).getByText("Read, 4 calls"),
    ).toBeInTheDocument()
    // Layout jsdom cannot compute: in a half-width cell the tool gives way
    // before the label and the figure beside it, which only the classes show.
    expect(within(row("Running tools")).getByText("Read, 4 calls")).toHaveClass(
      "truncate",
    )
    expect(within(row("Running tools")).getByText("Running tools")).toHaveClass(
      "shrink-0",
    )
  })

  it("folds the tools past the slowest five into one row", async () => {
    const user = userEvent.setup()
    const tools = Array.from({ length: 11 }, (_, i) => ({
      tool: `mcp__tool_${i}`,
      calls: 2,
      runningMs: (11 - i) * s,
    }))
    render(<SessionAtAGlance stats={stats({ tools })} cost={1.84} />)

    await user.click(screen.getByRole("button", { name: /11 tools/ }))

    expect(within(row("6 more tools")).getByText("12 calls")).toBeVisible()
    expect(screen.getByText("mcp__tool_4")).toBeVisible()
    expect(screen.queryByText("mcp__tool_5")).not.toBeInTheDocument()
  })

  it("says when idle is drawn shorter than it lasted", () => {
    render(
      <SessionAtAGlance
        stats={stats({ lengthMs: 520_604 * s, activeMs: 32_038 * s })}
        cost={5.6}
      />,
    )

    expect(
      screen.getByRole("heading", { name: "Took 6d 01h" }),
    ).toBeInTheDocument()
    expect(
      screen.getByText(
        /^Idle is 94% of the session, so the bar draws it shorter/,
      ),
    ).toBeInTheDocument()
  })

  it("explains the lane tools get when they ran inside the model's requests", () => {
    render(
      <SessionAtAGlance
        stats={stats({
          lengthMs: 40 * s,
          activeMs: 40 * s,
          time: { modelMs: 40 * s, toolsMs: 20 * s, waitingMs: 0, hooksMs: 0 },
          toolsInModelMs: 20 * s,
        })}
        cost={0.16}
      />,
    )

    expect(
      within(row("Model requests")).getByText("40.00 s"),
    ).toBeInTheDocument()
    expect(screen.getByText(/get a lane of their own/)).toBeInTheDocument()
  })

  it("leads with the cost and what it counts, split by meter, and says what the cache saved", () => {
    render(<SessionAtAGlance stats={stats()} cost={1.84} />)

    expect(
      screen.getByRole("heading", { name: "Cost $1.84" }),
    ).toBeInTheDocument()
    expect(screen.getByText("31 requests · 1.39M tokens")).toBeInTheDocument()
    expect(
      within(row("Writing the cache")).getByText("$0.71"),
    ).toBeInTheDocument()
    expect(screen.queryByText("39%")).not.toBeInTheDocument()
    expect(screen.queryByText(/1,240,388/)).not.toBeInTheDocument()
    expect(screen.getByText(/saved about \$5\.58/)).toBeInTheDocument()
  })

  it("names no model when the requests were priced whole", () => {
    render(
      <SessionAtAGlance
        stats={stats({
          cost: null,
          cacheSaved: null,
          models: [
            { model: "claude-sonnet-5-5", requests: 44, cost: 5.600345 },
          ],
        })}
        cost={5.600345}
      />,
    )

    expect(
      screen.getByRole("heading", { name: "Cost $5.60" }),
    ).toBeInTheDocument()
    expect(screen.queryByText("claude-sonnet-5-5")).not.toBeInTheDocument()
    expect(screen.getByText(/priced whole/)).toBeInTheDocument()
  })

  it("says there is no price rather than a cost of nothing", () => {
    render(
      <SessionAtAGlance
        stats={stats({
          requests: 3,
          cost: null,
          cacheSaved: null,
          pricedRequests: 0,
          models: [{ model: "claude-opus-5-5", requests: 3, cost: null }],
        })}
        cost={0}
      />,
    )

    expect(screen.getByText("No price")).toBeInTheDocument()
    expect(screen.queryByText("$0.00")).not.toBeInTheDocument()
    expect(
      screen.getByText(/^None of the 3 requests ran on a model with a price/),
    ).toBeInTheDocument()
  })

  it("reads a session priced at $0 as priced, not as missing a price", () => {
    render(
      <SessionAtAGlance
        stats={stats({
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
          cacheSaved: 0,
          models: [{ model: "llama-local", requests: 31, cost: 0 }],
        })}
        cost={0}
      />,
    )

    expect(screen.getByText("$0.00")).toBeInTheDocument()
    expect(screen.queryByText("No price")).not.toBeInTheDocument()
  })

  it("says what the judge gates cost apart from the split, and how much of it is priced", () => {
    render(
      <SessionAtAGlance
        stats={stats({ judges: { calls: 3, cost: 0.01, priced: 1 } })}
        cost={1.85}
      />,
    )

    expect(
      screen.getByText(
        "Judge gates made 3 model calls, for $0.01 the split above leaves out, from the 1 call priced.",
      ),
    ).toBeInTheDocument()
  })

  it("keeps a part too short to round to 1% on the panel", () => {
    render(
      <SessionAtAGlance
        stats={stats({
          lengthMs: 5_360,
          activeMs: 5_107,
          time: { modelMs: 4_810, toolsMs: 105, waitingMs: 6, hooksMs: 182 },
        })}
        cost={0}
      />,
    )

    expect(within(row("Waiting on you")).getByText("6 ms")).toBeInTheDocument()
  })

  it("counts the gates the hooks ran by verdict", () => {
    render(<SessionAtAGlance stats={stats()} cost={1.84} />)

    expect(screen.getByText("ran in hooks").closest("p")).toHaveTextContent(
      "9 gates ran in hooks",
    )
    expect(screen.getByText("8 passed")).toBeInTheDocument()
    expect(screen.getByText("1 flagged")).toBeInTheDocument()
    expect(screen.queryByText(/blocked/)).not.toBeInTheDocument()
  })

  it("leaves the gates out of a session whose hooks ran none", () => {
    render(
      <SessionAtAGlance
        stats={stats({ gates: { runs: 0, verdicts: {} } })}
        cost={1.84}
      />,
    )

    expect(screen.queryByText(/ran in hooks/)).not.toBeInTheDocument()
  })
})
